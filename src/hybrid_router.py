# src/hybrid_router.py
"""
Hybrid Router with Anchor-Based Trajectory Learning
Implements semantic neighborhood prefetching and reinforcement learning.

Core Innovation:
- Anchor-based prediction graph (not just caching)
- Prefetches TRAJECTORIES, not just neighbors
- Reinforcement learning on prediction hits
- Multi-tier anchor hierarchy (WEAK → MEDIUM → STRONG → PERMANENT)

Author: Saberzerker
Date: 2025-11-16 (VISION IMPLEMENTATION)
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.anchor_system import AnchorSystem
from src.semantic_cache import SemanticClusterCache
from src.local_vdb import LocalVDB
from src.cloud_client import QdrantCloudClient
from src.metrics import MetricsTracker
from src.config import (
    LOCAL_CONFIDENCE_THRESHOLD,
    PREFETCH_ENABLED,
    PREFETCH_K,
    VERBOSE,
    DEFAULT_SEARCH_K,
)

logger = logging.getLogger(__name__)


class HybridRouter:
    """
    Central orchestrator implementing trajectory-based learning.

    Flow:
    1. Query → Check prediction match → Reinforce anchor if hit
    2. Search local (permanent + dynamic)
    3. Cloud fallback if needed
    4. Cache result neighborhood (not just single result)
    5. Create/strengthen anchor
    6. Generate 5 trajectory predictions
    7. Prefetch predictions from cloud
    8. Register predictions for future matching
    """

    def __init__(
        self,
        local_vdb: LocalVDB,
        cloud_vdb: QdrantCloudClient,
        semantic_cache: SemanticClusterCache,
        anchor_system: AnchorSystem,
        metrics: MetricsTracker,
    ):
        """Initialize hybrid router."""
        self.local_vdb = local_vdb
        self.cloud_vdb = cloud_vdb
        self.semantic_cache = semantic_cache
        self.anchor_system = anchor_system
        self.metrics = metrics

        self.query_count = 0

        logger.info("[ROUTER] ✅ Initialized with trajectory learning")
        logger.info("[ROUTER] Prefetch strategy: Semantic neighborhoods")

    def search(
        self,
        query_vector: np.ndarray,
        query_id: str,
        query_text: str = "",
        k: int = DEFAULT_SEARCH_K,
    ) -> Dict:
        """
        Main search with complete trajectory learning pipeline.

        Returns detailed results including learning metadata.
        """
        self.query_count += 1
        start_time = time.time()

        if VERBOSE:
            print(f"\n{'='*70}")
            print(f'[QUERY {query_id}] "{query_text[:60]}..."')
            print(f"{'='*70}")

        result = {
            "query_id": query_id,
            "query_text": query_text,
            "timestamp": time.time(),
            "k": k,
        }

        # ══════════════════════════════════════════════════════════════
        # STEP 1: Check Anchor Prediction Match
        # ══════════════════════════════════════════════════════════════

        prediction_match = self.anchor_system.check_prediction_match(
            query_vector, query_id
        )

        if prediction_match:
            pred_id, source_anchor_id, similarity = prediction_match

            # 🎯 REINFORCE: Prediction was correct!
            self.anchor_system.strengthen_anchor(source_anchor_id, reward=1.0)

            result["prediction_hit"] = True
            result["source_anchor_id"] = source_anchor_id
            result["prediction_similarity"] = similarity

            logger.info(
                f"[ROUTER] 🎯 PREDICTION HIT! Anchor #{source_anchor_id} "
                f"predicted correctly (sim={similarity:.3f})"
            )
        else:
            result["prediction_hit"] = False
            logger.debug(f"[ROUTER] No prediction match")

        # ══════════════════════════════════════════════════════════════
        # STEP 2: Add to Semantic Cluster
        # ══════════════════════════════════════════════════════════════

        cluster_id, cluster_action, access_count = self.semantic_cache.add_query(
            query_vector, query_id
        )

        result["cluster_id"] = cluster_id
        result["cluster_action"] = cluster_action
        result["cluster_access_count"] = access_count

        logger.debug(
            f"[ROUTER] Cluster #{cluster_id} | {cluster_action} | "
            f"Access: {access_count}"
        )

        # ══════════════════════════════════════════════════════════════
        # STEP 3: Search Local VDB (Permanent + Dynamic)
        # ══════════════════════════════════════════════════════════════

        local_start = time.time()
        local_ids, local_scores = self.local_vdb.search(query_vector, k)
        local_latency_ms = (time.time() - local_start) * 1000

        result["local_latency_ms"] = local_latency_ms
        result["local_results_count"] = len(local_ids)

        # ══════════════════════════════════════════════════════════════
        # STEP 4: Check Local Confidence
        # ══════════════════════════════════════════════════════════════

        if local_ids and local_scores and local_scores[0] >= LOCAL_CONFIDENCE_THRESHOLD:
            # ✅ LOCAL HIT - High confidence!
            total_time = (time.time() - start_time) * 1000

            result.update(
                {
                    "ids": local_ids,
                    "scores": local_scores,
                    "source": "local",
                    "latency_ms": total_time,
                    "cloud_latency_ms": 0,
                    "confidence": local_scores[0],
                }
            )

            self.metrics.log_event("local_hit", latency=total_time)

            logger.info(
                f"[ROUTER] ✅ LOCAL HIT (confidence={local_scores[0]:.3f}, "
                f"{total_time:.1f}ms)"
            )

            # STILL do trajectory learning (prefetch next steps)
            if PREFETCH_ENABLED:
                self._learn_trajectory_and_prefetch(
                    query_vector,
                    query_id,
                    query_text,
                    result,
                    cluster_id,
                    prediction_match,
                )

            return result

        # ══════════════════════════════════════════════════════════════
        # STEP 5: Cloud Fallback
        # ══════════════════════════════════════════════════════════════

        logger.info(f"[ROUTER] ⬆️  Cloud fallback (local confidence too low)")

        try:
            cloud_start = time.time()
            cloud_ids, cloud_scores, cloud_latency_ms = self.cloud_vdb.search(
                query_vector, k
            )

            total_time = (time.time() - start_time) * 1000

            result.update(
                {
                    "ids": cloud_ids,
                    "scores": cloud_scores,
                    "source": "cloud",
                    "latency_ms": total_time,
                    "cloud_latency_ms": cloud_latency_ms,
                    "confidence": cloud_scores[0] if cloud_scores else 0.0,
                }
            )

            self.metrics.log_event("cloud_hit", latency=total_time)

            logger.info(
                f"[ROUTER] ☁️  CLOUD HIT (confidence={cloud_scores[0]:.3f}, "
                f"{cloud_latency_ms:.1f}ms)"
            )

            # ══════════════════════════════════════════════════════════════
            # STEP 6: Cache NEIGHBORHOOD (not just single result)
            # ══════════════════════════════════════════════════════════════

            if cloud_ids and cloud_scores:
                cached_count = self._cache_neighborhood(
                    cloud_ids[:3],  # Top 3 results form neighborhood
                    cloud_scores[:3],
                    query_vector,
                    cluster_id,
                )
                result["cached_to_local"] = cached_count
                logger.info(f"[ROUTER] 💾 Cached {cached_count} neighborhood vectors")

            # ══════════════════════════════════════════════════════════════
            # STEP 7: Learn Trajectory & Prefetch
            # ══════════════════════════════════════════════════════════════

            if PREFETCH_ENABLED:
                self._learn_trajectory_and_prefetch(
                    query_vector,
                    query_id,
                    query_text,
                    result,
                    cluster_id,
                    prediction_match,
                )

            return result

        except Exception as e:
            # ══════════════════════════════════════════════════════════════
            # STEP 8: Offline Graceful Degradation
            # ══════════════════════════════════════════════════════════════

            logger.error(f"[ROUTER] ❌ Cloud error: {e}")
            import traceback

            traceback.print_exc()

            total_time = (time.time() - start_time) * 1000

            result.update(
                {
                    "ids": local_ids if local_ids else [],
                    "scores": local_scores if local_scores else [],
                    "source": "offline_fallback",
                    "latency_ms": total_time,
                    "cloud_latency_ms": 0,
                    "error": str(e),
                    "confidence": local_scores[0] if local_scores else 0.0,
                }
            )

            self.metrics.log_event("offline_fallback", latency=total_time)

            logger.warning(f"[ROUTER] ⚠️  OFFLINE FALLBACK (best local result)")

            return result

    def _cache_neighborhood(
        self,
        cloud_ids: List[str],
        cloud_scores: List[float],
        query_vector: np.ndarray,
        cluster_id: int,
    ) -> int:
        """
        Cache SEMANTIC NEIGHBORHOOD (top-k results), not just single result.

        This implements your vision: "cache the entire neighborhood"
        """
        cached = 0

        for doc_id, score in zip(cloud_ids, cloud_scores):
            try:
                # Fetch full vector from cloud
                vectors = self.cloud_vdb.get_vectors_by_ids([doc_id])

                if vectors and len(vectors) > 0:
                    vector = vectors[0]

                    # Insert into local dynamic layer
                    self.local_vdb.insert_vector(
                        vectors=vector.reshape(1, -1),
                        ids=[doc_id],
                        layer="dynamic",
                        anchor_id=None,
                        metadata={
                            "source": "cloud_neighborhood",
                            "cached_at": time.time(),
                            "cluster_id": cluster_id,
                            "score": score,
                            "reason": "semantic_neighbor",
                        },
                    )

                    cached += 1
                    logger.debug(
                        f"[CACHE] Added neighborhood vector {doc_id} "
                        f"(score={score:.3f})"
                    )

            except Exception as e:
                logger.warning(f"[CACHE] Failed to cache {doc_id}: {e}")
                continue

        return cached

    def _learn_trajectory_and_prefetch(
        self,
        query_vector: np.ndarray,
        query_id: str,
        query_text: str,
        result: Dict,
        cluster_id: int,
        prediction_match: Optional[Tuple],
    ):
        """
        CORE INNOVATION: Learn trajectory and prefetch predicted next steps.

        Your vision:
        1. Create anchor for current query
        2. Generate 5 trajectory predictions (where user will go next)
        3. Prefetch predictions from cloud
        4. Register predictions for future matching
        5. Build anchor graph (parent-child relationships)

        This is NOT just "cache nearest neighbors"
        This is "predict and prefetch the trajectory"
        """
        logger.info(f"[LEARNING] {'─'*60}")
        logger.info(f"[LEARNING] Building trajectory from Query {query_id}...")

        # ══════════════════════════════════════════════════════════════
        # Create Anchor (with parent linkage if prediction hit)
        # ══════════════════════════════════════════════════════════════

        parent_anchor_id = prediction_match[1] if prediction_match else None

        anchor_id = self.anchor_system.create_anchor(
            query_vector=query_vector,
            query_id=query_id,
            query_text=query_text,
            parent_anchor_id=parent_anchor_id,
        )

        result["anchor_id"] = anchor_id
        result["anchor_created"] = True

        if parent_anchor_id:
            logger.info(
                f"[LEARNING] Created Anchor #{anchor_id} "
                f"(child of Anchor #{parent_anchor_id})"
            )
        else:
            logger.info(f"[LEARNING] Created Anchor #{anchor_id} (root)")

        # ══════════════════════════════════════════════════════════════
        # Generate Trajectory Predictions
        # ══════════════════════════════════════════════════════════════

        # Get cluster centroid to guide trajectory
        centroid = self.semantic_cache.get_centroid(cluster_id)

        prediction_vectors = self.anchor_system.generate_predictions(
            anchor_id=anchor_id, centroid=centroid
        )

        if centroid is not None:
            strategy = "trajectory-guided (toward cluster centroid)"
        else:
            strategy = "random-walk (no cluster yet)"

        logger.info(f"[LEARNING] Generated {len(prediction_vectors)} predictions")
        logger.info(f"[LEARNING] Strategy: {strategy}")

        # ══════════════════════════════════════════════════════════════
        # Prefetch Predictions from Cloud
        # ══════════════════════════════════════════════════════════════

        prefetched = 0
        prefetch_start = time.time()

        for i, pred_vec in enumerate(prediction_vectors):
            try:
                # Search cloud for this predicted vector
                pred_ids, pred_scores, _ = self.cloud_vdb.search(pred_vec, k=1)

                if not pred_ids or not pred_scores:
                    continue

                doc_id = pred_ids[0]
                score = pred_scores[0]

                # Fetch full vector
                vectors = self.cloud_vdb.get_vectors_by_ids([doc_id])

                if vectors and len(vectors) > 0:
                    vector = vectors[0]

                    # Cache to local dynamic layer
                    self.local_vdb.insert_vector(
                        vectors=vector.reshape(1, -1),
                        ids=[doc_id],
                        layer="dynamic",
                        anchor_id=anchor_id,
                        metadata={
                            "source": "trajectory_prefetch",
                            "prediction_index": i,
                            "cluster_id": cluster_id,
                            "prefetched_at": time.time(),
                            "score": score,
                            "parent_anchor": anchor_id,
                        },
                    )

                    # Register prediction for future matching
                    self.anchor_system.register_prediction(pred_vec, anchor_id)

                    prefetched += 1
                    logger.debug(
                        f"[PREFETCH] {i+1}/{len(prediction_vectors)}: "
                        f"doc={doc_id[:12]}... (score={score:.3f})"
                    )

            except Exception as e:
                logger.warning(f"[PREFETCH] Prediction {i} failed: {e}")
                continue

        prefetch_time = (time.time() - prefetch_start) * 1000

        result["prefetch_count"] = prefetched
        result["prefetch_total"] = len(prediction_vectors)
        result["prefetch_time_ms"] = prefetch_time

        # ══════════════════════════════════════════════════════════════
        # Update Cache Stats
        # ══════════════════════════════════════════════════════════════

        stats = self.local_vdb.get_stats()

        logger.info(
            f"[PREFETCH] ✅ Prefetched {prefetched}/{len(prediction_vectors)} "
            f"trajectory vectors ({prefetch_time:.1f}ms)"
        )
        logger.info(
            f"[CACHE] Local: {stats['total_vectors']} vectors "
            f"(perm={stats['permanent_layer_vectors']}, "
            f"dyn={stats['dynamic_layer_vectors']})"
        )

        # Show anchor graph
        anchor_stats = self.anchor_system.get_anchor_stats()
        logger.info(
            f"[ANCHORS] Total: {anchor_stats['total_anchors']}, "
            f"Active predictions: {anchor_stats['active_predictions']}"
        )
        logger.info(f"[LEARNING] {'─'*60}")

        self.metrics.log_event("prefetch_operation", count=prefetched)

    def get_stats(self) -> Dict:
        """Get comprehensive router statistics."""
        return {
            "total_queries": self.query_count,
            "local_vdb": self.local_vdb.get_stats(),
            "semantic_cache": self.semantic_cache.get_stats(),
            "anchor_system": self.anchor_system.get_anchor_stats(),
            "metrics": self.metrics.get_summary(),
        }
