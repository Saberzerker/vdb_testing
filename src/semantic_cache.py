# src/semantic_cache.py
"""
Semantic Cluster Cache with Momentum-Based Context Management.

Instead of caching individual queries, we cache SEMANTIC REGIONS:
- Cluster centroids represent active topics/contexts
- Momentum-based updates prevent false context shifts
- Drift detection creates new clusters when topics change
- Cluster-level eviction maintains semantic coherence

Author: Saberzerker
Date: 2025-11-16
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple

from src.config import (
    SEMANTIC_DRIFT_THRESHOLD,
    MOMENTUM_ALPHA,
    MAX_CLUSTERS,
    CLUSTER_TTL_SECONDS,
    MIN_CLUSTER_REINFORCEMENT_SCORE
)

logger = logging.getLogger(__name__)


class SemanticClusterCache:
    """
    Manages semantic clusters for context-aware caching.
    
    Key Innovation:
    - Clusters represent semantic regions, not individual queries
    - Centroids updated with momentum to resist noise
    - New clusters created only on genuine semantic drift
    - Cluster eviction preserves semantic neighborhoods
    """
    
    def __init__(self, drift_threshold: float = None, momentum_alpha: float = None, 
                 max_clusters: int = None):
        """
        Initialize semantic cache.
        
        Args:
            drift_threshold: Cosine distance threshold for creating new clusters
            momentum_alpha: Weight for old centroid in momentum update (0-1)
            max_clusters: Maximum number of clusters to maintain
        """
        self.drift_threshold = drift_threshold or SEMANTIC_DRIFT_THRESHOLD
        self.momentum_alpha = momentum_alpha or MOMENTUM_ALPHA
        self.max_clusters = max_clusters or MAX_CLUSTERS
        
        self.clusters = {}  # {cluster_id: cluster_metadata}
        self.cluster_counter = 0
        self.query_to_cluster = {}  # {query_id: cluster_id}
        
        # Metrics
        self.metrics = {
            "clusters_created": 0,
            "clusters_merged": 0,
            "clusters_evicted": 0,
            "drift_events": 0,
            "reinforcements": 0
        }
        
        logger.info(f"[SEMANTIC CACHE] Initialized (drift_threshold={self.drift_threshold}, "
                   f"momentum_alpha={self.momentum_alpha})")
    
    def add_query(self, query_vector: np.ndarray, query_id: str) -> Tuple[int, str, int]:
        """
        Add query to semantic cache with momentum-based clustering.
        
        Args:
            query_vector: Query embedding (normalized)
            query_id: Query identifier
        
        Returns:
            Tuple of (cluster_id, action, access_count)
            - cluster_id: Assigned cluster ID
            - action: "new_cluster" | "reinforced" | "drift_detected"
            - access_count: Current cluster access count
        """
        # Find nearest existing cluster
        nearest_cluster = self._find_nearest_cluster(query_vector)
        
        if nearest_cluster is None:
            # No clusters yet → create first cluster
            cluster_id = self._create_cluster(query_vector, query_id)
            return cluster_id, "new_cluster", 1
        
        cluster_id, distance = nearest_cluster
        
        if distance < self.drift_threshold:
            # Query fits in existing cluster → reinforce
            self._reinforce_cluster(cluster_id, query_vector, query_id)
            access_count = self.clusters[cluster_id]["access_count"]
            return cluster_id, "reinforced", access_count
        
        else:
            # Semantic drift detected → create new cluster
            self.metrics["drift_events"] += 1
            logger.debug(f"[SEMANTIC] Drift detected (distance={distance:.3f} > {self.drift_threshold})")
            
            # Check if we're at max capacity
            if len(self.clusters) >= self.max_clusters:
                # Evict least-used cluster to make room
                self._evict_lru_cluster()
            
            cluster_id = self._create_cluster(query_vector, query_id)
            return cluster_id, "drift_detected", 1
    
    def _find_nearest_cluster(self, query_vector: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Find the nearest cluster centroid to query vector.
        
        Args:
            query_vector: Query embedding
        
        Returns:
            (cluster_id, distance) or None if no clusters exist
        """
        if not self.clusters:
            return None
        
        min_distance = float('inf')
        nearest_id = None
        
        for cluster_id, cluster in self.clusters.items():
            centroid = cluster["centroid"]
            
            # Calculate cosine distance (1 - cosine_similarity)
            similarity = np.dot(query_vector, centroid) / (
                np.linalg.norm(query_vector) * np.linalg.norm(centroid)
            )
            distance = 1 - similarity
            
            if distance < min_distance:
                min_distance = distance
                nearest_id = cluster_id
        
        return (nearest_id, min_distance)
    
    def _create_cluster(self, query_vector: np.ndarray, query_id: str) -> int:
        """
        Create a new semantic cluster.
        
        Args:
            query_vector: Initial centroid
            query_id: First query in cluster
        
        Returns:
            New cluster ID
        """
        cluster_id = self.cluster_counter
        self.cluster_counter += 1
        
        self.clusters[cluster_id] = {
            "id": cluster_id,
            "centroid": query_vector.copy(),
            "created_at": time.time(),
            "last_access": time.time(),
            "access_count": 1,
            "queries": [query_id],
            "total_weight": 1.0
        }
        
        self.query_to_cluster[query_id] = cluster_id
        self.metrics["clusters_created"] += 1
        
        logger.info(f"[SEMANTIC] Created cluster {cluster_id} (total: {len(self.clusters)})")
        
        return cluster_id
    
    def _reinforce_cluster(self, cluster_id: int, query_vector: np.ndarray, query_id: str):
        """
        Reinforce existing cluster with momentum-based centroid update.
        
        Momentum Formula:
            new_centroid = α * old_centroid + (1 - α) * query_vector
        
        Where α (momentum_alpha) is typically 0.9, meaning:
        - 90% weight to existing centroid (stability)
        - 10% weight to new query (adaptation)
        
        This prevents noise from causing false context shifts while
        still allowing gradual adaptation to genuine topic changes.
        
        Args:
            cluster_id: Cluster to reinforce
            query_vector: New query embedding
            query_id: Query identifier
        """
        cluster = self.clusters[cluster_id]
        
        # Momentum-based centroid update
        old_centroid = cluster["centroid"]
        new_centroid = (
            self.momentum_alpha * old_centroid + 
            (1 - self.momentum_alpha) * query_vector
        )
        
        # Normalize to maintain unit length
        new_centroid = new_centroid / np.linalg.norm(new_centroid)
        
        # Update cluster
        cluster["centroid"] = new_centroid
        cluster["last_access"] = time.time()
        cluster["access_count"] += 1
        cluster["queries"].append(query_id)
        cluster["total_weight"] += 1.0
        
        self.query_to_cluster[query_id] = cluster_id
        self.metrics["reinforcements"] += 1
        
        logger.debug(f"[SEMANTIC] Reinforced cluster {cluster_id} "
                    f"(access_count={cluster['access_count']})")
    
    def get_centroid(self, cluster_id: int) -> Optional[np.ndarray]:
        """
        Get the centroid vector for a cluster.
        
        This is used by the anchor system to guide prediction trajectory:
        predictions are generated along the path from query toward centroid.
        
        Args:
            cluster_id: Cluster identifier
        
        Returns:
            Centroid vector if cluster exists, else None
        """
        if cluster_id in self.clusters:
            return self.clusters[cluster_id]["centroid"]
        return None
    
    def get_cluster_for_query(self, query_id: str) -> Optional[int]:
        """
        Get the cluster ID that a query belongs to.
        
        Args:
            query_id: Query identifier
        
        Returns:
            Cluster ID or None if query not found
        """
        return self.query_to_cluster.get(query_id)
    
    def _evict_lru_cluster(self):
        """
        Evict the least-recently-used cluster to make room for new clusters.
        
        This is called when we hit max_clusters limit.
        """
        if not self.clusters:
            return
        
        # Find cluster with oldest last_access time
        lru_id = min(
            self.clusters.keys(),
            key=lambda cid: self.clusters[cid]["last_access"]
        )
        
        cluster = self.clusters[lru_id]
        age = time.time() - cluster["created_at"]
        idle = time.time() - cluster["last_access"]
        
        logger.info(f"[SEMANTIC] ❌ Evicting LRU cluster {lru_id} "
                   f"(age={age:.0f}s, idle={idle:.0f}s, accesses={cluster['access_count']})")
        
        # Remove cluster and query mappings
        for query_id in cluster["queries"]:
            if query_id in self.query_to_cluster:
                del self.query_to_cluster[query_id]
        
        del self.clusters[lru_id]
        self.metrics["clusters_evicted"] += 1
    
    def evict_stale_clusters(self) -> List[int]:
        """
        Evict clusters that haven't been accessed recently.
        
        This is called periodically by the scheduler to clean up old clusters.
        
        Returns:
            List of evicted cluster IDs
        """
        current_time = time.time()
        to_evict = []
        
        for cluster_id, cluster in self.clusters.items():
            idle_time = current_time - cluster["last_access"]
            
            if idle_time > CLUSTER_TTL_SECONDS:
                to_evict.append(cluster_id)
        
        # Evict identified clusters
        for cluster_id in to_evict:
            cluster = self.clusters[cluster_id]
            logger.info(f"[SEMANTIC] ❌ Evicting stale cluster {cluster_id} "
                       f"(idle={idle_time:.0f}s, TTL={CLUSTER_TTL_SECONDS}s)")
            
            # Remove query mappings
            for query_id in cluster["queries"]:
                if query_id in self.query_to_cluster:
                    del self.query_to_cluster[query_id]
            
            del self.clusters[cluster_id]
            self.metrics["clusters_evicted"] += 1
        
        if to_evict:
            logger.info(f"[SEMANTIC] Evicted {len(to_evict)} stale clusters")
        
        return to_evict
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive statistics about semantic cache state.
        
        Returns:
            Dict with cluster counts, access patterns, metrics
        """
        if not self.clusters:
            return {
                "total_clusters": 0,
                "total_queries_tracked": 0,
                **self.metrics
            }
        
        access_counts = [c["access_count"] for c in self.clusters.values()]
        ages = [time.time() - c["created_at"] for c in self.clusters.values()]
        idle_times = [time.time() - c["last_access"] for c in self.clusters.values()]
        
        return {
            "total_clusters": len(self.clusters),
            "total_queries_tracked": len(self.query_to_cluster),
            "avg_access_count": np.mean(access_counts),
            "max_access_count": np.max(access_counts),
            "avg_cluster_age_seconds": np.mean(ages),
            "avg_idle_time_seconds": np.mean(idle_times),
            **self.metrics
        }