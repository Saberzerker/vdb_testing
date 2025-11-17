# src/storage_engine.py
"""
Storage Engine with Fixed-Size Dynamic Layer and Weight Tracking

Think of this as the BACKPACK that holds 700 cookies (vectors).

Key Features:
- Fixed 700 capacity (doesn't grow)
- Tracks "stars" (weights) for each cookie (vector)
- Can check "do I already have this cookie?" (neighborhood search)
- Can evict weakest cookies when full

Author: Saberzerker
Date: 2025-11-17 00:00 UTC
"""

import faiss
import numpy as np
import os
import glob
import json
import pickle
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from src.config import (
    BASE_LAYER_PATH,
    DYNAMIC_LAYER_PATH,
    VECTOR_DIMENSION,
    DYNAMIC_LAYER_CAPACITY,
    HOT_PARTITION_RAM_LIMIT,
)

logger = logging.getLogger(__name__)


class StorageEngine:
    """
    Two-tier storage with fixed-size dynamic layer.

    TIER 1 (Permanent):
    - 300 vectors (read-only)
    - Like kitchen cookies (always there)

    TIER 2 (Dynamic):
    - 700 vectors (read-write, FIXED SIZE)
    - Like backpack cookies (smart swapping)
    """

    def __init__(self, config):
        """Initialize two-tier storage."""
        self.config = config
        self.dimension = config.VECTOR_DIMENSION

        # Paths
        self.base_path = Path(config.BASE_LAYER_PATH)
        self.dynamic_path = Path(config.DYNAMIC_LAYER_PATH)

        self.base_path.mkdir(parents=True, exist_ok=True)
        self.dynamic_path.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self.lock = threading.RLock()

        # ═══════════════════════════════════════════════════════════
        # TIER 1: PERMANENT LAYER (Kitchen)
        # ═══════════════════════════════════════════════════════════

        self.permanent_partitions = []  # List of FAISS indexes
        self.permanent_metadata = {}  # {id: metadata}

        # ═══════════════════════════════════════════════════════════
        # TIER 2: DYNAMIC LAYER (Backpack with 700 cookie limit)
        # ═══════════════════════════════════════════════════════════

        # Hot partition (in-memory)
        self.dynamic_index = faiss.IndexFlatL2(self.dimension)
        self.dynamic_ids = []  # List of vector IDs
        self.dynamic_metadata = {}  # {id: {weight, timestamp, ...}}
        self.dynamic_vectors_cache = {}  # {id: vector} for quick access

        self.deleted_ids = set()

        # FIXED CAPACITY
        self.dynamic_capacity = DYNAMIC_LAYER_CAPACITY  # 700

        # Load existing data
        self._load_permanent_layer()
        self._load_dynamic_layer()

        logger.info(f"[STORAGE] Initialized")
        logger.info(f"[STORAGE] TIER 1 (Permanent): {self._count_permanent()} vectors")
        logger.info(
            f"[STORAGE] TIER 2 (Dynamic): {self._count_dynamic()}/{self.dynamic_capacity} vectors"
        )

    # ═══════════════════════════════════════════════════════════
    # TIER 1: PERMANENT LAYER (Kitchen Cookies)
    # ═══════════════════════════════════════════════════════════

    def _load_permanent_layer(self):
        """Load permanent layer from disk."""
        if not self.base_path.exists():
            logger.warning(f"[STORAGE] Permanent path doesn't exist: {self.base_path}")
            return

        partition_files = sorted(glob.glob(str(self.base_path / "partition_*.index")))

        for pfile in partition_files:
            try:
                index = faiss.read_index(pfile)
                self.permanent_partitions.append(
                    {"index": index, "file": pfile, "nvectors": index.ntotal}
                )
                logger.info(
                    f"[STORAGE] Loaded permanent: {pfile} ({index.ntotal} vectors)"
                )
            except Exception as e:
                logger.error(f"[STORAGE] Failed to load {pfile}: {e}")

        # Load metadata
        metadata_file = self.base_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.permanent_metadata = json.load(f)

    def _count_permanent(self) -> int:
        """Count vectors in permanent layer."""
        return sum(p["nvectors"] for p in self.permanent_partitions)

    def search_permanent(
        self, query_vector: np.ndarray, k: int
    ) -> Tuple[List[str], List[float]]:
        """Search only permanent layer."""
        with self.lock:
            all_ids = []
            all_distances = []

            query_vec = query_vector.reshape(1, -1).astype("float32")

            for partition in self.permanent_partitions:
                if partition["index"].ntotal == 0:
                    continue

                D, I = partition["index"].search(
                    query_vec, min(k, partition["index"].ntotal)
                )

                for local_idx, dist in zip(I[0], D[0]):
                    if local_idx != -1 and dist < float("inf"):
                        vec_id = self._get_permanent_id(partition, local_idx)
                        if vec_id:
                            all_ids.append(vec_id)
                            all_distances.append(dist)

            # Sort and return top-k
            combined = list(zip(all_ids, all_distances))
            combined.sort(key=lambda x: x[1])

            top_k = combined[:k]
            return [vid for vid, _ in top_k], [d for _, d in top_k]

    def _get_permanent_id(self, partition, local_idx) -> Optional[str]:
        """Get vector ID from permanent metadata."""
        for vid, meta in self.permanent_metadata.items():
            if (
                meta.get("partition_file") == partition["file"]
                and meta.get("local_idx") == local_idx
            ):
                return vid
        return None

    # ═══════════════════════════════════════════════════════════
    # TIER 2: DYNAMIC LAYER (Backpack - 700 Cookie Limit!)
    # ═══════════════════════════════════════════════════════════

    def _load_dynamic_layer(self):
        """Load dynamic layer from disk."""
        index_file = self.dynamic_path / "dynamic.index"
        ids_file = self.dynamic_path / "dynamic_ids.pkl"
        metadata_file = self.dynamic_path / "dynamic_metadata.pkl"

        if index_file.exists():
            try:
                self.dynamic_index = faiss.read_index(str(index_file))

                with open(ids_file, "rb") as f:
                    self.dynamic_ids = pickle.load(f)

                with open(metadata_file, "rb") as f:
                    self.dynamic_metadata = pickle.load(f)

                logger.info(
                    f"[STORAGE] Loaded dynamic: {len(self.dynamic_ids)} vectors"
                )
            except Exception as e:
                logger.error(f"[STORAGE] Failed to load dynamic: {e}")

    def _count_dynamic(self) -> int:
        """Count vectors in dynamic layer."""
        return len(self.dynamic_ids) - len(self.deleted_ids)

    def is_dynamic_full(self) -> bool:
        """Check if dynamic layer is at capacity."""
        return self._count_dynamic() >= self.dynamic_capacity

    def has_dynamic_space(self, n: int) -> bool:
        """Check if dynamic has space for n vectors."""
        return self._count_dynamic() + n <= self.dynamic_capacity

    def insert_dynamic(
        self, vectors: np.ndarray, ids: List[str], metadata: Optional[Dict] = None
    ):
        """
        Insert vectors into dynamic layer.

        Like adding cookies to backpack!
        If backpack is full, reject (caller must evict first).
        """
        with self.lock:
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            n_vectors = vectors.shape[0]

            # Check capacity (CRITICAL!)
            if not self.has_dynamic_space(n_vectors):
                raise ValueError(
                    f"Dynamic layer full! "
                    f"({self._count_dynamic()}/{self.dynamic_capacity}). "
                    f"Must evict {n_vectors} vectors first."
                )

            # Add to FAISS index
            self.dynamic_index.add(vectors.astype("float32"))

            # Track IDs and metadata
            for i, vid in enumerate(ids):
                self.dynamic_ids.append(vid)

                # Store vector for quick retrieval
                self.dynamic_vectors_cache[vid] = vectors[i]

                # Store metadata with weight (STARS!)
                meta = metadata.copy() if metadata else {}
                meta.update(
                    {
                        "weight": meta.get("weight", 1.0),  # ⭐ Default 1 star
                        "inserted_at": time.time(),
                        "local_idx": len(self.dynamic_ids) - 1,
                    }
                )
                self.dynamic_metadata[vid] = meta

            logger.debug(
                f"[STORAGE] Inserted {n_vectors} vectors to dynamic "
                f"({self._count_dynamic()}/{self.dynamic_capacity})"
            )

    def delete_dynamic(self, vec_id: str):
        """
        Mark vector as deleted (tombstone).

        Like removing a cookie from backpack.
        """
        with self.lock:
            if vec_id in self.dynamic_metadata:
                self.deleted_ids.add(vec_id)
                logger.debug(f"[STORAGE] Deleted {vec_id} from dynamic")

    def search_dynamic(
        self, query_vector: np.ndarray, k: int
    ) -> Tuple[List[str], List[float]]:
        """
        Search only dynamic layer.

        Like checking backpack for cookies!
        """
        with self.lock:
            if self.dynamic_index.ntotal == 0:
                return [], []

            query_vec = query_vector.reshape(1, -1).astype("float32")

            # Search
            D, I = self.dynamic_index.search(
                query_vec, min(k, self.dynamic_index.ntotal)
            )

            ids = []
            distances = []

            for local_idx, dist in zip(I[0], D[0]):
                if (
                    local_idx != -1
                    and dist < float("inf")
                    and local_idx < len(self.dynamic_ids)
                ):
                    vec_id = self.dynamic_ids[local_idx]

                    # Skip deleted
                    if vec_id not in self.deleted_ids:
                        ids.append(vec_id)
                        distances.append(dist)

            return ids[:k], distances[:k]

    def exists_in_dynamic_neighborhood(
        self, query_vector: np.ndarray, threshold: float = 0.90
    ) -> bool:
        """
        CHECK: Do I already have a cookie like this in backpack?

        This is the SMART part!
        Before fetching from bakery (cloud), check backpack first.

        Returns:
            True if similar vector exists (SKIP fetching!)
            False if new vector (GO fetch!)
        """
        ids, distances = self.search_dynamic(query_vector, k=1)

        if ids and distances:
            # Convert L2 distance to similarity
            similarity = 1.0 / (1.0 + distances[0])

            if similarity >= threshold:
                logger.debug(
                    f"[SMART CHECK] ✅ Found similar vector "
                    f"(sim={similarity:.3f} >= {threshold})"
                )
                return True  # Already have it!

        logger.debug(f"[SMART CHECK] ❌ No similar vector, need to fetch")
        return False  # Need to fetch

    def get_weakest_dynamic_vector(self) -> Optional[str]:
        """
        Find cookie with LEAST stars (weakest).

        Used for eviction when backpack is full.
        """
        with self.lock:
            if not self.dynamic_metadata:
                return None

            # Find vector with lowest weight
            weakest_id = None
            min_weight = float("inf")

            for vid, meta in self.dynamic_metadata.items():
                if vid in self.deleted_ids:
                    continue

                weight = meta.get("weight", 1.0)

                if weight < min_weight:
                    min_weight = weight
                    weakest_id = vid

            if weakest_id:
                logger.debug(
                    f"[EVICTION] Weakest vector: {weakest_id} (weight={min_weight})"
                )

            return weakest_id

    def update_dynamic_weight(self, vec_id: str, delta: float):
        """
        Update cookie stars!

        When prediction hits: delta = +1.0 (add star ⭐)
        When prediction misses: delta = -0.5 (remove half star)
        """
        with self.lock:
            if vec_id in self.dynamic_metadata:
                old_weight = self.dynamic_metadata[vec_id].get("weight", 1.0)
                new_weight = max(0.1, old_weight + delta)  # Min 0.1 stars

                self.dynamic_metadata[vec_id]["weight"] = new_weight

                logger.debug(f"[WEIGHT] {vec_id}: {old_weight:.1f} → {new_weight:.1f}")

    def get_dynamic_stats(self) -> Dict:
        """Get dynamic layer statistics."""
        with self.lock:
            if not self.dynamic_metadata:
                return {
                    "current_size": 0,
                    "capacity": self.dynamic_capacity,
                    "fill_rate": 0.0,
                    "avg_weight": 0.0,
                    "deleted_count": 0,
                }

            active_weights = [
                meta.get("weight", 1.0)
                for vid, meta in self.dynamic_metadata.items()
                if vid not in self.deleted_ids
            ]

            current_size = self._count_dynamic()

            return {
                "current_size": current_size,
                "capacity": self.dynamic_capacity,
                "fill_rate": current_size / self.dynamic_capacity * 100,
                "avg_weight": np.mean(active_weights) if active_weights else 0.0,
                "max_weight": np.max(active_weights) if active_weights else 0.0,
                "min_weight": np.min(active_weights) if active_weights else 0.0,
                "deleted_count": len(self.deleted_ids),
            }

    def save_dynamic_state(self):
        """Save dynamic layer to disk."""
        with self.lock:
            try:
                # Save index
                faiss.write_index(
                    self.dynamic_index, str(self.dynamic_path / "dynamic.index")
                )

                # Save IDs
                with open(self.dynamic_path / "dynamic_ids.pkl", "wb") as f:
                    pickle.dump(self.dynamic_ids, f)

                # Save metadata
                with open(self.dynamic_path / "dynamic_metadata.pkl", "wb") as f:
                    pickle.dump(self.dynamic_metadata, f)

                logger.info(
                    f"[STORAGE] Saved dynamic state ({self._count_dynamic()} vectors)"
                )

            except Exception as e:
                logger.error(f"[STORAGE] Failed to save dynamic: {e}")

    def get_stats(self) -> Dict:
        """Get complete storage statistics."""
        return {
            "permanent_vectors": self._count_permanent(),
            "dynamic_vectors": self._count_dynamic(),
            "dynamic_capacity": self.dynamic_capacity,
            "dynamic_fill_rate": self._count_dynamic() / self.dynamic_capacity * 100,
            "deleted_count": len(self.deleted_ids),
        }
