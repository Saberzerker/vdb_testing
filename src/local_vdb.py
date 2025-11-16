# src/local_vdb.py
"""
Local Vector Database Interface.

Provides a clean API wrapper around the storage engine.
Handles layer routing (permanent vs dynamic) and metadata management.

Author: Saberzerker
Date: 2025-11-16
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

from src.storage_engine import StorageEngine
from src.config import VECTOR_DIMENSION

logger = logging.getLogger(__name__)


class LocalVDB:
    """
    High-level interface for local vector database operations.
    
    Abstracts away storage engine details and provides simple methods
    for insert, search, delete operations with proper layer routing.
    """
    
    def __init__(self, config=None):
        """Initialize local VDB with optional config."""
        if config is None:
         # Import config module
            from src import config as cfg
            config = cfg
    
        self.storage = StorageEngine(config)
        self.dimension = config.VECTOR_DIMENSION
    
    logger.info("[LOCAL VDB] Initialized")
    
    def insert_vector(
        self,
        vectors: np.ndarray,
        ids: List[str],
        layer: str = "dynamic",
        anchor_id: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Insert vectors into specified layer.
        
        Args:
            vectors: Vector embeddings (shape: [n, dimension] or [dimension])
            ids: Vector IDs (list of strings)
            layer: "permanent" or "dynamic" (default: dynamic)
            anchor_id: Associated anchor ID for tracking
            metadata: Additional metadata (source, cluster_id, etc.)
        
        Raises:
            ValueError: If trying to insert into permanent layer during runtime
        """
        if layer == "permanent":
            raise ValueError(
                "Cannot insert into permanent layer during runtime. "
                "Use seed_database.py to populate permanent layer."
            )
        
        # Ensure vectors is 2D array
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # Validate dimensions
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )
        
        # Merge anchor_id into metadata
        if metadata is None:
            metadata = {}
        if anchor_id is not None:
            metadata["anchor_id"] = anchor_id
        
        # Insert into storage engine
        self.storage.insert(vectors, ids, metadata=metadata)
        
        logger.debug(f"[LOCAL VDB] Inserted {len(ids)} vectors into {layer} layer")
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        """
        Search across both permanent and dynamic layers.
        
        Args:
            query_vector: Query embedding (shape: [dimension])
            k: Number of results to return
        
        Returns:
            Tuple of (ids, scores)
            - ids: List of vector IDs
            - scores: List of similarity scores (distance-based, lower = more similar)
        """
        # Ensure query vector is 1D
        if query_vector.ndim != 1:
            query_vector = query_vector.flatten()
        
        # Search storage engine (searches both layers)
        ids, distances = self.storage.search(query_vector, k)
        
        # Convert L2 distances to similarity scores
        # Lower distance = higher similarity
        # Use inverse: score = 1 / (1 + distance)
        scores = [1.0 / (1.0 + d) for d in distances]
        
        logger.debug(f"[LOCAL VDB] Search returned {len(ids)} results")
        
        return ids, scores
    
    def delete_vector(self, ids: List[str]):
        """
        Delete vectors (only from dynamic layer).
        
        Args:
            ids: List of vector IDs to delete
        
        Raises:
            ValueError: If trying to delete from permanent layer
        """
        self.storage.delete(ids)
        logger.debug(f"[LOCAL VDB] Deleted {len(ids)} vectors")
    
    def trigger_compaction(self):
        """
        Manually trigger compaction of dynamic layer.
        
        This merges hot partition and delta segments into optimized indexes.
        Normally called by background scheduler.
        """
        logger.info("[LOCAL VDB] Triggering compaction...")
        self.storage.compact()
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive statistics about local VDB state.
        
        Returns:
            Dict with layer sizes, segment counts, etc.
        """
        stats = self.storage.get_stats()
        
        return {
            "total_vectors": stats["base_vectors"] + stats["cache_vectors"],
            "permanent_layer_vectors": stats["base_vectors"],
            "dynamic_layer_vectors": stats["cache_vectors"],
            "permanent_partitions": stats["base_partitions"],
            "dynamic_hot_size": stats["cache_hot_size"],
            "dynamic_delta_segments": stats["cache_delta_segments"],
            "deleted_count": stats["cache_deleted_count"]
        }
    
    def save_state(self):
        """
        Persist current state to disk.
        
        Saves:
        - Dynamic layer indexes
        - Metadata
        - Tombstone records
        """
        logger.info("[LOCAL VDB] Saving state to disk...")
        self.storage.save_state()
    
    def load_state(self):
        """
        Load previously saved state from disk.
        
        Called automatically during initialization if saved state exists.
        """
        logger.info("[LOCAL VDB] Loading state from disk...")
        self.storage.load_state()