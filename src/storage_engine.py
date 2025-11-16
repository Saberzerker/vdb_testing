# src/storage_engine.py
"""
Production Two-Tier Storage Engine with MicroNN-Inspired Incremental Design.

PERMANENT LAYER (30%):
- Pre-loaded, read-only optimized FAISS indexes
- Never modified during runtime
- Provides guaranteed baseline offline functionality

DYNAMIC LAYER (70%):
- Hot partition (in-memory, append-only)
- Delta segments (on-disk, micro-indexes)
- Background compaction merges segments
- Tombstone-based deletion

Author: Saberzerker
Date: 2025-11-16
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
    CACHE_LAYER_PATH,
    VECTOR_DIMENSION,
    HOT_PARTITION_RAM_LIMIT,
    PERMANENT_LAYER_CAPACITY,
    DYNAMIC_LAYER_CAPACITY
)

logger = logging.getLogger(__name__)


class StorageEngine:
    """
    Production storage engine with explicit two-tier architecture.
    
    Thread-safe operations with proper locking.
    Persistent state management.
    Efficient incremental updates for dynamic layer.
    """
    
    def __init__(self, config):
        """
        Initialize storage engine.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.dimension = config.VECTOR_DIMENSION
        
        # Paths
        self.base_path = Path(config.BASE_LAYER_PATH)
        self.cache_path = Path(config.CACHE_LAYER_PATH)
        
        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # ═══════════════════════════════════════════════════════════
        # PERMANENT LAYER (Read-only)
        # ═══════════════════════════════════════════════════════════
        
        self.base_partitions = []  # List of loaded FAISS indexes
        self.base_metadata = {}    # {vector_id: {partition_idx, local_idx, ...}}
        
        # ═══════════════════════════════════════════════════════════
        # DYNAMIC LAYER (Read-write)
        # ═══════════════════════════════════════════════════════════
        
        # Hot partition (in-memory, append-only)
        self.hot_partition = faiss.IndexFlatL2(self.dimension)
        self.hot_ids = []
        self.hot_metadata = []
        
        # Delta segments (on-disk, micro-indexes)
        self.delta_segments = []  # List of {index, ids, metadata, file}
        
        # Cache metadata tracking
        self.cache_metadata = {}  # {vector_id: {segment_type, segment_idx, ...}}
        
        # Tombstone deletion (mark as deleted without rebuilding)
        self.deleted_ids = set()
        
        # Statistics
        self.stats = {
            "inserts": 0,
            "deletes": 0,
            "searches": 0,
            "compactions": 0
        }
        
        # Load existing state
        self._load_base_layer()
        self._load_dynamic_layer()
        
        logger.info(f"[STORAGE] Initialized")
        logger.info(f"[STORAGE] Base layer: {self._count_base_vectors()} vectors (read-only)")
        logger.info(f"[STORAGE] Dynamic layer: {self._count_cache_vectors()} vectors (read-write)")
    
    # ═══════════════════════════════════════════════════════════
    # BASE LAYER OPERATIONS (Read-only)
    # ═══════════════════════════════════════════════════════════
    
    def _load_base_layer(self):
        """
        Load all permanent layer indexes from disk.
        
        Expected structure:
        BASE_LAYER_PATH/
            partition_0.index
            partition_1.index
            ...
            metadata.json
        """
        if not self.base_path.exists():
            logger.warning(f"[STORAGE] Base layer path {self.base_path} does not exist")
            return
        
        # Load all partition files
        partition_files = sorted(glob.glob(str(self.base_path / "partition_*.index")))
        
        for partition_file in partition_files:
            try:
                index = faiss.read_index(partition_file)
                self.base_partitions.append({
                    "index": index,
                    "file": partition_file,
                    "nvectors": index.ntotal
                })
                logger.info(f"[STORAGE] Loaded base partition: {partition_file} ({index.ntotal} vectors)")
            except Exception as e:
                logger.error(f"[STORAGE] Failed to load {partition_file}: {e}")
        
        # Load metadata
        metadata_file = self.base_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.base_metadata = json.load(f)
            logger.info(f"[STORAGE] Loaded base metadata ({len(self.base_metadata)} entries)")
    
    def _count_base_vectors(self) -> int:
        """Count total vectors in base layer."""
        return sum(p["nvectors"] for p in self.base_partitions)
    
    # ═══════════════════════════════════════════════════════════
    # DYNAMIC LAYER OPERATIONS (Read-write)
    # ═══════════════════════════════════════════════════════════
    
    def _load_dynamic_layer(self):
        """
        Load dynamic layer state from disk.
        
        Expected structure:
        CACHE_LAYER_PATH/
            hot_partition.index (if exists)
            hot_ids.pkl
            hot_metadata.pkl
            delta_0.index
            delta_0_ids.pkl
            delta_0_metadata.pkl
            ...
            cache_metadata.json
            deleted_ids.pkl
        """
        # Load hot partition
        hot_index_file = self.cache_path / "hot_partition.index"
        hot_ids_file = self.cache_path / "hot_ids.pkl"
        hot_metadata_file = self.cache_path / "hot_metadata.pkl"
        
        if hot_index_file.exists():
            try:
                self.hot_partition = faiss.read_index(str(hot_index_file))
                with open(hot_ids_file, 'rb') as f:
                    self.hot_ids = pickle.load(f)
                with open(hot_metadata_file, 'rb') as f:
                    self.hot_metadata = pickle.load(f)
                logger.info(f"[STORAGE] Loaded hot partition ({self.hot_partition.ntotal} vectors)")
            except Exception as e:
                logger.error(f"[STORAGE] Failed to load hot partition: {e}")
        
        # Load delta segments
        delta_files = sorted(glob.glob(str(self.cache_path / "delta_*.index")))
        
        for delta_file in delta_files:
            try:
                segment_name = Path(delta_file).stem
                ids_file = self.cache_path / f"{segment_name}_ids.pkl"
                metadata_file = self.cache_path / f"{segment_name}_metadata.pkl"
                
                index = faiss.read_index(delta_file)
                with open(ids_file, 'rb') as f:
                    ids = pickle.load(f)
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.delta_segments.append({
                    "index": index,
                    "ids": ids,
                    "metadata": metadata,
                    "file": delta_file
                })
                
                logger.info(f"[STORAGE] Loaded delta segment: {delta_file} ({index.ntotal} vectors)")
            except Exception as e:
                logger.error(f"[STORAGE] Failed to load {delta_file}: {e}")
        
        # Load cache metadata
        cache_metadata_file = self.cache_path / "cache_metadata.json"
        if cache_metadata_file.exists():
            with open(cache_metadata_file, 'r') as f:
                self.cache_metadata = json.load(f)
        
        # Load deleted IDs
        deleted_ids_file = self.cache_path / "deleted_ids.pkl"
        if deleted_ids_file.exists():
            with open(deleted_ids_file, 'rb') as f:
                self.deleted_ids = pickle.load(f)
            logger.info(f"[STORAGE] Loaded {len(self.deleted_ids)} deleted IDs")
    
    def _count_cache_vectors(self) -> int:
        """Count total vectors in dynamic layer (excluding deleted)."""
        total = self.hot_partition.ntotal
        for segment in self.delta_segments:
            total += segment["index"].ntotal
        return total - len(self.deleted_ids)
    
    # ═══════════════════════════════════════════════════════════
    # INSERT OPERATION (Dynamic Layer Only)
    # ═══════════════════════════════════════════════════════════
    
    def insert(self, vectors: np.ndarray, ids: List[str], metadata: Optional[Dict] = None):
        """
        Insert vectors into dynamic layer.
        
        Uses MicroNN-inspired append-only strategy:
        - Add to hot partition if space available
        - Create new delta segment if hot is full
        
        Args:
            vectors: Vector embeddings (shape: [n, dimension])
            ids: Vector IDs
            metadata: Optional metadata dict
        
        Raises:
            ValueError: If trying to insert vectors already in base layer
        """
        with self.lock:
            # Validate not in base layer
            for vid in ids:
                if vid in self.base_metadata:
                    raise ValueError(f"Cannot insert {vid}: already in permanent layer")
            
            # Ensure 2D array
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            
            # Validate dimension
            if vectors.shape[1] != self.dimension:
                raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
            
            n_vectors = vectors.shape[0]
            
            # Check if hot partition has space
            if self.hot_partition.ntotal + n_vectors <= HOT_PARTITION_RAM_LIMIT:
                # Add to hot partition
                self.hot_partition.add(vectors.astype('float32'))
                self.hot_ids.extend(ids)
                
                for i, vid in enumerate(ids):
                    meta = metadata.copy() if metadata else {}
                    meta.update({
                        "segment_type": "hot",
                        "local_idx": self.hot_partition.ntotal - n_vectors + i,
                        "inserted_at": time.time()
                    })
                    self.hot_metadata.append(meta)
                    self.cache_metadata[vid] = meta
                
                logger.debug(f"[STORAGE] Inserted {n_vectors} vectors into hot partition")
            
            else:
                # Hot partition full → create new delta segment
                self._create_delta_segment(vectors, ids, metadata)
            
            self.stats["inserts"] += n_vectors
    
    def _create_delta_segment(self, vectors: np.ndarray, ids: List[str], metadata: Optional[Dict] = None):
        """
        Create a new delta segment when hot partition is full.
        
        Args:
            vectors: Vectors to store
            ids: Vector IDs
            metadata: Optional metadata
        """
        segment_idx = len(self.delta_segments)
        segment_name = f"delta_{segment_idx}"
        
        # Create FAISS index
        index = faiss.IndexFlatL2(self.dimension)
        index.add(vectors.astype('float32'))
        
        # Prepare metadata
        segment_metadata = []
        for i, vid in enumerate(ids):
            meta = metadata.copy() if metadata else {}
            meta.update({
                "segment_type": "delta",
                "segment_idx": segment_idx,
                "local_idx": i,
                "inserted_at": time.time()
            })
            segment_metadata.append(meta)
            self.cache_metadata[vid] = meta
        
        # Save to disk
        segment_file = self.cache_path / f"{segment_name}.index"
        faiss.write_index(index, str(segment_file))
        
        with open(self.cache_path / f"{segment_name}_ids.pkl", 'wb') as f:
            pickle.dump(ids, f)
        with open(self.cache_path / f"{segment_name}_metadata.pkl", 'wb') as f:
            pickle.dump(segment_metadata, f)
        
        # Add to segments list
        self.delta_segments.append({
            "index": index,
            "ids": ids,
            "metadata": segment_metadata,
            "file": str(segment_file)
        })
        
        logger.info(f"[STORAGE] Created delta segment {segment_idx} with {len(ids)} vectors")
    
    # ═══════════════════════════════════════════════════════════
    # SEARCH OPERATION (Searches Both Layers)
    # ═══════════════════════════════════════════════════════════
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        """
        Search across both permanent and dynamic layers.
        
        Process:
        1. Search all base partitions
        2. Search hot partition
        3. Search all delta segments
        4. Merge results
        5. Filter deleted IDs
        6. Re-rank and return top-k
        
        Args:
            query_vector: Query embedding
            k: Number of results
        
        Returns:
            (ids, distances) - sorted by distance (ascending)
        """
        with self.lock:
            query_vector = query_vector.reshape(1, -1).astype('float32')
            
            all_distances = []
            all_ids = []
            
            # Search base layer (permanent partitions)
            for part_idx, partition in enumerate(self.base_partitions):
                if partition["index"].ntotal == 0:
                    continue
                
                D, I = partition["index"].search(query_vector, min(k, partition["index"].ntotal))
                
                for local_idx, distance in zip(I[0], D[0]):
                    if local_idx != -1 and distance < float('inf'):
                        # Find vector ID from metadata
                        vec_id = self._get_base_vector_id(part_idx, local_idx)
                        if vec_id:
                            all_distances.append(distance)
                            all_ids.append(vec_id)
            
            # Search dynamic layer - hot partition
            if self.hot_partition.ntotal > 0:
                D, I = self.hot_partition.search(query_vector, min(k, self.hot_partition.ntotal))
                
                for local_idx, distance in zip(I[0], D[0]):
                    if local_idx != -1 and distance < float('inf') and local_idx < len(self.hot_ids):
                        vec_id = self.hot_ids[local_idx]
                        all_distances.append(distance)
                        all_ids.append(vec_id)
            
            # Search dynamic layer - delta segments
            for segment in self.delta_segments:
                if segment["index"].ntotal == 0:
                    continue
                
                D, I = segment["index"].search(query_vector, min(k, segment["index"].ntotal))
                
                for local_idx, distance in zip(I[0], D[0]):
                    if local_idx != -1 and distance < float('inf') and local_idx < len(segment["ids"]):
                        vec_id = segment["ids"][local_idx]
                        all_distances.append(distance)
                        all_ids.append(vec_id)
            
            # Merge and sort
            combined = list(zip(all_ids, all_distances))
            
            # Filter deleted IDs
            combined = [(vid, d) for vid, d in combined if vid not in self.deleted_ids]
            
            # Sort by distance (ascending = more similar)
            combined.sort(key=lambda x: x[1])
            
            # Return top-k
            top_k = combined[:k]
            
            ids = [vid for vid, _ in top_k]
            distances = [d for _, d in top_k]
            
            self.stats["searches"] += 1
            
            return ids, distances
    
    def _get_base_vector_id(self, partition_idx: int, local_idx: int) -> Optional[str]:
        """
        Get vector ID from base metadata.
        
        Args:
            partition_idx: Partition index
            local_idx: Local index within partition
        
        Returns:
            Vector ID or None
        """
        # Linear search through metadata (can be optimized with reverse index)
        for vid, meta in self.base_metadata.items():
            if meta.get("partition_idx") == partition_idx and meta.get("local_idx") == local_idx:
                return vid
        return None
    
    # ═══════════════════════════════════════════════════════════
    # DELETE OPERATION (Tombstone-based)
    # ═══════════════════════════════════════════════════════════
    
    def delete(self, ids: List[str]):
        """
        Mark vectors as deleted (tombstone mechanism).
        
        Actual removal happens during compaction.
        
        Args:
            ids: Vector IDs to delete
        
        Raises:
            ValueError: If trying to delete from permanent layer
        """
        with self.lock:
            for vid in ids:
                # Validate not in base layer
                if vid in self.base_metadata:
                    raise ValueError(f"Cannot delete {vid}: in permanent layer (read-only)")
                
                # Mark as deleted
                if vid in self.cache_metadata:
                    self.deleted_ids.add(vid)
                    logger.debug(f"[STORAGE] Marked {vid} as deleted")
                    self.stats["deletes"] += 1
    
    # ═══════════════════════════════════════════════════════════
    # COMPACTION (Background Job)
    # ═══════════════════════════════════════════════════════════
    
    def compact(self):
        """
        Compact dynamic layer: merge hot + deltas into optimized index.
        
        Process:
        1. Collect all vectors from hot + delta segments
        2. Filter out deleted IDs
        3. Create new optimized FAISS index
        4. Save as new delta segment
        5. Clear hot partition and old deltas
        6. Clear tombstones
        
        This is called periodically by scheduler.
        """
        with self.lock:
            logger.info("[STORAGE] Starting compaction...")
            
            start_time = time.time()
            
            # Collect all vectors
            all_vectors = []
            all_ids = []
            all_metadata = []
            
            # From hot partition
            if self.hot_partition.ntotal > 0:
                for i in range(self.hot_partition.ntotal):
                    vid = self.hot_ids[i]
                    if vid not in self.deleted_ids:
                        vec = self.hot_partition.reconstruct(i)
                        all_vectors.append(vec)
                        all_ids.append(vid)
                        all_metadata.append(self.hot_metadata[i])
            
            # From delta segments
            for segment in self.delta_segments:
                for i in range(segment["index"].ntotal):
                    vid = segment["ids"][i]
                    if vid not in self.deleted_ids:
                        vec = segment["index"].reconstruct(i)
                        all_vectors.append(vec)
                        all_ids.append(vid)
                        all_metadata.append(segment["metadata"][i])
            
            if not all_vectors:
                logger.info("[STORAGE] No vectors to compact")
                return
            
            # Create new optimized index
            vectors_array = np.array(all_vectors).astype('float32')
            new_index = faiss.IndexFlatL2(self.dimension)
            new_index.add(vectors_array)
            
            # Save as compacted segment
            compacted_file = self.cache_path / f"compacted_{int(time.time())}.index"
            faiss.write_index(new_index, str(compacted_file))
            
            with open(str(compacted_file).replace('.index', '_ids.pkl'), 'wb') as f:
                pickle.dump(all_ids, f)
            with open(str(compacted_file).replace('.index', '_metadata.pkl'), 'wb') as f:
                pickle.dump(all_metadata, f)
            
            # Delete old files
            if (self.cache_path / "hot_partition.index").exists():
                os.remove(self.cache_path / "hot_partition.index")
                os.remove(self.cache_path / "hot_ids.pkl")
                os.remove(self.cache_path / "hot_metadata.pkl")
            
            for segment in self.delta_segments:
                if os.path.exists(segment["file"]):
                    os.remove(segment["file"])
                    os.remove(segment["file"].replace('.index', '_ids.pkl'))
                    os.remove(segment["file"].replace('.index', '_metadata.pkl'))
            
            # Update state
            self.hot_partition = faiss.IndexFlatL2(self.dimension)
            self.hot_ids = []
            self.hot_metadata = []
            
            self.delta_segments = [{
                "index": new_index,
                "ids": all_ids,
                "metadata": all_metadata,
                "file": str(compacted_file)
            }]
            
            self.deleted_ids.clear()
            
            # Update cache metadata
            for i, vid in enumerate(all_ids):
                self.cache_metadata[vid] = {
                    "segment_type": "compacted",
                    "segment_idx": 0,
                    "local_idx": i,
                    "compacted_at": time.time()
                }
            
            elapsed = time.time() - start_time
            self.stats["compactions"] += 1
            
            logger.info(f"[STORAGE] ✅ Compaction complete: {len(all_vectors)} vectors in {elapsed:.2f}s")
    
    # ═══════════════════════════════════════════════════════════
    # STATE PERSISTENCE
    # ═══════════════════════════════════════════════════════════
    
    def save_state(self):
        """Save current dynamic layer state to disk."""
        with self.lock:
            logger.info("[STORAGE] Saving dynamic layer state...")
            
            try:
                # Save hot partition
                if self.hot_partition.ntotal > 0:
                    faiss.write_index(self.hot_partition, str(self.cache_path / "hot_partition.index"))
                    with open(self.cache_path / "hot_ids.pkl", 'wb') as f:
                        pickle.dump(self.hot_ids, f)
                    with open(self.cache_path / "hot_metadata.pkl", 'wb') as f:
                        pickle.dump(self.hot_metadata, f)
                
                # Save cache metadata
                with open(self.cache_path / "cache_metadata.json", 'w') as f:
                    json.dump(self.cache_metadata, f)
                
                # Save deleted IDs
                with open(self.cache_path / "deleted_ids.pkl", 'wb') as f:
                    pickle.dump(self.deleted_ids, f)
                
                logger.info("[STORAGE] ✅ State saved")
            
            except Exception as e:
                logger.error(f"[STORAGE] Failed to save state: {e}")
    
    def get_stats(self) -> Dict:
        """Get comprehensive storage statistics."""
        with self.lock:
            return {
                "base_partitions": len(self.base_partitions),
                "base_vectors": self._count_base_vectors(),
                "cache_hot_size": self.hot_partition.ntotal,
                "cache_delta_segments": len(self.delta_segments),
                "cache_vectors": self._count_cache_vectors(),
                "cache_deleted_count": len(self.deleted_ids),
                "total_inserts": self.stats["inserts"],
                "total_deletes": self.stats["deletes"],
                "total_searches": self.stats["searches"],
                "total_compactions": self.stats["compactions"]
            }