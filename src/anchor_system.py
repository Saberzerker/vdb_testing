# src/anchor_system.py
"""
Anchor-Based Trajectory Learning System for Predictive Vector Caching.

NOVEL CONTRIBUTION:
This module implements a self-organizing semantic graph where query vectors
become "anchors" that predict likely next queries. When predictions are validated
(user asks a predicted query), anchors are strengthened through reinforcement
learning. Over time, frequently-used query paths solidify into STRONG/PERMANENT
anchors, while unused predictions decay and are evicted.

Core Innovation:
- Four-tier anchor hierarchy: WEAK → MEDIUM → STRONG → PERMANENT
- Prediction generation: 5 vectors per query via semantic expansion
- Hit detection: Check if incoming query matches active predictions
- Reinforcement: Correct predictions strengthen source anchors
- Decay: Unused anchors gradually weaken and evict

Author: Saberzerker
Date: 2025-11-16
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.config import (
    ANCHOR_STRENGTH_WEAK,
    ANCHOR_STRENGTH_MEDIUM,
    ANCHOR_STRENGTH_STRONG,
    ANCHOR_STRENGTH_PERMANENT,
    ANCHOR_DECAY_RATE_WEAK,
    ANCHOR_DECAY_RATE_MEDIUM,
    ANCHOR_DECAY_RATE_STRONG,
    ANCHOR_DECAY_RATE_PERMANENT,
    ANCHOR_WEAK_TIMEOUT,
    ANCHOR_MEDIUM_TIMEOUT,
    ANCHOR_STRONG_TIMEOUT,
    ANCHOR_DECAY_CHECK_INTERVAL,
    PREFETCH_K,
    PREDICTION_STEP_SIZE,
    PREDICTION_MATCH_THRESHOLD
)

logger = logging.getLogger(__name__)


class AnchorType(Enum):
    """
    Four-tier anchor hierarchy based on validation strength.
    
    WEAK: New predictions, not yet validated (fast decay)
    MEDIUM: Some validation, emerging patterns (medium decay)
    STRONG: Proven trajectories, frequently used (slow decay)
    PERMANENT: Core knowledge, never decays
    """
    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"
    PERMANENT = "PERMANENT"


@dataclass
class Anchor:
    """
    Represents a learned query point in semantic space.
    
    Anchors are created when queries arrive and strengthened when their
    predictions are validated. They form nodes in the trajectory graph.
    
    Attributes:
        id: Unique anchor identifier
        vector: Query embedding (normalized)
        query_id: Original query identifier
        query_text: Original query string (for debugging)
        strength: Reinforcement score (0 → 10+)
        hits: Number of times predictions were validated
        misses: Number of predictions that weren't used
        type: Current tier (WEAK/MEDIUM/STRONG/PERMANENT)
        created_at: Timestamp of creation
        last_hit: Timestamp of most recent validation
        last_decay: Timestamp of most recent decay check
        predictions_generated: Total predictions made from this anchor
        successful_predictions: How many predictions led to hits
        parent_anchor: ID of anchor that predicted this query (graph structure)
        child_anchors: IDs of anchors created from this anchor's predictions
    """
    id: int
    vector: np.ndarray
    query_id: str
    query_text: str
    
    # Reinforcement metrics
    strength: float = 0.0
    hits: int = 0
    misses: int = 0
    
    # Type (auto-upgraded based on strength)
    type: AnchorType = AnchorType.WEAK
    
    # Temporal tracking
    created_at: float = field(default_factory=time.time)
    last_hit: Optional[float] = None
    last_decay: float = field(default_factory=time.time)
    
    # Graph connectivity
    predictions_generated: int = 0
    successful_predictions: int = 0
    parent_anchor: Optional[int] = None
    child_anchors: List[int] = field(default_factory=list)
    
    def update_type(self):
        """
        Automatically upgrade/downgrade anchor type based on current strength.
        
        Thresholds (from config):
        - PERMANENT: 10+ strength
        - STRONG: 5-9 strength
        - MEDIUM: 2-4 strength
        - WEAK: 0-1 strength
        """
        if self.strength >= ANCHOR_STRENGTH_PERMANENT:
            self.type = AnchorType.PERMANENT
        elif self.strength >= ANCHOR_STRENGTH_STRONG:
            self.type = AnchorType.STRONG
        elif self.strength >= ANCHOR_STRENGTH_MEDIUM:
            self.type = AnchorType.MEDIUM
        else:
            self.type = AnchorType.WEAK
    
    def get_decay_rate(self) -> float:
        """Get the decay rate for this anchor's current type."""
        decay_rates = {
            AnchorType.WEAK: ANCHOR_DECAY_RATE_WEAK,
            AnchorType.MEDIUM: ANCHOR_DECAY_RATE_MEDIUM,
            AnchorType.STRONG: ANCHOR_DECAY_RATE_STRONG,
            AnchorType.PERMANENT: ANCHOR_DECAY_RATE_PERMANENT
        }
        return decay_rates[self.type]
    
    def should_be_evicted(self) -> bool:
        """
        Check if anchor should be evicted based on type and idle time.
        
        PERMANENT anchors never evict.
        Other types have timeout thresholds based on idle time.
        """
        if self.type == AnchorType.PERMANENT:
            return False
        
        # Calculate idle time (time since last hit, or creation if never hit)
        reference_time = self.last_hit if self.last_hit else self.created_at
        idle_time = time.time() - reference_time
        
        timeouts = {
            AnchorType.WEAK: ANCHOR_WEAK_TIMEOUT,
            AnchorType.MEDIUM: ANCHOR_MEDIUM_TIMEOUT,
            AnchorType.STRONG: ANCHOR_STRONG_TIMEOUT
        }
        
        return idle_time > timeouts.get(self.type, float('inf'))


@dataclass
class Prediction:
    """
    Represents a predicted future query vector.
    
    Predictions are generated by anchors and prefetched from cloud.
    When an incoming query matches a prediction (high similarity),
    the source anchor is strengthened.
    
    Attributes:
        id: Unique prediction identifier
        vector: Predicted query embedding
        source_anchor_id: Anchor that generated this prediction
        created_at: Timestamp of prediction generation
        status: "active" | "hit" | "expired"
        matched_query_id: If hit, the query that matched
        similarity_score: If hit, the cosine similarity score
    """
    id: int
    vector: np.ndarray
    source_anchor_id: int
    created_at: float = field(default_factory=time.time)
    status: str = "active"  # "active", "hit", "expired"
    matched_query_id: Optional[str] = None
    similarity_score: Optional[float] = None


class AnchorSystem:
    """
    Manages the anchor-based trajectory learning graph.
    
    This is the core innovation: a self-organizing semantic graph where:
    - Nodes = Anchors (learned query points)
    - Edges = Predictions (anticipated next queries)
    - Weights = Strength (reinforcement-based validation)
    
    The system learns which query paths users follow and proactively
    caches along those trajectories.
    """
    
    def __init__(self):
        """Initialize empty anchor system."""
        self.anchors: Dict[int, Anchor] = {}
        self.predictions: Dict[int, Prediction] = {}
        
        self.anchor_counter = 0
        self.prediction_counter = 0
        
        # Metrics for evaluation
        self.metrics = {
            "anchors_created": 0,
            "anchors_strengthened": 0,
            "anchors_weakened": 0,
            "anchors_evicted": 0,
            "predictions_generated": 0,
            "predictions_hit": 0,
            "predictions_missed": 0,
            "type_transitions": []  # Track WEAK→MEDIUM→STRONG→PERMANENT
        }
        
        logger.info("[ANCHOR SYSTEM] Initialized with four-tier hierarchy")
    
    def create_anchor(self, query_vector: np.ndarray, query_id: str, 
                     query_text: str, parent_anchor_id: Optional[int] = None) -> int:
        """
        Create a new anchor point in the semantic graph.
        
        Args:
            query_vector: Query embedding (will be normalized)
            query_id: Unique query identifier
            query_text: Original query string
            parent_anchor_id: If this query matched a prediction, ID of source anchor
        
        Returns:
            New anchor ID
        """
        anchor_id = self.anchor_counter
        self.anchor_counter += 1
        
        # Normalize vector
        normalized_vector = query_vector / np.linalg.norm(query_vector)
        
        anchor = Anchor(
            id=anchor_id,
            vector=normalized_vector,
            query_id=query_id,
            query_text=query_text,
            parent_anchor=parent_anchor_id
        )
        
        self.anchors[anchor_id] = anchor
        self.metrics["anchors_created"] += 1
        
        # Link to parent if exists (forms trajectory graph)
        if parent_anchor_id is not None and parent_anchor_id in self.anchors:
            self.anchors[parent_anchor_id].child_anchors.append(anchor_id)
        
        logger.debug(f"[ANCHOR] Created anchor {anchor_id} [{anchor.type.value}] for query: \"{query_text[:50]}...\"")
        
        return anchor_id
    
    def strengthen_anchor(self, anchor_id: int, reward: float = 1.0):
        """
        REINFORCEMENT: Strengthen anchor when its prediction was validated.
        
        This is the learning mechanism. When a user asks a query that we
        predicted, the anchor that made the prediction gets stronger.
        
        Args:
            anchor_id: Anchor to strengthen
            reward: Strength increment (default 1.0)
        """
        if anchor_id not in self.anchors:
            logger.warning(f"[ANCHOR] Cannot strengthen non-existent anchor {anchor_id}")
            return
        
        anchor = self.anchors[anchor_id]
        old_type = anchor.type
        
        # Apply reward
        anchor.strength += reward
        anchor.hits += 1
        anchor.last_hit = time.time()
        anchor.successful_predictions += 1
        
        self.metrics["anchors_strengthened"] += 1
        
        # Update type based on new strength
        anchor.update_type()
        
        # Track type transitions for analysis
        if old_type != anchor.type:
            transition = {
                "anchor_id": anchor_id,
                "from": old_type.value,
                "to": anchor.type.value,
                "strength": anchor.strength,
                "hits": anchor.hits,
                "timestamp": time.time()
            }
            self.metrics["type_transitions"].append(transition)
            
            logger.info(f"[REINFORCE] 🎯 Anchor {anchor_id}: {old_type.value} → {anchor.type.value} "
                       f"(strength={anchor.strength:.1f}, hits={anchor.hits})")
    
    def decay_anchors(self):
        """
        Apply time-based decay to anchors that haven't been validated recently.
        
        This prevents the graph from growing indefinitely with unused anchors.
        Decay rates depend on anchor type:
        - WEAK: Fast decay (1.0 per cycle)
        - MEDIUM: Medium decay (0.5 per cycle)
        - STRONG: Slow decay (0.1 per cycle)
        - PERMANENT: No decay (0.0)
        
        Called periodically by scheduler.
        """
        current_time = time.time()
        to_evict = []
        
        for anchor_id, anchor in self.anchors.items():
            # Check if enough time has passed since last decay
            time_since_decay = current_time - anchor.last_decay
            
            if time_since_decay >= ANCHOR_DECAY_CHECK_INTERVAL:
                decay_rate = anchor.get_decay_rate()
                
                if decay_rate > 0:
                    old_strength = anchor.strength
                    old_type = anchor.type
                    
                    # Apply decay
                    anchor.strength = max(0, anchor.strength - decay_rate)
                    anchor.last_decay = current_time
                    
                    # Update type (might downgrade)
                    anchor.update_type()
                    
                    if old_strength > 0:
                        self.metrics["anchors_weakened"] += 1
                    
                    # Track downgrades
                    if old_type != anchor.type:
                        self.metrics["type_transitions"].append({
                            "anchor_id": anchor_id,
                            "from": old_type.value,
                            "to": anchor.type.value,
                            "strength": anchor.strength,
                            "reason": "decay",
                            "timestamp": current_time
                        })
                        
                        logger.debug(f"[DECAY] Anchor {anchor_id}: {old_type.value} → {anchor.type.value} "
                                   f"(strength={anchor.strength:.1f})")
                
                # Check for eviction
                if anchor.should_be_evicted():
                    to_evict.append(anchor_id)
        
        # Evict expired anchors
        for anchor_id in to_evict:
            anchor = self.anchors[anchor_id]
            age = current_time - anchor.created_at
            logger.info(f"[ANCHOR] ❌ Evicting anchor {anchor_id} [{anchor.type.value}] "
                       f"(age={age:.0f}s, strength={anchor.strength:.1f}, hits={anchor.hits})")
            del self.anchors[anchor_id]
            self.metrics["anchors_evicted"] += 1
    
    def generate_predictions(self, anchor_id: int, centroid: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Generate N prediction vectors representing likely next queries.
        
        Strategy:
        - If centroid provided: Generate trajectory from anchor toward centroid
        - Else: Random walk around anchor
        
        Args:
            anchor_id: Anchor to generate predictions from
            centroid: Optional cluster centroid to guide trajectory
        
        Returns:
            List of PREFETCH_K prediction vectors
        """
        if anchor_id not in self.anchors:
            logger.warning(f"[PREDICT] Cannot generate from non-existent anchor {anchor_id}")
            return []
        
        anchor = self.anchors[anchor_id]
        anchor_vector = anchor.vector
        
        predictions = []
        
        if centroid is not None:
            # TRAJECTORY STRATEGY: Generate steps along path from anchor to centroid
            # This assumes user will continue exploring current topic
            for i in range(1, PREFETCH_K + 1):
                alpha = i * PREDICTION_STEP_SIZE  # e.g., 0.15, 0.30, 0.45, 0.60, 0.75
                pred_vec = (1 - alpha) * anchor_vector + alpha * centroid
                pred_vec = pred_vec / np.linalg.norm(pred_vec)  # Normalize
                predictions.append(pred_vec)
            
            logger.debug(f"[PREDICT] Generated {PREFETCH_K} predictions (trajectory strategy) from anchor {anchor_id}")
        
        else:
            # RANDOM WALK STRATEGY: Small perturbations around anchor
            # Used when no clear semantic context exists yet
            for i in range(PREFETCH_K):
                # Add Gaussian noise
                noise = np.random.normal(0, 0.08, anchor_vector.shape)
                pred_vec = anchor_vector + noise
                pred_vec = pred_vec / np.linalg.norm(pred_vec)
                predictions.append(pred_vec)
            
            logger.debug(f"[PREDICT] Generated {PREFETCH_K} predictions (random walk) from anchor {anchor_id}")
        
        # Track prediction generation
        anchor.predictions_generated += len(predictions)
        
        return predictions
    
    def register_prediction(self, pred_vector: np.ndarray, source_anchor_id: int) -> int:
        """
        Register a prediction for future hit detection.
        
        Args:
            pred_vector: Predicted query vector
            source_anchor_id: Anchor that generated this prediction
        
        Returns:
            New prediction ID
        """
        pred_id = self.prediction_counter
        self.prediction_counter += 1
        
        self.predictions[pred_id] = Prediction(
            id=pred_id,
            vector=pred_vector,
            source_anchor_id=source_anchor_id
        )
        
        self.metrics["predictions_generated"] += 1
        
        return pred_id
    
    def check_prediction_match(self, query_vector: np.ndarray, query_id: str) -> Optional[Tuple[int, int, float]]:
        """
        Check if incoming query matches any active prediction.
        
        This is where we detect if we correctly predicted the user's next query.
        
        Args:
            query_vector: Incoming query embedding
            query_id: Query identifier
        
        Returns:
            If match found: (prediction_id, source_anchor_id, similarity_score)
            If no match: None
        """
        best_match = None
        best_similarity = 0.0
        
        # Check all active predictions
        for pred_id, pred in self.predictions.items():
            if pred.status != "active":
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector, pred.vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(pred.vector)
            )
            
            # Check if similarity exceeds threshold
            if similarity >= PREDICTION_MATCH_THRESHOLD and similarity > best_similarity:
                best_similarity = similarity
                best_match = (pred_id, pred.source_anchor_id, similarity)
        
        if best_match:
            pred_id, anchor_id, similarity = best_match
            
            # Mark prediction as hit
            self.predictions[pred_id].status = "hit"
            self.predictions[pred_id].matched_query_id = query_id
            self.predictions[pred_id].similarity_score = similarity
            
            self.metrics["predictions_hit"] += 1
            
            logger.info(f"[PREDICT] 🎯 PREDICTION HIT! Prediction {pred_id} matched query {query_id} "
                       f"(similarity={similarity:.3f}, source_anchor={anchor_id})")
            
            return best_match
        
        # No match found
        self.metrics["predictions_missed"] += 1
        return None
    
    def expire_old_predictions(self, max_age_seconds: float = 300):
        """
        Mark old predictions as expired.
        
        Predictions that haven't been matched after a timeout are unlikely
        to ever match, so we mark them expired to reduce search space.
        
        Args:
            max_age_seconds: Age threshold (default 5 minutes)
        """
        current_time = time.time()
        expired_count = 0
        
        for pred_id, pred in self.predictions.items():
            if pred.status == "active":
                age = current_time - pred.created_at
                if age > max_age_seconds:
                    pred.status = "expired"
                    expired_count += 1
        
        if expired_count > 0:
            logger.debug(f"[PREDICT] Expired {expired_count} old predictions")
    
    def get_anchor_stats(self) -> Dict:
        """
        Get comprehensive statistics about anchor system state.
        
        Returns:
            Dict with anchor counts, type distribution, strength metrics, etc.
        """
        # Count by type
        type_counts = {
            "WEAK": 0,
            "MEDIUM": 0,
            "STRONG": 0,
            "PERMANENT": 0
        }
        
        for anchor in self.anchors.values():
            type_counts[anchor.type.value] += 1
        
        # Aggregate metrics
        total_strength = sum(a.strength for a in self.anchors.values())
        avg_strength = total_strength / len(self.anchors) if self.anchors else 0
        
        # Prediction metrics
        active_predictions = sum(1 for p in self.predictions.values() if p.status == "active")
        hit_predictions = sum(1 for p in self.predictions.values() if p.status == "hit")
        
        return {
            "total_anchors": len(self.anchors),
            "by_type": type_counts,
            "total_strength": total_strength,
            "average_strength": avg_strength,
            "active_predictions": active_predictions,
            "hit_predictions": hit_predictions,
            **self.metrics
        }
    
    def get_learning_trajectory(self) -> List[Dict]:
        """
        Get the timeline of type transitions showing learning progress.
        
        This is useful for visualizing how the system learns over time.
        
        Returns:
            List of transition events with timestamps
        """
        return self.metrics["type_transitions"]
    
    def get_anchor_graph(self) -> Dict:
        """
        Export the anchor graph structure for visualization.
        
        Returns:
            Dict with nodes (anchors) and edges (parent-child relationships)
        """
        nodes = []
        edges = []
        
        for anchor_id, anchor in self.anchors.items():
            nodes.append({
                "id": anchor_id,
                "type": anchor.type.value,
                "strength": anchor.strength,
                "hits": anchor.hits,
                "query_text": anchor.query_text[:50]
            })
            
            # Add edge from parent to this anchor
            if anchor.parent_anchor is not None:
                edges.append({
                    "from": anchor.parent_anchor,
                    "to": anchor_id
                })
        
        return {
            "nodes": nodes,
            "edges": edges
        }