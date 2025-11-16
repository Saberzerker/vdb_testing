# tests/test_anchor_system.py
"""Unit tests for anchor system."""

import unittest
import numpy as np
import time

from src.anchor_system import AnchorSystem, AnchorType


class TestAnchorSystem(unittest.TestCase):
    def setUp(self):
        """Create fresh anchor system for each test."""
        self.anchor_sys = AnchorSystem()
    
    def test_create_anchor(self):
        """Test anchor creation."""
        query_vec = np.random.rand(384)
        anchor_id = self.anchor_sys.create_anchor(
            query_vec, "q1", "test query"
        )
        
        self.assertEqual(anchor_id, 0)
        self.assertIn(anchor_id, self.anchor_sys.anchors)
        self.assertEqual(self.anchor_sys.anchors[anchor_id].type, AnchorType.WEAK)
    
    def test_strengthen_anchor(self):
        """Test anchor strengthening and type upgrades."""
        query_vec = np.random.rand(384)
        anchor_id = self.anchor_sys.create_anchor(query_vec, "q1", "test")
        
        # Initially WEAK
        self.assertEqual(self.anchor_sys.anchors[anchor_id].type, AnchorType.WEAK)
        
        # Strengthen to MEDIUM
        self.anchor_sys.strengthen_anchor(anchor_id)
        self.anchor_sys.strengthen_anchor(anchor_id)
        self.assertEqual(self.anchor_sys.anchors[anchor_id].type, AnchorType.MEDIUM)
        
        # Strengthen to STRONG
        for _ in range(3):
            self.anchor_sys.strengthen_anchor(anchor_id)
        self.assertEqual(self.anchor_sys.anchors[anchor_id].type, AnchorType.STRONG)
        
        # Strengthen to PERMANENT
        for _ in range(5):
            self.anchor_sys.strengthen_anchor(anchor_id)
        self.assertEqual(self.anchor_sys.anchors[anchor_id].type, AnchorType.PERMANENT)
    
    def test_prediction_generation(self):
        """Test prediction vector generation."""
        query_vec = np.random.rand(384)
        anchor_id = self.anchor_sys.create_anchor(query_vec, "q1", "test")
        
        # Generate predictions
        predictions = self.anchor_sys.generate_predictions(anchor_id)
        
        self.assertEqual(len(predictions), 5)  # PREFETCH_K = 5
        for pred in predictions:
            self.assertEqual(pred.shape, query_vec.shape)
            # Check normalized
            self.assertAlmostEqual(np.linalg.norm(pred), 1.0, places=5)
    
    def test_prediction_hit_detection(self):
        """Test prediction matching logic."""
        # Create anchor and generate predictions
        query_vec = np.random.rand(384)
        anchor_id = self.anchor_sys.create_anchor(query_vec, "q1", "test")
        predictions = self.anchor_sys.generate_predictions(anchor_id)
        
        # Register predictions
        for pred in predictions:
            self.anchor_sys.register_prediction(pred, anchor_id)
        
        # Check if similar query matches
        similar_query = predictions[0] + np.random.normal(0, 0.01, 384)
        similar_query = similar_query / np.linalg.norm(similar_query)
        
        match = self.anchor_sys.check_prediction_match(similar_query, "q2")
        
        # Should match (similarity > 0.85)
        self.assertIsNotNone(match)
        pred_id, matched_anchor_id, similarity = match
        self.assertEqual(matched_anchor_id, anchor_id)
    
    def test_anchor_decay(self):
        """Test decay mechanism."""
        query_vec = np.random.rand(384)
        anchor_id = self.anchor_sys.create_anchor(query_vec, "q1", "test")
        
        # Strengthen to MEDIUM
        self.anchor_sys.strengthen_anchor(anchor_id)
        self.anchor_sys.strengthen_anchor(anchor_id)
        
        initial_strength = self.anchor_sys.anchors[anchor_id].strength
        
        # Force decay by manipulating timestamp
        self.anchor_sys.anchors[anchor_id].last_decay = time.time() - 60
        
        # Run decay
        self.anchor_sys.decay_anchors()
        
        # Should have decayed
        final_strength = self.anchor_sys.anchors[anchor_id].strength
        self.assertLess(final_strength, initial_strength)


if __name__ == '__main__':
    unittest.main()