# src/metrics.py
"""
Metrics Tracker for Hybrid VDB System.

Tracks all telemetry: hit rates, latencies, anchor stats, predictions.
Thread-safe operations for concurrent access.

Author: Saberzerker
Date: 2025-11-16
"""

import time
import threading
import json
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


class MetricsTracker:
    """
    Comprehensive metrics tracking for the hybrid VDB system.
    
    Tracks:
    - Query hit rates (local_base, local_cache, cloud, offline)
    - Latencies per source (p50, p95, p99)
    - Anchor system stats (created, strengthened, type distribution)
    - Prediction accuracy
    - Cache statistics
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.lock = threading.Lock()
        
        # Event tracking
        self.events = {
            "local_hit": [],      # Queries answered from local (permanent or dynamic)
            "cloud_hit": [],      # Queries requiring cloud fallback
            "offline_fallback": [], # Queries answered locally when cloud unavailable
            "prefetch_operation": []  # Prefetch operations completed
        }
        
        # Query-level details
        self.query_log = []  # Full query history with metadata
        
        # Anchor system metrics
        self.anchor_metrics = {
            "anchors_created": 0,
            "anchors_strengthened": 0,
            "anchors_weakened": 0,
            "anchors_evicted": 0,
            "type_transitions": [],  # WEAK→MEDIUM→STRONG→PERMANENT
            "predictions_generated": 0,
            "predictions_hit": 0,
            "predictions_missed": 0
        }
        
        # Cache metrics
        self.cache_metrics = {
            "cache_adds": 0,
            "cache_evictions": 0,
            "compactions": 0
        }
        
        # Timestamps
        self.start_time = time.time()
        self.last_reset_time = time.time()
        
        print("[METRICS] Initialized")
    
    def log_event(self, event_type: str, latency: Optional[float] = None, **kwargs):
        """
        Log an event with optional latency and metadata.
        
        Args:
            event_type: "local_hit" | "cloud_hit" | "offline_fallback" | "prefetch_operation"
            latency: Latency in milliseconds
            **kwargs: Additional metadata (e.g., count for prefetch)
        """
        with self.lock:
            event_data = {
                "timestamp": time.time(),
                "latency_ms": latency,
                **kwargs
            }
            
            if event_type in self.events:
                self.events[event_type].append(event_data)
            else:
                # Handle unknown event types
                if event_type not in self.events:
                    self.events[event_type] = []
                self.events[event_type].append(event_data)
    
    def log_query(self, query_data: Dict):
        """
        Log complete query with all metadata.
        
        Args:
            query_data: Dict with query details (from hybrid_router)
        """
        with self.lock:
            self.query_log.append({
                "timestamp": time.time(),
                **query_data
            })
    
    def log_anchor_event(self, event_type: str, **kwargs):
        """
        Log anchor system events.
        
        Args:
            event_type: "created" | "strengthened" | "weakened" | "evicted" | "transition"
            **kwargs: Event metadata
        """
        with self.lock:
            if event_type == "created":
                self.anchor_metrics["anchors_created"] += 1
            elif event_type == "strengthened":
                self.anchor_metrics["anchors_strengthened"] += 1
            elif event_type == "weakened":
                self.anchor_metrics["anchors_weakened"] += 1
            elif event_type == "evicted":
                self.anchor_metrics["anchors_evicted"] += 1
            elif event_type == "transition":
                self.anchor_metrics["type_transitions"].append({
                    "timestamp": time.time(),
                    **kwargs
                })
            elif event_type == "prediction_generated":
                self.anchor_metrics["predictions_generated"] += kwargs.get("count", 1)
            elif event_type == "prediction_hit":
                self.anchor_metrics["predictions_hit"] += 1
            elif event_type == "prediction_missed":
                self.anchor_metrics["predictions_missed"] += 1
    
    def log_cache_event(self, event_type: str, count: int = 1):
        """
        Log cache operations.
        
        Args:
            event_type: "add" | "eviction" | "compaction"
            count: Number of items affected
        """
        with self.lock:
            if event_type == "add":
                self.cache_metrics["cache_adds"] += count
            elif event_type == "eviction":
                self.cache_metrics["cache_evictions"] += count
            elif event_type == "compaction":
                self.cache_metrics["compactions"] += 1
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dict with all metrics aggregated
        """
        with self.lock:
            # Calculate total queries
            total_queries = sum(len(events) for event_type, events in self.events.items() 
                              if event_type in ["local_hit", "cloud_hit", "offline_fallback"])
            
            if total_queries == 0:
                return {
                    "status": "no_queries",
                    "message": "No queries recorded yet"
                }
            
            # Hit rates
            local_hits = len(self.events["local_hit"])
            cloud_hits = len(self.events["cloud_hit"])
            offline_hits = len(self.events["offline_fallback"])
            
            summary = {
                "total_queries": total_queries,
                "uptime_seconds": time.time() - self.start_time,
                
                # Hit rates
                "local_hit_rate": local_hits / total_queries if total_queries > 0 else 0,
                "cloud_hit_rate": cloud_hits / total_queries if total_queries > 0 else 0,
                "offline_rate": offline_hits / total_queries if total_queries > 0 else 0,
                
                # Latencies
                "latencies": self._calculate_latencies(),
                
                # Anchor system
                "anchor_system": {
                    **self.anchor_metrics,
                    "prediction_accuracy": (
                        self.anchor_metrics["predictions_hit"] / 
                        (self.anchor_metrics["predictions_hit"] + self.anchor_metrics["predictions_missed"])
                        if (self.anchor_metrics["predictions_hit"] + self.anchor_metrics["predictions_missed"]) > 0
                        else 0
                    )
                },
                
                # Cache
                "cache": self.cache_metrics,
                
                # Prefetch stats
                "prefetch_operations": len(self.events.get("prefetch_operation", []))
            }
            
            return summary
    
    def _calculate_latencies(self) -> Dict:
        """
        Calculate latency statistics (p50, p95, p99) per source.
        
        Returns:
            Dict with latency stats per source
        """
        latency_stats = {}
        
        for event_type in ["local_hit", "cloud_hit", "offline_fallback"]:
            latencies = [
                event["latency_ms"] 
                for event in self.events.get(event_type, []) 
                if event.get("latency_ms") is not None
            ]
            
            if latencies:
                latency_stats[event_type] = {
                    "count": len(latencies),
                    "mean": np.mean(latencies),
                    "p50": np.percentile(latencies, 50),
                    "p95": np.percentile(latencies, 95),
                    "p99": np.percentile(latencies, 99),
                    "min": np.min(latencies),
                    "max": np.max(latencies)
                }
            else:
                latency_stats[event_type] = None
        
        return latency_stats
    
    def get_learning_curve(self, window_size: int = 10) -> List[Dict]:
        """
        Calculate hit rate over time in windows.
        
        Shows how the system improves as it learns.
        
        Args:
            window_size: Number of queries per window
        
        Returns:
            List of {window_id, start_query, end_query, local_hit_rate}
        """
        with self.lock:
            if not self.query_log:
                return []
            
            learning_curve = []
            
            for i in range(0, len(self.query_log), window_size):
                window = self.query_log[i:i+window_size]
                
                local_hits = sum(1 for q in window if q.get("source") == "local")
                
                learning_curve.append({
                    "window_id": i // window_size,
                    "start_query": i,
                    "end_query": min(i + window_size, len(self.query_log)),
                    "total_queries": len(window),
                    "local_hit_rate": local_hits / len(window) if window else 0,
                    "timestamp": window[0]["timestamp"] if window else None
                })
            
            return learning_curve
    
    def export_to_json(self, filepath: str):
        """
        Export all metrics to JSON file.
        
        Args:
            filepath: Output file path
        """
        with self.lock:
            data = {
                "summary": self.get_summary(),
                "learning_curve": self.get_learning_curve(),
                "query_log": self.query_log[-100:],  # Last 100 queries
                "anchor_transitions": self.anchor_metrics["type_transitions"][-50:],  # Last 50 transitions
                "exported_at": time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"[METRICS] Exported to {filepath}")
    
    def reset(self):
        """Reset all metrics (for testing)."""
        with self.lock:
            for event_type in self.events:
                self.events[event_type] = []
            
            self.query_log = []
            
            self.anchor_metrics = {
                "anchors_created": 0,
                "anchors_strengthened": 0,
                "anchors_weakened": 0,
                "anchors_evicted": 0,
                "type_transitions": [],
                "predictions_generated": 0,
                "predictions_hit": 0,
                "predictions_missed": 0
            }
            
            self.cache_metrics = {
                "cache_adds": 0,
                "cache_evictions": 0,
                "compactions": 0
            }
            
            self.last_reset_time = time.time()
            
            print("[METRICS] Reset complete")
    
    def print_summary(self):
        """Print human-readable summary to console."""
        summary = self.get_summary()
        
        if summary.get("status") == "no_queries":
            print("\n[METRICS] No queries recorded yet")
            return
        
        print("\n" + "="*70)
        print("METRICS SUMMARY")
        print("="*70)
        
        print(f"\n📊 Query Statistics:")
        print(f"   Total Queries:        {summary['total_queries']}")
        print(f"   Local Hit Rate:       {summary['local_hit_rate']:.1%}")
        print(f"   Cloud Hit Rate:       {summary['cloud_hit_rate']:.1%}")
        print(f"   Offline Rate:         {summary['offline_rate']:.1%}")
        
        if summary['latencies']:
            print(f"\n⚡ Latencies:")
            for source, stats in summary['latencies'].items():
                if stats:
                    print(f"   {source:20s} p50={stats['p50']:6.1f}ms  p95={stats['p95']:6.1f}ms")
        
        print(f"\n⚓ Anchor System:")
        anchor = summary['anchor_system']
        print(f"   Anchors Created:      {anchor['anchors_created']}")
        print(f"   Anchors Strengthened: {anchor['anchors_strengthened']}")
        print(f"   Predictions Generated:{anchor['predictions_generated']}")
        print(f"   Prediction Accuracy:  {anchor['prediction_accuracy']:.1%}")
        
        print(f"\n💾 Cache:")
        cache = summary['cache']
        print(f"   Cache Adds:           {cache['cache_adds']}")
        print(f"   Cache Evictions:      {cache['cache_evictions']}")
        print(f"   Compactions:          {cache['compactions']}")
        
        print(f"\n🔮 Prefetch:")
        print(f"   Operations:           {summary['prefetch_operations']}")
        
        print(f"\n⏱️  Uptime:              {summary['uptime_seconds']:.1f}s")
        print("="*70 + "\n")