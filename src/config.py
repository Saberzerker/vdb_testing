# src/config.py
"""
Hybrid VDB System Configuration
Optimized for 5K cloud + 1K local (300 permanent + 700 dynamic)

Author: Saberzerker
Date: 2025-11-16
"""

import os
import logging

# ================================================================
# QDRANT CLOUD CONFIGURATION
# ================================================================

QDRANT_URL = ""
QDRANT_API_KEY = ""
QDRANT_COLLECTION_NAME = ""

# ================================================================
# VECTOR & EMBEDDING CONFIGURATION
# ================================================================

VECTOR_DIMENSION = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ================================================================
# STORAGE PATHS
# ================================================================

# Base directory for VDB storage
STORAGE_BASE_DIR = "data/vdb"

# Permanent layer path
BASE_LAYER_PATH = os.path.join(STORAGE_BASE_DIR, "permanent_layer")

# Dynamic layer path
DYNAMIC_LAYER_PATH = os.path.join(STORAGE_BASE_DIR, "dynamic_layer")

# Hot partition path (RAM)
HOT_PARTITION_PATH = os.path.join(DYNAMIC_LAYER_PATH, "hot")

# Delta segments path (disk)
DELTA_SEGMENTS_PATH = os.path.join(DYNAMIC_LAYER_PATH, "delta_segments")

# Anchor storage path
ANCHOR_STORAGE_PATH = os.path.join(STORAGE_BASE_DIR, "anchors")

# Metrics output path
METRICS_PATH = "results"

# Logs path
LOGS_PATH = "logs"

# Cache layer path (alias for DYNAMIC_LAYER_PATH)
CACHE_LAYER_PATH = DYNAMIC_LAYER_PATH

# Compaction interval (in seconds, not just hours)

# ================================================================
# STORAGE CAPACITY (Optimized for 5K cloud + 1K local)
# ================================================================

# Cloud capacity (actual)
CLOUD_CAPACITY = 9482  # Actual vectors in cloud

# Total local VDB capacity (20% of cloud)
TOTAL_LOCAL_CAPACITY = 1000  # vectors

# Permanent layer (30% of local)
PERMANENT_LAYER_CAPACITY = 300  # essential medical facts

# Dynamic layer (70% of local)
DYNAMIC_LAYER_CAPACITY = 700  # learned + prefetched

# Hot partition (50% of dynamic = 350 in RAM)
HOT_PARTITION_CAPACITY = 350  # most recent queries

# Delta segments (50% of dynamic = 350 on disk)
DELTA_SEGMENT_CAPACITY = 350  # overflow storage

# Memory limits
HOT_PARTITION_RAM_LIMIT = 2 * 1024 * 1024  # 2 MB

# Compaction settings
COMPACTION_THRESHOLD = 3  # Merge when 3+ delta segments
COMPACTION_INTERVAL = 6 * 60 * 60  # 6 hours in seconds

COMPACTION_INTERVAL_SECONDS = COMPACTION_INTERVAL

# ================================================================
# ANCHOR SYSTEM CONFIGURATION (Novel Learning Layer)
# ================================================================

# Maximum concurrent anchors
MAX_ANCHORS = 50

# Anchor strength thresholds for tier promotion
WEAK_ANCHOR_THRESHOLD = 25.0      # 0-25: WEAK
MEDIUM_ANCHOR_THRESHOLD = 60.0    # 25-60: MEDIUM
STRONG_ANCHOR_THRESHOLD = 90.0    # 60-90: STRONG
# 90-100: PERMANENT

# Minimum anchor strength (prune anchors below this)
MIN_ANCHOR_STRENGTH = 5.0

# Anchor decay rates (multiplier per hour)
WEAK_DECAY_RATE = 0.5      # Lose 50% per hour
MEDIUM_DECAY_RATE = 0.8    # Lose 20% per hour
STRONG_DECAY_RATE = 0.9    # Lose 10% per hour
PERMANENT_DECAY_RATE = 1.0  # No decay

# Anchor strength changes
ANCHOR_CREATION_STRENGTH = 15.0      # Initial strength for new anchor
PREDICTION_HIT_REWARD = 10.0         # Reward for correct prediction
PREDICTION_MISS_PENALTY = 2.0        # Penalty for wrong prediction
QUERY_HIT_BOOST = 5.0               # Boost for direct query hit

# Cluster threshold for anchor creation
CLUSTER_THRESHOLD = 0.15  # Max distance to merge into existing anchor

# Time thresholds
ANCHOR_DECAY_CHECK_INTERVAL = 3600   # Check decay every hour (seconds)
ANCHOR_WEAK_TIMEOUT = 3600           # Remove weak anchors after 1 hour no hits
ANCHOR_MEDIUM_TIMEOUT = 21600        # 6 hours
ANCHOR_STRONG_TIMEOUT = 86400        # 24 hours

# ================================================================
# PREDICTION GENERATION CONFIGURATION
# ================================================================

# Prefetch settings
PREFETCH_ENABLED = True

# Predictions per anchor type
PREDICTIONS_PER_WEAK_ANCHOR = 3
PREDICTIONS_PER_MEDIUM_ANCHOR = 5
PREDICTIONS_PER_STRONG_ANCHOR = 7
PREDICTIONS_PER_PERMANENT_ANCHOR = 10

# Prediction generation strategy
PREDICTION_STRATEGY = "trajectory"  # Options: "trajectory", "centroid", "hybrid"
PREDICTION_STEP_SIZE = 0.15         # Distance to step in semantic space
PREDICTION_RADIUS = 0.10            # Spread radius for prediction diversity

# Prediction hit detection
PREDICTION_HIT_THRESHOLD = 0.85     # Cosine similarity to count as hit
PREDICTION_MATCH_THRESHOLD = 0.85   # Alias for consistency

# ================================================================
# HYBRID ROUTING CONFIGURATION
# ================================================================

# Local search first
LOCAL_SEARCH_FIRST = True

# Confidence threshold for local-only results
LOCAL_CONFIDENCE_THRESHOLD = 0.75

# Top-K results
DEFAULT_SEARCH_K = 5

# Fallback behavior
CLOUD_FALLBACK_ENABLED = True
CLOUD_TIMEOUT_SECONDS = 10.0

# ================================================================
# SEMANTIC CACHE CONFIGURATION
# ================================================================

# Semantic clustering
SEMANTIC_DRIFT_THRESHOLD = 0.35     # Cluster similarity threshold
MOMENTUM_ALPHA = 0.85               # Momentum decay factor
MAX_CLUSTERS = 10                   # Maximum semantic clusters

# Cluster momentum thresholds
CLUSTER_HOT_THRESHOLD = 5.0         # High momentum cluster
CLUSTER_WARM_THRESHOLD = 2.0        # Medium momentum
CLUSTER_COLD_THRESHOLD = 0.5        # Low momentum (evict)

# Momentum decay
MOMENTUM_DECAY_RATE = 0.95          # Per time unit
MOMENTUM_DECAY_INTERVAL = 60   

# Cluster TTL (Time To Live)
CLUSTER_TTL_SECONDS = 300           # 5 minutes (ADD THIS LINE)
CLUSTER_MAX_AGE = 1800    # Seconds

# ================================================================
# PERFORMANCE & RELIABILITY
# ================================================================

# Search settings
SEARCH_TIMEOUT_LOCAL = 1.0          # 1 second local timeout
SEARCH_TIMEOUT_CLOUD = 10.0         # 10 second cloud timeout

# Batch processing
BATCH_SIZE_EMBEDDING = 32           # Embed 32 queries at once
BATCH_SIZE_UPLOAD = 100             # Upload 100 vectors at once

# Retry settings
MAX_RETRIES_CLOUD = 3
RETRY_DELAY_SECONDS = 1.0

# Cache settings
ENABLE_QUERY_CACHE = True           # Cache recent query results
QUERY_CACHE_SIZE = 100              # Cache last 100 queries
QUERY_CACHE_TTL = 300               # 5 minutes

# ================================================================
# LOGGING & DEBUG
# ================================================================

LOG_LEVEL = logging.INFO
LOG_FILE = os.path.join(LOGS_PATH, "hybrid_vdb.log")

# Verbose logging flags
VERBOSE = True
LOG_ANCHOR_EVENTS = True            # Log anchor creation/promotion/deletion
LOG_PREDICTIONS = True              # Log prediction generation/hits
LOG_REINFORCEMENT = True            # Log learning signals
LOG_SEARCH_DETAILS = False          # Detailed search logging (noisy)
LOG_TIMING = True                   # Log operation timing

# Console output
CONSOLE_LOG_LEVEL = logging.INFO
SHOW_PROGRESS_BARS = True

# ================================================================
# METRICS & MONITORING
# ================================================================

# Metrics collection
ENABLE_METRICS = True
METRICS_EXPORT_INTERVAL = 10        # Export every 10 queries
METRICS_EXPORT_PATH = os.path.join(METRICS_PATH, "session_metrics.json")

# Tracked metrics
TRACK_LATENCY = True
TRACK_HIT_RATE = True
TRACK_PREDICTION_ACCURACY = True
TRACK_ANCHOR_STATS = True
TRACK_MEMORY_USAGE = True

# Performance baselines
ENABLE_BASELINE_COMPARISON = True   # Compare to cloud-only baseline
EXPORT_DETAILED_METRICS = True      # Export comprehensive metrics

# ================================================================
# EXPERIMENT CONFIGURATION
# ================================================================

# A/B Testing
ENABLE_AB_TESTING = False           # Enable A/B test mode
AB_TEST_RATIO = 0.5                 # 50% with anchors, 50% without

# Evaluation mode
EVALUATION_MODE = False             # Disable learning for evaluation
COLD_START_MODE = False             # Start with empty local cache

# Simulation settings (for testing)
SIMULATE_CLOUD_LATENCY = False      # Add artificial cloud delay
SIMULATED_CLOUD_LATENCY_MS = 200    # Simulated latency

# ================================================================
# OLLAMA / LLM CONFIGURATION (for demo)
# ================================================================

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"        # Lightweight for demo
OLLAMA_TIMEOUT = 30                 # 30 seconds for generation

# RAG settings
RAG_CONTEXT_SIZE = 3                # Use top-3 retrieved documents
RAG_MAX_CONTEXT_LENGTH = 1500       # Max characters per document
RAG_TEMPERATURE = 0.7               # Generation temperature

# ================================================================
# SYSTEM LIMITS
# ================================================================

# Safety limits
MAX_QUERY_LENGTH = 1000             # Max characters in query
MAX_DOCUMENT_LENGTH = 10000         # Max characters in document
MAX_BATCH_SIZE = 1000               # Max vectors in single operation

# Resource limits
MAX_MEMORY_MB = 100                 # Max RAM usage for VDB
MAX_DISK_MB = 500                   # Max disk usage for VDB

# ================================================================
# FEATURE FLAGS
# ================================================================

# Core features
ENABLE_ANCHOR_SYSTEM = True         # Novel anchor-based learning
ENABLE_PREDICTIVE_PREFETCH = True   # Trajectory-based prefetching
ENABLE_SEMANTIC_CACHE = True        # Momentum-based clustering
ENABLE_TWO_TIER_STORAGE = True      # Permanent + dynamic layers

# Advanced features
ENABLE_FEDERATED_LEARNING = False   # Multi-device learning (future)
ENABLE_PERSONALIZATION = False      # Per-user anchors (future)
ENABLE_DRIFT_DETECTION = True       # Query distribution drift detection



# ================================================================
# SEMANTIC CACHE - ADDITIONAL SETTINGS
# ================================================================

# Cluster reinforcement
MIN_CLUSTER_REINFORCEMENT_SCORE = 10.0
MAX_CLUSTER_REINFORCEMENT_SCORE = 100.0
CLUSTER_REINFORCEMENT_DECAY = 0.95
CLUSTER_REINFORCEMENT_BOOST = 5.0
CLUSTER_CREATION_THRESHOLD = 3      # Min queries to create cluster
CLUSTER_MERGE_THRESHOLD = 0.85      # Similarity to merge clusters
MIN_CLUSTER_SIZE = 2                # Min vectors per cluster

# Cache eviction
CACHE_EVICTION_POLICY = "MOMENTUM"  # LRU, LFU, or MOMENTUM
MAX_CACHE_SIZE_MB = 50              # Max cache size
CACHE_CLEANUP_INTERVAL = 300        # Cleanup every 5 minutes


# ================================================================
# BACKWARD COMPATIBILITY ALIASES (for anchor_system.py)
# ================================================================

# Old anchor strength threshold names
ANCHOR_STRENGTH_WEAK = 0
ANCHOR_STRENGTH_MEDIUM = WEAK_ANCHOR_THRESHOLD      # 25.0
ANCHOR_STRENGTH_STRONG = MEDIUM_ANCHOR_THRESHOLD    # 60.0
ANCHOR_STRENGTH_PERMANENT = STRONG_ANCHOR_THRESHOLD # 90.0

# Old decay rate names (inverted logic: old rates are per-cycle loss amounts)
ANCHOR_DECAY_RATE_WEAK = 0.5        # Loses 0.5 per cycle
ANCHOR_DECAY_RATE_MEDIUM = 0.2      # Loses 0.2 per cycle
ANCHOR_DECAY_RATE_STRONG = 0.1      # Loses 0.1 per cycle
ANCHOR_DECAY_RATE_PERMANENT = 0.0   # No loss

# Prefetch config alias
PREFETCH_K = 5

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def create_directories():
    """Create necessary directories for VDB system."""
    dirs = [
        STORAGE_BASE_DIR,
        BASE_LAYER_PATH,
        DYNAMIC_LAYER_PATH,
        HOT_PARTITION_PATH,
        DELTA_SEGMENTS_PATH,
        ANCHOR_STORAGE_PATH,
        METRICS_PATH,
        LOGS_PATH
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def get_config_summary():
    """Return configuration summary as dict."""
    return {
        "cloud": {
            "url": QDRANT_URL,
            "collection": QDRANT_COLLECTION_NAME,
            "capacity": CLOUD_CAPACITY
        },
        "local": {
            "total_capacity": TOTAL_LOCAL_CAPACITY,
            "permanent": PERMANENT_LAYER_CAPACITY,
            "dynamic": DYNAMIC_LAYER_CAPACITY,
            "hot_partition": HOT_PARTITION_CAPACITY,
            "delta_segments": DELTA_SEGMENT_CAPACITY
        },
        "anchors": {
            "max_anchors": MAX_ANCHORS,
            "weak_threshold": WEAK_ANCHOR_THRESHOLD,
            "medium_threshold": MEDIUM_ANCHOR_THRESHOLD,
            "strong_threshold": STRONG_ANCHOR_THRESHOLD
        },
        "features": {
            "anchor_system": ENABLE_ANCHOR_SYSTEM,
            "prefetch": ENABLE_PREDICTIVE_PREFETCH,
            "semantic_cache": ENABLE_SEMANTIC_CACHE,
            "two_tier": ENABLE_TWO_TIER_STORAGE
        }
    }


# Auto-create directories on import

create_directories()
