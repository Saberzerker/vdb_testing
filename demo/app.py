# demo/app.py
"""
Hybrid VDB Demo Application - Production Web Interface

Flask-based web interface showing:
- Real-time query processing
- Three-tier architecture visualization
- Learning curve over time (TIER 2 prefetch effectiveness)
- Anchor graph visualization
- Dynamic space fill rate
- Prefetch efficiency metrics
- Retrieval vs total latency breakdown

Key Features (2025-11-17 Update):
- TIER 2 hit rate = prefetch effectiveness (neighborhood coverage)
- Avg latency = retrieval time only (excludes LLM)
- Query count tracking
- Hit count display (X/Y format)
- Learning curve shows TIER 2 hits over time

Author: Saberzerker
Date: 2025-11-17
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import logging
import requests

# Import hybrid VDB components
from src.hybrid_router import HybridRouter
from src.local_vdb import LocalVDB
from src.cloud_client import QdrantCloudClient
from src.semantic_cache import SemanticClusterCache
from src.anchor_system import AnchorSystem
from src.metrics import MetricsTracker
from src.config import (
    Config,
    DEMO_HOST,
    DEMO_PORT,
    DEMO_DEBUG,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    RAG_CONTEXT_SIZE,
    USE_REAL_SYSTEM,
    VERBOSE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════════════
# GLOBAL INITIALIZATION
# ═══════════════════════════════════════════════════════════

config = Config()

# Initialize embedding model
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info("[APP] ✅ Loaded embedding model")
except Exception as e:
    logger.error(f"[APP] Failed to load embeddings: {e}")
    embedder = None

# Global system components
local_vdb = None
cloud_client = None
semantic_cache = None
anchor_system = None
metrics_tracker = None
router = None

if USE_REAL_SYSTEM:
    try:
        logger.info("[APP] Initializing hybrid VDB system...")
        
        # Initialize local VDB
        local_vdb = LocalVDB(config)
        
        # Initialize cloud client
        cloud_client = QdrantCloudClient(
            url=config.CLOUD_URL,
            api_key=config.CLOUD_API_KEY,
            collection_name=config.CLOUD_COLLECTION_NAME,
            dimension=config.VECTOR_DIMENSION
        )
        
        # Initialize other components
        semantic_cache = SemanticClusterCache()
        anchor_system = AnchorSystem()
        metrics_tracker = MetricsTracker()
        
        # Initialize router
        router = HybridRouter(
            local_vdb=local_vdb,
            cloud_vdb=cloud_client,
            semantic_cache=semantic_cache,
            anchor_system=anchor_system,
            metrics=metrics_tracker
        )
        
        logger.info("[APP] ✅ Hybrid VDB system initialized")
        logger.info(f"[APP] Cloud: {config.CLOUD_URL}")
        logger.info(f"[APP] Collection: {config.CLOUD_COLLECTION_NAME}")
        
    except Exception as e:
        logger.error(f"[APP] Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        USE_REAL_SYSTEM = False
        logger.warning("[APP] Falling back to non-real system mode")

# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def query_llama(question: str, context_docs: list) -> str:
    """
    Query Ollama LLM with RAG context.
    
    Args:
        question: User question
        context_docs: Retrieved documents for context
    
    Returns:
        Generated answer
    """
    try:
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.get('text', str(doc))[:500]}"
            for i, doc in enumerate(context_docs[:RAG_CONTEXT_SIZE])
        ])
        
        # Medical-friendly prompt
        prompt = f"""You are a medical information assistant. Answer the user's question using the provided medical documents.

Medical Documents:
{context}

User Question: {question}

Instructions:
- Provide clear, factual medical information
- Base your answer on the documents above
- Be concise (2-3 paragraphs maximum)
- If the documents don't contain enough information, say so clearly
- Use simple language that patients can understand

Answer:"""
        
        # Query Ollama
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 250
                }
            },
            timeout=45
        )
        
        if response.status_code == 200:
            answer = response.json().get("response", "No response generated")
            return answer.strip()
        else:
            return f"Error: Ollama returned status {response.status_code}"
    
    except Exception as e:
        logger.error(f"[LLM] Error: {e}")
        return f"Error generating answer: {str(e)}"


def format_time_delta(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def handle_query():
    """
    Handle medical query with full visualization data.
    
    NEW (2025-11-17):
    - Tracks TIER 2 prefetch effectiveness
    - Separates retrieval vs total latency
    - Shows neighborhood coverage status
    - Returns query count and hit count
    
    Returns:
        JSON with:
        - answer: LLM-generated response
        - retrieved_docs: Context documents
        - flow: Step-by-step processing visualization
        - metrics: Comprehensive performance metrics
    """
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    if not embedder:
        return jsonify({'error': 'Embedding model not loaded'}), 500
    
    start_time = time.time()
    
    # ══════════════════════════════════════════════════════════
    # STEP 1: Generate Embedding
    # ══════════════════════════════════════════════════════════
    
    embedding_start = time.time()
    try:
        query_vector = embedder.encode([question])[0]
    except Exception as e:
        return jsonify({'error': f'Embedding failed: {str(e)}'}), 500
    
    embedding_time = (time.time() - embedding_start) * 1000
    
    # ══════════════════════════════════════════════════════════
    # STEP 2: Search Hybrid VDB
    # ══════════════════════════════════════════════════════════
    
    if USE_REAL_SYSTEM and router:
        query_id = f"q_{int(time.time() * 1000)}"
        
        try:
            search_start = time.time()
            
            # Search through hybrid router
            results = router.search(
                query_vector=query_vector,
                query_id=query_id,
                query_text=question,
                k=RAG_CONTEXT_SIZE
            )
            
            search_time = (time.time() - search_start) * 1000
            
            # Extract documents
            retrieved_docs = []
            if 'ids' in results and results['ids']:
                for doc_id in results['ids'][:RAG_CONTEXT_SIZE]:
                    retrieved_docs.append({
                        'id': doc_id,
                        'text': f"Medical document {doc_id}"
                    })
            
            source = results.get('source', 'unknown')
            
        except Exception as e:
            logger.error(f"[APP] Router search failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Search failed: {str(e)}'}), 500
    
    else:
        # Fallback mode (when USE_REAL_SYSTEM=False)
        search_start = time.time()
        retrieved_docs = [
            {'id': 'fallback_1', 'text': 'Fallback document 1'},
            {'id': 'fallback_2', 'text': 'Fallback document 2'}
        ]
        search_time = (time.time() - search_start) * 1000
        source = 'fallback'
        results = {}
    
    # ══════════════════════════════════════════════════════════
    # STEP 3: Generate Answer with LLM
    # ══════════════════════════════════════════════════════════
    
    llm_start = time.time()
    answer = query_llama(question, retrieved_docs)
    llm_time = (time.time() - llm_start) * 1000
    
    total_time = (time.time() - start_time) * 1000
    
    # ══════════════════════════════════════════════════════════
    # STEP 4: Get Metrics FIRST (before building flow)
    # ══════════════════════════════════════════════════════════
    
    if USE_REAL_SYSTEM and metrics_tracker:
        metrics = metrics_tracker.get_summary()
        
        # Get storage stats
        if local_vdb:
            storage_stats = local_vdb.get_stats()
        else:
            storage_stats = {
                'dynamic_layer_vectors': 0,
                'dynamic_capacity': 700,
                'permanent_layer_vectors': 0
            }
        
        # Get anchor stats
        if anchor_system:
            anchor_stats = anchor_system.get_anchor_stats()
        else:
            anchor_stats = {
                'total_anchors': 0,
                'active_predictions': 0,
                'prediction_accuracy': 0
            }
    else:
        # Fallback metrics
        metrics = {
            'total_queries': 1,
            'local_hit_rate': 0,
            'tier2_prefetch_rate': 0,
            'avg_latency': search_time,
            'tier1_hit_rate': 0,
            'tier2_hit_rate': 0,
            'tier3_hit_rate': 100,
            'prefetch_cache_hit_rate': 0,
            'tier2_hits': 0,
            'tier2_attempts': 1
        }
        storage_stats = {
            'dynamic_layer_vectors': 0,
            'dynamic_capacity': 700,
            'permanent_layer_vectors': 0
        }
        anchor_stats = {
            'total_anchors': 0,
            'active_predictions': 0,
            'prediction_accuracy': 0
        }
    
    # ══════════════════════════════════════════════════════════
    # STEP 5: Build Detailed Flow Visualization
    # ══════════════════════════════════════════════════════════
    
    flow_steps = []
    
    # Step 1: Embedding
    flow_steps.append({
        'step': 1,
        'name': '🧠 Generate Embedding',
        'time_ms': round(embedding_time, 1),
        'status': 'success',
        'detail': 'Convert query to 384-dim vector'
    })
    
    if USE_REAL_SYSTEM and router:
        # Step 2: Neighborhood Coverage (shows prefetch effectiveness)
        if source == 'tier2_dynamic':
            flow_steps.append({
                'step': 2,
                'name': '🎯 Neighborhood Coverage',
                'time_ms': 1.0,
                'status': 'hit',
                'detail': 'Query in prefetched neighborhood! Prediction worked ✓'
            })
        else:
            flow_steps.append({
                'step': 2,
                'name': '🎯 Neighborhood Coverage',
                'time_ms': 1.0,
                'status': 'miss',
                'detail': 'Query outside coverage zones, fetching from cloud'
            })
        
        # Step 3: Semantic Clustering
        flow_steps.append({
            'step': 3,
            'name': '📊 Semantic Clustering',
            'time_ms': 1.0,
            'status': 'success',
            'detail': f"Cluster #{results.get('cluster_id', 0)} | {results.get('cluster_action', 'check')} | Access: {results.get('cluster_access_count', 0)}"
        })
        
        # Step 4: TIER 1 Check
        tier1_status = 'hit' if source == 'tier1_permanent' else 'miss'
        flow_steps.append({
            'step': 4,
            'name': '🏠 TIER 1: Permanent',
            'time_ms': round(results.get('tier1_latency_ms', 0), 1),
            'status': tier1_status,
            'detail': f"Privacy layer | {results.get('tier1_results', 0)} results"
        })
        
        # Step 5: TIER 2 Check (DYNAMIC - Learning Layer)
        tier2_status = 'hit' if source == 'tier2_dynamic' else 'miss'
        
        # Get dynamic fill rate
        dynamic_vecs = storage_stats.get('dynamic_layer_vectors', 0)
        dynamic_cap = storage_stats.get('dynamic_capacity', 700)
        fill_rate_str = f"{dynamic_vecs}/{dynamic_cap}"
        
        if source == 'tier2_dynamic':
            tier2_detail = f"Learning space | {fill_rate_str} | {results.get('tier2_results', 0)} results | LOCAL ✓"
        else:
            tier2_detail = f"Learning space | {fill_rate_str} | 0 results"
        
        flow_steps.append({
            'step': 5,
            'name': '🎒 TIER 2: Dynamic',
            'time_ms': round(results.get('tier2_latency_ms', 0), 1),
            'status': tier2_status,
            'detail': tier2_detail
        })
        
        # Step 6: TIER 3 Check (only if cloud hit or offline)
        if source == 'tier3_cloud':
            flow_steps.append({
                'step': 6,
                'name': '☁️ TIER 3: Cloud',
                'time_ms': round(results.get('tier3_latency_ms', 0), 1),
                'status': 'hit',
                'detail': f"Canonical truth | Cached {results.get('cached_to_tier2', 3)} to TIER 2"
            })
        elif source == 'offline_fallback':
            flow_steps.append({
                'step': 6,
                'name': '⚠️ Offline Fallback',
                'time_ms': 0,
                'status': 'offline',
                'detail': 'Cloud unreachable, using best local result'
            })
    
    # Final Step: LLM Generation
    flow_steps.append({
        'step': len(flow_steps) + 1,
        'name': '🤖 Generate Answer',
        'time_ms': round(llm_time, 1),
        'status': 'success',
        'detail': f'{OLLAMA_MODEL}'
    })
    
    # ══════════════════════════════════════════════════════════
    # STEP 6: Build Response
    # ══════════════════════════════════════════════════════════
    
    # Extract document texts
    doc_texts = []
    for doc in retrieved_docs[:3]:
        if isinstance(doc, dict):
            text = doc.get('text', str(doc))
        else:
            text = str(doc)
        doc_texts.append(text[:300] + '...' if len(text) > 300 else text)
    
    response = {
        'answer': answer,
        'retrieved_docs': doc_texts,
        'flow': flow_steps,
        'metrics': {
            # Latency breakdown
            'total_latency_ms': round(total_time, 1),
            'search_latency_ms': round(search_time, 1),  # Retrieval time
            'llm_latency_ms': round(llm_time, 1),
            
            # Average retrieval latency (for dashboard display)
            'avg_retrieval_latency_ms': round(metrics.get('avg_latency', 0), 1),
            'avg_latency_ms': round(metrics.get('avg_latency', 0), 1),  # Backwards compat
            
            # Source & query info
            'source': source,
            'query_number': metrics.get('total_queries', 0),
            'total_queries': metrics.get('total_queries', 0),
            
            # Hit rates (percentage of total queries)
            'tier1_hit_rate': round(metrics.get('tier1_hit_rate', 0), 1),
            'tier2_hit_rate': round(metrics.get('tier2_hit_rate', 0), 1),
            'tier3_hit_rate': round(metrics.get('tier3_hit_rate', 0), 1),
            'local_hit_rate': round(metrics.get('local_hit_rate', 0), 1),
            
            # TIER 2 prefetch performance (for learning curve!) ⭐
            # This is the KEY metric showing neighborhood coverage effectiveness
            'tier2_prefetch_rate': round(metrics.get('tier2_prefetch_rate', 0), 1),
            'tier2_hits': metrics.get('tier2_hits', 0),
            'tier2_attempts': metrics.get('tier2_attempts', 0),
            
            # Storage
            'dynamic_vectors': storage_stats.get('dynamic_layer_vectors', 0),
            'dynamic_capacity': storage_stats.get('dynamic_capacity', 700),
            'dynamic_fill_rate': round(
                (storage_stats.get('dynamic_layer_vectors', 0) / 700 * 100), 1
            ) if storage_stats.get('dynamic_layer_vectors', 0) > 0 else 0,
            'permanent_vectors': storage_stats.get('permanent_layer_vectors', 0),
            
            # Anchors
            'total_anchors': anchor_stats.get('total_anchors', 0),
            'active_predictions': anchor_stats.get('active_predictions', 0),
            'prediction_accuracy': round(anchor_stats.get('prediction_accuracy', 0), 1),
            
            # Prefetch efficiency (cache hit rate during prefetch operations)
            'prefetch_cache_hit_rate': round(metrics.get('prefetch_cache_hit_rate', 0), 1),
            
            # Learning curve data
            'learning_curve': metrics_tracker.get_learning_curve() if metrics_tracker else {'has_data': False}
        }
    }
    
    return jsonify(response)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get comprehensive system statistics.
    
    Returns:
        - router: Router statistics
        - metrics: Metrics summary
        - learning_curve: Learning progression data
        - anchor_graph: Anchor network visualization data
    """
    if not USE_REAL_SYSTEM:
        return jsonify({'error': 'Real system not initialized'}), 503
    
    try:
        stats = {
            'router': router.get_stats() if router else {},
            'metrics': metrics_tracker.get_summary() if metrics_tracker else {},
            'learning_curve': metrics_tracker.get_learning_curve() if metrics_tracker else {},
            'anchor_graph': anchor_system.get_anchor_graph() if anchor_system else {'nodes': [], 'edges': []}
        }
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"[APP] Stats error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset_system():
    """
    Reset system metrics (start fresh session).
    
    Clears:
    - All query history
    - Hit rate tracking
    - Latency history
    - Learning curve data
    """
    if metrics_tracker:
        metrics_tracker.reset()
        logger.info("[APP] Metrics reset by user")
    
    return jsonify({'status': 'reset', 'message': 'All metrics cleared'})


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        System status and component readiness
    """
    return jsonify({
        'status': 'healthy',
        'system_enabled': USE_REAL_SYSTEM,
        'embedding_loaded': embedder is not None,
        'router_initialized': router is not None,
        'metrics_tracking': metrics_tracker is not None
    })

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info(f"[APP] Starting demo app on {DEMO_HOST}:{DEMO_PORT}")
    logger.info(f"[APP] Real system: {USE_REAL_SYSTEM}")
    logger.info(f"[APP] Verbose mode: {VERBOSE}")
    
    app.run(
        host=DEMO_HOST,
        port=DEMO_PORT,
        debug=DEMO_DEBUG
    )