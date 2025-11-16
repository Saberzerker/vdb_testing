# demo/app.py
"""
Medical RAG Demo with REAL Hybrid VDB + Anchor System
Shows actual learning and predictive caching.

Author: Saberzerker
Date: 2025-11-16
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify
import requests
import time

# Import YOUR actual components
try:
    from src.hybrid_router import HybridRouter
    from src.anchor_system import AnchorSystem
    from src.cloud_client import QdrantCloudClient
    from src.local_vdb import LocalVDB
    from src.metrics import MetricsTracker
    from sentence_transformers import SentenceTransformer

    USE_REAL_SYSTEM = True
    print("✅ Using REAL Hybrid VDB system")
except ImportError as e:
    print(f"⚠️  Could not import real components: {e}")
    print("⚠️  Falling back to simple wrappers")
    from src.simple_wrapper import (
        SimpleCloudClient,
        SimpleLocalVDB,
        get_embedding_model,
    )

    USE_REAL_SYSTEM = False

from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    VECTOR_DIMENSION,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAG_CONTEXT_SIZE,
    RAG_MAX_CONTEXT_LENGTH,
    RAG_TEMPERATURE,
)

# Initialize Flask app
app = Flask(__name__)

# Global system components
router = None
metrics_tracker = None
embedder = None
cloud_client = None
local_vdb = None
anchor_system = None


def initialize_system():
    """Initialize the system (real or fallback)."""
    global router, metrics_tracker, embedder, cloud_client, local_vdb, anchor_system

    print("\n" + "=" * 70)
    print("INITIALIZING MEDICAL RAG SYSTEM")
    print("=" * 70 + "\n")

    if USE_REAL_SYSTEM:
        return initialize_real_system()
    else:
        return initialize_simple_system()


def initialize_real_system():
    """Initialize with YOUR actual Hybrid VDB components."""
    global router, metrics_tracker, embedder, cloud_client, local_vdb, anchor_system

    # Load embedding model
    print("[1/7] Loading embedding model...")  # Changed from 1/5
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model ready\n")

    # Initialize cloud client
    print("[2/7] Connecting to Qdrant Cloud...")
    try:
        cloud_client = QdrantCloudClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=QDRANT_COLLECTION_NAME,
            dimension=VECTOR_DIMENSION,
        )
        print(f"✅ Cloud ready\n")
    except Exception as e:
        print(f"❌ Cloud connection failed: {e}\n")
        return False

    # Initialize local VDB
    print("[3/7] Loading local VDB...")
    try:
        local_vdb = LocalVDB()
        print(f"✅ Local VDB ready\n")
    except Exception as e:
        print(f"❌ Local VDB failed: {e}\n")
        return False

    # Initialize semantic cache
    print("[4/7] Initializing semantic cache...")
    try:
        from src.semantic_cache import SemanticClusterCache

        semantic_cache = SemanticClusterCache()
        print("✅ Semantic cache ready\n")
    except Exception as e:
        print(f"❌ Semantic cache failed: {e}\n")
        return False

    # Initialize anchor system
    print("[5/7] Initializing anchor system...")
    try:
        anchor_system = AnchorSystem()
        print("✅ Anchor system ready\n")
    except Exception as e:
        print(f"❌ Anchor system failed: {e}\n")
        return False

    # Initialize metrics
    print("[6/7] Initializing metrics tracker...")
    try:
        metrics_tracker = MetricsTracker()
        print("✅ Metrics ready\n")
    except Exception as e:
        print(f"❌ Metrics failed: {e}\n")
        return False

    # Initialize updater
    print("[7/7] Initializing updater and router...")
    try:
        from src.updater import Updater
        from src import config

        updater = Updater(
            local_vdb=local_vdb,
            cloud_vdb=cloud_client,
            semantic_cache=semantic_cache,
            config=config,
        )

        # Initialize hybrid router
        router = HybridRouter(
            local_vdb=local_vdb,
            cloud_vdb=cloud_client,
            semantic_cache=semantic_cache,
            anchor_system=anchor_system,
            updater=updater,
            metrics=metrics_tracker,
        )
        print("✅ Hybrid router ready\n")
    except Exception as e:
        print(f"❌ Router initialization failed: {e}\n")
        import traceback

        traceback.print_exc()
        return False

    print("=" * 70)
    print("✅ REAL SYSTEM READY - Learning enabled!")
    print("=" * 70 + "\n")
    return True


def initialize_simple_system():
    """Fallback to simple wrappers."""
    global embedder, cloud_client, local_vdb

    print("[1/3] Loading embedding model...")
    embedder = get_embedding_model()
    print("✅ Embedding model ready\n")

    print("[2/3] Connecting to Qdrant Cloud...")
    cloud_client = SimpleCloudClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
        dimension=VECTOR_DIMENSION,
    )
    print("✅ Cloud ready\n")

    print("[3/3] Loading local VDB...")
    local_vdb = SimpleLocalVDB()
    print("✅ Local ready\n")

    print("=" * 70)
    print("✅ SIMPLE SYSTEM READY")
    print("=" * 70 + "\n")
    return True


def query_llama(question, context_docs):
    """Query Ollama Llama 3.2 with retrieved context."""

    # Extract text from docs (handle both formats)
    texts = []
    for doc in context_docs[:RAG_CONTEXT_SIZE]:
        if isinstance(doc, dict):
            text = doc.get("payload", {}).get("text", "") or doc.get("text", "")
        else:
            text = str(doc)
        texts.append(text[:RAG_MAX_CONTEXT_LENGTH])

    context = "\n\n".join([f"[Doc {i+1}] {t}" for i, t in enumerate(texts)])

    prompt = f"""You are a medical knowledge assistant. Answer based on the medical information provided.

MEDICAL CONTEXT:
{context}

QUESTION: {question}

ANSWER (concise and accurate):"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": RAG_TEMPERATURE, "num_predict": 300},
            },
            timeout=30,
        )

        if response.status_code == 200:
            return response.json().get("response", "No response generated.")
        return f"❌ Ollama error: HTTP {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to Ollama. Run: `ollama serve`"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


@app.route("/")
def index():
    """Render main page."""
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
@app.route("/api/query", methods=["POST"])
def handle_query():
    """Handle medical query."""

    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    start_time = time.time()

    # Generate embedding
    embedding_start = time.time()
    query_vector = embedder.encode([question])[0]
    embedding_time = (time.time() - embedding_start) * 1000

    if USE_REAL_SYSTEM and router:
        # Use REAL hybrid router
        search_start = time.time()

        # Generate unique query ID
        query_id = f"q_{int(time.time() * 1000)}"

        try:
            results = router.search(
                query_vector=query_vector,
                query_id=query_id,  # FIXED: Added query_id
                query_text=question,
                k=RAG_CONTEXT_SIZE,
            )
            search_time = (time.time() - search_start) * 1000

            # Extract documents from results
            retrieved_docs = []
            if "ids" in results and results["ids"]:
                # Get actual documents from cloud/local
                for doc_id in results["ids"][:RAG_CONTEXT_SIZE]:
                    retrieved_docs.append(
                        {
                            "id": doc_id,
                            "text": f"Retrieved document {doc_id}",  # Placeholder
                        }
                    )

            source = results.get("source", "unknown")

        except Exception as e:
            print(f"[ERROR] Router search failed: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"error": f"Search failed: {str(e)}"}), 500

    else:
        # Simple fallback
        search_start = time.time()
        cloud_results = cloud_client.search(query_vector, k=RAG_CONTEXT_SIZE)
        search_time = (time.time() - search_start) * 1000
        retrieved_docs = cloud_results
        source = "cloud"

    # Generate answer with LLM
    llm_start = time.time()
    answer = query_llama(question, retrieved_docs)
    llm_time = (time.time() - llm_start) * 1000

    total_time = (time.time() - start_time) * 1000

    # Get metrics
    if USE_REAL_SYSTEM and metrics_tracker:
        current_metrics = metrics_tracker.get_summary()
    else:
        current_metrics = {
            "total_queries": 1,
            "local_hit_rate": 0,
            "avg_latency": total_time,
        }

    # Build flow visualization
    flow_steps = []

    if USE_REAL_SYSTEM and router:
        # Extract flow details from router results
        flow_steps.append(
            {
                "step": 1,
                "name": "Generate Embedding",
                "time_ms": round(embedding_time, 1),
                "status": "success",
            }
        )

        # Add detailed router flow
        if "prediction_hit" in results:
            flow_steps.append(
                {
                    "step": 2,
                    "name": "🎯 Prediction Check",
                    "time_ms": 1.0,
                    "status": "hit" if results["prediction_hit"] else "miss",
                    "detail": f"{'HIT' if results['prediction_hit'] else 'MISS'}",
                }
            )

        if "cluster_id" in results:
            flow_steps.append(
                {
                    "step": 3,
                    "name": "📊 Semantic Clustering",
                    "time_ms": 1.0,
                    "status": "success",
                    "detail": f"Cluster {results['cluster_id']}",
                }
            )

        # Local search
        local_latency = results.get("local_latency_ms", 0)
        if local_latency > 0:
            flow_steps.append(
                {
                    "step": 4,
                    "name": "💾 Search Local VDB",
                    "time_ms": round(local_latency, 1),
                    "status": "hit" if source == "local" else "miss",
                }
            )

        # Cloud search
        cloud_latency = results.get("cloud_latency_ms", 0)
        if cloud_latency > 0:
            flow_steps.append(
                {
                    "step": 5,
                    "name": "☁️ Search Cloud VDB",
                    "time_ms": round(cloud_latency, 1),
                    "status": "success",
                }
            )

        # Anchor creation
        if "anchor_id" in results:
            flow_steps.append(
                {
                    "step": 6,
                    "name": "⚓ Create/Update Anchor",
                    "time_ms": 1.0,
                    "status": "success",
                    "detail": f"Anchor #{results['anchor_id']}",
                }
            )

        # LLM
        flow_steps.append(
            {
                "step": 7,
                "name": "🤖 Generate Answer",
                "time_ms": round(llm_time, 1),
                "status": "success",
            }
        )

    else:
        # Fallback simple flow
        flow_steps = [
            {
                "step": 1,
                "name": "Generate Embedding",
                "time_ms": round(embedding_time, 1),
                "status": "success",
            },
            {
                "step": 2,
                "name": f"Search {source.title()} VDB",
                "time_ms": round(search_time, 1),
                "status": "success",
            },
            {
                "step": 3,
                "name": "🤖 Generate Answer",
                "time_ms": round(llm_time, 1),
                "status": "success",
            },
        ]

    # Extract text for display
    doc_texts = []
    for doc in retrieved_docs[:3]:
        if isinstance(doc, dict):
            text = doc.get("payload", {}).get("text", "") or doc.get("text", "")
        else:
            text = str(doc)
        doc_texts.append(text[:300] + "...")

    response = {
        "answer": answer,
        "retrieved_docs": doc_texts,
        "flow": flow_steps,
        "metrics": {
            "total_latency_ms": round(total_time, 1),
            "search_latency_ms": round(search_time, 1),
            "llm_latency_ms": round(llm_time, 1),
            "source": source,
            "total_queries": current_metrics.get("total_queries", 1),
            "local_hit_rate": round(current_metrics.get("local_hit_rate", 0), 1),
            "avg_latency_ms": round(current_metrics.get("avg_latency", total_time), 1),
        },
    }

    return jsonify(response)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check."""
    return jsonify(
        {
            "status": "healthy",
            "system_type": "REAL" if USE_REAL_SYSTEM else "SIMPLE",
            "components_loaded": {
                "router": router is not None,
                "anchor_system": anchor_system is not None,
                "metrics": metrics_tracker is not None,
            },
        }
    )


if __name__ == "__main__":
    success = initialize_system()

    if not success:
        print("❌ Initialization failed.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("STARTING WEB SERVER")
    print("=" * 70)
    print(f"\n🌐 Open: http://localhost:5000")
    print(f"🔬 System: {'REAL Hybrid VDB' if USE_REAL_SYSTEM else 'Simple fallback'}")
    print("⚡ Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
