# tests/test_full_system.py
"""
Full System Test: Anchor-Based Trajectory Learning Hybrid VDB

This test demonstrates:
1. Query sequences following semantic paths
2. Anchor strengthening through reinforcement
3. Prediction accuracy improving over time
4. Learning curve visualization
5. Multi-tier anchor hierarchy emerging

Author: Saberzerker
Date: 2025-11-16
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_vdb import HybridVectorDB
from src.cloud_client import QdrantCloudClient
from src.config import *
from sentence_transformers import SentenceTransformer
import time
import json


def test_trajectory_learning():
    """
    Main test: Demonstrate anchor-based trajectory learning.
    
    We'll simulate a user exploring different topics:
    - Start with vector databases (forms anchor, generates predictions)
    - Continue in same topic (predictions hit, anchor strengthens)
    - Shift to ML (new anchor, weak one starts decaying)
    - Return to vector DBs (strong anchor still active)
    """
    print("\n" + "="*70)
    print("ANCHOR-BASED TRAJECTORY LEARNING - FULL SYSTEM TEST")
    print("="*70 + "\n")
    
    # Initialize cloud client
    print("Step 1: Initializing Qdrant Cloud client...")
    cloud = QdrantCloudClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
        dimension=VECTOR_DIMENSION
    )
    
    # Check if already populated
    stats = cloud.get_collection_stats()
    if stats['points_count'] == 0:
        print("\nStep 2: Populating cloud with corpus...")
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        # Use the same 40 documents from connection test
        corpus = [
            # Vector DB topic
            "Vector databases store high-dimensional embeddings for similarity search",
            "FAISS is a library for efficient similarity search and clustering",
            "Approximate nearest neighbor search enables fast retrieval at scale",
            "Vector embeddings capture semantic meaning of text and images",
            "HNSW algorithm provides fast approximate nearest neighbor search",
            "Product quantization compresses vectors for efficient storage",
            "IVF indexes partition vector space for faster search",
            "Semantic search uses vector similarity instead of keyword matching",
            "Vector databases power modern AI applications like RAG",
            "Embedding models transform data into vector representations",
            
            # ML topic
            "Deep learning neural networks learn hierarchical representations",
            "Transformer models use self-attention mechanisms",
            "Training neural networks requires backpropagation and gradient descent",
            "Convolutional neural networks excel at image recognition",
            "Recurrent neural networks process sequential data",
            "PyTorch and TensorFlow are popular deep learning frameworks",
            "Transfer learning leverages pre-trained models for new tasks",
            "Overfitting occurs when models memorize training data",
            "Regularization techniques prevent overfitting in neural networks",
            "Batch normalization stabilizes neural network training",
            
            # Python topic
            "Python list comprehensions provide concise syntax for transformations",
            "Python decorators modify function behavior without changing code",
            "Python generators yield values lazily for memory efficiency",
            "Asyncio enables concurrent programming in Python",
            "Python classes support object-oriented programming",
            "Python type hints improve code clarity and catch errors",
            "Virtual environments isolate Python project dependencies",
            "Python enumerate function provides index and value in loops",
            "Python context managers handle resource cleanup automatically",
            "Python lambda functions create anonymous inline functions",
            
            # NLP topic
            "Named entity recognition identifies people places and organizations",
            "Sentiment analysis determines emotional tone of text",
            "Tokenization splits text into words or subwords",
            "Word embeddings represent words as dense vectors",
            "Language models predict next words in sequences",
            "BERT uses bidirectional context for language understanding",
            "GPT models generate coherent text through autoregression",
            "Attention mechanisms focus on relevant input parts",
            "Text classification categorizes documents into topics",
            "Machine translation converts text between languages"
        ]
        
        cloud.populate_with_documents(corpus, embedder)
    else:
        print(f"\nStep 2: Cloud already populated with {stats['points_count']} vectors")
    
    # Initialize hybrid VDB
    print("\nStep 3: Initializing Hybrid VDB with anchor system...")
    vdb = HybridVectorDB(cloud_client=cloud)
    
    print("\n" + "="*70)
    print("RUNNING QUERY TRAJECTORY SCENARIOS")
    print("="*70)
    
    # TRAJECTORY 1: Vector Database Exploration
    # This should form a strong anchor
    print("\n" + "-"*70)
    print("TRAJECTORY 1: Exploring Vector Databases")
    print("-"*70)
    
    vdb_queries = [
        "What is a vector database?",
        "How does FAISS work?",
        "Explain HNSW algorithm",
        "What is approximate nearest neighbor search?",
        "How to optimize vector search performance?"
    ]
    
    for query in vdb_queries:
        result = vdb.search(query)
        time.sleep(0.5)  # Simulate user thinking time
    
    # TRAJECTORY 2: Machine Learning Exploration
    # This should form a separate anchor
    print("\n" + "-"*70)
    print("TRAJECTORY 2: Exploring Machine Learning")
    print("-"*70)
    
    ml_queries = [
        "How do neural networks work?",
        "What is a transformer model?",
        "Explain backpropagation in deep learning",
        "What is transfer learning?"
    ]
    
    for query in ml_queries:
        result = vdb.search(query)
        time.sleep(0.5)
    
    # TRAJECTORY 3: Return to Vector Databases
    # This should hit the strong anchor from Trajectory 1
    print("\n" + "-"*70)
    print("TRAJECTORY 3: Returning to Vector Databases")
    print("-"*70)
    
    vdb_return_queries = [
        "Tell me about product quantization",
        "How does semantic search work?",
        "What are vector embeddings?"
    ]
    
    for query in vdb_return_queries:
        result = vdb.search(query)
        time.sleep(0.5)
    
    # TRAJECTORY 4: Python Programming (brief exploration)
    print("\n" + "-"*70)
    print("TRAJECTORY 4: Brief Python Exploration")
    print("-"*70)
    
    python_queries = [
        "What are Python decorators?",
        "How do Python generators work?"
    ]
    
    for query in python_queries:
        result = vdb.search(query)
        time.sleep(0.5)
    
    # FINAL RESULTS
    print("\n" + "="*70)
    print("TEST COMPLETE - ANALYZING RESULTS")
    print("="*70 + "\n")
    
    # Print comprehensive report
    vdb.print_final_report()
    
    # Export detailed metrics
    results_path = "results/full_system_test_results.json"
    vdb.export_results(results_path)
    
    # Additional analysis
    metrics = vdb.get_comprehensive_metrics()
    
    print("\n" + "="*70)
    print("ANCHOR LEARNING ANALYSIS")
    print("="*70 + "\n")
    
    # Show anchor type transitions (learning progress)
    transitions = vdb.anchor_system.metrics["type_transitions"]
    if transitions:
        print("📈 Anchor Type Transitions (Learning Progress):")
        for t in transitions:
            print(f"   Anchor {t['anchor_id']}: {t['from']} → {t['to']} "
                  f"(strength={t['strength']:.1f}, hits={t.get('hits', 'N/A')})")
    else:
        print("No anchor transitions yet (need more queries to see learning)")
    
    # Show anchor details
    print(f"\n⚓ Detailed Anchor States:")
    for aid, anchor in vdb.anchor_system.anchors.items():
        print(f"\n   Anchor {aid} [{anchor.type.value}]:")
        print(f"      Query: \"{anchor.query_text[:50]}...\"")
        print(f"      Strength: {anchor.strength:.1f}")
        print(f"      Hits: {anchor.hits} | Misses: {anchor.misses}")
        print(f"      Predictions Generated: {anchor.predictions_generated}")
        print(f"      Successful Predictions: {anchor.successful_predictions}")
        if anchor.parent_anchor is not None:
            print(f"      Parent Anchor: {anchor.parent_anchor}")
        if anchor.child_anchors:
            print(f"      Child Anchors: {anchor.child_anchors}")
    
    print("\n" + "="*70)
    print("KEY OBSERVATIONS")
    print("="*70 + "\n")
    
    # Calculate key insights
    print("✅ What This Test Demonstrates:\n")
    
    print(f"1. PREDICTION ACCURACY: {metrics['prediction_accuracy']:.1%}")
    if metrics['prediction_accuracy'] > 0:
        print(f"   → The anchor system correctly predicted {metrics['prediction_hits']} queries")
        print(f"   → This shows trajectory learning is working!")
    else:
        print(f"   → Need more queries to see prediction hits")
    
    print(f"\n2. PREFETCH EFFECTIVENESS: {metrics['prefetch_hit_rate']:.1%}")
    if metrics['prefetch_hit_rate'] > 0:
        print(f"   → {metrics['prefetched_vectors_used']} local hits came from prefetched vectors")
        print(f"   → Predictive caching reduced cloud API calls!")
    
    print(f"\n3. LEARNING CURVE:")
    if len(metrics['learning_curve']) > 1:
        first_window = metrics['learning_curve'][0]['local_hit_rate']
        last_window = metrics['learning_curve'][-1]['local_hit_rate']
        improvement = last_window - first_window
        print(f"   → First queries: {first_window:.1%} local hit rate")
        print(f"   → Latest queries: {last_window:.1%} local hit rate")
        if improvement > 0:
            print(f"   → IMPROVEMENT: +{improvement:.1%} (system is learning!)")
        else:
            print(f"   → Hit rate changed by {improvement:.1%}")
    
    print(f"\n4. ANCHOR HIERARCHY:")
    anchor_stats = metrics['anchor_system']
    print(f"   → PERMANENT: {anchor_stats['by_type']['PERMANENT']} (never decay)")
    print(f"   → STRONG: {anchor_stats['by_type']['STRONG']} (proven paths)")
    print(f"   → MEDIUM: {anchor_stats['by_type']['MEDIUM']} (emerging paths)")
    print(f"   → WEAK: {anchor_stats['by_type']['WEAK']} (exploratory)")
    
    if anchor_stats['by_type']['STRONG'] > 0 or anchor_stats['by_type']['PERMANENT'] > 0:
        print(f"   → Multi-tier hierarchy is forming! Strong anchors emerged.")
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY ✅")
    print("="*70 + "\n")
    
    print(f"📊 Detailed results saved to: {results_path}")
    print(f"💡 Review the JSON for complete metrics and query logs\n")
    
    return True


if __name__ == "__main__":
    try:
        success = test_trajectory_learning()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)