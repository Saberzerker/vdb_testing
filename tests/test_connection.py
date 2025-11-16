# tests/test_connection.py
"""
Quick test to verify Qdrant Cloud connection and basic functionality.
Run this first to ensure credentials are correct.
"""
from src.cloud_client import QdrantCloudClient
from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, VECTOR_DIMENSION
from sentence_transformers import SentenceTransformer

def test_qdrant_connection():
    """
    Test 1: Can we connect to Qdrant Cloud?
    """
    print("\n" + "="*70)
    print("QDRANT CLOUD CONNECTION TEST")
    print("="*70 + "\n")
    
    try:
        # Initialize client
        client = QdrantCloudClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=QDRANT_COLLECTION_NAME,
            dimension=VECTOR_DIMENSION
        )
        
        print("✅ Connection successful!\n")
        
        # Test 2: Can we populate with sample data?
        print("="*70)
        print("POPULATING WITH SAMPLE DATA")
        print("="*70 + "\n")
        
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sample documents covering different topics
        sample_docs = [
            # Topic 1: Vector Databases (10 docs)
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
            
            # Topic 2: Machine Learning (10 docs)
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
            
            # Topic 3: Python Programming (10 docs)
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
            
            # Topic 4: Natural Language Processing (10 docs)
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
        
        client.populate_with_documents(sample_docs, embedder)
        
        # Test 3: Can we search?
        print("\n" + "="*70)
        print("TESTING SEARCH")
        print("="*70 + "\n")
        
        test_query = "What is a vector database?"
        query_vector = embedder.encode([test_query])[0]
        
        ids, scores, latency = client.search(query_vector, k=3)
        
        print(f"Query: '{test_query}'")
        print(f"Cloud Latency: {latency:.2f}ms")
        print(f"\nTop 3 Results:")
        for i, (id_, score) in enumerate(zip(ids, scores), 1):
            print(f"  {i}. ID={id_}, Score={score:.3f}")
        
        # Get stats
        stats = client.get_collection_stats()
        print(f"\n✅ Collection contains {stats['points_count']} vectors")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✅")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease check:")
        print("1. QDRANT_URL is correct in src/config.py")
        print("2. QDRANT_API_KEY is correct in src/config.py")
        print("3. You have internet connection")
        print("4. Qdrant cluster is running")
        return False

if __name__ == "__main__":
    success = test_qdrant_connection()
    exit(0 if success else 1)