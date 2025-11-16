# test_components.py
"""Test each component individually."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_config():
    print("\n[1/10] Testing config...")
    try:
        from src import config
        assert hasattr(config, 'CACHE_LAYER_PATH')
        assert hasattr(config, 'COMPACTION_INTERVAL_SECONDS')
        print("✅ Config OK")
        return True
    except Exception as e:
        print(f"❌ Config FAILED: {e}")
        return False

def test_cloud_client():
    print("\n[2/10] Testing cloud_client...")
    try:
        from src.cloud_client import QdrantCloudClient
        from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, VECTOR_DIMENSION
        
        client = QdrantCloudClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=QDRANT_COLLECTION_NAME,
            dimension=VECTOR_DIMENSION
        )
        stats = client.get_collection_stats()
        print(f"✅ Cloud client OK ({stats.get('points_count', 0)} vectors)")
        return True
    except Exception as e:
        print(f"❌ Cloud client FAILED: {e}")
        return False

def test_anchor_system():
    print("\n[3/10] Testing anchor_system...")
    try:
        from src.anchor_system import AnchorSystem
        anchor_sys = AnchorSystem()
        print("✅ Anchor system OK")
        return True
    except Exception as e:
        print(f"❌ Anchor system FAILED: {e}")
        return False

def test_semantic_cache():
    print("\n[4/10] Testing semantic_cache...")
    try:
        from src.semantic_cache import SemanticClusterCache
        cache = SemanticClusterCache()
        print("✅ Semantic cache OK")
        return True
    except Exception as e:
        print(f"❌ Semantic cache FAILED: {e}")
        return False

def test_storage_engine():
    print("\n[5/10] Testing storage_engine...")
    try:
        from src.storage_engine import StorageEngine
        from src import config
        storage = StorageEngine(config)
        print("✅ Storage engine OK")
        return True
    except Exception as e:
        print(f"❌ Storage engine FAILED: {e}")
        return False

def test_local_vdb():
    print("\n[6/10] Testing local_vdb...")
    try:
        from src.local_vdb import LocalVDB
        local = LocalVDB()
        print("✅ Local VDB OK")
        return True
    except Exception as e:
        print(f"❌ Local VDB FAILED: {e}")
        return False

def test_metrics():
    print("\n[7/10] Testing metrics...")
    try:
        from src.metrics import MetricsTracker
        metrics = MetricsTracker()
        print("✅ Metrics OK")
        return True
    except Exception as e:
        print(f"❌ Metrics FAILED: {e}")
        return False

def test_updater():
    print("\n[8/10] Testing updater...")
    try:
        from src.updater import Updater
        print("✅ Updater OK (not instantiating, needs dependencies)")
        return True
    except Exception as e:
        print(f"❌ Updater FAILED: {e}")
        return False

def test_scheduler():
    print("\n[9/10] Testing scheduler...")
    try:
        from src.scheduler import BackgroundScheduler
        print("✅ Scheduler OK (not instantiating, needs dependencies)")
        return True
    except Exception as e:
        print(f"❌ Scheduler FAILED: {e}")
        return False

def test_hybrid_router():
    print("\n[10/10] Testing hybrid_router...")
    try:
        from src.hybrid_router import HybridRouter
        print("✅ Hybrid router OK (not instantiating, needs dependencies)")
        return True
    except Exception as e:
        print(f"❌ Hybrid router FAILED: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("COMPONENT TEST SUITE")
    print("="*70)
    
    tests = [
        test_config,
        test_cloud_client,
        test_anchor_system,
        test_semantic_cache,
        test_storage_engine,
        test_local_vdb,
        test_metrics,
        test_updater,
        test_scheduler,
        test_hybrid_router
    ]
    
    passed = sum(1 for test in tests if test())
    total = len(tests)
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n✅ ALL COMPONENTS OK! Ready to run demo.")
    else:
        print(f"\n❌ {total - passed} component(s) failed. Fix errors above.")