# test_100_queries.py
"""
Run 100 queries against the demo app
No code changes needed - just run this script while app is running
"""
import requests
import time
import json

BASE_URL = "http://localhost:5000"

QUERIES = [
    # [Same 100 queries as above JavaScript version]
    "What is diabetes?",
    "What are the symptoms of diabetes?",
    "What is the treatment of diabetes?",
    "Do I have diabetes?",
    "Am I prone to diabetes?",
    "What causes diabetes?",
    "How to prevent diabetes?",
    "What is type 1 diabetes?",
    "What is type 2 diabetes?",
    "What is gestational diabetes?",
    "Can diabetes be cured?",
    "What foods should diabetics avoid?",
    "What is insulin resistance?",
    "What is diabetic ketoacidosis?",
    "How to manage diabetes?",
    "What is hypertension?",
    "What causes high blood pressure?",
    "What are symptoms of high blood pressure?",
    "How to lower blood pressure naturally?",
    "What is normal blood pressure?",
    "What medications treat hypertension?",
    "Is hypertension dangerous?",
    "Can hypertension cause stroke?",
    "What is white coat hypertension?",
    "How to measure blood pressure?",
    "What is coronary artery disease?",
    "What are symptoms of heart attack?",
    "What is angina?",
    "What is an ECG?",
    "What is an angiogram?",
    "What causes heart disease?",
    "How to prevent heart disease?",
    "What is heart failure?",
    "What is atrial fibrillation?",
    "What is proper heart health care?",
    "What is asthma?",
    "What causes asthma?",
    "How to treat asthma?",
    "What is COPD?",
    "What is bronchitis?",
    "What is pneumonia?",
    "What causes shortness of breath?",
    "What is a pulmonary function test?",
    "What is an inhaler?",
    "How to prevent respiratory infections?",
    "How to heal a broken leg?",
    "Why did my leg break?",
    "What is osteoporosis?",
    "What is arthritis?",
    "What causes joint pain?",
    "How to treat back pain?",
    "What is a sprain?",
    "What is a fracture?",
    "How long does a broken bone take to heal?",
    "What is physical therapy?",
    "What is Alzheimer's disease?",
    "What causes headaches?",
    "What is a migraine?",
    "What is epilepsy?",
    "What is Parkinson's disease?",
    "What is multiple sclerosis?",
    "What causes dizziness?",
    "What is a stroke?",
    "What are symptoms of stroke?",
    "How to prevent stroke?",
    "What is acid reflux?",
    "What is IBS?",
    "What causes stomach pain?",
    "What is Crohn's disease?",
    "What is ulcerative colitis?",
    "What is celiac disease?",
    "What causes constipation?",
    "What causes diarrhea?",
    "What is gastritis?",
    "What is a peptic ulcer?",
    "What is depression?",
    "What is anxiety?",
    "What is PTSD?",
    "What is bipolar disorder?",
    "What is schizophrenia?",
    "How to treat depression?",
    "What is therapy?",
    "What are antidepressants?",
    "What is panic attack?",
    "How to manage stress?",
    "What is COVID-19?",
    "What is influenza?",
    "What is tuberculosis?",
    "What is HIV?",
    "What is hepatitis?",
    "What is the meaning of life?",
    "How to fix my car?",
    "What is quantum physics?",
    "What is a rare genetic disorder?",
    "What is fibromyalgia?",
    "What is chronic fatigue syndrome?",
    "What is lupus?",
    "What is sarcoidosis?",
    "What is amyotrophic lateral sclerosis?",
    "What is Huntington's disease?"
]

def run_100_queries():
    print("="*70)
    print("🚀 RUNNING 100 QUERIES")
    print("="*70)
    
    start_time = time.time()
    
    tier1_hits = 0
    tier2_hits = 0
    tier3_hits = 0
    total_latency = 0
    
    for i, question in enumerate(QUERIES):
        try:
            print(f"[{i+1}/100] {question[:50]}...")
            
            response = requests.post(
                f"{BASE_URL}/api/query",
                json={"question": question},
                timeout=60
            )
            
            data = response.json()
            
            # Track stats
            source = data.get('metrics', {}).get('source', '')
            if source == 'tier1_permanent':
                tier1_hits += 1
            elif source == 'tier2_dynamic':
                tier2_hits += 1
            elif source == 'tier3_cloud':
                tier3_hits += 1
            
            total_latency += data.get('metrics', {}).get('search_latency_ms', 0)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"✅ Progress: {i+1}/100 ({(i+1)/100*100:.0f}%)")
                print(f"   TIER 2 Hit Rate: {tier2_hits}/{i+1} ({tier2_hits/(i+1)*100:.1f}%)")
            
            time.sleep(0.1)  # Small delay
            
        except Exception as e:
            print(f"❌ Query {i+1} failed: {e}")
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("🎉 100 QUERIES COMPLETE!")
    print("="*70)
    print(f"✅ Total Time: {elapsed:.1f}s")
    print(f"⚡ Queries/sec: {100/elapsed:.2f}")
    print(f"📊 TIER 1 Hits: {tier1_hits}")
    print(f"📊 TIER 2 Hits: {tier2_hits} ({tier2_hits/100*100:.1f}%)")
    print(f"📊 TIER 3 Hits: {tier3_hits}")
    print(f"⏱️  Avg Latency: {total_latency/100:.1f}ms")
    print("="*70)
    
    # Get final stats
    try:
        stats_response = requests.get(f"{BASE_URL}/api/stats")
        stats = stats_response.json()
        
        print("\n📈 Final System Stats:")
        print(f"   Total Queries: {stats['metrics']['total_queries']}")
        print(f"   TIER 2 Hit Rate: {stats['metrics']['tier2_prefetch_rate']:.1f}%")
        print(f"   Dynamic Space: {stats['metrics'].get('dynamic_vectors', 0)}/700")
        print(f"   Anchors: {stats['metrics'].get('total_anchors', 0)}")
    except:
        pass

if __name__ == "__main__":
    run_100_queries()