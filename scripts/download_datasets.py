# scripts/download_datasets.py (FIXED VERSION)
"""
Download datasets for the 3 use cases.
Uses direct Hugging Face datasets API (updated for 2024).
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
import pickle
import json


def download_wikipedia(output_dir="datasets"):
    """
    Download text dataset for RAG.
    Using SQuAD dataset as alternative (easier to load).
    """
    print("\n" + "="*70)
    print("DOWNLOADING DATASET 1: Text Corpus (SQuAD for RAG)")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Use SQuAD dataset (questions + context passages)
        # This is perfect for RAG - has real Wikipedia passages
        print("Loading SQuAD dataset from Hugging Face...")
        dataset = load_dataset("squad", split="train[:5000]")
        
        # Extract unique context passages
        contexts = list(set([item['context'] for item in dataset]))
        
        print(f"Extracted {len(contexts)} unique passages")
        
        # Save to disk
        save_path = output_path / "text_corpus.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(contexts, f)
        
        print(f"✅ Saved {len(contexts)} text passages to {save_path}")
        return save_path
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        
        # FALLBACK: Create synthetic dataset
        print("Creating synthetic dataset instead...")
        synthetic_docs = [
            "Quantum mechanics is a fundamental theory in physics that describes nature at small scales.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Python is a high-level programming language known for its simplicity and readability.",
            "Deep learning uses neural networks with multiple layers to model complex patterns.",
            "Vector databases store high-dimensional embeddings for similarity search operations.",
            # Add more synthetic documents here
        ] * 100  # Repeat to create more data
        
        save_path = output_path / "text_corpus.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(synthetic_docs, f)
        
        print(f"✅ Saved {len(synthetic_docs)} synthetic passages to {save_path}")
        return save_path


def download_cifar10(output_dir="datasets"):
    """Download CIFAR-10 dataset for image search."""
    print("\n" + "="*70)
    print("DOWNLOADING DATASET 2: CIFAR-10 (Images)")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        print("Loading CIFAR-10 from Hugging Face...")
        # Load first 1000 images for faster demo
        dataset = load_dataset("cifar10", split="train[:1000]")
        
        save_path = output_path / "cifar10.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"✅ Saved {len(dataset)} images to {save_path}")
        return save_path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def download_code_dataset(output_dir="datasets"):
    """Download code dataset for code search."""
    print("\n" + "="*70)
    print("DOWNLOADING DATASET 3: Code Snippets")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        print("Loading code dataset from Hugging Face...")
        # Use a smaller, well-maintained code dataset
        dataset = load_dataset("codeparrot/github-code", 
                              streaming=True,
                              split="train",
                              languages=["Python"])
        
        # Take first 1000 samples
        code_samples = []
        for i, item in enumerate(dataset):
            if i >= 1000:
                break
            code_samples.append(item)
            if (i + 1) % 100 == 0:
                print(f"Loaded {i+1} code samples...")
        
        save_path = output_path / "code_snippets.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(code_samples, f)
        
        print(f"✅ Saved {len(code_samples)} code snippets to {save_path}")
        return save_path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # FALLBACK: Synthetic code snippets
        print("Creating synthetic code dataset...")
        synthetic_code = [
            {"code": "def hello_world():\n    print('Hello, World!')", "language": "Python"},
            {"code": "import numpy as np\narr = np.array([1, 2, 3])", "language": "Python"},
            {"code": "class MyClass:\n    def __init__(self):\n        self.value = 0", "language": "Python"},
            # Add more samples
        ] * 100
        
        save_path = output_path / "code_snippets.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(synthetic_code, f)
        
        print(f"✅ Saved {len(synthetic_code)} synthetic code snippets to {save_path}")
        return save_path


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DATASET DOWNLOADER - ALL 3 USE CASES")
    print("="*70)
    
    # Create datasets directory
    Path("datasets").mkdir(exist_ok=True)
    
    # Download all datasets
    wiki_path = download_wikipedia()
    cifar_path = download_cifar10()
    code_path = download_code_dataset()
    
    print("\n" + "="*70)
    print("DATASET DOWNLOAD COMPLETE")
    print("="*70)
    print(f"\n1. Text Corpus: {wiki_path}")
    print(f"2. CIFAR-10: {cifar_path}")
    print(f"3. Code Snippets: {code_path}\n")