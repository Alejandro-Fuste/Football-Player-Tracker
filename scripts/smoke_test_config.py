import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_config():
    print("="*60)
    print("RUNNING SMOKE TEST: CONFIG AND PATH RESOLUTION")
    print("="*60)
    
    try:
        from src.config import (
            DATASET_ROOT, VIDEOS_DIR, ANNOT_ROOT, PROJECT_ROOT, MODEL_REGISTRY, BOTSORT_CONFIG, EXPORT_DIR
        )
        print(f"DATASET_ROOT:   {DATASET_ROOT}")
        print(f"VIDEOS_DIR:     {VIDEOS_DIR}")
        print(f"ANNOT_ROOT:     {ANNOT_ROOT}")
        print(f"PROJECT_ROOT:   {PROJECT_ROOT}")
        print(f"MODEL_REGISTRY: {MODEL_REGISTRY}")
        print(f"BOTSORT_CONFIG: {BOTSORT_CONFIG}")
        print(f"EXPORT_DIR:     {EXPORT_DIR}")
        
        # Check folder structure existences
        print("\nChecking file and directory presence:")
        print(f"  - PROJECT_ROOT exists: {PROJECT_ROOT.exists()}")
        print(f"  - MODEL_REGISTRY exists: {MODEL_REGISTRY.exists()}")
        print(f"  - BOTSORT_CONFIG exists: {BOTSORT_CONFIG.exists()}")
        
        print("\nConfig smoke test passed.")
        print("="*60)
    except Exception as e:
        print(f"\nConfig smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_config()
