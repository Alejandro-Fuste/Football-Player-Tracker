import sys
import yaml
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY
from src.model_loader import load_model_from_registry

def test_model_loading():
    print("="*60)
    print("RUNNING SMOKE TEST: MODEL LOADING")
    print("="*60)
    
    if not MODEL_REGISTRY.exists():
        print(f"Registry configuration missing at: {MODEL_REGISTRY}")
        sys.exit(1)
        
    with open(MODEL_REGISTRY, "r") as f:
        config = yaml.safe_load(f)
        
    available_keys = list(config.get("models", {}).keys())
    print(f"Available registry models: {available_keys}")
    
    for key in available_keys:
        model_info = config["models"][key]
        weights = Path(model_info.get("weights_path", ""))
        print(f"\nChecking model: '{key}'")
        print(f"  - Family:       {model_info.get('family')}")
        print(f"  - Variant:      {model_info.get('variant')}")
        print(f"  - Weights path: {weights}")
        print(f"  - Weights file exists: {weights.exists()}")
        
        # Only try to load if weights file actually exists to prevent hard crash during environment checks
        if weights.exists():
            try:
                print(f"  - Loading model key '{key}'...")
                model, info = load_model_from_registry(MODEL_REGISTRY, key)
                print(f"  - Successfully loaded model. Family: {info.get('family')}")
            except Exception as e:
                print(f"  - Error loading model: {e}")
        else:
            print(f"  - Skipping load test: weights file is missing. Set actual path in model_registry.yaml.")
            
    print("\nModel loader smoke test completed.")
    print("="*60)

if __name__ == "__main__":
    test_model_loading()
