import yaml
from pathlib import Path
from ultralytics import RTDETR, YOLO

def load_model_from_registry(model_config_path: Path, model_key: str):
    """
    Reads the model registry YAML file and returns the loaded Ultralytics model
    and the corresponding metadata/configuration dictionary.

    Args:
        model_config_path: Path to model_registry.yaml
        model_key: Key under the 'models' configuration section.

    Returns:
        model: Loaded Ultralytics RTDETR or YOLO instance.
        model_info: Dictionary containing model metadata (family, allowed classes, imgsz, etc.)
    """
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model registry configuration file not found at: {model_config_path}")

    with open(model_config_path, "r") as f:
        registry = yaml.safe_load(f)

    if not registry or "models" not in registry:
        raise ValueError(f"Invalid model registry format at: {model_config_path}. Missing 'models' section.")

    if model_key not in registry["models"]:
        available_keys = list(registry["models"].keys())
        raise ValueError(f"Model key '{model_key}' not found in registry models. Available keys: {available_keys}")

    model_info = registry["models"][model_key]
    weights_path = Path(model_info.get("weights_path", ""))

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file not found for model key '{model_key}' at path: {weights_path}\n"
            f"Please ensure the file exists or update its path in {model_config_path}."
        )

    family = model_info.get("family", "").lower()

    if family == "rtdetr":
        model = RTDETR(str(weights_path))
    elif family == "yolo":
        model = YOLO(str(weights_path))
    else:
        raise ValueError(
            f"Unsupported model family '{family}' for model key '{model_key}'. "
            f"Supported families are 'rtdetr' and 'yolo'."
        )

    return model, model_info
