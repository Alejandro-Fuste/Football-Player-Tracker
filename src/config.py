import sys
from pathlib import Path

# Add project root to sys.path if not present to ensure we can load config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from configs.project_config import (
        DATASET_ROOT,
        VIDEOS_DIR,
        ANNOT_ROOT,
        PROJECT_ROOT,
        CONFIG_DIR,
        MODEL_REGISTRY,
        BOTSORT_CONFIG,
        EXPORT_DIR,
        CVAT_EXPORT_DIR,
        PREVIEW_DIR,
        METRICS_DIR,
        LOG_DIR,
        TO_BE_TRACKED,
        DEFAULT_MODEL_KEY,
        DEFAULT_CONF_THRESH,
        DEFAULT_IMG_SIZE,
        CLASS_NAMES,
        PLAYER_CLASS_ID,
        ALLOWED_CLASS_IDS,
        SPLITS,
        get_annot_dir_name
    )
except ImportError as e:
    # Fallback to local defaults if needed
    print(f"Warning: Could not import configs.project_config, using defaults. Error: {e}")
    DATASET_ROOT = PROJECT_ROOT / "datasets" / "AF-MOT"
    VIDEOS_DIR = DATASET_ROOT / "videos"
    ANNOT_ROOT = DATASET_ROOT / "annotations"
    CONFIG_DIR = PROJECT_ROOT / "configs"
    MODEL_REGISTRY = CONFIG_DIR / "model_registry.yaml"
    BOTSORT_CONFIG = CONFIG_DIR / "football_botsort.yaml"
    EXPORT_DIR = PROJECT_ROOT / "exports"
    CVAT_EXPORT_DIR = EXPORT_DIR / "cvat_mot_tracks"
    PREVIEW_DIR = EXPORT_DIR / "previews"
    METRICS_DIR = EXPORT_DIR / "metrics"
    LOG_DIR = EXPORT_DIR / "logs"
    TO_BE_TRACKED = '/content/drive/MyDrive/UCF/Courses/Spring2026/IndependentStudy/Datasets/AmericanFootballDataset/videos'
    DEFAULT_MODEL_KEY = "rtdetrx_player_best"
    DEFAULT_CONF_THRESH = 0.64
    DEFAULT_IMG_SIZE = 640
    CLASS_NAMES = ["player"]
    PLAYER_CLASS_ID = 0
    ALLOWED_CLASS_IDS = {0}
    SPLITS = {}
    def get_annot_dir_name(seq_name):
        return seq_name
