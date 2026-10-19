"""
Project-level configuration for the AF-MOT inference, tracking, evaluation, and export pipeline.
"""

from pathlib import Path

# Target Dataset Layout (AF-MOT)
DATASET_ROOT = Path("/content/drive/MyDrive/UCF/Courses/Spring2026/IndependentStudy/Datasets/OtherDatasets/AF-MOT")
VIDEOS_DIR = DATASET_ROOT / "videos"
ANNOT_ROOT = DATASET_ROOT / "annotations"

# Project Code Directories
PROJECT_ROOT = Path("/content/drive/MyDrive/UCF/Courses/Spring2026/IndependentStudy/Utilities/Football_MOT_Exporting")

# Robust local environment fallback
if not PROJECT_ROOT.exists():
    local_path = Path(__file__).resolve().parents[1]
    if (local_path / "configs").exists():
        PROJECT_ROOT = local_path

CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_REGISTRY = CONFIG_DIR / "model_registry.yaml"
BOTSORT_CONFIG = CONFIG_DIR / "football_botsort.yaml"

# Output organized export folders
EXPORT_DIR = PROJECT_ROOT / "exports"
CVAT_EXPORT_DIR = EXPORT_DIR / "cvat_mot_tracks"
PREVIEW_DIR = EXPORT_DIR / "previews"
METRICS_DIR = EXPORT_DIR / "metrics"
LOG_DIR = EXPORT_DIR / "logs"

# Input clips folder for batch export
TO_BE_TRACKED = Path('/content/drive/MyDrive/UCF/Courses/Spring2026/IndependentStudy/Datasets/AmericanFootballDataset/videos')

# Defaults
DEFAULT_MODEL_KEY = "rtdetrx_player_best"
DEFAULT_CONF_THRESH = 0.64
DEFAULT_IMG_SIZE = 640

CLASS_NAMES = ["player", "ball"]
PLAYER_CLASS_ID = 0
ALLOWED_CLASS_IDS = {0, 1}

# Dataset split info for evaluation / dataset validation compatibility
TRAIN_VIDEOS = [
    "HTight009.mp4", "HWide005.mp4", "HWide009.mp4",
    "MTight013.mp4", "MTight016.mp4", "MWide013.mp4", "MWide016.mp4",
    "RTight012.mp4", "RTight014.mp4", "RWide012.mp4", "RWide014.mp4",
    "STight008.mp4", "STight024.mp4",
]

VAL_VIDEOS = [
    "SWide008.mp4", "SWide024.mp4", "WTight006.mp4", "WTight012.mp4",
]

TEST_VIDEOS = [
    "WWide006.mp4", "WWide012.mp4",
]

SPLITS = {
    "train": [Path(v).stem for v in TRAIN_VIDEOS],
    "val": [Path(v).stem for v in VAL_VIDEOS],
    "test": [Path(v).stem for v in TEST_VIDEOS],
}

def get_annot_dir_name(seq_name: str) -> str:
    """
    Resolves the correct annotation directory name inside ANNOT_ROOT.
    Supports both direct name mapping (e.g. JetSweep_1, Counter_1) and legacy keio prefix mapping.
    """
    if not seq_name:
        return seq_name
        
    # Strip extension if present (e.g. JetSweep_1.mp4 -> JetSweep_1)
    stem = Path(seq_name).stem
    
    # Candidates to check
    candidates = [
        stem,
        stem.replace('-', '_'),
        seq_name,
        seq_name.replace('-', '_')
    ]
    
    # Legacy prefix mapping logic
    prefix_map = {
        'H': 'hosei',
        'M': 'meiji',
        'R': 'rikkyo',
        'S': 'sensyu',
        'W': 'waseda',
    }
    
    prefix = 'unknown'
    if stem and stem[0] in prefix_map:
        prefix = prefix_map[stem[0]]
        
    if 'Tight' in stem:
        view = 'tight'
        num = stem.split('Tight')[-1]
    elif 'Wide' in stem:
        view = 'wide'
        num = stem.split('Wide')[-1]
    else:
        view = 'unknown'
        num = '000'
        
    legacy_name = f"keio_{prefix}_{view}_{num}"
    candidates.append(legacy_name)
    
    # Check which candidate directory exists under ANNOT_ROOT
    if ANNOT_ROOT and ANNOT_ROOT.exists():
        for cand in candidates:
            if (ANNOT_ROOT / cand).exists():
                return cand
                
    # Default to the first candidate (stem)
    return stem