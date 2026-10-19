"""Verify the paths and directories configured in configs/project_config.py."""

from pathlib import Path
import sys

PROJECT_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_ROOT))

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
        TO_BE_TRACKED
    )
except ImportError as e:
    print(f"Error: Could not import configs.project_config. Error: {e}")
    sys.exit(1)


def verify_paths() -> None:
    print("=" * 60)
    print("VERIFYING PROJECT PATHS & DIRECTORIES CONFIGURATION")
    print("=" * 60)

    # 1. Essential Input Paths (Must exist)
    input_paths = {
        "PROJECT_ROOT": (PROJECT_ROOT, "Directory", "Code repository root directory"),
        "DATASET_ROOT": (DATASET_ROOT, "Directory", "Root folder for the datasets"),
        "VIDEOS_DIR": (VIDEOS_DIR, "Directory", "Raw video clips folder"),
        "ANNOT_ROOT": (ANNOT_ROOT, "Directory", "Ground-truth annotations folder"),
        "CONFIG_DIR": (CONFIG_DIR, "Directory", "Configuration folder containing model and tracker YAMLs"),
        "MODEL_REGISTRY": (MODEL_REGISTRY, "File", "Model registry YAML configuration file"),
        "BOTSORT_CONFIG": (BOTSORT_CONFIG, "File", "BoT-SORT tracker YAML configuration file"),
        "TO_BE_TRACKED": (TO_BE_TRACKED, "Directory", "Target video folder for batch processing"),
    }

    # 2. Output Paths (Will be created if missing)
    output_paths = {
        "EXPORT_DIR": (EXPORT_DIR, "Directory", "General exports root directory"),
        "CVAT_EXPORT_DIR": (CVAT_EXPORT_DIR, "Directory", "Folder for exported CVAT MOT 1.1 tracking files"),
        "PREVIEW_DIR": (PREVIEW_DIR, "Directory", "Folder for visual tracking prediction preview MP4s"),
        "METRICS_DIR": (METRICS_DIR, "Directory", "Folder for tracking evaluation metrics report outputs"),
        "LOG_DIR": (LOG_DIR, "Directory", "Folder for tracking error logs and execution records"),
    }

    errors = 0

    print("\n--- Checking Input Paths (Required to exist) ---")
    for name, (path, path_type, desc) in input_paths.items():
        print(f"\n{name} ({desc}):")
        print(f"  Path: {path}")
        if path.exists():
            # Check type matching
            if path_type == "Directory" and not path.is_dir():
                print(f"  [ERROR] Expected directory but found a file.")
                errors += 1
            elif path_type == "File" and not path.is_file():
                print(f"  [ERROR] Expected file but found a directory.")
                errors += 1
            else:
                print(f"  [OK] Found and validated.")
        else:
            print(f"  [ERROR] Path does not exist.")
            errors += 1

    print("\n--- Checking/Creating Output Paths ---")
    for name, (path, path_type, desc) in output_paths.items():
        print(f"\n{name} ({desc}):")
        print(f"  Path: {path}")
        if path.exists():
            print(f"  [OK] Directory exists.")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  [OK] Created successfully.")
            except Exception as e:
                print(f"  [ERROR] Directory is missing and could not be created. Error: {e}")
                errors += 1

    print("\n" + "=" * 60)
    if errors > 0:
        print(f"Verification completed with {errors} error(s). Please fix the configuration.")
        sys.exit(1)
    else:
        print("All path and directory validations passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    verify_paths()