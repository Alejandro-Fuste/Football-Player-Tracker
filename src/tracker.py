import yaml
import sys
import torch
import inspect
from pathlib import Path
import numpy as np

def create_botsort_tracker(tracker_config_path: Path, device: str = None):
    """
    Dynamically imports BoT-SORT from BoxMOT and instantiates it with configuration parameters.

    Args:
        tracker_config_path: Path to the tracker yaml config (e.g. configs/football_botsort.yaml)
        device: CUDA device or 'cpu'

    Returns:
        tracker: Instantiated BoxMOT tracker.
    """
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    half = torch.cuda.is_available()

    if not tracker_config_path.exists():
        raise FileNotFoundError(f"Tracker configuration file not found at: {tracker_config_path}")

    with open(tracker_config_path, "r") as f:
        tracker_params = yaml.safe_load(f)

    # ReID model weight path setup
    reid_model = tracker_params.get("model", "osnet_x0_25_msmt17.pt")
    reid_weights = Path(reid_model)

    # Method 1: Try using the official create_tracker API from boxmot (recommended)
    create_tracker = None
    for path in ["boxmot.tracker_zoo", "boxmot.trackers.tracker_zoo", "boxmot"]:
        try:
            mod = __import__(path, fromlist=["create_tracker"])
            create_tracker = getattr(mod, "create_tracker", None)
            if create_tracker is not None:
                print(f"[FootballMOT] Successfully imported create_tracker from {path}")
                break
        except Exception:
            continue

    # Determine tracker type from config
    tracker_type = tracker_params.get("tracker_type", "botsort")

    # Method 1: Try using the official create_tracker API from boxmot (recommended)
    create_tracker = None
    for path in ["boxmot.tracker_zoo", "boxmot.trackers.tracker_zoo", "boxmot"]:
        try:
            mod = __import__(path, fromlist=["create_tracker"])
            create_tracker = getattr(mod, "create_tracker", None)
            if create_tracker is not None:
                print(f"[FootballMOT] Successfully imported create_tracker from {path}")
                break
        except Exception:
            continue

    if create_tracker is not None:
        try:
            sig = inspect.signature(create_tracker)
            tracker_kwargs = {
                "tracker_type": tracker_type,
                "reid_weights": reid_weights,
                "model_weights": reid_weights,
                "device": device,
                "half": half,
                "fp16": half
            }

            # Check if tracker_params is flat (meaning it represents resolved tracker_args rather than evolution configs)
            is_flat = True
            for k, v in tracker_params.items():
                if isinstance(v, dict) and "default" in v:
                    is_flat = False
                    break

            if is_flat and "evolve_param_dict" in sig.parameters:
                tracker_kwargs["evolve_param_dict"] = tracker_params.copy()
                tracker_kwargs["tracker_config"] = None
            else:
                tracker_kwargs["tracker_config"] = tracker_config_path

            filtered_tracker_kwargs = {k: v for k, v in tracker_kwargs.items() if k in sig.parameters}
            tracker = create_tracker(**filtered_tracker_kwargs)
            print(f"[FootballMOT] Successfully initialized tracker using {create_tracker.__name__}")
            return tracker
        except Exception as e:
            print(f"[FootballMOT] Method 1 (create_tracker) execution failed: {e}. Trying Method 2...")

    # Method 2: Fallback to importing ReidModel/ReIDDetectMultiBackend and tracker class directly
    tracker_cls = None
    import_errors = []

    if tracker_type == "occluboost":
        try:
            from boxmot.trackers.bbox.occluboost.occluboost import OccluBoost
            tracker_cls = OccluBoost
        except Exception as e:
            import_errors.append(repr(e))
    else:
        try:
            from boxmot import BoTSORT
            tracker_cls = BoTSORT
        except Exception as e:
            import_errors.append(repr(e))

        if tracker_cls is None:
            try:
                from boxmot import BotSort
                tracker_cls = BotSort
            except Exception as e:
                import_errors.append(repr(e))

        # BoxMOT v20.0.0+ subpaths
        if tracker_cls is None:
            try:
                from boxmot.trackers.bbox.botsort import BotSort
                tracker_cls = BotSort
            except Exception as e:
                import_errors.append(repr(e))

        if tracker_cls is None:
            try:
                from boxmot.trackers.bbox.botsort import BoTSORT
                tracker_cls = BoTSORT
            except Exception as e:
                import_errors.append(repr(e))

        # Legacy/Old BoxMOT subpaths
        if tracker_cls is None:
            try:
                from boxmot.trackers.bbox.botsort import BoTSORT
                tracker_cls = BoTSORT
            except Exception as e:
                import_errors.append(repr(e))

    if tracker_cls is None:
        try:
            from boxmot.trackers.bbox.botsort import BotSort
            tracker_cls = BotSort
        except Exception as e:
            import_errors.append(repr(e))

    if tracker_cls is None:
        try:
            from boxmot.trackers.bbox.botsort import BOTSORT
            tracker_cls = BOTSORT
        except Exception as e:
            import_errors.append(repr(e))

    if tracker_cls is None:
        raise ImportError(
            "Could not import BoT-SORT/BotSort from boxmot library. "
            "Please ensure boxmot is installed correctly. Errors: " + " | ".join(import_errors)
        )

    # Build ReID model object dynamically if ReID is enabled
    reid_model_obj = None
    with_reid = tracker_params.get("with_reid", True)
    
    if with_reid:
        for import_path in [
            "boxmot.reid.core.reid",
            "boxmot.reid.core",
            "boxmot.reid",
            "boxmot.appearance.reid_multibackend",
            "boxmot.appearance.reid_auto_backend",
            "boxmot.appearance.reid_model",
            "boxmot.models.reid",
            "boxmot"
        ]:
            try:
                mod = __import__(import_path, fromlist=["ReIDDetectMultiBackend", "ReidDetectMultiBackend", "ReidAutoBackend", "ReIDAutoBackend", "ReIDModel", "ReidModel", "ReID"])
                ReIDModel_class = (
                    getattr(mod, "ReID", None) or
                    getattr(mod, "ReIDDetectMultiBackend", None) or 
                    getattr(mod, "ReidDetectMultiBackend", None) or
                    getattr(mod, "ReidAutoBackend", None) or
                    getattr(mod, "ReIDAutoBackend", None) or
                    getattr(mod, "ReIDModel", None) or 
                    getattr(mod, "ReidModel", None)
                )
                if ReIDModel_class is None:
                    continue
                sig = inspect.signature(ReIDModel_class)
                reid_kwargs = {
                    "weights": reid_weights,
                    "weights_path": str(reid_weights),
                    "model_path": str(reid_weights),
                    "modelPath": str(reid_weights),
                    "device": device,
                    "half": half,
                    "fp16": half
                }
                filtered_reid_kwargs = {k: v for k, v in reid_kwargs.items() if k in sig.parameters}
                reid_model_obj = ReIDModel_class(**filtered_reid_kwargs)
                print(f"[FootballMOT] Successfully instantiated {ReIDModel_class.__name__} for ReID.")
                break
            except Exception as e_reid:
                print(f"[FootballMOT] Failed importing ReIDModel from {import_path}: {e_reid}")
                continue

    # Fallback if ReID was requested but ReID backend couldn't be loaded
    if with_reid and reid_model_obj is None:
        print("[FootballMOT] WARNING: ReID model loading failed (missing boxmot.appearance or dependencies). "
              "Falling back to tracking without ReID (with_reid=False).")
        with_reid = False

    kwargs = {
        "device": device,
        "half": half,
        "per_class": False,
        "track_high_thresh": tracker_params.get("track_high_thresh", 0.25),
        "track_low_thresh": tracker_params.get("track_low_thresh", 0.10),
        "new_track_thresh": tracker_params.get("new_track_thresh", 0.25),
        "track_buffer": tracker_params.get("track_buffer", 90),
        "match_thresh": tracker_params.get("match_thresh", 0.80),
        "proximity_thresh": tracker_params.get("proximity_thresh", 0.50),
        "appearance_thresh": tracker_params.get("appearance_thresh", 0.25),
        "with_reid": with_reid,
    }

    if with_reid:
        kwargs["model"] = reid_model_obj
        kwargs["reid_model"] = reid_model_obj
        kwargs["reid_weights"] = reid_model_obj

    # Filter arguments to match class signature
    signature = inspect.signature(tracker_cls)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}

    return tracker_cls(**filtered_kwargs)

def parse_tracker_output(tracks):
    """
    Parses tracking results arrays into standard python formats.

    Returns:
        parsed_tracks: List of tuples (x1, y1, x2, y2, track_id, score, class_id)
    """
    if tracks is None:
        return []
    
    arr = np.asarray(tracks)
    if arr.size == 0:
        return []
        
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    parsed = []
    for row in arr:
        if len(row) < 5:
            continue
        x1, y1, x2, y2 = map(float, row[:4])
        track_id = int(row[4])
        conf = float(row[5]) if len(row) > 5 else 1.0
        class_id = int(row[6]) if len(row) > 6 else 0
        parsed.append((x1, y1, x2, y2, track_id, conf, class_id))
        
    return parsed
