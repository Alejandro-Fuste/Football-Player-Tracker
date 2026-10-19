import argparse
import json
import time
import sys
import cv2
from pathlib import Path
from tqdm.auto import tqdm

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- START BOXMOT PATH FORCE (Inserted by Colab Agent) ---
_boxmot_repo_path = '/tmp/boxmot_repo'
# Explicitly remove any other boxmot installations from sys.path if they were loaded
# This is a robust way to ensure the editable install is used by the subprocess
_paths_to_remove = []
for p in list(sys.path): # Iterate over a copy as sys.path may be modified
    if 'boxmot' in p and _boxmot_repo_path not in p:
        _paths_to_remove.append(p)
for p in _paths_to_remove:
    sys.path.remove(p)
    print(f"DEBUG: Removed '{p}' from sys.path to force correct boxmot usage.")

# Ensure the cloned boxmot repo is at the very beginning of sys.path
if _boxmot_repo_path not in sys.path:
    sys.path.insert(0, _boxmot_repo_path)
    print(f"DEBUG: Inserted '{_boxmot_repo_path}' to sys.path.")

# Diagnostic prints to verify boxmot import source
try:
    import boxmot
    print(f"DEBUG: boxmot imported from: {boxmot.__file__}")
    from boxmot.trackers.common.appearance import resolve_batch_embeddings
    print(f"DEBUG: resolve_batch_embeddings loaded from: {resolve_batch_embeddings.__code__.co_filename}")
except Exception as e:
    print(f"DEBUG: Error during boxmot inspection in detect_track_export: {e}")
# --- END BOXMOT PATH FORCE ---



# --- START BOXMOT PATH FORCE (Inserted by Colab Agent) ---
_boxmot_repo_path = '/tmp/boxmot_repo'
# Explicitly remove any other boxmot installations from sys.path if they were loaded
# This is a robust way to ensure the editable install is used by the subprocess
_paths_to_remove = []
for p in list(sys.path): # Iterate over a copy as sys.path may be modified
    if 'boxmot' in p and _boxmot_repo_path not in p:
        _paths_to_remove.append(p)
for p in _paths_to_remove:
    sys.path.remove(p)
    print(f"DEBUG: Removed '{p}' from sys.path to force correct boxmot usage.")

# Ensure the cloned boxmot repo is at the very beginning of sys.path
if _boxmot_repo_path not in sys.path:
    sys.path.insert(0, _boxmot_repo_path)
    print(f"DEBUG: Inserted '{_boxmot_repo_path}' to sys.path.")

# Diagnostic prints to verify boxmot import source
try:
    import boxmot
    print(f"DEBUG: boxmot imported from: {boxmot.__file__}")
    from boxmot.trackers.common.appearance import resolve_batch_embeddings
    print(f"DEBUG: resolve_batch_embeddings loaded from: {resolve_batch_embeddings.__code__.co_filename}")
except Exception as e:
    print(f"DEBUG: Error during boxmot inspection in detect_track_export: {e}")
# --- END BOXMOT PATH FORCE ---




from src.config import MODEL_REGISTRY, BOTSORT_CONFIG, EXPORT_DIR, DEFAULT_CONF_THRESH, DEFAULT_IMG_SIZE, ALLOWED_CLASS_IDS, CLASS_NAMES
from src.model_loader import load_model_from_registry
from src.detector import get_detections
from src.tracker import create_botsort_tracker, parse_tracker_output
from src.cvat_mot_export import write_cvat_mot_zip
from src.video_io import get_video_properties, draw_track_overlay
from src.logging_utils import setup_logger

logger = setup_logger()

def run_tracking_pipeline(
    video_path: Path,
    model_config: Path,
    model_key: str,
    tracker_config: Path,
    out_zip: Path,
    conf_thresh: float,
    imgsz: int,
    save_preview: bool,
    max_frames: int | None = None,
    weights_override: Path | None = None,
    preview_dir: Path | None = None
) -> dict:
    start_time = time.time()
    
    if not video_path.exists():
        # Check if the path exists with a standard video extension appended
        resolved_path = None
        for ext in [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"]:
            temp_path = video_path.with_suffix(ext)
            if temp_path.exists():
                resolved_path = temp_path
                break
        if resolved_path:
            logger.info(f"Video file not found at: {video_path}. Automatically resolved to: {resolved_path}")
            video_path = resolved_path
        else:
            raise FileNotFoundError(f"Video file not found at: {video_path}")
        
    logger.info(f"Loading model info for key: {model_key}...")
    model, model_info = load_model_from_registry(model_config, model_key)
    
    # Apply weights override if provided
    if weights_override:
        if not weights_override.exists():
            raise FileNotFoundError(f"Override weights file not found at: {weights_override}")
        logger.info(f"Overriding weights from registry with: {weights_override}")
        # Ultralytics model can load new weights via load() or reinstantiating
        from ultralytics import RTDETR, YOLO
        family = model_info.get("family", "rtdetr").lower()
        if family == "rtdetr":
            model = RTDETR(str(weights_override))
        else:
            model = YOLO(str(weights_override))
        model_info["weights_path"] = str(weights_override)
        
    logger.info("Initializing tracker...")
    tracker = create_botsort_tracker(tracker_config)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    video_props = get_video_properties(video_path)
    fps = video_props["fps"]
    frame_count = video_props["frame_count"]
    frame_w = video_props["width"]
    frame_h = video_props["height"]
    
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    
    preview_writer = None
    preview_path = ""
    temp_preview_out = None
    if save_preview:
        if preview_dir is None:
            # Previews go to preview dir or alongside out_zip
            preview_dir = out_zip.parent / "previews"
            if "exports" in out_zip.parts:
                # Try to place in the standard preview directory: exports/previews/
                preview_dir = PROJECT_ROOT / "exports" / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_out = preview_dir / f"{video_path.stem}_preview.mp4"
        preview_path = str(preview_out)
        
        # Write to a temporary file first so we can add timecode track with ffmpeg
        temp_preview_out = preview_dir / f"{video_path.stem}_preview_temp.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_writer = cv2.VideoWriter(str(temp_preview_out), fourcc, fps, (frame_w, frame_h))
        logger.info(f"Saving tracking preview video (temporary) to: {temp_preview_out}")

    mot_rows = []
    # If max_frames is positive, use it. If -1, process all.
    limit_frames = frame_count
    if max_frames is not None and max_frames > 0:
        limit_frames = min(frame_count, max_frames)
        
    frame_id = 1
    detections_total = 0
    unique_tracks = set()
    empty_frames_count = 0
    
    logger.info(f"Processing video: {video_path.name} ({limit_frames} frames)...")
    
    allowed_classes = set(model_info.get("allowed_class_ids", list(ALLOWED_CLASS_IDS)))
    
    with tqdm(total=limit_frames, desc=f"Tracking {video_path.name}") as pbar:
        while True:
            if max_frames is not None and max_frames > 0 and frame_id > max_frames:
                break
                
            ok, frame = cap.read()
            if not ok:
                break
                
            # Run detection
            detections = get_detections(model, frame, imgsz, conf_thresh, allowed_classes)
            detections_total += len(detections)
            
            if len(detections) == 0:
                empty_frames_count += 1
                
            # Run tracking
            tracks = tracker.update(detections, frame)
            parsed_tracks = parse_tracker_output(tracks)
            
            # Format and save MOT rows
            # Columns: frame_id, track_id, x, y, w, h, not_ignored (1), class_id, visibility (track_conf)
            for x1, y1, x2, y2, track_id, track_conf, class_id in parsed_tracks:
                w = x2 - x1
                h = y2 - y1
                if w <= 1 or h <= 1:
                    continue
                    
                x = x1
                y = y1
                
                # CVAT MOT 1.1 classes are 1-indexed (player=1, ball=2, etc.)
                cvat_class_id = int(class_id) + 1
                visibility = max(0.0, min(1.0, float(track_conf)))
                mot_rows.append([
                    frame_id, 
                    track_id, 
                    round(x, 2), 
                    round(y, 2), 
                    round(w, 2), 
                    round(h, 2), 
                    1, 
                    cvat_class_id, 
                    round(visibility, 5)
                ])
                unique_tracks.add(track_id)
                
            # Render preview frame if enabled
            if preview_writer is not None:
                annotated_frame = draw_track_overlay(frame.copy(), parsed_tracks, allowed_classes)
                preview_writer.write(annotated_frame)
                
            frame_id += 1
            pbar.update(1)
            
    cap.release()
    if preview_writer is not None:
        preview_writer.release()
        
        # Post-process with FFmpeg to add timecode track natively recognized by QuickTime
        import subprocess
        temp_str = str(temp_preview_out)
        final_str = str(preview_path)
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_str, "-c", "copy", "-timecode", "00:00:00:00", final_str]
        
        try:
            logger.info(f"Running ffmpeg to add QuickTime timecode: {' '.join(ffmpeg_cmd)}")
            res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if res.returncode == 0:
                logger.info(f"Successfully added QuickTime timecode to: {preview_path}")
                if Path(temp_str).exists():
                    Path(temp_str).unlink()
            else:
                logger.warning(
                    f"FFmpeg returned non-zero code {res.returncode}. Error: {res.stderr}\n"
                    "Falling back to standard OpenCV preview output."
                )
                if Path(temp_str).exists():
                    if Path(final_str).exists():
                        Path(final_str).unlink()
                    Path(temp_str).rename(final_str)
        except Exception as e:
            logger.warning(
                f"FFmpeg not found or failed: {e}. Falling back to standard OpenCV preview output."
            )
            if Path(temp_str).exists():
                if Path(final_str).exists():
                    Path(final_str).unlink()
                Path(temp_str).rename(final_str)
        
    # Write MOT 1.1 zip file
    write_cvat_mot_zip(mot_rows, out_zip, model_info.get("class_names", list(CLASS_NAMES)))
    
    elapsed_time = time.time() - start_time
    
    summary = {
        "video": video_path.name,
        "model_key": model_key,
        "weights_path": model_info.get("weights_path"),
        "tracker_config": str(tracker_config),
        "conf": conf_thresh,
        "imgsz": imgsz,
        "frames_processed": frame_id - 1,
        "tracks_created": len(unique_tracks),
        "detections_total": detections_total,
        "empty_frames_count": empty_frames_count,
        "output_zip": str(out_zip),
        "preview_path": preview_path,
        "runtime_seconds": round(elapsed_time, 2)
    }
    
    # Save JSON sidecar alongside ZIP
    json_summary_path = out_zip.with_suffix(".json")
    json_summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote JSON sidecar summary: {json_summary_path}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Run detection + BoT-SORT on a video and export to CVAT MOT 1.1 format.")
    parser.add_argument("--video", required=True, type=Path, help="Path to input video file.")
    parser.add_argument("--model-config", default=MODEL_REGISTRY, type=Path, help="Path to model registry YAML.")
    parser.add_argument("--model-key", default="rtdetrx_player_best", type=str, help="Model registry key.")
    parser.add_argument("--tracker-config", default=BOTSORT_CONFIG, type=Path, help="Path to tracker YAML config.")
    parser.add_argument("--out", required=True, type=Path, help="Path to output CVAT MOT ZIP file.")
    parser.add_argument("--conf", default=DEFAULT_CONF_THRESH, type=float, help="Confidence threshold.")
    parser.add_argument("--imgsz", default=DEFAULT_IMG_SIZE, type=int, help="Inference resolution.")
    parser.add_argument("--save-preview", action="store_true", help="Save visualization preview MP4.")
    parser.add_argument("--preview-dir", default=None, type=Path, help="Optional custom folder to save previews.")
    parser.add_argument("--max-frames", default=None, type=int, help="Limit number of frames to track.")
    parser.add_argument("--weights", default=None, type=Path, help="Override weights path.")
    
    args = parser.parse_args()
    
    try:
        summary = run_tracking_pipeline(
            video_path=args.video,
            model_config=args.model_config,
            model_key=args.model_key,
            tracker_config=args.tracker_config,
            out_zip=args.out,
            conf_thresh=args.conf,
            imgsz=args.imgsz,
            save_preview=args.save_preview,
            max_frames=args.max_frames,
            weights_override=args.weights,
            preview_dir=args.preview_dir
        )
        print(json.dumps(summary, indent=2))
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
