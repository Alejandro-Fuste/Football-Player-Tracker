import argparse
import csv
import json
import time
import sys
import traceback
from pathlib import Path

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, BOTSORT_CONFIG, EXPORT_DIR, VIDEOS_DIR, DEFAULT_CONF_THRESH, DEFAULT_IMG_SIZE
from scripts.detect_track_export import run_tracking_pipeline
from src.logging_utils import setup_logger

logger = setup_logger()

# Supported video patterns
VIDEO_EXTENSIONS = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV"]

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process videos and export tracker predictions to CVAT MOT 1.1 format.")
    parser.add_argument("--model-config", default=MODEL_REGISTRY, type=Path, help="Path to model registry YAML.")
    parser.add_argument("--model-key", default="rtdetrx_player_best", type=str, help="Model registry key.")
    parser.add_argument("--tracker-config", default=BOTSORT_CONFIG, type=Path, help="Path to tracker YAML config.")
    parser.add_argument("--clips-dir", default=VIDEOS_DIR, type=Path, help="Directory containing input clips.")
    parser.add_argument("--out-dir", default=EXPORT_DIR, type=Path, help="Output folder to write tracks to.")
    parser.add_argument("--conf", default=DEFAULT_CONF_THRESH, type=float, help="Confidence threshold.")
    parser.add_argument("--imgsz", default=DEFAULT_IMG_SIZE, type=int, help="Inference resolution size.")
    parser.add_argument("--save-preview", action="store_true", help="Save visualization preview MP4s.")
    parser.add_argument("--max-videos", default=None, type=int, help="Limit total videos processed.")
    parser.add_argument("--max-frames", default=-1, type=int, help="Limit number of frames processed per video (-1 for unlimited).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output ZIP files.")
    parser.add_argument("--write-summary", action="store_true", help="Write summary CSV/JSON files.")
    parser.add_argument("--weights", default=None, type=Path, help="Override weights path.")
    parser.add_argument("--csv-list", default=None, type=Path, help="Optional CSV file containing a list of videos to track.")

    args = parser.parse_args()

    # Validation
    if not args.clips_dir.exists():
        logger.error(f"Clips directory does not exist: {args.clips_dir}")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Gather all video files
    video_paths = []
    if args.csv_list:
        if not args.csv_list.exists():
            logger.error(f"CSV list file not found: {args.csv_list}")
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(1)
            
        import csv
        video_names = []
        with open(args.csv_list, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            
            if headers:
                # Normalize headers: strip and lowercase
                headers_norm = [h.strip().lower() for h in headers]
                
                # Check for "filename" first, fallback to "shortname"
                file_idx = -1
                if "filename" in headers_norm:
                    file_idx = headers_norm.index("filename")
                elif "shortname" in headers_norm:
                    file_idx = headers_norm.index("shortname")
                
                if file_idx != -1:
                    for row in reader:
                        if row and len(row) > file_idx:
                            val = row[file_idx].strip()
                            if val:
                                video_names.append(val)
                else:
                    # Fallback to the first column if headers do not match
                    for row in reader:
                        if row and row[0].strip():
                            video_names.append(row[0].strip())
            else:
                logger.warning("Uploaded CSV file is empty.")
                
        # Resolve names against the clips-dir
        for name in video_names:
            has_ext = any(name.lower().endswith(ext.replace('*', '').lower()) for ext in VIDEO_EXTENSIONS)
            
            resolved_path = None
            if has_ext:
                cand = args.clips_dir / name
                if cand.exists():
                    resolved_path = cand
            else:
                # Try standard video extensions
                for ext in [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"]:
                    cand = args.clips_dir / f"{name}{ext}"
                    if cand.exists():
                        resolved_path = cand
                        break
            
            if resolved_path:
                video_paths.append(resolved_path)
            else:
                logger.warning(f"Could not find video in clips-dir for CSV entry: {name}")
                
        # Remove duplicates while preserving CSV order
        seen = set()
        video_paths = [x for x in video_paths if not (x in seen or seen.add(x))]
    else:
        # Standard folder scanning behavior
        for ext in VIDEO_EXTENSIONS:
            video_paths.extend(args.clips_dir.glob(ext))
        video_paths = sorted(list(set(video_paths)))
    
    if args.max_videos is not None:
        video_paths = video_paths[:args.max_videos]

    logger.info(f"Found {len(video_paths)} videos in {args.clips_dir}")

    batch_summaries = []
    success_count = 0
    failure_count = 0
    total_start_time = time.time()

    for idx, video_path in enumerate(video_paths):
        out_zip = args.out_dir / f"{video_path.stem}_cvat_mot.zip"
        
        # Skip logic
        if out_zip.exists() and not args.overwrite:
            logger.info(f"[{idx+1}/{len(video_paths)}] Skipping existing: {out_zip.name}")
            # Load sidecar if exists to count in summary
            sidecar_path = out_zip.with_suffix(".json")
            if sidecar_path.exists():
                try:
                    summary_data = json.loads(sidecar_path.read_text())
                    summary_data["status"] = "skipped"
                    batch_summaries.append(summary_data)
                    continue
                except Exception:
                    pass
            batch_summaries.append({
                "video_name": video_path.name,
                "status": "skipped",
                "frames_processed": 0,
                "detections_total": 0,
                "tracks_created": 0,
                "output_zip": str(out_zip),
                "preview_path": "",
                "error_message": "",
                "runtime_seconds": 0.0
            })
            continue

        logger.info(f"[{idx+1}/{len(video_paths)}] Processing: {video_path.name}")
        
        # Determine max frames parameter
        max_fr = None if args.max_frames == -1 else args.max_frames
        
        # Save batch previews to a previews subfolder in out-dir
        batch_preview_dir = args.out_dir / "previews" if args.save_preview else None

        try:
            summary = run_tracking_pipeline(
                video_path=video_path,
                model_config=args.model_config,
                model_key=args.model_key,
                tracker_config=args.tracker_config,
                out_zip=out_zip,
                conf_thresh=args.conf,
                imgsz=args.imgsz,
                save_preview=args.save_preview,
                max_frames=max_fr,
                weights_override=args.weights,
                preview_dir=batch_preview_dir
            )
            summary["video_name"] = video_path.name
            summary["status"] = "success"
            summary["error_message"] = ""
            batch_summaries.append(summary)
            success_count += 1
            logger.info(f"Successfully processed {video_path.name}. Tracks: {summary['tracks_created']}")
        except Exception as e:
            failure_count += 1
            err_msg = str(e)
            logger.error(f"Failed to process {video_path.name}: {err_msg}")
            
            # Log traceback to logs folder
            trace_path = logs_dir / f"{video_path.stem}_error.log"
            with open(trace_path, "w") as f:
                traceback.print_exc(file=f)
            
            batch_summaries.append({
                "video_name": video_path.name,
                "status": "failed",
                "frames_processed": 0,
                "detections_total": 0,
                "tracks_created": 0,
                "output_zip": str(out_zip),
                "preview_path": "",
                "error_message": err_msg,
                "runtime_seconds": 0.0
            })

    total_elapsed = time.time() - total_start_time
    logger.info(f"Batch processing completed in {total_elapsed:.2f} seconds. Successes: {success_count}, Failures: {failure_count}")

    # Write summary files if write-summary or default is requested
    if args.write_summary or not args.write_summary: # Write summary files unconditionally for robust export summaries
        # JSON summary
        json_summary_path = args.out_dir / "batch_tracking_summary.json"
        json_summary_path.write_text(json.dumps(batch_summaries, indent=2))
        logger.info(f"Wrote batch summary JSON: {json_summary_path}")

        # CSV summary
        csv_summary_path = args.out_dir / "batch_tracking_summary.csv"
        csv_columns = [
            "video_name", "status", "frames_processed", "detections_total", 
            "tracks_created", "output_zip", "preview_path", "error_message", "runtime_seconds"
        ]
        
        with open(csv_summary_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns, extrasaction="ignore")
            writer.writeheader()
            for summary in batch_summaries:
                # Ensure fields match columns
                row = {
                    "video_name": summary.get("video_name", summary.get("video")),
                    "status": summary.get("status", "success"),
                    "frames_processed": summary.get("frames_processed", 0),
                    "detections_total": summary.get("detections_total", 0),
                    "tracks_created": summary.get("tracks_created", 0),
                    "output_zip": summary.get("output_zip", ""),
                    "preview_path": summary.get("preview_path", ""),
                    "error_message": summary.get("error_message", ""),
                    "runtime_seconds": summary.get("runtime_seconds", 0.0)
                }
                writer.writerow(row)
                
        logger.info(f"Wrote batch summary CSV: {csv_summary_path}")

if __name__ == "__main__":
    main()