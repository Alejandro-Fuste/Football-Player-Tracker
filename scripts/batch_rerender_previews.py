import argparse
import csv
import io
import json
import time
import sys
import zipfile
import subprocess
import traceback
import cv2
from pathlib import Path
from tqdm.auto import tqdm

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPORT_DIR, VIDEOS_DIR, CVAT_EXPORT_DIR, PREVIEW_DIR
from src.video_io import get_video_properties, draw_track_overlay
from src.logging_utils import setup_logger

logger = setup_logger()

# Supported video patterns
VIDEO_EXTENSIONS = [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"]

def load_tracks_from_mot_zip(zip_path: Path) -> dict[int, list[tuple]]:
    """
    Reads tracking records from a CVAT MOT 1.1 ZIP archive (gt/gt.txt)
    and returns a mapping of frame_id -> list of track tuples:
    [(x1, y1, x2, y2, track_id, conf, class_id), ...]
    """
    frame_tracks: dict[int, list[tuple]] = {}
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find gt.txt within the archive
        gt_filename = None
        for name in zf.namelist():
            if name.endswith("gt.txt"):
                gt_filename = name
                break
                
        if not gt_filename:
            raise ValueError(f"No 'gt.txt' found inside ZIP archive: {zip_path}")
            
        with zf.open(gt_filename) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                if not row or len(row) < 6:
                    continue
                try:
                    frame_id = int(row[0])
                    track_id = int(row[1])
                    x1 = float(row[2])
                    y1 = float(row[3])
                    w = float(row[4])
                    h = float(row[5])
                    x2 = x1 + w
                    y2 = y1 + h
                    
                    # CVAT MOT 1.1 format: class_id is 1-indexed at column 7 if present
                    cvat_class_id = int(row[7]) if len(row) > 7 else 1
                    class_id = max(0, cvat_class_id - 1)
                    
                    # Visibility / confidence at column 8 if present
                    conf = float(row[8]) if len(row) > 8 else 1.0
                    
                    if frame_id not in frame_tracks:
                        frame_tracks[frame_id] = []
                    frame_tracks[frame_id].append((x1, y1, x2, y2, track_id, conf, class_id))
                except (ValueError, IndexError):
                    continue
                    
    return frame_tracks

def resolve_video_path(zip_path: Path, clips_dir: Path) -> Path | None:
    """
    Resolves matching raw video file in clips_dir corresponding to zip_path.
    Strips '_cvat_mot' suffix from zip stem if present.
    """
    base_name = zip_path.stem
    if base_name.endswith("_cvat_mot"):
        base_name = base_name[:-9]
        
    # Check direct match with extensions
    for ext in VIDEO_EXTENSIONS:
        cand = clips_dir / f"{base_name}{ext}"
        if cand.exists():
            return cand
            
    # Fallback: exact filename search without suffix changes
    for ext in VIDEO_EXTENSIONS:
        cand = clips_dir / f"{zip_path.stem}{ext}"
        if cand.exists():
            return cand
            
    return None

def rerender_preview(
    video_path: Path,
    frame_tracks: dict[int, list[tuple]],
    out_preview_path: Path,
    font_scale: float | None = None,
    box_thickness: int | None = None
) -> dict:
    """
    Renders preview MP4 video with updated overlay settings for the given frame_tracks.
    """
    start_time = time.time()
    video_props = get_video_properties(video_path)
    fps = video_props["fps"]
    frame_count = video_props["frame_count"]
    frame_w = video_props["width"]
    frame_h = video_props["height"]

    out_preview_path.parent.mkdir(parents=True, exist_ok=True)
    temp_preview_out = out_preview_path.parent / f"{out_preview_path.stem}_temp.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video source at: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_preview_out), fourcc, fps, (frame_w, frame_h))

    frame_id = 1
    with tqdm(total=frame_count, desc=f"Re-rendering {video_path.name}") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            parsed_tracks = frame_tracks.get(frame_id, [])
            annotated_frame = draw_track_overlay(
                frame, 
                parsed_tracks, 
                allowed_classes=None,
                font_scale=font_scale,
                box_thickness=box_thickness
            )
            writer.write(annotated_frame)
            frame_id += 1
            pbar.update(1)

    cap.release()
    writer.release()

    # Post-process with FFmpeg to add timecode / ensure playback compatibility
    temp_str = str(temp_preview_out)
    final_str = str(out_preview_path)
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_str, "-c", "copy", "-timecode", "00:00:00:00", final_str]

    try:
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res.returncode == 0:
            if Path(temp_str).exists():
                Path(temp_str).unlink()
        else:
            if Path(temp_str).exists():
                if Path(final_str).exists():
                    Path(final_str).unlink()
                Path(temp_str).rename(final_str)
    except Exception:
        if Path(temp_str).exists():
            if Path(final_str).exists():
                Path(final_str).unlink()
            Path(temp_str).rename(final_str)

    elapsed_time = time.time() - start_time
    return {
        "video_name": video_path.name,
        "preview_path": final_str,
        "frames_rendered": frame_id - 1,
        "runtime_seconds": round(elapsed_time, 2)
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch re-render video previews from MOT 1.1 ZIP files with dynamic/custom bounding box scaling."
    )
    parser.add_argument("--zips-dir", default=CVAT_EXPORT_DIR, type=Path, help="Directory containing MOT 1.1 ZIP files.")
    parser.add_argument("--clips-dir", default=VIDEOS_DIR, type=Path, help="Directory containing original input videos.")
    parser.add_argument("--out-dir", default=PREVIEW_DIR, type=Path, help="Output directory to save re-rendered preview MP4s.")
    parser.add_argument("--csv-list", default=None, type=Path, help="Optional CSV file listing specific filenames to process.")
    parser.add_argument("--font-scale", default=None, type=float, help="Explicit text font scale (e.g. 0.3, 0.4). Auto-scaled if omitted.")
    parser.add_argument("--box-thickness", default=None, type=int, help="Explicit line thickness (e.g. 1, 2). Auto-scaled if omitted.")
    parser.add_argument("--max-videos", default=None, type=int, help="Limit number of videos to process for testing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preview videos.")

    args = parser.parse_args()

    if not args.zips_dir.exists():
        logger.error(f"ZIP directory does not exist: {args.zips_dir}")
        sys.exit(1)

    if not args.clips_dir.exists():
        logger.error(f"Clips directory does not exist: {args.clips_dir}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Gather ZIP files
    zip_paths: list[Path] = []
    if args.csv_list:
        if not args.csv_list.exists():
            logger.error(f"CSV list file not found: {args.csv_list}")
            sys.exit(1)

        raw_entries: list[str] = []
        with open(args.csv_list, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            target_idx = 0
            if headers:
                headers_norm = [h.strip().lower() for h in headers]
                if "filename" in headers_norm:
                    target_idx = headers_norm.index("filename")
                elif "shortname" in headers_norm:
                    target_idx = headers_norm.index("shortname")
                elif "zip_name" in headers_norm:
                    target_idx = headers_norm.index("zip_name")
                else:
                    if any(ext in headers[0].lower() for ext in [".mp4", ".mov", ".avi", ".mkv", ".zip"]):
                        val = headers[0].strip()
                        if val:
                            raw_entries.append(val)

            for row in reader:
                if row and len(row) > target_idx:
                    val = row[target_idx].strip()
                    if val:
                        raw_entries.append(val)

        # Build lookup table of existing ZIP files in zips_dir
        existing_zips = sorted(list(args.zips_dir.glob("*.zip")))
        zip_map: dict[str, Path] = {}
        for zp in existing_zips:
            zip_map[zp.name.lower()] = zp
            zip_map[zp.stem.lower()] = zp
            if zp.stem.lower().endswith("_cvat_mot"):
                zip_map[zp.stem[:-9].lower()] = zp

        for val in raw_entries:
            p_val = Path(val)
            stem = p_val.stem.lower()
            name = p_val.name.lower()
            base_stem = stem[:-9] if stem.endswith("_cvat_mot") else stem

            # Search order:
            # 1. Exact filename (e.g. JetSweep_66.mp4 or JetSweep_66_cvat_mot.zip)
            # 2. Exact stem (e.g. JetSweep_66_cvat_mot)
            # 3. Base stem without _cvat_mot (e.g. JetSweep_66)
            matched_zip = (
                zip_map.get(name) or
                zip_map.get(stem) or
                zip_map.get(base_stem)
            )

            if matched_zip:
                if matched_zip not in zip_paths:
                    zip_paths.append(matched_zip)
            else:
                logger.warning(f"CSV entry '{val}' (stem: '{stem}') could not be matched to any ZIP file in {args.zips_dir}")
    else:
        zip_paths = sorted(list(args.zips_dir.glob("*.zip")))

    if args.max_videos is not None:
        zip_paths = zip_paths[:args.max_videos]

    logger.info(f"Found {len(zip_paths)} MOT ZIP files in {args.zips_dir}")

    summaries = []
    success_count = 0
    failure_count = 0
    start_total = time.time()

    for idx, zip_path in enumerate(zip_paths):
        video_path = resolve_video_path(zip_path, args.clips_dir)
        if not video_path:
            logger.warning(f"[{idx+1}/{len(zip_paths)}] Skipping {zip_path.name}: matching video not found in {args.clips_dir}")
            summaries.append({
                "zip_name": zip_path.name,
                "status": "failed",
                "error_message": f"Matching video not found in {args.clips_dir}"
            })
            failure_count += 1
            continue

        preview_out = args.out_dir / f"{video_path.stem}_preview.mp4"
        if preview_out.exists() and not args.overwrite:
            logger.info(f"[{idx+1}/{len(zip_paths)}] Skipping existing: {preview_out.name}")
            summaries.append({
                "zip_name": zip_path.name,
                "video_name": video_path.name,
                "status": "skipped",
                "preview_path": str(preview_out)
            })
            continue

        logger.info(f"[{idx+1}/{len(zip_paths)}] Re-rendering preview for: {video_path.name}")
        try:
            tracks = load_tracks_from_mot_zip(zip_path)
            res = rerender_preview(
                video_path=video_path,
                frame_tracks=tracks,
                out_preview_path=preview_out,
                font_scale=args.font_scale,
                box_thickness=args.box_thickness
            )
            res["zip_name"] = zip_path.name
            res["status"] = "success"
            summaries.append(res)
            success_count += 1
            logger.info(f"Successfully re-rendered: {preview_out.name}")
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Failed to re-render {zip_path.name}: {err_msg}")
            trace_path = logs_dir / f"{zip_path.stem}_error.log"
            with open(trace_path, "w") as f:
                traceback.print_exc(file=f)
            summaries.append({
                "zip_name": zip_path.name,
                "video_name": video_path.name,
                "status": "failed",
                "error_message": err_msg
            })
            failure_count += 1

    total_elapsed = time.time() - start_total
    logger.info(f"Batch re-rendering completed in {total_elapsed:.2f}s. Successes: {success_count}, Failures: {failure_count}")

    # Save summary
    json_path = args.out_dir / "batch_rerender_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2))
    logger.info(f"Wrote summary JSON: {json_path}")

if __name__ == "__main__":
    main()
