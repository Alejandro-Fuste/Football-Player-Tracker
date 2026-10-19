import argparse
import sys
import json
from pathlib import Path

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_REGISTRY, 
    BOTSORT_CONFIG, 
    DEFAULT_IMG_SIZE,
    PLAYER_CONF_THRESH,
    BALL_CONF_THRESH
)
from scripts.detect_track_export import run_tracking_pipeline, logger

def main():
    parser = argparse.ArgumentParser(description="Wrapper script running detection + BoT-SORT on a video and exporting to CVAT MOT 1.1 format.")
    parser.add_argument("--video", required=True, type=Path, help="Path to input video file.")
    parser.add_argument("--model-config", default=MODEL_REGISTRY, type=Path, help="Path to model registry YAML.")
    parser.add_argument("--model-key", default="rtdetrx_player_best", type=str, help="Model registry key.")
    parser.add_argument("--tracker-config", default=BOTSORT_CONFIG, type=Path, help="Path to tracker YAML config.")
    parser.add_argument("--out", required=True, type=Path, help="Path to output CVAT MOT ZIP file.")
    parser.add_argument("--conf", default=None, type=float, help="Deprecated global confidence threshold.")
    parser.add_argument("--player-conf", default=PLAYER_CONF_THRESH, type=float, help="Player confidence threshold.")
    parser.add_argument("--ball-conf", default=BALL_CONF_THRESH, type=float, help="Ball confidence threshold.")
    parser.add_argument("--imgsz", default=DEFAULT_IMG_SIZE, type=int, help="Inference resolution.")
    parser.add_argument("--save-preview", action="store_true", help="Save visualization preview MP4.")
    parser.add_argument("--max-frames", default=None, type=int, help="Limit number of frames to track.")
    parser.add_argument("--weights", default=None, type=Path, help="Override weights path.")
    
    args = parser.parse_args()

    player_conf = args.player_conf
    ball_conf = args.ball_conf
    if args.conf is not None:
        logger.warning(f"Using deprecated --conf argument ({args.conf}) as player confidence threshold.")
        player_conf = args.conf

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
            player_conf=player_conf,
            ball_conf=ball_conf
        )
        print(json.dumps(summary, indent=2))
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
