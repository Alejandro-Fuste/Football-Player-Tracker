import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def load_summary(summary_path: Path) -> list:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found at: {summary_path}")
    with open(summary_path, "r") as f:
        return json.load(f)

def print_summary_report(summaries: list) -> None:
    total_videos = len(summaries)
    successes = sum(1 for s in summaries if s.get("status") == "success")
    failures = sum(1 for s in summaries if s.get("status") == "failed")
    skipped = sum(1 for s in summaries if s.get("status") == "skipped")

    total_frames = sum(s.get("frames_processed", 0) for s in summaries)
    total_detections = sum(s.get("detections_total", 0) for s in summaries)
    total_tracks = sum(s.get("tracks_created", 0) for s in summaries)
    total_runtime = sum(s.get("runtime_seconds", 0.0) for s in summaries)

    print("\n" + "="*80)
    print(" BATCH TRACKING EXECUTION SUMMARY")
    print("="*80)
    print(f"Total Videos Scanned: {total_videos}")
    print(f"  - Successful runs: {successes}")
    print(f"  - Failed runs:     {failures}")
    print(f"  - Skipped runs:    {skipped}")
    print(f"Total Frames Processed: {total_frames}")
    print(f"Total Detections Made: {total_detections}")
    print(f"Total Unique Tracks:   {total_tracks}")
    print(f"Cumulative Run Time:   {total_runtime:.2f} seconds")
    print("="*80)
    
    # Print detail table header
    print(f"{'Video Name':<30} | {'Status':<10} | {'Frames':<6} | {'Tracks':<6} | {'Runtime (s)':<12}")
    print("-"*80)
    
    for s in summaries:
        name = s.get("video_name", s.get("video", "unknown"))
        # Truncate long names
        if len(name) > 30:
            name = name[:27] + "..."
            
        status = s.get("status", "success")
        frames = s.get("frames_processed", 0)
        tracks = s.get("tracks_created", 0)
        runtime = s.get("runtime_seconds", 0.0)
        
        print(f"{name:<30} | {status:<10} | {frames:<6} | {tracks:<6} | {runtime:<12.2f}")
        
    print("="*80 + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregates and formats batch video tracking results.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to batch_tracking_summary.json.")
    args = parser.parse_args()

    try:
        summaries = load_summary(args.summary)
        print_summary_report(summaries)
    except Exception as e:
        print(f"Error loading summary: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
