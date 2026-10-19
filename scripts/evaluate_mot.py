import argparse
import sys
import json
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SPLITS, ANNOT_ROOT, EXPORT_DIR, get_annot_dir_name
from src.mot_eval import evaluate_one_sequence, save_metrics_report, save_summary_report
from src.logging_utils import setup_logger

logger = setup_logger()

def parse_class_ids(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}

def evaluate_split_workflow(
    split_name: str,
    out_dir: Path,
    iou_threshold: float,
    gt_class_ids: set[int] | None = None,
    pred_class_ids: set[int] | None = None
):
    if split_name not in SPLITS:
        raise ValueError(f"Unknown split: {split_name}. Available: {list(SPLITS.keys())}")

    if ANNOT_ROOT is None or EXPORT_DIR is None:
        raise RuntimeError("ANNOT_ROOT or EXPORT_DIR is not configured. Define them in configs/project_config.py")

    metrics_list = []

    for sequence_name in SPLITS[split_name]:
        gt_path = ANNOT_ROOT / get_annot_dir_name(sequence_name) / "gt" / "gt.txt"
        
        # Check standard predictions path and batch paths
        pred_path_candidates = [
            EXPORT_DIR / f"{sequence_name}_cvat_mot.zip",
            EXPORT_DIR / "cvat_mot_tracks" / f"{sequence_name}_cvat_mot.zip",
            EXPORT_DIR / "cvat_mot_tracks" / "single" / f"{sequence_name}_cvat_mot.zip",
            EXPORT_DIR / "cvat_mot_tracks" / "batch" / f"{sequence_name}_cvat_mot.zip",
            PROJECT_ROOT / "exports" / "cvat_mot_tracks" / f"{sequence_name}_cvat_mot.zip"
        ]
        
        pred_path = None
        for cand in pred_path_candidates:
            if cand.exists():
                pred_path = cand
                break
                
        if not gt_path.exists():
            logger.warning(f"Skipping {sequence_name}: missing GT at {gt_path}")
            continue

        if not pred_path:
            logger.warning(f"Skipping {sequence_name}: missing prediction ZIP under candidates.")
            continue

        logger.info(f"Evaluating sequence: {sequence_name}")
        metrics = evaluate_one_sequence(
            gt_path=gt_path,
            pred_path=pred_path,
            sequence_name=sequence_name,
            out_dir=out_dir,
            iou_threshold=iou_threshold,
            gt_class_ids=gt_class_ids,
            pred_class_ids=pred_class_ids,
        )
        metrics_list.append(metrics)

    if metrics_list:
        save_summary_report(metrics_list, out_dir, summary_name=split_name)
        logger.info(f"Split '{split_name}' summary reports written to {out_dir}")
    else:
        logger.warning(f"No sequences were evaluated for split: {split_name}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MOT tracking predictions against GT.")
    parser.add_argument("--gt", type=Path, default=None, help="Path to ground truth gt.txt file.")
    parser.add_argument("--pred", type=Path, default=None, help="Path to prediction zip file or txt file.")
    parser.add_argument("--sequence-name", type=str, default=None, help="Sequence name (e.g. SWide008).")
    parser.add_argument("--batch-split", type=str, choices=list(SPLITS.keys()) if SPLITS else None, default=None, help="Evaluate a split of sequences.")
    parser.add_argument("--out-json", type=Path, default=None, help="Specific output JSON path for single evaluation.")
    parser.add_argument("--out-md", type=Path, default=None, help="Specific output Markdown report path for single evaluation.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory where JSON/CSV metrics will be saved.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for frame matching.")
    parser.add_argument("--gt-class-ids", type=str, default=None, help="Comma-separated GT class IDs to evaluate.")
    parser.add_argument("--pred-class-ids", type=str, default=None, help="Comma-separated prediction class IDs to evaluate (player=1).")

    args = parser.parse_args()

    gt_class_ids = parse_class_ids(args.gt_class_ids)
    pred_class_ids = parse_class_ids(args.pred_class_ids)

    # In MOT ground truth class is usually also 1 for player or 0 depending on format, let's keep all GT if not specified.

    if args.batch_split:
        out_dir = args.out_dir or (PROJECT_ROOT / "exports" / "metrics")
        evaluate_split_workflow(
            split_name=args.batch_split,
            out_dir=out_dir,
            iou_threshold=args.iou_threshold,
            gt_class_ids=gt_class_ids,
            pred_class_ids=pred_class_ids
        )
        return

    if args.gt is None or args.pred is None:
        # Do not crash if files are missing, print clear warning/error
        logger.error("Error: --gt and --pred paths must be provided for single evaluation, or use --batch-split.")
        sys.exit(1)

    sequence_name = args.sequence_name
    if not sequence_name:
        # Use filename stem or sequence dir name
        sequence_name = args.gt.parent.parent.name if args.gt else "sequence"
    
    # Strip common video extensions from sequence_name if present
    for ext in [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"]:
        if sequence_name.endswith(ext):
            sequence_name = sequence_name[:-len(ext)]
            break

    if not args.gt.exists():
        logger.warning(
            f"Ground truth not found for sequence {sequence_name} at: {args.gt}\n"
            "  * Note: Custom video clips (like JetSweep_1) do not have ground truth annotations in the dataset.\n"
            "  * If you want to test the evaluation pipeline, please use a benchmark validation sequence name like 'SWide008' in Cell 9.\n"
            "Skipping MOT evaluation."
        )
        return

    logger.info(f"Evaluating {sequence_name}...")
    
    # Setup metrics out_dir or specific paths
    out_dir = args.out_dir
    if not out_dir and not args.out_json and not args.out_md:
        out_dir = PROJECT_ROOT / "exports" / "metrics"

    metrics = evaluate_one_sequence(
        gt_path=args.gt,
        pred_path=args.pred,
        sequence_name=sequence_name,
        out_json_path=args.out_json,
        out_md_path=args.out_md,
        out_dir=out_dir,
        iou_threshold=args.iou_threshold,
        gt_class_ids=gt_class_ids,
        pred_class_ids=pred_class_ids,
    )
    
    print("\nEvaluation metrics:")
    print(json.dumps(asdict(metrics), indent=2))

if __name__ == "__main__":
    main()