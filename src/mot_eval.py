from __future__ import annotations

import csv
import json
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.config import SPLITS, ANNOT_ROOT, EXPORT_DIR, get_annot_dir_name

@dataclass
class MotMetrics:
    """Container for per-sequence MOT metrics."""
    sequence: str
    iou_threshold: float
    hota: float | str
    mota: float
    motp: float
    idf1: float
    precision: float
    recall: float
    mostly_tracked: int
    mostly_lost: int
    tp: int
    fp: int
    fn: int
    id_switches: int
    gt_detections: int
    pred_detections: int
    avg_iou: float
    mota_error_rate: float
    det_a: float
    ass_a: float

MOT_COLUMNS = [
    "frame",
    "track_id",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "class_id",
    "visibility",
]

def read_mot_txt(path: Path) -> pd.DataFrame:
    """Read a MOT-format gt.txt or prediction txt file."""
    if not path.exists():
        raise FileNotFoundError(f"MOT file not found: {path}")

    if path.stat().st_size == 0:
        return pd.DataFrame(columns=MOT_COLUMNS)

    df = pd.read_csv(path, header=None)

    if df.empty:
        return pd.DataFrame(columns=MOT_COLUMNS)

    df = df.iloc[:, :9]
    df.columns = MOT_COLUMNS[: df.shape[1]]

    for col in MOT_COLUMNS:
        if col not in df.columns:
            df[col] = 1.0

    for col in MOT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["frame", "track_id", "x", "y", "w", "h"])
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["class_id"] = df["class_id"].astype(int)

    return df[(df["w"] > 0) & (df["h"] > 0)].copy()

def read_pred_from_cvat_mot_zip(zip_path: Path) -> pd.DataFrame:
    """Read predicted tracks from a CVAT MOT 1.1 ZIP file."""
    if not zip_path.exists():
        # Try to resolve path by stripping common video extensions from the filename
        filename = zip_path.name
        cleaned_filename = filename
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"]:
            if ext + "_cvat_mot" in filename:
                cleaned_filename = filename.replace(ext + "_cvat_mot", "_cvat_mot")
                break
            elif ext in filename and filename.endswith(ext):
                cleaned_filename = filename[:-len(ext)]
        
        alt_path = zip_path.parent / cleaned_filename
        if alt_path.exists():
            zip_path = alt_path
        else:
            raise FileNotFoundError(f"Prediction ZIP not found: {zip_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)

        gt_path = tmpdir_path / "gt" / "gt.txt"
        if not gt_path.exists():
            raise FileNotFoundError(f"Prediction ZIP does not contain gt/gt.txt: {zip_path}")

        return read_mot_txt(gt_path)

def filter_classes(df: pd.DataFrame, class_ids: set[int] | None = None) -> pd.DataFrame:
    """Optionally filter a MOT DataFrame by class IDs."""
    if class_ids is None:
        return df.copy()
    return df[df["class_id"].isin(class_ids)].copy()

def xywh_to_xyxy_array(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame x,y,w,h columns to Nx4 xyxy boxes."""
    if df.empty:
        return np.empty((0, 4), dtype=np.float32)

    x1 = df["x"].to_numpy(dtype=np.float32)
    y1 = df["y"].to_numpy(dtype=np.float32)
    x2 = x1 + df["w"].to_numpy(dtype=np.float32)
    y2 = y1 + df["h"].to_numpy(dtype=np.float32)

    return np.stack([x1, y1, x2, y2], axis=1)

def compute_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between GT boxes and predicted boxes."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)

    gt = gt_boxes.astype(np.float32)
    pr = pred_boxes.astype(np.float32)

    gt_area = np.maximum(0, gt[:, 2] - gt[:, 0]) * np.maximum(0, gt[:, 3] - gt[:, 1])
    pr_area = np.maximum(0, pr[:, 2] - pr[:, 0]) * np.maximum(0, pr[:, 3] - pr[:, 1])

    x1 = np.maximum(gt[:, None, 0], pr[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], pr[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pr[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], pr[None, :, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h

    union = gt_area[:, None] + pr_area[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    return iou.astype(np.float32)

def match_frame_by_iou(
    gt_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Match GT and predicted detections in one frame using Hungarian matching."""
    if gt_frame.empty or pred_frame.empty:
        return []

    gt_boxes = xywh_to_xyxy_array(gt_frame)
    pred_boxes = xywh_to_xyxy_array(pred_frame)

    iou_matrix = compute_iou_matrix(gt_boxes, pred_boxes)

    # Hungarian solves minimum cost; use negative IoU to maximize IoU.
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches = []
    gt_ids = gt_frame["track_id"].to_numpy(dtype=int)
    pred_ids = pred_frame["track_id"].to_numpy(dtype=int)

    for r, c in zip(row_ind, col_ind):
        iou = float(iou_matrix[r, c])
        if iou >= iou_threshold:
            matches.append((int(gt_ids[r]), int(pred_ids[c]), iou))

    return matches

def collect_matches(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_threshold: float,
) -> dict:
    """Match every frame and collect counts needed for MOT metrics."""
    frames = sorted(set(gt_df["frame"].tolist()) | set(pred_df["frame"].tolist()))

    gt_by_frame = {f: g for f, g in gt_df.groupby("frame")}
    pred_by_frame = {f: g for f, g in pred_df.groupby("frame")}

    matches_by_frame: dict[int, list[tuple[int, int, float]]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ious = []

    previous_match_for_gt: dict[int, int] = {}
    id_switches = 0
    pair_match_counts: dict[tuple[int, int], int] = defaultdict(int)

    for frame in frames:
        gt_frame = gt_by_frame.get(frame, pd.DataFrame(columns=gt_df.columns))
        pred_frame = pred_by_frame.get(frame, pd.DataFrame(columns=pred_df.columns))

        matches = match_frame_by_iou(gt_frame, pred_frame, iou_threshold)
        matches_by_frame[frame] = matches

        tp = len(matches)
        fp = len(pred_frame) - tp
        fn = len(gt_frame) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for gt_id, pred_id, iou in matches:
            ious.append(iou)
            pair_match_counts[(gt_id, pred_id)] += 1

            if gt_id in previous_match_for_gt:
                if previous_match_for_gt[gt_id] != pred_id:
                    id_switches += 1

            previous_match_for_gt[gt_id] = pred_id

    return {
        "frames": frames,
        "matches_by_frame": matches_by_frame,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "id_switches": id_switches,
        "ious": ious,
        "pair_match_counts": pair_match_counts,
    }

def compute_mota(gt_count: int, fp: int, fn: int, id_switches: int) -> tuple[float, float]:
    """Compute MOTA."""
    if gt_count == 0:
        return 0.0, 0.0

    error_rate = (fn + fp + id_switches) / gt_count
    mota = 1.0 - error_rate
    return float(mota), float(error_rate)

def compute_motp(ious: list[float]) -> float:
    """Compute MOTP (average IoU of matched bounding boxes)."""
    if not ious:
        return 0.0
    return float(np.mean(ious))

def compute_idf1(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    pair_match_counts: dict[tuple[int, int], int],
) -> float:
    """Compute IDF1 score."""
    gt_ids = sorted(gt_df["track_id"].unique().tolist())
    pred_ids = sorted(pred_df["track_id"].unique().tolist())

    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    gt_index = {track_id: i for i, track_id in enumerate(gt_ids)}
    pred_index = {track_id: j for j, track_id in enumerate(pred_ids)}

    score_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)

    for (gt_id, pred_id), count in pair_match_counts.items():
        if gt_id in gt_index and pred_id in pred_index:
            score_matrix[gt_index[gt_id], pred_index[pred_id]] = count

    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    idtp = 0
    for r, c in zip(row_ind, col_ind):
        idtp += int(score_matrix[r, c])

    total_gt = len(gt_df)
    total_pred = len(pred_df)

    idfn = total_gt - idtp
    idfp = total_pred - idtp

    denom = (2 * idtp) + idfp + idfn
    if denom == 0:
        return 0.0

    return float((2 * idtp) / denom)

def compute_hota(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    alpha_values: list[float] | None = None,
) -> tuple[float | str, float, float]:
    """Compute HOTA (Higher Order Tracking Accuracy). Handles missing scipy/numpy dependencies gracefully."""
    try:
        if alpha_values is None:
            alpha_values = [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]

        gt_total_by_id = gt_df.groupby("track_id").size().to_dict()
        pred_total_by_id = pred_df.groupby("track_id").size().to_dict()

        hota_values = []
        det_a_values = []
        ass_a_values = []

        for alpha in alpha_values:
            match_data = collect_matches(gt_df, pred_df, iou_threshold=alpha)

            tp = match_data["tp"]
            fp = match_data["fp"]
            fn = match_data["fn"]
            pair_match_counts = match_data["pair_match_counts"]

            det_denom = tp + fp + fn
            det_a = tp / det_denom if det_denom > 0 else 0.0

            ass_scores = []
            for (gt_id, pred_id), pair_tp in pair_match_counts.items():
                gt_total = gt_total_by_id.get(gt_id, 0)
                pred_total = pred_total_by_id.get(pred_id, 0)

                pair_fn = gt_total - pair_tp
                pair_fp = pred_total - pair_tp

                ass_denom = pair_tp + pair_fn + pair_fp
                ass_score = pair_tp / ass_denom if ass_denom > 0 else 0.0
                ass_scores.extend([ass_score] * pair_tp)

            ass_a = float(np.mean(ass_scores)) if ass_scores else 0.0
            hota_alpha = float(np.sqrt(det_a * ass_a)) if det_a > 0 and ass_a > 0 else 0.0

            hota_values.append(hota_alpha)
            det_a_values.append(det_a)
            ass_a_values.append(ass_a)

        hota = float(np.mean(hota_values)) if hota_values else 0.0
        mean_det_a = float(np.mean(det_a_values)) if det_a_values else 0.0
        mean_ass_a = float(np.mean(ass_a_values)) if ass_a_values else 0.0

        return hota, mean_det_a, mean_ass_a
    except Exception as e:
        print(f"Warning: HOTA calculation failed ({e}). Returning 'HOTA unavailable: dependency not installed.'")
        return "HOTA unavailable: dependency not installed.", 0.0, 0.0

def evaluate_mot(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    sequence_name: str,
    iou_threshold: float = 0.5,
) -> MotMetrics:
    """Compute MOT metrics for one sequence."""
    match_data = collect_matches(gt_df, pred_df, iou_threshold=iou_threshold)

    tp = match_data["tp"]
    fp = match_data["fp"]
    fn = match_data["fn"]
    id_switches = match_data["id_switches"]
    ious = match_data["ious"]
    pair_match_counts = match_data["pair_match_counts"]

    gt_count = len(gt_df)
    pred_count = len(pred_df)

    mota, mota_error_rate = compute_mota(
        gt_count=gt_count,
        fp=fp,
        fn=fn,
        id_switches=id_switches,
    )

    motp = compute_motp(ious)
    idf1 = compute_idf1(gt_df, pred_df, pair_match_counts)
    hota, det_a, ass_a = compute_hota(gt_df, pred_df)

    # Compute precision & recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Compute Mostly Tracked (MT) and Mostly Lost (ML)
    mostly_tracked = 0
    mostly_lost = 0
    if not gt_df.empty:
        gt_ids = gt_df["track_id"].unique()
        if len(gt_ids) > 0:
            gt_lifetimes = gt_df.groupby("track_id").size().to_dict()
            gt_match_counts = defaultdict(int)
            for frame, matches in match_data["matches_by_frame"].items():
                for gt_id, pred_id, iou in matches:
                    gt_match_counts[gt_id] += 1
            for gt_id in gt_ids:
                lifetime = gt_lifetimes.get(gt_id, 0)
                if lifetime == 0:
                    continue
                matched = gt_match_counts.get(gt_id, 0)
                ratio = matched / lifetime
                if ratio >= 0.8:
                    mostly_tracked += 1
                elif ratio < 0.2:
                    mostly_lost += 1

    return MotMetrics(
        sequence=sequence_name,
        iou_threshold=float(iou_threshold),
        hota=hota if isinstance(hota, str) else round(hota, 6),
        mota=round(mota, 6),
        motp=round(motp, 6),
        idf1=round(idf1, 6),
        precision=round(precision, 6),
        recall=round(recall, 6),
        mostly_tracked=mostly_tracked,
        mostly_lost=mostly_lost,
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        id_switches=int(id_switches),
        gt_detections=int(gt_count),
        pred_detections=int(pred_count),
        avg_iou=round(motp, 6),
        mota_error_rate=round(mota_error_rate, 6),
        det_a=round(det_a, 6),
        ass_a=round(ass_a, 6),
    )

def generate_markdown_report(metrics: MotMetrics) -> str:
    """Generate a clean Markdown table from evaluation metrics."""
    hota_display = metrics.hota if isinstance(metrics.hota, str) else f"{metrics.hota:.6f}"
    
    return f"""# MOT Evaluation Report: {metrics.sequence}

| Metric | Value | Description |
| :--- | :--- | :--- |
| **HOTA** | {hota_display} | Higher Order Tracking Accuracy |
| **MOTA** | {metrics.mota:.6f} | Multi-Object Tracking Accuracy |
| **MOTP** | {metrics.motp:.6f} | Multi-Object Tracking Precision (Avg IoU) |
| **IDF1** | {metrics.idf1:.6f} | ID F1 Score |
| **Precision** | {metrics.precision:.6f} | TP / (TP + FP) |
| **Recall** | {metrics.recall:.6f} | TP / (TP + FN) |
| **Mostly Tracked (MT)** | {metrics.mostly_tracked} | GT tracks tracked >= 80% |
| **Mostly Lost (ML)** | {metrics.mostly_lost} | GT tracks tracked < 20% |
| **TP (True Positives)** | {metrics.tp} | Count of correctly matched detections |
| **FP (False Positives)** | {metrics.fp} | Count of extra tracker detections |
| **FN (False Negatives)** | {metrics.fn} | Count of missed ground truth detections |
| **ID Switches** | {metrics.id_switches} | Number of times a track ID changed |
| **GT Detections** | {metrics.gt_detections} | Total ground truth bounding boxes |
| **Pred Detections** | {metrics.pred_detections} | Total predictions from the tracker |
| **Average IoU** | {metrics.avg_iou:.6f} | Mean overlap of correctly matched targets |
| **MOTA Error Rate** | {metrics.mota_error_rate:.6f} | Cumulative error rate vs ground truth |
"""

def save_metrics_report(
    metrics: MotMetrics,
    out_json_path: Path | None = None,
    out_md_path: Path | None = None,
    out_dir: Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Save metrics to JSON, Markdown, and CSV reports."""
    row = asdict(metrics)
    
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{metrics.sequence}_metrics.json"
        csv_path = out_dir / f"{metrics.sequence}_metrics.csv"
        
        json_path.write_text(json.dumps(row, indent=2))
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
            
        md_path = out_dir / f"{metrics.sequence}_metrics.md"
        md_path.write_text(generate_markdown_report(metrics))
        return json_path, md_path

    if out_json_path:
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(row, indent=2))

    if out_md_path:
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        out_md_path.write_text(generate_markdown_report(metrics))

    return out_json_path, out_md_path

def save_summary_report(
    metrics_list: list[MotMetrics],
    out_dir: Path,
    summary_name: str,
) -> tuple[Path, Path]:
    """Save batch summary as JSON and CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [asdict(m) for m in metrics_list]

    json_path = out_dir / f"{summary_name}_summary.json"
    csv_path = out_dir / f"{summary_name}_summary.csv"

    json_path.write_text(json.dumps(rows, indent=2))

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")

    return json_path, csv_path

def evaluate_one_sequence(
    gt_path: Path,
    pred_path: Path,
    sequence_name: str,
    out_json_path: Path | None = None,
    out_md_path: Path | None = None,
    out_dir: Path | None = None,
    iou_threshold: float = 0.5,
    gt_class_ids: set[int] | None = None,
    pred_class_ids: set[int] | None = None,
) -> MotMetrics:
    """Evaluate one sequence prediction file/ZIP against target ground truth."""
    gt_df = read_mot_txt(gt_path)

    if pred_path.suffix.lower() == ".zip":
        pred_df = read_pred_from_cvat_mot_zip(pred_path)
    else:
        pred_df = read_mot_txt(pred_path)

    gt_df = filter_classes(gt_df, gt_class_ids)
    pred_df = filter_classes(pred_df, pred_class_ids)

    metrics = evaluate_mot(
        gt_df=gt_df,
        pred_df=pred_df,
        sequence_name=sequence_name,
        iou_threshold=iou_threshold,
    )

    save_metrics_report(metrics, out_json_path, out_md_path, out_dir)
    return metrics
