# AF-MOT Football Tracking & Export Pipeline

This repository contains the testing, tracking, and export pipeline for football player tracking. It uses pre-trained object detectors and the **BoT-SORT** multi-object tracker to output annotation-ready CVAT MOT 1.1 format datasets.

> [!NOTE]
> **This codebase is strictly for inference, testing, and pre-annotation exports. Model training has been removed/disabled.**

---

## 1. Pipeline Overview

- **Detector:** Pre-trained **Baidu RT-DETRx** (or YOLO) loaded via dynamic weight configuration in the model registry.
- **Tracker:** BoT-SORT tracker via the BoxMOT library.
- **Target Dataset:** AF-MOT (football player videos & annotations).
- **Target Export:** CVAT MOT 1.1 ZIP files (containing `gt/gt.txt` and `gt/labels.txt` pre-annotations).
- **Evaluation:** Evaluates predicted tracking outputs against MOT ground-truth datasets.

---

## 2. Directory Layout

```text
Football_MOT_Exporting/
├── README.md
├── configs/
│   ├── project_config.py          # Paths, class filterings, and options
│   ├── model_registry.yaml        # dynamic model configurations
│   └── football_botsort.yaml      # BoT-SORT tracker hyper-parameters
├── src/                           # Core tracking package modules
│   ├── __init__.py
│   ├── config.py                  # configuration loader interface
│   ├── model_loader.py            # loads model based on registry key
│   ├── detector.py                # frame detection inference & box cleaning
│   ├── tracker.py                 # creates and parses BoT-SORT tracks
│   ├── cvat_mot_export.py         # zips tracks into CVAT-compatible packages
│   ├── mot_eval.py                # HOTA, MOTA, MOTP, IDF1 evaluation engine
│   ├── video_io.py                # frames and visualization writing
│   └── logging_utils.py           # stream logger
├── scripts/
│   ├── install_dependencies.py    # installs python libraries
│   ├── verify_dataset.py          # validates dataset paths and splits
│   ├── detect_track_export.py     # single video tracker script
│   ├── evaluate_mot.py            # evaluates predicted tracks against ground truth
│   ├── batch_export.py            # batch tracking clip processor
│   ├── summarize_exports.py       # prints batch execution reports
│   ├── smoke_test_config.py       # smoketests project path logic
│   └── smoke_test_model_load.py   # smoketests registry weight loader
├── notebooks/
│   └── Football_MOT_Export.ipynb  # Main pipeline Colab notebook
└── legacy/                        # Archived legacy training scripts
    ├── prepare_yolo_dataset.py
    └── train_rtdetr.py
```

---

## 3. Dataset Setup

By default, the configuration points to the Google Drive dataset root:
`/content/drive/MyDrive/UCF/Courses/Spring2026/IndependentStudy/Datasets/OtherDatasets/AF-MOT`

Expected folder layout:
```text
AF-MOT/
├── videos/
│   └── [sequence].mp4
└── annotations/
    └── [keio_sequence_folder]/
        ├── gt/
        │   └── gt.txt
        └── labels.txt
```

---

## 4. Model Registry

To swap model weights, add or select a configured key inside `configs/model_registry.yaml`:

```yaml
models:
  rtdetrx_player_best:
    family: rtdetr
    variant: rtdetr-x
    weights_path: "/path/to/best.pt"
    class_names: ["player"]
    allowed_class_ids: [0]
    imgsz: 640
```

---

## 5. Usage Commands

All commands can be executed locally or on Google Colab.

### Install Dependencies
```bash
python scripts/install_dependencies.py
```

### Validate Config and Models
```bash
python scripts/smoke_test_config.py
python scripts/smoke_test_model_load.py
```

### 1. Single Video Tracking & CVAT Export (Cell 8)
Tracks a single video and exports a CVAT MOT 1.1 ZIP:
```bash
python scripts/detect_track_export.py \
    --video "/path/to/video.mp4" \
    --model-config configs/model_registry.yaml \
    --model-key rtdetrx_player_best \
    --tracker-config configs/football_botsort.yaml \
    --out "exports/cvat_mot_tracks/single/video_cvat_mot.zip" \
    --conf 0.64 \
    --save-preview
```

### 2. Track Evaluation (Cell 9)
Evaluates prediction ZIPs against ground-truth MOT `gt.txt`:
```bash
python scripts/evaluate_mot.py \
    --gt "/path/to/annotations/keio_sequence/gt/gt.txt" \
    --pred "exports/cvat_mot_tracks/single/video_cvat_mot.zip" \
    --sequence-name "SWide008" \
    --out-json "exports/metrics/json/SWide008_metrics.json" \
    --out-md "exports/metrics/reports/SWide008_metrics.md"
```

### 3. Batch Clip Export & Summarization (Cell 10 & 11)
Processes multiple video clips and aggregates results:
```bash
python scripts/batch_export.py \
    --model-config configs/model_registry.yaml \
    --model-key rtdetrx_player_best \
    --tracker-config configs/football_botsort.yaml \
    --clips-dir "/path/to/ToBeTracked" \
    --out-dir "exports/cvat_mot_tracks/batch" \
    --conf 0.64 \
    --save-preview \
    --write-summary
```

Print batch summaries in your console:
```bash
python scripts/summarize_exports.py \
    --summary "exports/cvat_mot_tracks/batch/batch_tracking_summary.json"
```

---

## 6. Organized Exports Structure

Outputs will be saved in organized subfolders inside `exports/`:
```text
exports/
├── cvat_mot_tracks/            # CVAT MOT 1.1 ZIP files
│   ├── single/
│   └── batch/
├── previews/                   # MP4 tracking preview videos
├── metrics/
│   ├── json/                   # raw evaluation json files
│   └── reports/                # markdown evaluation tables
└── logs/                       # batch error log trace files
```

---

## 7. Troubleshooting

### Missing Weights File
If you receive a `FileNotFoundError` during model loading, check the `weights_path` defined in `configs/model_registry.yaml`. Make sure you have mounted Google Drive if running in Google Colab.

### Ground Truth Not Found
Evaluation will skip gracefully if ground truth `gt.txt` is missing for a target sequence. Make sure dataset folder structures match the expected layout.

### HOTA Metric Unavailable
If the HOTA score says `HOTA unavailable: dependency not installed.`, check that you have installed correct package dependencies (specifically `scipy` and `numpy`).
