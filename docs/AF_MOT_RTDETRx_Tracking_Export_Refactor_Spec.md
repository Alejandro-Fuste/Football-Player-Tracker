# AI Specification Document — AF-MOT RT-DETRx Detection + Tracking Export Refactor

## Working Title

**AF-MOT RT-DETRx Tracking Export Refactor**

This specification is intended for use in **Antigravity** to refactor the existing `Football_MOT_Exporting` codebase into a cleaner object detection, tracking, evaluation, and batch-export codebase that uses a previously trained **Baidu RT-DETRx** detector stored as a `best.pt` weights file.

---

## 1. Goal

Refactor the current `Football_MOT_Exporting` project so it becomes a **testing and export pipeline** for football player detection and multi-object tracking, not a model-training pipeline.

The refactored codebase must:

1. Use a trained **Baidu RT-DETRx** object detector through Ultralytics-compatible `.pt` weights.
2. Load the detector from a configurable `best.pt` path.
3. Run detection + BoT-SORT tracking on AF-MOT / AF-HandHeld football videos.
4. Export tracker output to **CVAT MOT 1.1** ZIP files.
5. Evaluate predicted tracks against MOT ground truth when GT exists.
6. Batch process many clips for automated annotation support.
7. Make it easy to swap:
   - model family
   - model weights
   - confidence threshold
   - tracker config
   - input clips directory
   - output export directory

The main operational focus should be the notebook’s logical **Cell 8, Cell 9, and Cell 10**:

- **Cell 8:** Export one sample video to CVAT MOT 1.1.
- **Cell 9:** Evaluate exported predictions against MOT ground truth.
- **Cell 10:** Batch export many videos/clips.

The refactor must remove or disable training-specific workflow steps because the object detector has already been trained.

---

## 2. Inputs

### 2.1 Existing Codebase

Refactor the existing codebase:

```text
Football_MOT_Exporting/
├── README.md
├── configs/
│   ├── project_config.py
│   └── football_botsort.yaml
├── scripts/
│   ├── install_dependencies.py
│   ├── verify_dataset.py
│   ├── prepare_yolo_dataset.py
│   ├── train_rtdetr.py
│   ├── track_to_cvat_mot.py
│   ├── evaluate_mot.py
│   └── batch_export.py
└── notebooks/
    └── AF_HandHeld_RTDETR_BoTSORT_Modular.ipynb
```

### 2.2 Existing Important Notebook Cells

The current notebook already contains the three important testing/export cells. Refactor these into clean, reusable, config-driven cells.

#### Logical Cell 8 — Single Video Export

Current behavior:

```python
from configs.project_config import VIDEOS_DIR, EXPORT_DIR, RUNS_DIR

weights = RUNS_DIR / 'rtdetr_af_handheld_player_ball-3' / 'weights' / 'best.pt'
sample_video = Path('/content/JetSweep_1.mp4')
out_zip = EXPORT_DIR / f"{sample_video.stem}_t2_conf_64_cvat_mot.zip"

!python scripts/track_to_cvat_mot.py --video "$sample_video" --weights "$weights" --out "$out_zip" --conf 0.64 --save-preview --max-frames 1
```

Refactor goal:

```python
!python scripts/detect_track_export.py \
    --video "$SAMPLE_VIDEO" \
    --model-config configs/model_registry.yaml \
    --model-key rtdetrx_player_best \
    --tracker-config configs/football_botsort.yaml \
    --out "$OUT_ZIP" \
    --conf "$CONF_THRESH" \
    --save-preview
```

#### Logical Cell 9 — MOT Evaluation

Current behavior:

```python
from configs.project_config import ANNOT_ROOT, EXPORT_DIR, get_annot_dir_name

sequence_name = "SWide008"
gt_path = ANNOT_ROOT / get_annot_dir_name(sequence_name) / "gt" / "gt.txt"
pred_zip = EXPORT_DIR / f"{sequence_name}_t2_conf_70_cvat_mot.zip"

!python scripts/evaluate_mot.py \
    --gt "$gt_path" \
    --pred "$pred_zip" \
    --sequence-name "$sequence_name"
```

Refactor goal:

```python
!python scripts/evaluate_mot.py \
    --gt "$GT_PATH" \
    --pred "$PRED_ZIP" \
    --sequence-name "$SEQUENCE_NAME" \
    --out-json "$METRICS_JSON" \
    --out-md "$METRICS_REPORT"
```

#### Logical Cell 10 — Batch Export

Current behavior:

```python
from configs.project_config import TO_BE_TRACKED, EXPORT_DIR, RUNS_DIR

weights = RUNS_DIR / 'rtdetr_af_handheld_player_ball' / 'weights' / 'best.pt'
out_dir = EXPORT_DIR / "Batch" / "Conf_64_5-13_version1"

!python scripts/batch_export.py \
    --weights "$weights" \
    --clips-dir "$TO_BE_TRACKED" \
    --out-dir "$out_dir" \
    --conf 0.64 \
    --save-preview \
    --max-frames 1
```

Refactor goal:

```python
!python scripts/batch_export.py \
    --model-config configs/model_registry.yaml \
    --model-key rtdetrx_player_best \
    --tracker-config configs/football_botsort.yaml \
    --clips-dir "$TO_BE_TRACKED" \
    --out-dir "$BATCH_EXPORT_DIR" \
    --conf "$CONF_THRESH" \
    --save-preview \
    --write-summary
```

---

## 3. Rules / Constraints

### 3.1 No Training

Do **not** train RT-DETR, YOLO, BoT-SORT, ReID, or any other model in this refactor.

The following scripts/cells must be removed from the main path, archived, or explicitly marked as legacy:

```text
scripts/prepare_yolo_dataset.py
scripts/train_rtdetr.py
training-specific notebook cells
dataset-conversion-only cells used only for detector training
```

Allowed exception: keep old training scripts in a `legacy/` or `archive_training/` folder only if preserving history is useful.

### 3.2 Primary Task

The refactored project is for:

- object detection inference
- multi-object tracking
- single-video export
- batch export
- MOT evaluation
- CVAT MOT 1.1 ZIP generation
- preview video/image generation
- export summaries and reports

It is **not** for detector training.

### 3.3 Default Detector

Default detector:

```yaml
model_key: rtdetrx_player_best
model_family: rtdetr
model_variant: rtdetr-x
weights_path: /content/drive/MyDrive/.../best.pt
class_names:
  - player
allowed_class_ids:
  - 0
```

The detector should be loaded through the Ultralytics API if the `.pt` file is compatible:

```python
from ultralytics import RTDETR
model = RTDETR(weights_path)
```

### 3.4 Model Swapping Requirement

The refactor must support swapping models without editing script internals.

Use a model registry file:

```text
configs/model_registry.yaml
```

Example:

```yaml
default_model: rtdetrx_player_best

models:
  rtdetrx_player_best:
    family: rtdetr
    variant: rtdetr-x
    weights_path: "/content/drive/MyDrive/path/to/best.pt"
    class_names: ["player"]
    allowed_class_ids: [0]
    imgsz: 640

  yolo26x_player_best:
    family: yolo
    variant: yolo26x
    weights_path: "/content/drive/MyDrive/path/to/yolo26x/best.pt"
    class_names: ["player"]
    allowed_class_ids: [0]
    imgsz: 640
```

The command line should accept:

```bash
--model-config configs/model_registry.yaml
--model-key rtdetrx_player_best
```

If `--weights` is provided directly, it may override the registry entry.

### 3.5 Dataset Requirement

Use the AF-MOT dataset as the target dataset.

Expected dataset layout:

```text
AF-MOT/
├── videos/
│   └── [video_name].mp4
└── annotations/
    └── [annotation_sequence_name]/
        ├── gt/
        │   └── gt.txt
        ├── img1/
        └── labels.txt
```

The code must support the sequence-name mapping already used in `project_config.py`:

```python
HTight005 -> keio_hosei_tight_005
HWide005  -> keio_hosei_wide_005
MTight013 -> keio_meiji_tight_013
RWide014  -> keio_rikkyo_wide_014
SWide008  -> keio_sensyu_wide_008
WWide012  -> keio_waseda_wide_012
```

### 3.6 Class Filtering

Default output should export only football player detections.

Rules:

- Keep `CLASS_NAMES = ["player"]`.
- Keep `PLAYER_CLASS_ID = 0`.
- Keep `ALLOWED_CLASS_IDS = {0}` by default.
- Do not export referees, coaches, sideline personnel, or ball unless explicitly enabled later.
- If the trained detector has multiple classes, filter predictions to `allowed_class_ids`.

### 3.7 Tracker

Use BoT-SORT as the default tracker.

Default tracker config:

```yaml
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: 90
match_thresh: 0.80
fuse_score: true
gmc_method: sparseOptFlow
proximity_thresh: 0.50
appearance_thresh: 0.25
with_reid: true
model: osnet_x0_25_msmt17.pt
```

The tracker config must be swappable with:

```bash
--tracker-config configs/football_botsort.yaml
```

### 3.8 Output Format

Primary export format:

```text
CVAT MOT 1.1 ZIP
```

Each ZIP should be importable into CVAT as preannotations.

Also save optional side outputs:

```text
exports/
├── cvat_mot_1_1/
│   ├── single/
│   └── batch/
├── previews/
├── metrics/
│   ├── json/
│   └── reports/
└── logs/
```

### 3.9 Google Colab Compatibility

The refactor must remain Google Colab friendly.

Requirements:

- Use absolute Google Drive paths in config.
- Keep a notebook entry point.
- Avoid requiring local-only dependencies.
- Provide clear cells for install, config check, sample export, evaluation, and batch export.
- Do not require GPU for script import or dataset verification, but inference should use GPU if available.

### 3.10 Mac Compatibility Where Reasonable

The code should not be Colab-only unless unavoidable.

Use `pathlib.Path`, clean CLI arguments, and Python scripts that can run locally on a MacBook if dependencies are installed.

---

## 4. Steps / Logic



### Step 1 — Refactor Folder Structure

Target structure:

```text
Football_MOT_Exporting/
├── README.md
├── configs/
│   ├── project_config.py
│   ├── model_registry.yaml
│   └── football_botsort.yaml
├── notebooks/
│   └── Football_MOT_Export.ipynb
├── scripts/
│   ├── install_dependencies.py
│   ├── verify_dataset.py
│   ├── detect_track_export.py
│   ├── evaluate_mot.py
│   ├── batch_export.py
│   └── summarize_exports.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model_loader.py
│   ├── detector.py
│   ├── tracker.py
│   ├── cvat_mot_export.py
│   ├── mot_eval.py
│   ├── video_io.py
│   └── logging_utils.py
├── exports/
│   ├── cvat_mot_tracks/
│   ├── previews/
│   ├── metrics/
│   └── logs/
└── legacy/
    ├── prepare_yolo_dataset.py
    └── train_rtdetr.py
```

### Step 2 — Create Model Loader

Create:

```text
src/model_loader.py
```

Responsibilities:

- Read `configs/model_registry.yaml`.
- Select model by `model_key`.
- Support RT-DETR and YOLO families.
- Return a loaded model and metadata.

Required API:

```python
def load_model_from_registry(model_config_path: Path, model_key: str):
    """
    Returns:
        model: loaded Ultralytics model
        model_info: dict containing family, variant, weights_path, class_names,
                    allowed_class_ids, imgsz
    """
```

Expected behavior:

```python
if family == "rtdetr":
    from ultralytics import RTDETR
    model = RTDETR(weights_path)
elif family == "yolo":
    from ultralytics import YOLO
    model = YOLO(weights_path)
else:
    raise ValueError(f"Unsupported model family: {family}")
```

### Step 3 — Refactor Single-Video Export Script

Rename or replace:

```text
scripts/track_to_cvat_mot.py
```

with:

```text
scripts/detect_track_export.py
```

Required CLI:

```bash
python scripts/detect_track_export.py \
  --video /path/to/video.mp4 \
  --model-config configs/model_registry.yaml \
  --model-key rtdetrx_player_best \
  --tracker-config configs/football_botsort.yaml \
  --out /path/to/export.zip \
  --conf 0.64 \
  --imgsz 640 \
  --save-preview \
  --max-frames 300
```

Required features:

- Load detector from registry.
- Run frame-by-frame or video-level prediction/tracking.
- Use BoT-SORT tracker.
- Filter detections to allowed classes.
- Write CVAT MOT 1.1 ZIP.
- Save optional preview file.
- Save a JSON sidecar summary.

Summary JSON example:

```json
{
  "video": "SWide008.mp4",
  "model_key": "rtdetrx_player_best",
  "weights_path": ".../best.pt",
  "tracker_config": "configs/football_botsort.yaml",
  "conf": 0.64,
  "imgsz": 640,
  "frames_processed": 300,
  "tracks_created": 34,
  "detections_total": 4120,
  "output_zip": ".../SWide008_cvat_mot.zip",
  "preview_path": ".../SWide008_preview.mp4"
}
```

### Step 4 — Refactor Evaluation Script

Keep:

```text
scripts/evaluate_mot.py
```

but make it more robust.

Required CLI:

```bash
python scripts/evaluate_mot.py \
  --gt /path/to/gt.txt \
  --pred /path/to/prediction.zip \
  --sequence-name SWide008 \
  --out-json exports/metrics/json/SWide008_metrics.json \
  --out-md exports/metrics/reports/SWide008_metrics.md
```

Required metrics if supported by available dependencies:

- HOTA
- MOTA
- MOTP
- IDF1
- precision
- recall
- false positives
- false negatives
- ID switches
- mostly tracked
- mostly lost


Important: if HOTA is unavailable, do not crash. Report:

```text
HOTA unavailable: dependency not installed.
```

### Step 6 — Refactor Batch Export Script

Keep:

```text
scripts/batch_export.py
```

but make it registry-driven.

Required CLI:

```bash
python scripts/batch_export.py \
  --model-config configs/model_registry.yaml \
  --model-key rtdetrx_player_best \
  --tracker-config configs/football_botsort.yaml \
  --clips-dir /path/to/clips \
  --out-dir /path/to/export/batch \
  --conf 0.64 \
  --imgsz 640 \
  --save-preview \
  --max-frames -1 \
  --write-summary
```

Required behavior:

- Iterate over `.mp4`, `.mov`, `.avi`, and `.mkv`.
- Skip files that already have output ZIPs unless `--overwrite` is passed.
- Continue processing if one video fails.
- Write one summary CSV and one summary JSON for the whole batch.
- Save per-video error logs.

Batch summary CSV columns:

```text
video_name, status, frames_processed, detections_total, tracks_created,
output_zip, preview_path, error_message, runtime_seconds
```

### Step 7 — Refactor Notebook

Create or update notebook:

```text
notebooks/Football_MOT_Export.ipynb
```

Notebook should contain only testing/export workflow cells:

1. Mount Drive.
2. Set project path.
3. Install dependencies.
4. Verify dataset and config.
5. Select model and weights.
6. Select sample video.
7. Optional quick detector smoke test.
8. Export one sample CVAT MOT 1.1 ZIP.
9. Evaluate MOT output against GT.
10. Batch export clips.
11. Summarize batch results.

Remove main workflow cells for:

- preparing YOLO training data
- training RT-DETR
- resuming training
- validation during training

### Step 5 — Update `project_config.py`

Replace training-focused variables with inference/export variables.

Target contents:

```python
from pathlib import Path

DATASET_ROOT = Path("/content/drive/MyDrive/UCF/Courses/Spring2026/IndependentStudy/Datasets/OtherDatasets/AF-MOT")
VIDEOS_DIR = DATASET_ROOT / "videos"
ANNOT_ROOT = DATASET_ROOT / "annotations"

PROJECT_ROOT = Path("/content/drive/MyDrive/UCF/Courses/Spring2026/IndependentStudy/Utilities/Football_MOT_Exporting/")

CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_REGISTRY = CONFIG_DIR / "model_registry.yaml"
BOTSORT_CONFIG = CONFIG_DIR / "football_botsort.yaml"

EXPORT_DIR = PROJECT_ROOT / "exports"
CVAT_EXPORT_DIR = EXPORT_DIR / "cvat_mot_tracks"
PREVIEW_DIR = EXPORT_DIR / "previews"
METRICS_DIR = EXPORT_DIR / "metrics"
LOG_DIR = EXPORT_DIR / "logs"

TO_BE_TRACKED = PROJECT_ROOT / "datasets" / "ToBeTracked"

DEFAULT_MODEL_KEY = "rtdetrx_player_best"
DEFAULT_CONF_THRESH = 0.64
DEFAULT_IMG_SIZE = 640

CLASS_NAMES = ["player"]
PLAYER_CLASS_ID = 0
ALLOWED_CLASS_IDS = {0}
```

Keep the existing `get_annot_dir_name(seq_name)` helper.

### Step 6 — Update README

The README must clearly state:

- This codebase does not train models.
- It uses trained detector weights.
- Default detector is Baidu RT-DETRx with `best.pt`.
- Default tracker is BoT-SORT.
- Default dataset is AF-MOT.
- Main outputs are CVAT MOT 1.1 ZIPs, previews, metrics, and batch summaries.
- Main notebook cells are 8, 9, and 10.

### Step 7 — Add Smoke Tests

Add simple tests or smoke scripts:

```text
scripts/smoke_test_config.py
scripts/smoke_test_model_load.py
scripts/smoke_test_single_export.py
```

Smoke test goals:

- Config files load.
- Weight path exists.
- One video path exists.
- Model can be loaded.
- One frame or first 5 frames can be processed.
- Export folder is writable.

---

## 5. Output

Antigravity should produce the following final deliverables.



### 5.1 Updated Notebook

```text
notebooks/Football_MOT_Exporting.ipynb
```

The notebook should make cells 8, 9, and 10 the primary workflow.

### 5.2 Updated Scripts

Required scripts:

```text
scripts/detect_track_export.py
scripts/evaluate_mot.py
scripts/batch_export.py
scripts/summarize_exports.py
```

Optional but recommended:

```text
scripts/smoke_test_config.py
scripts/smoke_test_model_load.py
scripts/smoke_test_single_export.py
```

### 5.3 Updated Configs

Required configs:

```text
configs/project_config.py
configs/model_registry.yaml
configs/football_botsort.yaml
```

### 5.4 Updated README

The README should include:

- purpose
- dataset path
- model registry usage
- single-video export command
- evaluation command
- batch export command
- expected output folder structure
- troubleshooting section

### 5.5 Export Outputs

Generated exports should follow:

```text
exports/
├── cvat_mot_1_1/
│   ├── single/
│   │   └── [video_stem]_cvat_mot.zip
│   └── batch/
│       └── [run_name]/
├── previews/
├── metrics/
│   ├── json/
│   └── reports/
└── logs/
```

---

## 6. Edge Cases

### 6.1 Missing Weights

If the selected `best.pt` does not exist:

- Stop before inference.
- Print the missing path.
- Tell the user to update `configs/model_registry.yaml`.

### 6.2 Unsupported Model Family

If the model family is unsupported:

```text
Unsupported model family: [family]. Supported families: rtdetr, yolo.
```

### 6.3 Empty Detections

If the detector returns no detections for a frame:

- Do not crash.
- Write no detections for that frame.
- Continue tracking.
- Include empty-frame count in summary.

### 6.4 No Ground Truth Available

If GT does not exist for evaluation:

- Do not crash.
- Print and log:

```text
Ground truth not found for sequence [sequence_name].
Skipping MOT evaluation.
```

### 6.5 CVAT Import Compatibility

If CVAT rejects an output ZIP:

- Validate ZIP contains expected MOT files.
- Add a debug export mode that writes the raw MOT `.txt`.
- Add a README troubleshooting note for CVAT import format.

### 6.6 Batch Failure

If one video fails during batch export:

- Continue with the next video.
- Record the failure in summary CSV.
- Save error trace to `exports/logs/`.

### 6.7 Max Frames

`--max-frames` behavior:

- `--max-frames 1`: process first frame only for smoke testing.
- `--max-frames 300`: process first 300 frames.
- `--max-frames -1`: process the full video.

### 6.8 Existing Output Files

Default behavior:

- Skip if output ZIP already exists.

Override:

```bash
--overwrite
```

---

## 7. Acceptance Criteria

The refactor is complete when all of the following are true:

- [ ] The notebook no longer requires training cells to run detection/tracking/export.
- [ ] A trained RT-DETRx `best.pt` can be selected through `configs/model_registry.yaml`.
- [ ] A different YOLO or RT-DETR `.pt` file can be selected without editing script internals.
- [ ] Cell 8 exports one sample CVAT MOT 1.1 ZIP.
- [ ] Cell 9 evaluates a prediction ZIP against an AF-MOT `gt.txt` file when available.
- [ ] Cell 10 batch exports all clips in `TO_BE_TRACKED`.
- [ ] Outputs are written to organized export folders.
- [ ] Batch export writes a CSV/JSON summary.
- [ ] Missing weights, missing GT, empty detections, and failed videos are handled gracefully.
- [ ] README explains the new workflow clearly.


---

