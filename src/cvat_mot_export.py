import csv
import shutil
import zipfile
from pathlib import Path
from src.config import CLASS_NAMES

def write_cvat_mot_zip(mot_rows: list[list], out_zip_path: Path, class_names: list[str] | None = None) -> Path:
    """
    Compresses target tracking lines into a CVAT-importable MOT 1.1 ZIP structure:
    archive.zip
    └── gt/
        ├── gt.txt
        └── labels.txt

    Args:
        mot_rows: List of MOT 1.1 tracking records.
        out_zip_path: Output archive filepath.
        class_names: Optional list of class names to write to labels.txt.

    Returns:
        out_zip_path: Path to the generated ZIP.
    """
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_zip_path.parent / f"{out_zip_path.stem}_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    gt_dir = temp_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / "gt.txt"
    labels_path = gt_dir / "labels.txt"

    # Write GT records
    with open(gt_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in mot_rows:
            writer.writerow(row)

    # Write label files
    if class_names is None:
        class_names = CLASS_NAMES
    labels_path.write_text("\n".join(class_names) + "\n")

    # Package into zip file
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(gt_path, arcname="gt/gt.txt")
        zf.write(labels_path, arcname="gt/labels.txt")

    # Cleanup temp folder
    shutil.rmtree(temp_dir)
    return out_zip_path
