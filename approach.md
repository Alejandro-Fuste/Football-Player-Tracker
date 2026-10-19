Viewed tracker.py:1-38
Edited prompt.md

I will read the rest of the [detect_track_export.py](file:///Users/alejandro/Desktop/Projects/FilmBreakdownAI/Utilities/Football_MOT_Exporting/scripts/detect_track_export.py) script to see how the preview video is written, what styling is used for bounding boxes/track IDs, and how batch previews are generated.

Viewed detect_track_export.py:80-239

I will inspect `src/video_io.py` to see how bounding boxes and track IDs are currently drawn on preview video frames.

Viewed video_io.py:1-46

I have researched the technical requirements for each of the features you requested. Below are the possible approaches, details on why they work, and their pros/cons.

---

### Feature 1: Dark, High-Contrast Bounding Boxes and Text
Right now, the drawing function in [video_io.py](file:///Users/alejandro/Desktop/Projects/FilmBreakdownAI/Utilities/Football_MOT_Exporting/src/video_io.py#L30) draws boxes in bright highlighter green `(0, 255, 0)` for players and orange `(0, 165, 255)` for the ball. These colors blend in or clash with the green grass.

#### Proposed Approaches:

* **Approach A: Dark Solid Colors with Dual-Pass Drawing (Recommended)**
  * **How it works:** We draw the bounding boxes and text in two passes. 
    1. **Pass 1:** Draw a slightly thicker white box/text (thickness 4) to act as a border/glow.
    2. **Pass 2:** Draw the actual dark box/text (thickness 2) over it in a dark color like **Charcoal Black** `(25, 25, 25)` or **Dark Navy** `(100, 0, 0)` in BGR.
  * **Pros:** This creates a drop-shadow/border effect. The dark color contrasts perfectly with the green grass, while the white border ensures the box remains visible if a player runs over a black jersey or dark shadow.
  * **Cons:** Slightly more lines of code in the rendering function (drawing twice per target), but computationally negligible.

* **Approach B: Semi-Transparent Dark Fills**
  * **How it works:** Draw the player boxes as semi-transparent dark overlays (e.g., 20% opacity black fill) with white or bright cyan text.
  * **Pros:** Looks premium and highlights the player area clearly.
  * **Cons:** Overlays make it harder to see details on the player's body (like numbers or jersey wrinkles) which are useful for review.

---

### Feature 2: Native QuickTime Player "Frame Count" Toggle
By default, QuickTime Player (on macOS) only displays elapsed time in HH:MM:SS format. It disables the native menu option to switch to **Frame Numbers** unless the video container contains a **Timecode Track (`tmcd`)**. OpenCV's `VideoWriter` writes a raw H.264 stream that lacks this metadata.

#### Proposed Approaches:

* **Approach A: Lossless FFmpeg Post-Processing (Highly Recommended)**
  * **How it works:** Once OpenCV finishes writing the preview video, we run a quick, lossless `ffmpeg` command in the background to copy the video streams and embed a standard timecode track at the correct frame rate (e.g., starting at `00:00:00:00`):
    ```bash
    ffmpeg -y -i input_temp.mp4 -c copy -timecode 00:00:00:00 final_output.mp4
    ```
  * **Pros:** 
    * Native QuickTime compatibility—allows you to click the timeline in QuickTime and toggle it to display frame numbers natively.
    * Lossless and near-instantaneous (less than 0.5 seconds per video) because `-c copy` bypasses re-encoding.
    * Works on macOS natively if `ffmpeg` is installed (and in Google Colab).
  * **Cons:** Requires `ffmpeg` to be installed on the machine running the code (pre-installed in Colab, easy to install via Homebrew on Mac).

* **Approach B: Burn-in Frame Counters (Visual Overlay)**
  * **How it works:** In [video_io.py](file:///Users/alejandro/Desktop/Projects/FilmBreakdownAI/Utilities/Football_MOT_Exporting/src/video_io.py), we draw the current frame number directly in a corner of each frame (e.g., top-left corner in a small charcoal box with white text: `Frame: 145`).
  * **Pros:** 
    * Works in *every* video player (VLC, QuickTime, browser, Google Drive preview, Colab).
    * Zero dependency on file metadata.
  * **Cons:** Visually occupies a small portion of the screen (though it can be placed in a non-disruptive corner).
  * *Note: We can combine **Approach A** (Timecode metadata) and **Approach B** (Burn-in text) to give you the ultimate review layout.*

---

### Feature 3: Batch Preview Video Generation
Your codebase already contains the foundations for batch exporting. However, the current Jupyter Notebook cell and `batch_export.py` don't save visual previews in a dedicated batch manner by default.

#### Proposed Approaches:

* **Approach A: CLI Flag Forwarding via `batch_export.py`**
  * **How it works:** Ensure `batch_export.py` forwards the `--save-preview` flag down to `detect_track_export.py` for every clip.
  * **File Organization:** Previews will be written to a dedicated folder under your batch export path, for example: `exports/cvat_mot_tracks/batch/previews/[video_name]_preview.mp4`.
  * **Pros:** Clean, automated, and follows the existing structure of your pipeline.
  * **Cons:** Generating previews increases the processing time slightly, but this is unavoidable if you want visual outputs.

---

### Recommended Plan of Action

If you want to move forward, we can implement these changes by:
1. Modifying [video_io.py](file:///Users/alejandro/Desktop/Projects/FilmBreakdownAI/Utilities/Football_MOT_Exporting/src/video_io.py) to implement **Dual-Pass Dark Drawing** (Approach A for Feature 1) and optionally a **Burn-in Frame Counter** in the top-left corner (Approach B for Feature 2).
2. Modifying `run_tracking_pipeline` in [detect_track_export.py](file:///Users/alejandro/Desktop/Projects/FilmBreakdownAI/Utilities/Football_MOT_Exporting/scripts/detect_track_export.py) to post-process the saved video using `ffmpeg -timecode` (Approach A for Feature 2).
3. Ensuring that [batch_export.py](file:///Users/alejandro/Desktop/Projects/FilmBreakdownAI/Utilities/Football_MOT_Exporting/scripts/batch_export.py) and the notebook support these new preview styling and metadata features when processing in batches.

Please let me know which approaches you prefer or if you have any questions before I draft the implementation plan!