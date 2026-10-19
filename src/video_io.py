import cv2
from pathlib import Path
from src.config import CLASS_NAMES

def get_video_properties(video_path: Path) -> dict:
    """Reads video dimensions, frame count, and FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video source at: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height
    }

def draw_track_overlay(
    frame, 
    parsed_tracks, 
    allowed_classes: set[int] | None = None,
    font_scale: float | None = None,
    box_thickness: int | None = None
):
    """Draws tracking box rectangles and track IDs onto the image frame.
    
    Dynamically scales bounding box line thickness and text size based on frame resolution
    if explicit values are not provided.
    """
    h, w = frame.shape[:2]
    
    # Calculate scale factor relative to standard 1080p height
    scale_factor = max(0.35, min(w, h) / 1080.0)
    
    # Determine box line thicknesses
    if box_thickness is None:
        inner_box_thick = max(1, int(round(2 * scale_factor)))
        outer_box_thick = inner_box_thick + max(1, int(round(2 * scale_factor)))
    else:
        inner_box_thick = max(1, box_thickness)
        outer_box_thick = inner_box_thick + max(1, int(round(2 * scale_factor)))
        
    # Determine font scale and text line thicknesses
    if font_scale is None:
        actual_font_scale = max(0.3, round(0.6 * scale_factor, 2))
    else:
        actual_font_scale = font_scale
        
    text_inner_thick = max(1, int(round(1 * scale_factor)))
    text_outer_thick = text_inner_thick + max(1, int(round(3 * scale_factor)))
    
    for x1, y1, x2, y2, track_id, conf, class_id in parsed_tracks:
        if allowed_classes is not None and int(class_id) not in allowed_classes:
            continue
            
        # BGR format: Navy Blue (120, 0, 0) for class 0 (player), Dark Red (0, 0, 150) for class 1 (ball)
        dark_color = (120, 0, 0) if int(class_id) == 0 else (0, 0, 150)
        white_color = (255, 255, 255)
        
        # Dual-pass bounding box: thick white outline, then standard dark box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), white_color, outer_box_thick)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), dark_color, inner_box_thick)
        
        overlay_text = f"{track_id}"
        
        # Dual-pass text: thick white background outline, then thin dark foreground text
        text_margin = max(12, int(round(20 * scale_factor)))
        text_pos = (int(x1), max(text_margin, int(y1) - 5))
        cv2.putText(
            frame,
            overlay_text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            actual_font_scale,
            white_color,
            text_outer_thick
        )
        cv2.putText(
            frame,
            overlay_text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            actual_font_scale,
            dark_color,
            text_inner_thick
        )
    return frame


