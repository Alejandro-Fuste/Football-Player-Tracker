import numpy as np
import torch

def clip_box_xyxy(xyxy, frame_w: int, frame_h: int):
    """
    Clip coordinates to stay within frame boundary limits, and resolve cases
    where box bounds are inverted.
    """
    x1, y1, x2, y2 = map(float, xyxy)
    x1 = max(0.0, min(frame_w - 1.0, x1))
    y1 = max(0.0, min(frame_h - 1.0, y1))
    x2 = max(0.0, min(frame_w - 1.0, x2))
    y2 = max(0.0, min(frame_h - 1.0, y2))
    
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
        
    return [x1, y1, x2, y2]

def get_detections(model, frame, imgsz: int, conf_thresh: float, allowed_class_ids: set[int] | None = None) -> np.ndarray:
    """
    Runs model inference on a single frame and returns filtered detections.

    Args:
        model: Ultralytics RTDETR/YOLO instance.
        frame: OpenCV image frame (BGR).
        imgsz: Inference size (e.g. 640).
        conf_thresh: Confidence threshold.
        allowed_class_ids: Optional set of class IDs to filter detections.

    Returns:
        detections: NumPy array of shape (N, 6) with rows [x1, y1, x2, y2, confidence, class_id].
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(source=frame, imgsz=imgsz, conf=conf_thresh, verbose=False, device=device)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    
    rows = []
    frame_h, frame_w = frame.shape[:2]

    for box, score, class_id in zip(xyxy, conf, cls):
        if float(score) < conf_thresh:
            continue
        if allowed_class_ids is not None and int(class_id) not in allowed_class_ids:
            continue
            
        x1, y1, x2, y2 = clip_box_xyxy(box, frame_w, frame_h)
        # Filter out invalid, tiny boxes
        if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
            continue
            
        rows.append([x1, y1, x2, y2, float(score), int(class_id)])

    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 6), dtype=np.float32)
