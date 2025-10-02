import cv2
import numpy as np
from typing import Tuple, Dict, Sequence
from skuDetection.entity.config_entity import Facing, OOSResult

try:
    from ultralytics import YOLO 
except Exception:
    YOLO = None # Fallback so the module can still be imported for non-inference

# -------------------------------
#  GEOMETRY HELPERS
# -------------------------------

def clamp(v: int, lo: int, hi: int) -> int:
    """
    Clamp integer `v` into the inclusive range [lo, hi].
    
    Useful to sanitize ROI inputs coming from a UI where users might specify values
    outside the valid image range.
    """
    return max(lo, min(hi, v))


def _area(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    Compute area (in pixels) of an axis-aligned rectangle.
    
    Returns 0 if coordinates are invalid (non-positive width/height).
    """
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou_rect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """
    Intersection-over-Union (IoU) between two rectangles.

    IoU = intersection_area / union_area, in [0,1]. Returns 0 if there is no overlap or if union is zero.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b  
    
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    
    inter = _area(ix1, iy1, ix2, iy2)
    if inter == 0:
        return 0.0
    
    area_a = _area(ax1, ay1, ax2, ay2)
    area_b = _area(bx1, by1, bx2, by2)
    union = area_a + area_b - inter
    
    return float(inter) / float(union) if union > 0 else 0.0


def rect_intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    """
    Intersection area (in pixels) between rectangles `a` and `b`.

    This is used by `occupancy_ratio` to approximate how much of a facing is "filled".
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = _area(ix1, iy1, ix2, iy2)
    return inter


# -------------------------------
# OOS / OCCUPANCY LOGIC
# -------------------------------
def occupancy_ratio(facing: Facing, boxes_xyxy: np.ndarray) -> float:
    """
    Compute the fraction of a facing's area covered by detection boxes.
    
    Parameters
    ----------
    facing: Facing - The shelf slot rectangle.
    boxes_xyxy: np.ndarray of shape (N, 4) - Detected bounding boxes in **XYXY** format, same coordinate space as the facing.
    
    Returns
    ----------
    float - A value in [0,1]. 0 means no coverage; 1 means the facing is fully covered by
            the *union* of boxes (approximation: we sum intersections without exact union).
            
    Notes
    ----------
    This implementation sums the intersection areas of each box with the facing, and
    divides by the facing area. This can slightly **overestimate** coverage in cases
    where many boxes overlap each other (double-counting). For OOS heuristics this is
    usually acceptable and fast. If you need exact coverage, rasterize to a mask.
    
    """
    fx1, fy1, fx2, fy2 = facing.x1, facing.y1, facing.x2, facing.y2
    farea = max(1, _area(fx1, fy1, fx2, fy2)) # avoid division by zero
    F = (fx1, fy1, fx2, fy2)
    
    inter_total = 0
    for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
        inter_total += rect_intersection_area(F, (int(x1), int(y1), int(x2), int(y2)))
    return float(inter_total) / float(farea)


def facing_status(facing: Facing, boxes_xyxy: np.ndarray, 
                  occ_empty: float, occ_low: float,
                  iou_threshold: float = 0.1) -> OOSResult:
    """
    Evaluate a single facing against detection boxes.
    
    Logic
    ----------
    1) Count check: if the number of detected boxes overlapping the facing by IoU
                    > `iou_threshold` is less than `min_count`, we flag **OOS**.
    2) Occupancy check: compute `occupancy_ratio`. If it is below `occ_empty`, flag
                        **OOS**. Else if below `occ_low`, flag **LOW**. Else **OK**.
    
    Parameters
    ----------
    facing: Facing - Slot rectangle with id and `min_count`.
    boxes_xyxy: np.ndarray - YOLO detections (N,4) in XYXY.
    occ_empty: float - Occupancy threshold for OOS (e.g., 0.10)
    occ_low: float - Occupancy threshold for LOW (e.g., 0.35)
    iou_threshold: float - Overlap threshold to consider a detection as belonging to this facing for purpose
                           of counting `min_count`. Default to 0.1
                           
    Returns
    ----------
    OOSResult - Per-facing status including the raw count and occupancy.
    
    """
    F = (facing.x1, facing.y1, facing.x2, facing.y2)
    
    # Count boxes overlapping facing by IOU > 0.1
    count = 0
    for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
        if iou_rect(F, (int(x1), int(y1), int(x2), int(y2))) > iou_threshold:
            count += 1
            
        # Approximate how much of the slot is visually occupied by products. 
        occ = occupancy_ratio(facing, boxes_xyxy)
        
        if count < facing.min_count or occ < occ_empty:
            status = "OOS"
        elif occ < occ_low:
            status = "LOW"
        else:
            status = "OK"
            
        return OOSResult(facing_id=facing.id, count=count, occupancy=occ, status=status)
    
    
# -------------------------------
# DRAWING HELPERS (OpenCV BGR)
# -------------------------------
def draw_boxes(img_bgr: np.ndarray, boxes_xyxy: np.ndarray, 
               color: Tuple[int, int, int] = (0, 255, 0),
               thickness: int = 2) -> np.ndarray:
    """
    Draw YOLO detection boxes on a copy of the image. 
    
    Parameters
    ----------
    img_bgr: np.ndarray (H,W,3) - Input image in BGR order (OpenCV default).
    boxes_xyxy: np.ndarray (N,4) - Detection boxes (x1,y1,x2,y2) in pixel coordinates.
    color: Tuple - Box color (B,G,R). Default to (0, 255, 0).
    thickness: int - Rectangle thickness in pixels. Default to 2.
    
    Returns
    ----------
    np.ndarray - A copy of the input image with rectangles drawn. 
     
    """
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), color=color, thickness=thickness)
    return out


def draw_facings(img_bgr: np.ndarray, facings: Sequence[Facing], 
                 status_map: Dict[str, str]) -> np.ndarray:
    """
    Draw facings with color-coded status banner. 
    
    Colors
    ---------
    - 'OK'  → green
    - 'LOW' → yellow/cyan
    - 'OOS' → red
    
    A small filled banner with text like "A1:OOS" is drawn above each facing for quick visual inspection.
    
    """
    out = img_bgr.copy()
    for f in facings:
        status = status_map.get(f.id, "OK")
        if status == "OK":
            color = (0, 200, 0)
        elif status == "LOW":
            color = (0, 200, 200)
        else:
            color = (0, 0, 255)
             
        cv2.rectangle(out, (f.x1, f.y1), (f.x2, f.y2), color=color, thickness=2)
        label = f"{f.id}:{status}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (f.x1, f.y1 - th - 6), (f.x1 + tw + 6, f.y1), color, -1)
        cv2.putText(out, label, (f.x1 + 3, f.y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return out


# -------------------------------
# MODEL INFERENCE HELPERS (Ultralytics)
# -------------------------------
def load_model(weights_path: str):
    """
    Load an Ultralytics YOLO model from local weights.

    Notes
    -----
    - We import Ultralytics lazily at module import-time to keep lightweight.
    - If Ultralytics is not installed, this function raises a clear error message.
    
    """
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Please `pip install ultralytics`")
    model = YOLO(weights_path)
    return model


def run_yolo_on_image(model, img_bgr: np.ndarray, conf: float = 0.25, 
                      iou: float = 0.60, imgsz: int = 640) -> np.ndarray:
    """
    Run YOLO on a single image and return detections as a NumPy array.


    Parameters
    ----------
    model : ultralytics.YOLO - A loaded YOLO model instance.
    img_bgr : np.ndarray (H,W,3) - Input image in **BGR** order (OpenCV default). Ultralytics accepts RGB, 
                                   so we will convert before inference.
    conf : float, default=0.25 - Confidence threshold for post-NMS filtering.
    iou : float, default=0.60 - NMS IoU threshold.
    imgsz : int, default=640 - Inference size (short side). Larger values help with dense small objects.


    Returns
    -------
    np.ndarray of shape (N, 4) - Detections in **XYXY** pixel coordinates. Returns an empty array 
                                 if there are no boxes.
                                 
    """
    # Ultralytics' `predict` accepts RGB NumPy arrays directly.
    rgb = img_bgr[..., ::-1]
    res = model.predict(rgb, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    
    # Defensive checks: `.boxes` may be None if no detections.
    if getattr(res, "boxes", None) is None or getattr(res.boxes, "xyxy", None) is None:
        return np.zeros((0, 4), dtype=np.float32)
    
    boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy()
    # Ensure a consistent dtype for downstream operations.
    return boxes_xyxy.astype(np.float32)
