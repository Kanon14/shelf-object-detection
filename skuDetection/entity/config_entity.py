from dataclasses import dataclass

@dataclass
class Facing:
    """
    A *facing* describes a rectangular product slot on a shelf.
    
    ----------
    Parameters
    ----------
    id : str - Identifier (e.g., "A1", "Row2-Col5"). Used in UI tables and overlays.
    x1, y1, x2, y2 : int - Pixel coordinates of the top-left (x1,y1) and bottom-right (x2,y2) corners.
                           These should be provided in the coordinate space of the *full* image (not a crop).
    min_count : int, default=1 - Minimum number of detected product boxes that must overlap this facing to be
                                 considered *in stock*. If lower than this, it will be flagged OOS.
    """
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    min_count: int = 1
    
@dataclass
class OOSResult:
    """
    Per-facing evaluation summary.

    ----------
    Attributes
    ----------
    facing_id : str - The facing identifier this result corresponds to.
    count : int - Number of detected boxes that overlap this facing (IoU > 0.1 by default).
    occupancy : float - Fraction in [0,1] of the facing area covered by *any* detections. This is
                        computed as the sum of intersection areas (clamped by facing area), which is a
                        simple proxy for how "filled" the slot looks in 2D.
    status : {"OK", "LOW", "OOS"} - Status derived from `count` and `occupancy` against user thresholds.
    """
    facing_id: str
    count: int
    occupancy: float
    status: str # "OK" | "LOW" | "OOS"