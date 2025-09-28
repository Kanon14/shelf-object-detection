import os
import sys
import yaml
import base64
import cv2
import math
import numpy as np
import time

from typing import Tuple
from skuDetection.exception import AppException
from skuDetection.logger import logging
from skuDetection.entity.config_entity import Facing, OOSResult


# -------------------------------
# Geometry helpers
# -------------------------------

def clamp(v: int, lo: int, hi: int) -> int:
    '''Clamps a value into an inclusive range.'''
    return max(lo, min(hi, v))


def iou_rect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    '''
    Compute the Intersection-over-Union (IoU) between two axis-aligned reactangle.
    '''
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b  
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def rect_intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    '''
    Compute the intersection area between two axis-aligned reactangle.
    '''
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter


# -------------------------------
# OOS / Occupancy Logic
# -------------------------------
def occupancy_ratio(facing: Facing, boxes_xyxy: np.ndarray) -> float:
    fx1, fy1, fx2, fy2 = facing.x1, facing.y1, facing.x2, facing.y2
    farea = max(1, (fx2 - fx1) * (fy2 - fy1)) # avoid div/0
    inter_total = 0
    F = (fx1, fy1, fx2, fy2)
    for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
        inter_total += rect_intersection_area(F, (int(x1), int(y1), int(x2), int(y2)))
    return float(inter_total) / float(farea)