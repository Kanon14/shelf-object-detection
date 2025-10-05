from __future__ import annotations

import os
import json
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st 

from skuDetection.entity.config_entity import Facing, OOSResult
from skuDetection.utils.main_utils import (
    clamp, draw_boxes, draw_facings, 
    facing_status, load_model, run_yolo_on_image)

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Shelf Counter + OOS Detector", layout="wide")
st.title("🛒 Shelf Item Counter & OOS Detector")
st.caption("YOLO (SKU-110k, class-agnostic) • Count products • Detect out-of-stock facings")


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    weights_src = st.radio("Load weights via", ["Path", "Upload"], index=0)
    
    weights_path: Optional[str] = None
    if weights_src == "Path":
        weights_path = st.text_input("Weights path", value="model/sku110k-yolo11n.pt", 
                                     help="Local path to your trained Ultralytics YOLO weights (.pt)")
    else:
        up = st.file_uploader("Upload .pt weights", type=["pt"], accept_multiple_files=False)
        if up is not None:
            _ = os.mkdir("model")
            weights_path = f"/model/{up.name}"
            with open(weights_path, "wb") as f:
                f.write(up.read())
                
    st.divider()
    st.header("Detection Settings")
    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("NMS IoU", 0.10, 0.95, 0.60, 0.01)
    imgsz = st.select_slider("Image size (short side)", options=[320, 416, 512, 640, 800, 960, 1280], value=640)
    
    st.divider()
    st.header("ROI (Optional)")
    use_roi = st.checkbox("Crop to ROI before detection", value=False)
    roi_x1 = st.number_input("ROI x1", value=0, min_value=0)
    roi_y1 = st.number_input("ROI y1", value=0, min_value=0)
    roi_x2 = st.number_input("ROI x2 (0 = max width)", value=0, min_value=0)
    roi_y2 = st.number_input("ROI y2 (0 = max height)", value=0, min_value=0)
    
    st.divider()
    st.header("Planogram (optional)")
    st.markdown(
    "Upload a planogram JSON or paste below. Example:\n\n"
    "```{\n \"shelf_1\": {\n \"facings\": [\n {\"id\": \"A1\", \"x1\":120, \"y1\":220, \"x2\":220, \"y2\":420, \"min_count\":1},\n {\"id\": \"A2\", \"x1\":230, \"y1\":220, \"x2\":330, \"y2\":420, \"min_count\":1}\n ]\n }\n}\n```")
    planogram_file = st.file_uploader("Upload planogram.json", type=["json"], accept_multiple_files=False)
    planogram_text = st.text_area("or paste JSON here", value="", height=150)
    
    st.divider()
    st.header("OOS Thresholds")
    occ_empty = st.slider("Occupancy → OOS", 0.0, 1.0, 0.10, 0.01)
    occ_low = st.slider("Occupancy → LOW", 0.0, 1.0, 0.35, 0.01)
    iou_face = st.slider("IoU for facing count", 0.05, 0.50, 0.10, 0.01, help="IoU threshold to count a detection as inside a facing")
    
    st.divider()
    st.header("Video Settings")
    frame_skip = st.slider("Process every Nth frame", 1, 10, 3)
    max_frames = st.number_input("Max frames (0 = all)", value=0, min_value=0)
    
# ----------------------------------------------------------------------------
# Model load (lazy)
# ----------------------------------------------------------------------------
model = None
if weights_path:
    try:
        model = load_model(weights_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")