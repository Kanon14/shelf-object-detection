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
        
# ----------------------------------------------------------------------------
# Planogram parsing → List[Facing]
# ----------------------------------------------------------------------------
facings: List[Facing] = []
planogram_data = None
if planogram_file is not None:
    try: 
        planogram_data = json.load(planogram_file)
    except Exception as e:
        st.error(f"Error reading uploaded planogram: {e}")
elif planogram_text.strip():
    try:
        planogram_data = json.loads(planogram_text)
    except Exception as e:
        st.error(f"Error parsing planogram JSON: {e}")
        
if planogram_data:
    try:
        shelf_key = next(iter(planogram_data.keys()))
        for f in planogram_data[shelf_key]["facings"]:
            facings.append(Facing(id=str(f.get("id", "F")), 
                                  x1=int(f["x1"]), y1=int(f["y1"]),
                                  x2=int(f["x2"]), y2=int(f["y2"]),
                                  min_count=int(f.get("min_count", 1)),)
            )
    except Exception as e:
        st.error(f"Invalid planogram schema {e}")

# ----------------------------------------------------------------------------
# Detection Mode
# ----------------------------------------------------------------------------
st.divider()
mode = st.radio("Mode", ["Image", "Video", "Livecam"], horizontal=True)

# ----------------------------------------------------------------------------
# Image Mode
# ----------------------------------------------------------------------------
if mode == "Image":
    up_img = st.file_uploader("Upload shelf image", type=["jpg", "jpeg", "png"])
    
    if up_img is not None and model is not None:
        pil = Image.open(up_img).convert("RGB")
        img = np.array(pil)[..., ::-1] # RGB→BGR for OpenCV drawing
        H, W = img.shape[:2]
        
        # ROI crop (optional)
        view = img
        x_off = y_off = 0
        if use_roi:
            rx1 = clamp(int(roi_x1), 0, W)
            ry1 = clamp(int(roi_y1), 0, H)
            rx2 = clamp(int(roi_x2) if roi_x2 > 0 else W, 0, W)
            ry2 = clamp(int(roi_y2) if roi_y2 > 0 else H, 0, H)
            if rx2 > rx1 and ry2 > ry1:
                view = img[ry1:ry2, rx1:rx2]
                x_off, y_off = rx1, ry1
                
        # Inference 
        boxes = run_yolo_on_image(model, view, conf=conf, iou=iou, imgsz=imgsz)
        if use_roi and boxes.size > 0:
            boxes[:, [0, 2]] += x_off
            boxes[:, [1, 3]] += y_off
            
        total_count = int(boxes.shape[0])
        
        # OOS evaluation
        status_map: Dict[str, str] = {}
        oos_rows: List[OOSResult] = []
        if facings:
            for f in facings:
                res = facing_status(f, boxes, occ_empty=occ_empty, occ_low=occ_low, iou_threshold=iou_face)
                status_map[f.id] = res.status
                oos_rows.append(res)
                
        # Visualization
        vis = draw_boxes(img, boxes, color=(0, 255, 0))
        if facings:
            vis = draw_facings(vis, facings, status_map)
            
        st.subheader("Result")
        st.image(vis[..., ::-1], caption=f"Detections: {total_count}", width='stretch')
        st.markdown(f"**Total detected items:** {total_count}")
        
        if oos_rows:
            df =  pd.DataFrame(
                [{"Facing": r.facing_id, "Count": r.count, "Occupancy": round(r.occupancy, 3), "Status": r.status} for r in oos_rows]
                )
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="oos_report.csv",
                mime="text/csv"
            )
            
# ----------------------------------------------------------------------------
# Video Mode
# ----------------------------------------------------------------------------    
else:
    up_vid = st.file_uploader("Upload shelf video (mp4/mov/avi)", type=["mp4", "mov", "avi"])
    
    if up_vid is not None and model is not None:
        vid_path = f"saved_video/{up_vid.name}"
        with open(vid_path, "wb") as f:
            f.write(up_vid.read())
            
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            st.error("Failed to open video.")
        else:
            stframe = st.empty()
            info = st.empty()
            
            from collections import deque
            count_window = deque(maxlen=7)
            
            # Aggregate worst status per facing across frames
            status_order = {"OK": 0, "LOW": 1, "OOS": 2}
            per_facing_scores: Dict[str, List[int]] = {f.id: [] for f in facings} if facings else {}


            processed = 0
            frame_idx = 0
            t0 = time.time()
                        
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue
                
                H, W = frame.shape[:2]
                view = frame
                x_off = y_off = 0
                if use_roi:
                    rx1 = clamp(int(roi_x1), 0, W)
                    ry1 = clamp(int(roi_y1), 0, H)
                    rx2 = clamp(int(roi_x2) if roi_x2 > 0 else W, 0, W)
                    ry2 = clamp(int(roi_y2) if roi_y2 > 0 else H, 0, H)
                    if rx2 > rx1 and ry2 > ry1:
                        view = frame[ry1:ry2, rx1:rx2]
                        x_off, y_off = rx1, ry1
                        
                    boxes = run_yolo_on_image(model, view, conf=conf, iou=iou, imgsz=imgsz)
                    if use_roi and boxes.size > 0:
                        boxes[:, [0, 2]] += x_off
                        boxes[:, [1, 3]] += y_off
                        
                    total = int(boxes.shape[0])
                    count_window.append(total)
                    smoothed = int(np.median(count_window))
                    
                    # Per-frame facing status
                    status_map: Dict[str, str] = {}
                    if facings:
                        for f in facings:
                            res = facing_status(f, boxes, occ_empty=occ_empty, occ_low=occ_low, iou_threshold=iou_face)
                            status_map[f.id] = res.status
                            per_facing_scores[f.id].append(status_order[res.status])
                            
                    vis = draw_boxes(frame, boxes)
                    if facings:
                        vis = draw_facings(vis, facings, status_map)
                        
                    info.markdown(f"**Frame:** {frame_idx} | **Detected:** {total} | **Smoothed:** {smoothed}")
                    stframe.image(vis[..., ::-1], use_container_width=True)


                    processed += 1
                    if max_frames > 0 and processed >= max_frames:
                        break
                
                cap.release()
                st.success(f"Done. Processed {processed} frames in {time.time() - t0:.2f}s")
                
                if facings and per_facing_scores:
                    inv = {0: "OK", 1: "LOW", 2: "OOS"}
                    rows = []
                    for fid, scores in per_facing_scores.items():
                        if scores:
                            worst = max(scores)
                            rows.append({"Facing": fid, "Status_over_Video": inv[worst]})
                        else:
                            rows.append({"Facing": fid, "Status_over_Video": "UNKNOWN"})
                            df = pd.DataFrame(rows)
                            st.subheader("Video OOS Summary")
                            st.dataframe(df, use_container_width=True)
                            st.download_button("Download Video OOS Summary CSV",
                                                data=df.to_csv(index=False).encode("utf-8"),
                                                file_name="oos_video_summary.csv",
                                                mime="text/csv",)
                            
                            

        
# ----------------------------------------------------------------------------
# Tips
# ----------------------------------------------------------------------------
st.info(
"Tips: Start with confidence=0.25 and IoU=0.6. Increase image size (800-960) if small items are missed. "
"For dense shelves, high-resolution inputs and ROI cropping improve both speed and accuracy."
)