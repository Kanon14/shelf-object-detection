# Shelf Item Counter & Out-of-Stock (OOS) Detector

A lightweight Streamlit application powered by YOLOv11-nano (Ultralytics) and the **SKU-110k** dataset for dense product detection on retail shelves. It supports: 
* **Item counting** (class-agnostic product boxes)
* **Out-of-stock (OOS) / low-stock** detection via **planogram** facings
* **Image**, **Video**, and **Livecam** (webcam/RTSP) modes
* **ROI cropping**, adjustable threshold, and CSV exports

> **Note:** SKU‑110k is **class‑agnostic**. The model detects “product/object” without brand/category IDs. To recognize brands/SKUs, add a second‑stage classifier or train a multi‑class detector.

## ✨ Features

* 🔢 Accurate item counts on dense shelves

* 🧩 Planogram‑aware OOS status per facing (OK / LOW / OOS)

* 🖼️ Image inference with visualization overlays

* 🎞️ Video inference with smoothing + video‑wide OOS summary

* 📹 Livecam: webcam index or RTSP/HTTP network stream

* 🟩 ROI cropping to speed up inference and reduce noise

* 📄 CSV export for audits and analytics

## 📁 Project Structure
```bash
.
├─ notebook/sku110k_detection.ipynb # Notebook for the SKU-110k training
├─ skuDetection/utils/main_utils.py # Core logic (data classes, geometry, OOS, drawing, YOLO helpers)
├─ streamlit_app.py # Streamlit UI (Image, Video, Livecam modes)
├─ pyproject.toml # Package requirements 
└─ README.md # This file
```

## ⚙️ Setup Instructions
1. **Clone the repository**
```bash
git clone https://github.com/Kanon14/shelf-object-detection.git
cd shelf-object-detection
```

2. **Create and activate a virtual environment**
```bash
# conda setup
conda create -n sku110k-env python=3.11 -y
conda activate sku110k-env

# uv setup
uv venv --python 3.11
.venv/Scripts/activate
```

3. **Install dependencies**
```bash
uv pip install -r pyproject.toml
```

## 🤖 How to Run
1. Execute the project:
```python
streamlit run streamlit_app.py
```

2. Then, access the application via your web browser:
```python
open http://localhost:<port>
```

## Acknowledgements
## Acknowledgements
- **[Roboflow](https://roboflow.com/):** For dataset hosting and augmentation tools.
- **[Ultralytics](https://www.ultralytics.com/):** For the YOLO object detection framework.