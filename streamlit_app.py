from __future__ import annotations
import io
import json
import time
import base64
from dataclasses import dataclass
from typing import List, Tuple, Optional, dict

import numpy as np
import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO