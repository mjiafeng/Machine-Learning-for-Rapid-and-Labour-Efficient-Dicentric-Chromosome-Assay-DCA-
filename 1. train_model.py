# -*- coding: utf-8 -*-
"""
YOLOv8 Training + Instance Segmentation + Clean Crop + Mask Overlay + Polygon Highlights
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np

# --------------------------
# Paths and setup
# --------------------------
yaml_file = "chromosome.yaml"
train_model_name = "chromosome_yolo"

# --------------------------
# Load YOLO model (pre-trained)
# --------------------------
# model = YOLO("yolov8n-seg.pt")
model = YOLO("yolov8m-seg.pt")

# --------------------------
# Train model
# --------------------------
model.train(
    data=yaml_file,
    epochs=300,
    imgsz=256, # best
    batch=4, # better than '2'. Not going higher because using CPU
    lr0=0.01, # no change, so use default
    name=train_model_name,
)

# --------------------------
# Validation
# --------------------------
metrics = model.val()
print(metrics)
