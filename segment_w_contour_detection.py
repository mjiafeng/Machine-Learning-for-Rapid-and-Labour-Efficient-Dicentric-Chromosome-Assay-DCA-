# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 13:57:17 2025

@author: mjiafeng
"""

import cv2
import os

# --- Parameters ---
input_folder = "Image"  # folder containing all input images
output_folder = "chromosome_segments"  # single folder for all outputs
min_contour_area = 100  # filter out very small objects (adjust as needed)

# --- Create output folder if it doesn't exist ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Loop over all image files in input folder ---
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp")):
        continue  # skip non-image files

    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {filename}, skipping...")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Thresholding ---
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = 255 - thresh  # invert if chromosomes are dark on light background

    # --- Find contours ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Loop over contours and save each ---
    count = 0
    base_name = os.path.splitext(filename)[0]
    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            continue  # skip very small noise
        
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = img[y:y+h, x:x+w]
        # include image name and contour index in filename to avoid overwriting
        output_filename = f"{base_name}_chromosome_{count}.jpg"
        cv2.imwrite(os.path.join(output_folder, output_filename), cropped)
        count += 1

    print(f"Saved {count} chromosome images from '{filename}' to '{output_folder}'")
