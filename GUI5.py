# -*- coding: utf-8 -*-
"""
DCA Viewer GUI with Zoom, Pan, Crop, Analyze
Interactive Polygons:
- Click polygon → selects corresponding crop in table
- Edit class in table → polygon updates color and class
- Supports 4 classes: Normal, Dicentric, Irrelevant, Consult_Expert
- Live summary panel with manual 'Other' count
- Results table sortable
- Filter results table by class
- Lasso selection feature to group-modify polygon class
- Adjustable Minimum Polygon Area via clickable slider
- NEW: Per-Class Confidence Threshold Sliders for sensitivity tuning
"""

import sys
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout,
    QWidget, QScrollArea, QColorDialog, QCheckBox, QMessageBox, QFrame,
    QGroupBox, QGridLayout, QSlider, QTableWidget, QTableWidgetItem, QSplitter,
    QComboBox, QStyle, QLineEdit, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QIntValidator
from PyQt5.QtCore import Qt, QPoint

# -------------------- PARAMETERS --------------------
MIN_CONTOUR_AREA = 100 # Original, but overridden by slider in run_analyze now
CLASS_NAMES = ["Normal", "Dicentric", "Irrelevant", "Consult_Expert"]

CLASS_START_COLORS = {
    "Normal": (0, 255, 0),
    "Dicentric": (255, 0, 0),
    "Irrelevant": (0, 0, 255),
    "Consult_Expert": (255, 255, 0)
}

CROP_START_COLOR = (50, 50, 50)  # Dark grey

# -------------------- CLICKABLE SLIDER --------------------
class ClickableSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            val = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(),
                                                 event.pos().x(), self.width())
            self.setValue(val)
            event.accept()
            self.sliderMoved.emit(val)
        super().mousePressEvent(event)

# -------------------- COMBOBOX WITHOUT SCROLL --------------------
class NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()

# -------------------- IMAGE LABEL --------------------
class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.zoom_factor = 1.0
        self.pixmap_original = None
        self.pan_offset = QPoint(0,0)
        self.last_mouse_pos = None
        self.parent_widget = None

        # --- Lasso State ---
        self.is_drawing_lasso = False
        self.lasso_points = []
        self.lasso_color = (255, 255, 255) # White
        self.highlight_color = (255, 165, 0) # Orange for selection

    def setPixmap(self, pixmap: QPixmap):
        self.pixmap_original = pixmap
        self.update_pixmap()

    def update_pixmap(self):
        if self.pixmap_original is not None:
            scaled = self.pixmap_original.scaled(
                self.pixmap_original.size() * self.zoom_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

    def wheelEvent(self, event):
        if self.pixmap_original is None:
            return
    
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9
    
        scroll_area = self.parent().parent()
        hbar = scroll_area.horizontalScrollBar()
        vbar = scroll_area.verticalScrollBar()
    
        cursor_pos = event.pos()
    
        current_pixmap = self.pixmap()
        if current_pixmap is None:
            return
            
        h_ratio = (cursor_pos.x() + hbar.value()) / current_pixmap.width()
        v_ratio = (cursor_pos.y() + vbar.value()) / current_pixmap.height()
    
        self.zoom_factor *= factor
        self.parent_widget.update_display()
    
        updated_pixmap = self.pixmap()
        if updated_pixmap is None:
            return
            
        hbar.setValue(int(h_ratio * updated_pixmap.width() - cursor_pos.x()))
        vbar.setValue(int(v_ratio * updated_pixmap.height() - cursor_pos.y()))
        
    def get_original_coords(self, pos):
        """Converts mouse position to original image coordinates."""
        if self.pixmap() is None:
             return None, None
        x = int(pos.x() / self.zoom_factor)
        y = int(pos.y() / self.zoom_factor)
        return x, y

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()
            if self.parent_widget:
                self.parent_widget.check_polygon_click(event.pos())
        elif event.button() == Qt.RightButton and self.pixmap_original is not None:
            # Start Lasso Drawing
            self.is_drawing_lasso = True
            self.lasso_points = []
            x, y = self.get_original_coords(event.pos())
            if x is not None:
                self.lasso_points.append((x, y))
            self.parent_widget.update_display()


    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.last_mouse_pos is not None:
            # Pan
            delta = event.pos() - self.last_mouse_pos
            self.parent().parent().horizontalScrollBar().setValue(
                self.parent().parent().horizontalScrollBar().value() - delta.x())
            self.parent().parent().verticalScrollBar().setValue(
                self.parent().parent().verticalScrollBar().value() - delta.y())
            self.last_mouse_pos = event.pos()
        elif event.buttons() & Qt.RightButton and self.is_drawing_lasso:
            # Lasso Drawing
            x, y = self.get_original_coords(event.pos())
            if x is not None:
                self.lasso_points.append((x, y))
            self.parent_widget.update_display()
            

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None
        elif event.button() == Qt.RightButton and self.is_drawing_lasso:
            # End Lasso Drawing and perform selection
            self.is_drawing_lasso = False
            if self.parent_widget and len(self.lasso_points) > 2:
                self.parent_widget.select_polygons_by_lasso(self.lasso_points)
            self.lasso_points = []
            self.parent_widget.update_display()

# -------------------- MAIN GUI --------------------
class DCA_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DCA Analyzer with Lasso Selection and Fine-Tuning")
        self.setGeometry(50, 50, 1800, 1000)

        # ---------------- DATA ----------------
        self.loaded_image = None
        self.crop_rects = []
        self.polygons = []
        self.selected_polygon_idx = None
        self.lasso_selected_polygons = []
        # NEW: Store per-class confidence thresholds (0.25 is default)
        self.class_confidence_thresholds = {cls: 0.25 for cls in CLASS_NAMES} 

        # YOLO model
        try:
            # Assuming 'best.pt' is in the same directory as the script
            self.model = YOLO("best.pt")
        except:
            QMessageBox.warning(self, "YOLO Warning", "YOLO model 'best.pt' not found. Analysis will fail.")
            self.model = None

        # HORIZONTAL SPLITTER
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        # ---------------- LEFT PANEL ----------------
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.splitter.addWidget(self.left_panel)

        self.loaded_file_label = QLabel("No image loaded")
        self.left_layout.addWidget(self.loaded_file_label)

        self.load_btn = QPushButton("Load DCA Image")
        self.load_btn.clicked.connect(self.load_image)
        self.left_layout.addWidget(self.load_btn)

        self.crop_btn = QPushButton("Run Crop")
        self.crop_btn.clicked.connect(self.run_crop)
        self.left_layout.addWidget(self.crop_btn)

        crop_group = QGroupBox("Crop Options")
        crop_layout = QGridLayout()
        crop_group.setLayout(crop_layout)
        self.left_layout.addWidget(crop_group)

        self.show_crop_cb = QCheckBox("Show Crop Rectangles")
        self.show_crop_cb.setChecked(True)
        self.show_crop_cb.stateChanged.connect(self.update_display)
        crop_layout.addWidget(self.show_crop_cb, 0, 0)

        self.crop_color_btn = QPushButton("Pick Crop Color")
        self.crop_color_btn.clicked.connect(self.pick_crop_color)
        crop_layout.addWidget(self.crop_color_btn, 0, 1)

        self.crop_preview = QFrame()
        self.crop_preview.setFixedSize(25, 25)
        self.crop_color = CROP_START_COLOR
        r, g, b = self.crop_color
        self.crop_preview.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")

        crop_layout.addWidget(self.crop_preview, 0, 2)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.run_analyze)
        self.left_layout.addWidget(self.analyze_btn)

        # --- MIN CONTOUR AREA SLIDER ---
        contour_group = QGroupBox("Minimum Polygon Area (px²)")
        contour_layout = QVBoxLayout()
        contour_group.setLayout(contour_layout)
        self.left_layout.addWidget(contour_group)

        # Label to show current value
        self.min_area_label = QLabel("Current Min Area: 150") # Initial value
        contour_layout.addWidget(self.min_area_label)

        self.min_contour_area_slider = ClickableSlider(Qt.Horizontal)
        self.min_contour_area_slider.setMinimum(1)
        self.min_contour_area_slider.setMaximum(500)
        self.min_contour_area_slider.setValue(150)
        self.min_contour_area_slider.setTickPosition(QSlider.TicksBelow)
        self.min_contour_area_slider.setTickInterval(50)
        self.min_contour_area_slider.valueChanged.connect(self.update_min_area_label)
        contour_layout.addWidget(self.min_contour_area_slider)
        # -----------------------------

        # --- NEW: IOU MERGE THRESHOLD SLIDER ---
        iou_group = QGroupBox("IoU Merge Threshold (0.01 - 1.0)")
        iou_layout = QVBoxLayout()
        iou_group.setLayout(iou_layout)
        self.left_layout.addWidget(iou_group)

        # Label to show current value
        self.iou_threshold_label = QLabel("Current IoU Threshold: 0.10") # Initial value
        iou_layout.addWidget(self.iou_threshold_label)

        self.iou_threshold_slider = ClickableSlider(Qt.Horizontal)
        self.iou_threshold_slider.setMinimum(1)    # Represents 0.01
        self.iou_threshold_slider.setMaximum(100)  # Represents 1.00
        self.iou_threshold_slider.setValue(10)     # Initial value: 0.10
        self.iou_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_threshold_slider.setTickInterval(10)
        self.iou_threshold_slider.valueChanged.connect(self.update_iou_threshold_label)
        iou_layout.addWidget(self.iou_threshold_slider)
        # -----------------------------

        self.left_layout.addWidget(QLabel("Polygon Thickness"))
        self.poly_thickness_slider = ClickableSlider(Qt.Horizontal)
        self.poly_thickness_slider.setMinimum(1)
        self.poly_thickness_slider.setMaximum(10)
        self.poly_thickness_slider.setValue(2)
        self.poly_thickness_slider.setTickPosition(QSlider.TicksBelow)
        self.poly_thickness_slider.setTickInterval(1)
        self.poly_thickness_slider.valueChanged.connect(self.update_display)
        self.left_layout.addWidget(self.poly_thickness_slider)
        
        # --- NEW: CLASS CONFIDENCE THRESHOLD GROUP ---
        self.conf_group = QGroupBox("Class Confidence Thresholds (Sens. Tuning)")
        self.conf_layout = QGridLayout()
        self.conf_group.setLayout(self.conf_layout)
        self.left_layout.addWidget(self.conf_group)
        
        self.conf_labels = {}
        for i, cls_name in enumerate(CLASS_NAMES):
            # Label
            lbl = QLabel(f"{cls_name}: 0.25")
            self.conf_layout.addWidget(lbl, i, 0)
            self.conf_labels[cls_name] = lbl

            # Slider
            slider = ClickableSlider(Qt.Horizontal)
            slider.setMinimum(1)  # Represents 0.01
            slider.setMaximum(100) # Represents 1.00
            slider.setValue(25)   # Represents 0.25
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            # Use lambda to connect the slider change to the update function
            slider.valueChanged.connect(lambda val, c=cls_name: self.update_confidence_threshold_label(c, val))
            self.conf_layout.addWidget(slider, i, 1)
        # ----------------------------------------------


        # --- BATCH UPDATE UI ---
        self.batch_group = QGroupBox("Batch Lasso Update (Right Click)")
        batch_layout = QGridLayout()
        self.batch_group.setLayout(batch_layout)
        self.left_layout.addWidget(self.batch_group)

        self.batch_class_combo = NoScrollComboBox()
        self.batch_class_combo.addItems(CLASS_NAMES)
        batch_layout.addWidget(QLabel("Change Selected to:"), 0, 0)
        batch_layout.addWidget(self.batch_class_combo, 0, 1)

        self.batch_update_btn = QPushButton("Update Selection")
        self.batch_update_btn.clicked.connect(self.batch_update_polygons)
        batch_layout.addWidget(self.batch_update_btn, 1, 0, 1, 2)
        
        self.selected_count_label = QLabel("Polygons Selected: 0")
        batch_layout.addWidget(self.selected_count_label, 2, 0, 1, 2)

        self.clear_selection_btn = QPushButton("Clear Selection")
        self.clear_selection_btn.clicked.connect(self.clear_lasso_selection)
        batch_layout.addWidget(self.clear_selection_btn, 3, 0, 1, 2)
        # ---------------------------

        self.class_group = QGroupBox("Class Polygon Options")
        self.class_layout = QGridLayout()
        self.class_group.setLayout(self.class_layout)
        self.left_layout.addWidget(self.class_group)

        self.class_settings = {}
        for i, cls_name in enumerate(CLASS_NAMES):
            cb = QCheckBox(f"Show {cls_name}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_display)
            self.class_layout.addWidget(cb, i, 0)

            btn = QPushButton("Pick Color")
            btn.clicked.connect(lambda _, c=cls_name: self.pick_class_color(c))
            self.class_layout.addWidget(btn, i, 1)

            preview = QFrame()
            preview.setFixedSize(25, 25)
            start_color = CLASS_START_COLORS[cls_name]
            preview.setStyleSheet(f"background-color: rgb({start_color[0]}, {start_color[1]}, {start_color[2]});")
            self.class_layout.addWidget(preview, i, 2)

            self.class_settings[cls_name] = {"color": start_color, "checkbox": cb, "preview": preview}

        self.uncheck_all_btn = QPushButton("Toggle All Polygons")
        self.uncheck_all_state = False
        self.uncheck_all_btn.clicked.connect(self.toggle_all_polygons)
        self.left_layout.addWidget(self.uncheck_all_btn)

        # --- Summary Group ---
        self.summary_group = QGroupBox("Summary")
        self.summary_layout = QVBoxLayout()
        self.summary_group.setLayout(self.summary_layout)
        self.left_layout.addWidget(self.summary_group)
        
        self.summary_labels = {}
        for cls_name in CLASS_NAMES:
            lbl = QLabel(f"{cls_name}: 0")
            self.summary_labels[cls_name] = lbl
            self.summary_layout.addWidget(lbl)
            
        # NEW: Other Count Input
        other_widget = QWidget()
        other_layout = QHBoxLayout(other_widget)
        other_layout.setContentsMargins(0, 0, 0, 0)
        other_layout.addWidget(QLabel("Other:"))
        
        self.other_count_input = QLineEdit("0")
        self.other_count_input.setFixedWidth(50)
        self.other_count_input.setValidator(QIntValidator(0, 10000))
        self.other_count_input.setStyleSheet("font-size: 5.5pt;")
        self.other_count_input.textChanged.connect(self.update_summary_counts)
        other_layout.addWidget(self.other_count_input)
        
        self.summary_layout.addWidget(other_widget)
        # ------------------------------

        # Total Label
        self.total_label = QLabel("Total Normal + Dicentric + Other: 0")
        self.summary_layout.addWidget(self.total_label)

        self.left_layout.addStretch()

        # ---------------- MIDDLE PANEL ----------------
        self.middle_panel = QWidget()
        self.middle_layout = QVBoxLayout(self.middle_panel)
        self.splitter.addWidget(self.middle_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.middle_layout.addWidget(self.scroll_area)

        self.image_label = ImageLabel()
        self.image_label.parent_widget = self
        self.scroll_area.setWidget(self.image_label)

        # ---------------- RIGHT PANEL ----------------
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.splitter.addWidget(self.right_panel)

        # Filter combo box
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Classes")
        self.filter_combo.addItems(CLASS_NAMES)
        self.filter_combo.currentTextChanged.connect(self.apply_filter)
        self.right_layout.addWidget(QLabel("Filter by Class:"))
        self.right_layout.addWidget(self.filter_combo)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Class", "Confidence"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSortingEnabled(True)
        self.results_table.itemSelectionChanged.connect(self.table_selection_changed)
        self.right_layout.addWidget(self.results_table)

        self.splitter.setSizes([350, 1000, 450])


    # -------------------- LASSO LOGIC --------------------
    def is_polygon_in_lasso(self, polygon_points, lasso_points):
        """Checks if a polygon's centroid is inside the closed lasso path."""
        lasso_poly = np.array(lasso_points, dtype=np.int32)

        M = cv2.moments(polygon_points)
        if M["m00"] == 0:
            return False
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return cv2.pointPolygonTest(lasso_poly, (cx, cy), False) >= 0

    def select_polygons_by_lasso(self, lasso_points):
        """Identifies polygons within the lasso path."""
        self.lasso_selected_polygons = []
        for i, poly in enumerate(self.polygons):
            if self.is_polygon_in_lasso(poly["points"], lasso_points):
                self.lasso_selected_polygons.append(i)
        
        self.selected_count_label.setText(f"Polygons Selected: {len(self.lasso_selected_polygons)}")
        self.update_display()
        
    def clear_lasso_selection(self):
        """Clears the current lasso selection."""
        self.lasso_selected_polygons = []
        self.selected_count_label.setText("Polygons Selected: 0")
        self.update_display()
        
    def batch_update_polygons(self):
        """Updates the class of all lasso-selected polygons."""
        if not self.lasso_selected_polygons:
            QMessageBox.warning(self.batch_group, "Warning", "No polygons are selected with the lasso.")
            return

        new_class = self.batch_class_combo.currentText()
        count = 0
        
        for idx in self.lasso_selected_polygons:
            if idx < len(self.polygons):
                self.polygons[idx]["class"] = new_class
                count += 1
                
                # Update the corresponding QComboBox in the results table
                widget = self.results_table.cellWidget(idx, 0)
                if isinstance(widget, QComboBox):
                    widget.setCurrentText(new_class)
        
        QMessageBox.information(self.batch_group, "Done", f"Updated class of {count} polygons to '{new_class}'.")
        self.clear_lasso_selection()
        self.update_summary_counts()
        self.apply_filter(self.filter_combo.currentText())
        self.update_display()

    # -------------------- HELPER FUNCTIONS --------------------
    def update_confidence_threshold_label(self, cls_name, value):
        """Updates the label and stores the new confidence threshold for a class."""
        threshold = value / 100.0  # Convert 1-100 range back to 0.01-1.00
        self.conf_labels[cls_name].setText(f"{cls_name}: {threshold:.2f}")
        self.class_confidence_thresholds[cls_name] = threshold
        
    def bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
        boxBArea = (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open DCA Image", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)"
        )
        if file_path:
            self.loaded_image = cv2.imread(file_path)
            if self.loaded_image is not None:
                self.crop_rects = []
                self.polygons = []
                self.lasso_selected_polygons = []
                self.loaded_file_label.setText(f"Loaded: {os.path.basename(file_path)}")
                self.update_display()
                self.update_summary_counts()
                self.results_table.setRowCount(0)
                self.selected_count_label.setText("Polygons Selected: 0")

    def pick_crop_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            # Store as RGB tuple
            self.crop_color = (color.red(), color.green(), color.blue())
            
            # FIX: Explicitly set the stylesheet using the stored RGB values
            r, g, b = self.crop_color
            self.crop_preview.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b});"
            )
            
            self.update_display()


    def pick_class_color(self, cls_name):
        color = QColorDialog.getColor()
        if color.isValid():
            # Store as RGB tuple
            self.class_settings[cls_name]["color"] = (color.red(), color.green(), color.blue())
            r, g, b = self.class_settings[cls_name]["color"]
            self.class_settings[cls_name]["preview"].setStyleSheet(
                f"background-color: rgb({r}, {g}, {b});"
            )
            self.update_display()


    def toggle_all_polygons(self):
        for cls_name in CLASS_NAMES:
            self.class_settings[cls_name]["checkbox"].setChecked(self.uncheck_all_state)
        self.update_display()
        self.uncheck_all_state = not self.uncheck_all_state

    def update_min_area_label(self, value):
        """Updates the label to show the current minimum contour area value."""
        self.min_area_label.setText(f"Current Min Area: {value}")

    def update_iou_threshold_label(self, value):
        """Updates the label to show the current IoU threshold value."""
        threshold = value / 100.0 # Convert 1-100 range back to 0.01-1.00
        self.iou_threshold_label.setText(f"Current IoU Threshold: {threshold:.2f}")
        
    def run_crop(self):
        if self.loaded_image is None:
            QMessageBox.warning(self, "Warning", "Load an image first!")
            return

        gray = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = 255 - thresh
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.crop_rects = []
        for cnt in contours:
            # Note: Using the original hardcoded MIN_CONTOUR_AREA for cropping, not the slider value
            if cv2.contourArea(cnt) < 100: 
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            self.crop_rects.append((x, y, w, h))

        self.update_display()
        QMessageBox.information(self, "Done", f"{len(self.crop_rects)} crops detected.")

    def run_analyze(self):
        if self.loaded_image is None:
            QMessageBox.warning(self, "Warning", "Load an image first!")
            return
        if len(self.crop_rects) == 0:
            QMessageBox.warning(self, "Warning", "Run crop first!")
            return
        if self.model is None:
            QMessageBox.critical(self, "Error", "YOLO model not loaded. Cannot analyze.")
            return

        # Fetch the current value from the new area slider
        min_area_threshold = self.min_contour_area_slider.value()
        
        # NEW: Fetch the current value from the IoU slider
        iou_threshold = self.iou_threshold_slider.value() / 100.0 # Convert to 0.01-1.0 range
        
        # Determine the overall minimum confidence to pass to the YOLO model
        min_global_conf = min(self.class_confidence_thresholds.values())

        self.polygons = []
        self.lasso_selected_polygons = []
        self.selected_count_label.setText("Polygons Selected: 0")
        df_results = pd.DataFrame(columns=["crop_idx", "class_name", "confidence", "x1", "y1", "x2", "y2"])

        for idx, (x, y, w, h) in enumerate(self.crop_rects):
            crop_img = self.loaded_image[y:y+h, x:x+w]
            # Use the minimum global threshold for the initial prediction
            results = self.model.predict(source=crop_img, imgsz=(256), conf=min_global_conf, verbose=False)
            
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                scores = r.boxes.conf.cpu().numpy()
                masks = r.masks.data.cpu().numpy() if r.masks is not None else []

                for i, cls_id in enumerate(class_ids):
                    # Safely get class name and confidence
                    cls_name = r.names.get(cls_id, "Unknown")
                    conf = float(scores[i])
                    
                    # Apply class-specific confidence threshold
                    threshold = self.class_confidence_thresholds.get(cls_name, 0.25)
                    if conf < threshold:
                        # Check if all boxes fail their class thresholds
                        if all(float(scores[j]) < self.class_confidence_thresholds.get(r.names.get(class_ids[j], "Unknown"), 0.25)
                               for j in range(len(class_ids))):
                            # Force as Normal if nothing passed
                            cls_name = "Normal"
                            conf = max(conf, 0.01)  # optional minimal confidence
                        else:
                            continue

                    # ----------------------------------------------------

                    if len(masks) > 0:
                        mask = masks[i].astype(np.uint8)
                        if mask.shape != crop_img.shape[:2]:
                            mask = cv2.resize(mask, (crop_img.shape[1], crop_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for cnt in contours:
                            cnt_area = cv2.contourArea(cnt)
                            # --- USE MIN AREA SLIDER VALUE HERE ---
                            if cnt_area < min_area_threshold:
                                continue
                            # -----------------------------
                                
                            cnt[:,0,0] += x
                            cnt[:,0,1] += y
                            new_poly = {"class": cls_name, "points": cnt, "confidence": conf}
                            
                            new_box = cv2.boundingRect(cnt)
                            new_box = (new_box[0], new_box[1], new_box[0]+new_box[2], new_box[1]+new_box[3])

                            merge = False
                            for existing_poly in self.polygons:
                                ex_box = cv2.boundingRect(existing_poly["points"])
                                ex_box = (ex_box[0], ex_box[1], ex_box[0]+ex_box[2], ex_box[1]+ex_box[3])
                                iou = self.bb_intersection_over_union(new_box, ex_box)
                                    
                                # --- USE IOU SLIDER VALUE HERE ---
                                if iou > iou_threshold: 
                                    if new_poly["confidence"] > existing_poly["confidence"]:
                                        existing_poly.update(new_poly)
                                    merge = True
                                    break
                            if not merge:
                                self.polygons.append(new_poly)






                    if len(boxes) > 0:
                        bx, by, bx2, by2 = map(int, boxes[i])
                        df_results.loc[len(df_results)] = [idx, cls_name, conf, x+bx, y+by, x+bx2, y+by2]

        # TABLE UPDATE WITH POLYGON REFERENCE
        self.results_table.blockSignals(True)
        self.results_table.setRowCount(0)
        for row_idx, poly in enumerate(self.polygons):
            self.results_table.insertRow(row_idx)
            combo = NoScrollComboBox()
            combo.addItems(CLASS_NAMES)
            combo.setCurrentText(poly["class"])
            combo.polygon_ref = poly
            combo.currentTextChanged.connect(lambda val, c=combo: self.combo_class_changed(c, val))
            self.results_table.setCellWidget(row_idx, 0, combo)
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(f"{poly['confidence']:.2f}"))
        self.results_table.blockSignals(False)

        out_folder = "results"
        os.makedirs(out_folder, exist_ok=True)
        excel_path = os.path.join(out_folder, "detection_results.xlsx")
        if not df_results.empty:
            df_results.to_excel(excel_path, index=False)
            QMessageBox.information(self, "Analyze Done", f"Results saved to {excel_path}")
        else:
            QMessageBox.warning(self, "Analyze Done", "No objects detected. No results saved.")

        self.apply_filter(self.filter_combo.currentText())
        self.update_display()
        self.update_summary_counts()


    def combo_class_changed(self, combo, new_class):
        poly = getattr(combo, "polygon_ref", None)
        if poly is not None:
            poly["class"] = new_class
            try:
                row = self.results_table.indexAt(combo.pos()).row()
                if row in self.lasso_selected_polygons:
                     self.lasso_selected_polygons.remove(row)
                     self.selected_count_label.setText(f"Polygons Selected: {len(self.lasso_selected_polygons)}")
            except:
                pass
                
            self.update_display()
            self.update_summary_counts()
            self.apply_filter(self.filter_combo.currentText())

    def apply_filter(self, selected_class):
        for row in range(self.results_table.rowCount()):
            widget = self.results_table.cellWidget(row, 0)
            cls = widget.currentText() if widget else ""
            show = True
            if selected_class != "All Classes" and cls != selected_class:
                show = False
            self.results_table.setRowHidden(row, not show)

    def table_selection_changed(self):
        """Clears lasso selection when a table item is selected."""
        selected_ranges = self.results_table.selectedRanges()
        if selected_ranges and self.lasso_selected_polygons:
            self.clear_lasso_selection()
            
    def update_summary_counts(self):
        """Calculates and updates the count for each class, including the manual 'Other' count."""
        counts = {cls:0 for cls in CLASS_NAMES}
        for poly in self.polygons:
            if poly["class"] in counts:
                counts[poly["class"]] += 1
                
        # 1. Update individual class labels (based on detection)
        for cls, lbl in self.summary_labels.items():
            lbl.setText(f"{cls}: {counts[cls]}")
            
        # 2. Get manual 'Other' count
        try:
            other_count = int(self.other_count_input.text())
        except ValueError:
            other_count = 0
            
        # 3. Calculate new total: Normal + Dicentric + Other (manual)
        total = counts.get("Normal", 0) + counts.get("Dicentric", 0) + other_count
        
        # 4. Update total label
        self.total_label.setText(f"Total Normal + Dicentric + Other: {total}")

    def update_display(self):
        if self.loaded_image is None:
            return
    
        display_img = self.loaded_image.copy()
        thickness = self.poly_thickness_slider.value()
    
        # Draw crops
        if self.show_crop_cb.isChecked():
            for x, y, w, h in self.crop_rects:
                color_bgr = (self.crop_color[2], self.crop_color[1], self.crop_color[0])
                cv2.rectangle(display_img, (x, y), (x+w, y+h), color_bgr, 2)
    
        # Draw polygons
        for idx, poly in enumerate(self.polygons):
            cls_name = poly["class"]
            
            if not self.class_settings[cls_name]["checkbox"].isChecked():
                continue
            
            pts = poly["points"]
            
            # Lasso selection overrides class color
            if idx in self.lasso_selected_polygons:
                r, g, b = self.image_label.highlight_color
            else:
                r, g, b = self.class_settings[cls_name]["color"]

            color_bgr = (b, g, r)
            cv2.polylines(display_img, [pts], True, color_bgr, thickness)
            
        # Draw Lasso path
        if self.image_label.is_drawing_lasso and len(self.image_label.lasso_points) > 1:
            lasso_pts = np.array(self.image_label.lasso_points, dtype=np.int32)
            r, g, b = self.image_label.lasso_color
            lasso_color_bgr = (b, g, r)
            cv2.polylines(display_img, [lasso_pts], False, lasso_color_bgr, 2)
    
        qimg = QImage(display_img.data, display_img.shape[1], display_img.shape[0],
                      display_img.strides[0], QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))


    def check_polygon_click(self, pos):
        if self.loaded_image is None:
            return
            
        if self.lasso_selected_polygons:
            self.clear_lasso_selection()
            
        x = int(pos.x() / self.image_label.zoom_factor)
        y = int(pos.y() / self.image_label.zoom_factor)
    
        for row, poly in reversed(list(enumerate(self.polygons))):
            cls_name = poly["class"]
            if not self.class_settings[cls_name]["checkbox"].isChecked():
                continue
    
            if cv2.pointPolygonTest(poly["points"], (x, y), False) >= 0:
                for table_row in range(self.results_table.rowCount()):
                    widget = self.results_table.cellWidget(table_row, 0)
                    if getattr(widget, "polygon_ref", None) is poly:
                        self.results_table.selectRow(table_row)
                        self.results_table.scrollToItem(self.results_table.item(table_row, 1))
                        break
                break

# -------------------- RUN --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DCA_GUI()
    gui.show()
    sys.exit(app.exec_())