import sys
import os
import re
import json
import cv2
import numpy as np
from collections import defaultdict
from PyQt5.QtWidgets import (
    QApplication, QWidget, QListWidget, QVBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSplitter, QPushButton,
    QComboBox, QMessageBox, QLabel, QFrame, QHBoxLayout, QLineEdit, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt

# === SETTINGS ===
TM_THRESHOLD = 0.6
LABEL_COLOR_RGB = (128, 0, 128) # Purple (R, G, B) - for OpenCV it will be (B, G, R)

# --- Classes ---
CLASS_NAMES = {1: "Normal", 2: "Dicentric", 3: "Irrelevant", 4: "Consult_Expert"}
CLASS_COLORS = {
    1: (0, 255, 0),       # Normal → Green
    2: (255, 0, 0),       # Dicentric → Red
    3: (0, 0, 255),       # Irrelevant → Blue
    4: (255, 255, 0),    # Consult_Expert → Yellow
}

KEY_CLASS_MAP = {Qt.Key_N:1, Qt.Key_D:2, Qt.Key_I:3, Qt.Key_C:4}

# --- Helper functions ---
def cropped_to_original_filename(cropped_filename, original_dir):
    base, _ = os.path.splitext(cropped_filename)
    base = re.sub(r'_chromosome_\d+$', '', base)
    for f in os.listdir(original_dir):
        name, _ = os.path.splitext(f)
        if name == base:
            return f
    return None

def extract_chromosome_number(cropped_filename):
    """Extracts the chromosome number from a filename like 'Case_X_chromosome_12.png'."""
    match = re.search(r'_chromosome_(\d+)', cropped_filename)
    if match:
        return int(match.group(1))
    return None # Or 0, or handle as an error/missing

def find_crop_location(original, crop):
    # Ensure both images are loaded (sometimes OpenCV fails silently)
    if original is None or crop is None:
        return None
        
    # Check if images are grayscale or have different channels
    if original.shape[:2] != crop.shape[:2]:
        # Convert to grayscale for robust template matching if channels differ
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        # Use the image directly if they match (e.g., both are already grayscale or BGR)
        # We ensure they are 2D/grayscale for cv2.matchTemplate
        if len(original.shape) == 3:
             original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
             crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
             original_gray = original
             crop_gray = crop
        
    res = cv2.matchTemplate(original_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val < TM_THRESHOLD:
        return None
    return max_loc

# --- Image Viewer with Zoom/Pan ---
class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__()
        self.parent_widget = parent
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.pixmap_item = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def set_image(self, pixmap, reset_transform=True):
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.setSceneRect(self.pixmap_item.boundingRect())
        if reset_transform:
            self.resetTransform()

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor,factor)

    def mousePressEvent(self,event):
        if self.parent_widget and event.button()==Qt.LeftButton:
            pos=self.mapToScene(event.pos())
            self.parent_widget.check_polygon_click(pos.x(),pos.y())
        super().mousePressEvent(event)

# --- Main GUI ---
class ImageBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chromosome Viewer GUI")
        self.setGeometry(50,50,1400,900)

        # Directories
        # NOTE: Replace these paths with your actual paths
        self.original_dir = r"C:\Users\mjiafeng.DSONET\Desktop\SD6106 Capstone Project_2 (C)_best\1b. Renamed_COCO_JSON_checked\original_image"
        self.crop_dir = r"C:\Users\mjiafeng.DSONET\Desktop\SD6106 Capstone Project_2 (C)_best\1b. Renamed_COCO_JSON_checked\chromsome_segment"
        self.coco_dir = r"C:\Users\mjiafeng.DSONET\Desktop\SD6106 Capstone Project_2 (C)_best\1b. Renamed_COCO_JSON_checked\cocojson"
        self.output_dir = r"C:\Users\mjiafeng.DSONET\Desktop\SD6106 Capstone Project_2 (C)_best\1b. Renamed_COCO_JSON_checked\checked_cocojson"
        self.remark_dir = os.path.join(self.output_dir,"remarks") 
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.remark_dir, exist_ok=True)

        # Layout
        main_layout=QVBoxLayout()
        self.setLayout(main_layout)
        splitter=QSplitter(Qt.Horizontal)

        left_panel=QVBoxLayout()
        self.list_widget=QListWidget()
        self.list_widget.itemClicked.connect(self.display_image)
        
        # --- Drawing and Display Controls ---
        self.show_button=QPushButton("Draw Polygons")
        self.show_button.clicked.connect(self.show_polygons)
        
        self.toggle_button=QPushButton("Turn Polygons Off")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.clicked.connect(self.toggle_polygons)
        
        # New Checkboxes for Toggling Labels
        self.show_crop_num_cb = QCheckBox("Show Chromosome Number")
        self.show_crop_num_cb.stateChanged.connect(self.redraw_polygons)
        self.show_ann_id_cb = QCheckBox("Show Annotation ID (Ann0, Ann1...)")
        self.show_ann_id_cb.stateChanged.connect(self.redraw_polygons)
        
        self.class_combo=QComboBox()
        for k,name in CLASS_NAMES.items():
            self.class_combo.addItem(name,k)
        self.class_combo.currentIndexChanged.connect(self.change_polygon_class)
        self.save_button=QPushButton("Save COCO JSON")
        self.save_button.clicked.connect(self.save_coco_json)

        # Remark input
        self.remark_input = QLineEdit()
        self.remark_input.setPlaceholderText("Enter remark for selected polygon")
        self.save_remark_button = QPushButton("Save Remark")
        self.save_remark_button.clicked.connect(self.save_remark)

        # --- Add graphical color legend ---
        self.legend_container = QWidget()
        legend_layout = QVBoxLayout()
        for k, name in CLASS_NAMES.items():
            row = QHBoxLayout()
            color_box = QFrame()
            color_box.setFixedSize(20, 20)
            r, g, b = CLASS_COLORS[k]
            color_box.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid black;")
            label = QLabel(name)
            row.addWidget(color_box)
            row.addWidget(label)
            row.addStretch()
            legend_layout.addLayout(row)
        self.legend_container.setLayout(legend_layout)

        # Add widgets to left panel
        left_panel.addWidget(self.list_widget)
        left_panel.addWidget(self.show_button)
        left_panel.addWidget(self.toggle_button)
        left_panel.addWidget(self.show_crop_num_cb) # New checkbox
        left_panel.addWidget(self.show_ann_id_cb) # New checkbox
        left_panel.addWidget(self.class_combo)
        left_panel.addWidget(self.save_button)
        left_panel.addWidget(self.remark_input)
        left_panel.addWidget(self.save_remark_button)
        left_panel.addWidget(self.legend_container)
        left_container=QWidget()
        left_container.setLayout(left_panel)
        splitter.addWidget(left_container)

        self.image_viewer=ImageViewer(self)
        splitter.addWidget(self.image_viewer)
        splitter.setSizes([300,1000])
        main_layout.addWidget(splitter)

        # Variables
        self.current_image_name=None
        self.original_cv_img=None
        self.original_pixmap=None
        self.pixmap_with_polygons=None
        self.current_polygons=[]
        self.selected_polygon_index=None

        self.load_images()
        self.annotations_by_original=self.collect_annotations()

    def keyPressEvent(self,event):
        if self.selected_polygon_index is not None and event.key() in KEY_CLASS_MAP:
            self.current_polygons[self.selected_polygon_index]['class_id']=KEY_CLASS_MAP[event.key()]
            index=self.class_combo.findData(KEY_CLASS_MAP[event.key()])
            if index!=-1:
                self.class_combo.setCurrentIndex(index)
            self.redraw_polygons()

    def load_images(self):
        self.image_files=[f for f in os.listdir(self.original_dir)
                          if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        self.list_widget.addItems(self.image_files)

    def display_image(self,item):
        self.current_image_name=item.text()
        img_path=os.path.join(self.original_dir,item.text())
        self.original_cv_img=cv2.imread(img_path)
        self.original_pixmap=QPixmap(img_path)
        self.pixmap_with_polygons=None
        self.current_polygons=[]
        self.selected_polygon_index=None
        self.image_viewer.set_image(self.original_pixmap)
        self.toggle_button.setChecked(True)
        self.toggle_button.setText("Turn Polygons Off")
        self.show_crop_num_cb.setChecked(False)
        self.show_ann_id_cb.setChecked(False)

    def collect_annotations(self):
        annotations_by_original=defaultdict(list)
        for json_file in os.listdir(self.coco_dir):
            if not json_file.endswith(".json"):
                continue
            
            json_path = os.path.join(self.coco_dir,json_file)
            try:
                with open(json_path,"r") as f:
                    data=json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {json_file}")
                continue

            if "images" not in data or not data["images"]:
                continue
            
            # The JSON file name base is the crop image name base
            cropped_filename=data["images"][0]["file_name"] 
            original_filename=cropped_to_original_filename(cropped_filename,self.original_dir)
            if original_filename is None:
                continue
            
            # Extract chromosome number from the cropped filename
            chromosome_number = extract_chromosome_number(cropped_filename)

            for ann in data.get("annotations",[]):
                category_id=ann.get("category_id",1)
                
                # COCO segmentation is a list of polygons (list of points)
                for seg in ann.get("segmentation",[]):
                    annotations_by_original[original_filename].append({
                        "seg":seg,
                        "crop_filename":cropped_filename,
                        "class_id":category_id,
                        "annotation_id": ann.get("id"), # Store the unique annotation ID
                        "chromosome_number": chromosome_number # Store the extracted chromosome number
                    })
        print(f"Collected annotations for {len(annotations_by_original)} original images.")
        return annotations_by_original

    def show_polygons(self):
        if self.current_image_name is None or self.original_cv_img is None:
            return
        
        # If polygons are already drawn, just display the existing pixmap
        if self.pixmap_with_polygons is not None:
            self.image_viewer.set_image(self.pixmap_with_polygons,reset_transform=False)
            self.toggle_button.setText("Turn Polygons Off")
            return

        # Redraw the polygons from scratch if not already done
        self.current_polygons=[]
        annotations=self.annotations_by_original.get(self.current_image_name,[])

        for ann in annotations:
            seg=ann['seg']
            crop_filename=ann['crop_filename']
            class_id=ann['class_id']
            annotation_id = ann['annotation_id']
            chromosome_number = ann['chromosome_number']

            crop_path=os.path.join(self.crop_dir,crop_filename)
            crop_img=cv2.imread(crop_path)
            
            if crop_img is None:
                continue

            loc=find_crop_location(self.original_cv_img,crop_img)
            if loc is None:
                continue 
                
            dx,dy=loc
            
            # Reshape segment points (x1, y1, x2, y2, ...) into [[x1, y1], [x2, y2], ...]
            pts=np.array(seg).reshape(-1,2) 
            pts[:,0]+=dx # Offset X
            pts[:,1]+=dy # Offset Y
            pts=pts.astype(np.int32)

            self.current_polygons.append({
                "seg":seg,
                "crop_filename":crop_filename,
                "class_id":class_id,
                "annotation_id":annotation_id,
                "dx":dx,
                "dy":dy,
                "pts":pts,
                "chromosome_number": chromosome_number # Store the chromosome number here
            })
            
        # Draw the collected polygons
        self.redraw_polygons()
        self.toggle_button.setText("Turn Polygons Off")

    def toggle_polygons(self):
        if self.original_cv_img is None:
            return
        if self.toggle_button.isChecked():
            self.toggle_button.setText("Turn Polygons Off")
            if self.pixmap_with_polygons is None:
                self.show_polygons()
            else:
                self.image_viewer.set_image(self.pixmap_with_polygons,reset_transform=False)
        else:
            self.toggle_button.setText("Turn Polygons On")
            self.image_viewer.set_image(self.original_pixmap,reset_transform=False)

    def check_polygon_click(self,x,y):
        if not self.current_polygons or not self.toggle_button.isChecked():
            return
        clicked_index=None
        # Check against all polygons, starting from the last (often drawn on top)
        for i in range(len(self.current_polygons) - 1, -1, -1):
            poly = self.current_polygons[i]
            pts=poly['pts']
            # cv2.pointPolygonTest returns > 0 if inside, 0 if on edge, < 0 if outside
            if cv2.pointPolygonTest(pts,(x,y),False)>=0:
                clicked_index=i
                break
                
        self.selected_polygon_index=clicked_index
        self.redraw_polygons()
        if clicked_index is not None:
            class_id=self.current_polygons[clicked_index]['class_id']
            index=self.class_combo.findData(class_id)
            if index!=-1:
                self.class_combo.setCurrentIndex(index)

    def redraw_polygons(self):
        if self.original_cv_img is None or not self.current_polygons:
            # Revert to original image if no polygons exist or if polygons are toggled off
            if self.original_pixmap:
                 self.image_viewer.set_image(self.original_pixmap, reset_transform=False)
            return
            
        canvas=self.original_cv_img.copy()
        
        show_chromosome_number = self.show_crop_num_cb.isChecked()
        show_ann_id = self.show_ann_id_cb.isChecked()
        
        # OpenCV uses BGR
        label_color_bgr = (LABEL_COLOR_RGB[2], LABEL_COLOR_RGB[1], LABEL_COLOR_RGB[0]) 

        for i,poly in enumerate(self.current_polygons):
            pts=poly['pts']
            class_id=poly['class_id']
            ann_id = poly.get('annotation_id')
            chromosome_number = poly.get('chromosome_number')

            r,g,b = CLASS_COLORS.get(class_id,(0,255,0))
            color = (b,g,r) # convert RGB → BGR for OpenCV
            thickness=2 if i==self.selected_polygon_index else 1
            
            # Draw the polygon line
            cv2.polylines(canvas,[pts],True,color,thickness)
            
            # --- Draw Labels (Chromosome Number and/or AnnID) ---
            if show_chromosome_number or show_ann_id:
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    text_parts = []
                    
                    if show_chromosome_number:
                        if chromosome_number is not None:
                            text_parts.append(f"Chr{chromosome_number}")
                        else:
                            text_parts.append("Chr?") # Fallback if number not found
                            
                    if show_ann_id:
                        # Logic to format AnnID (Ann0, Ann1, AnnNoID)
                        if ann_id is None:
                            text_parts.append("AnnNoID")
                        elif ann_id == 0:
                            text_parts.append("Ann0")
                        else:
                            text_parts.append(f"Ann{ann_id}")
                            
                    final_text = " | ".join(text_parts)
                    
                    # Draw the text label in purple
                    cv2.putText(canvas, final_text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color_bgr, 1)

        canvas_rgb=cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)
        h,w,ch=canvas_rgb.shape
        bytes_per_line=ch*w
        qimg=QImage(canvas_rgb.data,w,h,bytes_per_line,QImage.Format_RGB888)
        self.pixmap_with_polygons=QPixmap.fromImage(qimg)
        self.image_viewer.set_image(self.pixmap_with_polygons,reset_transform=False)

    def change_polygon_class(self):
        if self.selected_polygon_index is None:
            return
        class_id=self.class_combo.currentData()
        self.current_polygons[self.selected_polygon_index]['class_id']=class_id
        self.redraw_polygons()

    def save_coco_json(self):
        if self.selected_polygon_index is None:
            QMessageBox.warning(self,"No selection","Please select a polygon first!")
            return
        
        poly = self.current_polygons[self.selected_polygon_index]
        crop_filename = poly['crop_filename']
        class_id = poly['class_id']
        annotation_id = poly.get('annotation_id')
        
        if annotation_id is None:
            QMessageBox.critical(self,"Error","Annotation ID is missing. Cannot save reliably.")
            return

        base_name, _ = os.path.splitext(crop_filename)
        output_path = os.path.join(self.output_dir, base_name + ".json")
        original_json_path = os.path.join(self.coco_dir, base_name + ".json")
        
        data = None
        
        # 1. Try to load the already modified JSON from the output directory
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                data = json.load(f)
        # 2. If not found in output, load the original JSON
        elif os.path.exists(original_json_path):
            with open(original_json_path, "r") as f:
                data = json.load(f)
        else:
            QMessageBox.critical(self, "Error", f"COCO JSON not found for crop: {crop_filename}")
            return

        # 3. Find the annotation by its unique ID and update its category_id
        found = False
        if "annotations" in data:
            for ann in data["annotations"]:
                if ann.get("id") == annotation_id:
                    ann["category_id"] = class_id
                    found = True
                    break

        if not found:
            QMessageBox.warning(self,"Update Error",f"Could not find annotation with ID {annotation_id} in {crop_filename}'s JSON. No changes saved.")
            return
        
        # 4. Save the updated JSON to the output directory
        try:
            with open(output_path,"w") as f:
                json.dump(data,f,indent=2)
            QMessageBox.information(self,"Saved",f"COCO JSON updated for Annotation ID {annotation_id} and saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to write JSON file: {e}")

    def save_remark(self):
        if self.selected_polygon_index is None:
            QMessageBox.warning(self,"No selection","Please select a polygon first!")
            return
        remark_text = self.remark_input.text().strip()
        if not remark_text:
            QMessageBox.warning(self,"Empty remark","Please enter a remark text!")
            return
            
        poly = self.current_polygons[self.selected_polygon_index]
        base_name=os.path.splitext(poly['crop_filename'])[0]
        annotation_id = poly.get('annotation_id')
        
        # Determine the string for the filename
        if annotation_id is None:
            ann_id_str = "NoID"
        elif annotation_id == 0:
            ann_id_str = "0"
        else:
            ann_id_str = str(annotation_id)
            
        # Base name for the remark file, uniquely identifying the crop and the annotation ID
        remark_base_name = f"{base_name}_ann{ann_id_str}"
            
        remark_file=os.path.join(self.remark_dir, remark_base_name+".txt")
            
        # Handle multiple remarks for the same annotation by adding a counter
        count=1
        final_remark_file=remark_file
        while os.path.exists(final_remark_file):
            # If the file exists, try appending a counter before the extension
            final_remark_file=os.path.join(self.remark_dir,f"{remark_base_name}_{count}.txt")
            count+=1
            
        try:
            with open(final_remark_file,"w") as f:
                f.write(remark_text)
            QMessageBox.information(self,"Saved",f"Remark saved for Ann ID {ann_id_str} at:\n{final_remark_file}")
            self.remark_input.clear()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to write remark file: {e}")


if __name__=="__main__":
    app=QApplication(sys.argv)
    window=ImageBrowser()
    window.show()
    sys.exit(app.exec_())