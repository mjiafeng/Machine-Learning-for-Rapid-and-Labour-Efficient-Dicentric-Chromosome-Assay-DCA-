import os
import json
import cv2

# -------------------------
# Paths
# -------------------------
input_folder = r"C:\Users\mjiafeng.DSONET\Desktop\SD6106 Capstone Project_2\1c. Augmented_COCO_JSON"
output_folder = os.path.join(input_folder, "augmented")
os.makedirs(output_folder, exist_ok=True)

# -------------------------
# Helper functions for polygon transformations
# -------------------------
def flip_vertical(seg, width, height):
    new_seg = []
    for s in seg:
        new_coords = []
        for i in range(0, len(s), 2):
            x, y = s[i], s[i+1]
            new_coords.extend([x, height - y])
        new_seg.append(new_coords)
    return new_seg

def flip_horizontal(seg, width, height):
    new_seg = []
    for s in seg:
        new_coords = []
        for i in range(0, len(s), 2):
            x, y = s[i], s[i+1]
            new_coords.extend([width - x, y])
        new_seg.append(new_coords)
    return new_seg

def rotate_90(seg, width, height):
    # 90° clockwise rotation
    new_seg = []
    for s in seg:
        new_coords = []
        for i in range(0, len(s), 2):
            x, y = s[i], s[i+1]
            new_coords.extend([height - y, x])  # correct
        new_seg.append(new_coords)
    return new_seg

def rotate_180(seg, width, height):
    new_seg = []
    for s in seg:
        new_coords = []
        for i in range(0, len(s), 2):
            x, y = s[i], s[i+1]
            new_coords.extend([width - x, height - y])
        new_seg.append(new_coords)
    return new_seg

def rotate_270(seg, width, height):
    # 270° clockwise (90° counter-clockwise)
    new_seg = []
    for s in seg:
        new_coords = []
        for i in range(0, len(s), 2):
            x, y = s[i], s[i+1]
            new_coords.extend([y, width - x])  # correct
        new_seg.append(new_coords)
    return new_seg

# -------------------------
# Load all JSONs and map image filenames to JSON paths
# -------------------------
json_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".json")]
json_map = {}

for jf in json_files:
    path = os.path.join(input_folder, jf)
    with open(path, "r") as f:
        data = json.load(f)
        image_filename = data["images"][0]["file_name"]
        json_map[image_filename] = path

# -------------------------
# Process each image
# -------------------------
for img_filename in os.listdir(input_folder):
    if img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, img_filename)

        if img_filename not in json_map:
            print(f"⚠️ No matching JSON for {img_filename}")
            continue

        # Load image and JSON
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        with open(json_map[img_filename], "r") as f:
            coco_data = json.load(f)

        # Define transformations
        transformations = {
            "flipV": flip_vertical,
            "flipH": flip_horizontal,
            "rot90": rotate_90,
            "rot180": rotate_180,
            "rot270": rotate_270
        }

        # Apply transformations
        for suffix, func in transformations.items():
            new_img_name = f"{os.path.splitext(img_filename)[0]}_{suffix}.jpg"
            new_json_name = f"{os.path.splitext(img_filename)[0]}_{suffix}.json"

            # Image transform
            if suffix == "flipV":
                new_img = cv2.flip(img, 0)
            elif suffix == "flipH":
                new_img = cv2.flip(img, 1)
            elif suffix == "rot90":
                new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif suffix == "rot180":
                new_img = cv2.rotate(img, cv2.ROTATE_180)
            elif suffix == "rot270":
                new_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Save new image
            cv2.imwrite(os.path.join(output_folder, new_img_name), new_img)

            # Transform annotations
            new_coco = json.loads(json.dumps(coco_data))  # deep copy
            new_coco["images"][0]["file_name"] = new_img_name

            # Update width/height for rotated images
            if suffix in ["rot90", "rot270"]:
                new_coco["images"][0]["width"], new_coco["images"][0]["height"] = height, width
            else:
                new_coco["images"][0]["width"], new_coco["images"][0]["height"] = width, height

            # Apply polygon transformation
            for ann in new_coco["annotations"]:
                ann["segmentation"] = func(ann["segmentation"], width, height)

            # Save new JSON
            with open(os.path.join(output_folder, new_json_name), "w") as f:
                json.dump(new_coco, f, indent=4)

            print(f"✅ Created {new_img_name} and {new_json_name} in {output_folder}")
