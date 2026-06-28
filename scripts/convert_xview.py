import json
import os
from PIL import Image

# Suppress PIL decompression bomb warnings for massive xView TIFFs
Image.MAX_IMAGE_PIXELS = None 

def main():
    geojson_path = "/home/o/uav_pipeline/datasets/xview_raw/xView_train.geojson"
    coco_out_path = "/home/o/uav_pipeline/datasets/xview_raw/instances_train.json"
    raw_images_dir = "/home/o/uav_pipeline/datasets/xview_raw/train_images"

    print("Loading massive xView GeoJSON... (This will take a minute)")
    try:
        with open(geojson_path, 'r') as f:
            geojson = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Could not find {geojson_path}. Make sure the file is there.")
        return

    # Maps xView's granular classes to our 6 pipeline target classes
    XVIEW_TO_TARGET = {
        17: 2, 18: 2, 19: 4, 20: 5, 21: 5, 23: 5, 24: 5, 
        25: 5, 26: 5, 28: 5, 29: 5, 32: 5, 82: 0, 83: 3
    }

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "bicycle"},
            {"id": 2, "name": "car"},
            {"id": 3, "name": "motorcycle"},
            {"id": 4, "name": "bus"},
            {"id": 5, "name": "truck"}
        ]
    }

    features = geojson.get("features", [])
    print(f"Found {len(features)} total annotations in xView.")

    image_name_to_id = {}
    img_id_counter = 1
    ann_id_counter = 1

    print("Filtering classes, reading image dimensions, and converting bounding boxes...")
    for feat in features:
        props = feat.get("properties", {})
        type_id = props.get("type_id")

        if type_id not in XVIEW_TO_TARGET:
            continue

        target_class_id = XVIEW_TO_TARGET[type_id]

        image_name = props.get("image_id")
        if not image_name:
            continue

        # Register unique image AND dynamically get its dimensions
        if image_name not in image_name_to_id:
            image_path = os.path.join(raw_images_dir, image_name)
            
            # Verify image exists on disk
            if not os.path.exists(image_path):
                continue

            # Open image header to extract width and height
            try:
                with Image.open(image_path) as img:
                    w, h = img.size
            except Exception as e:
                print(f"Warning: Could not read {image_path}: {e}")
                continue

            image_name_to_id[image_name] = img_id_counter
            coco["images"].append({
                "id": img_id_counter,
                "file_name": image_name,
                "width": w,      # <-- THIS FIXES THE SAHI ERROR
                "height": h      # <-- THIS FIXES THE SAHI ERROR
            })
            img_id_counter += 1

        # Retrieve the ID we just created (or an existing one)
        image_id = image_name_to_id.get(image_name)
        if not image_id:
            continue

        bounds_str = props.get("bounds_imcoords")
        if not bounds_str:
            continue

        try:
            xmin, ymin, xmax, ymax = map(int, bounds_str.split(","))
        except ValueError:
            continue

        width = xmax - xmin
        height = ymax - ymin

        if width <= 0 or height <= 0:
            continue

        coco["annotations"].append({
            "id": ann_id_counter,
            "image_id": image_id,
            "category_id": target_class_id,
            "bbox": [xmin, ymin, width, height],
            "area": width * height,
            "iscrowd": 0
        })
        ann_id_counter += 1

    print(f"Kept {len(coco['annotations'])} relevant vehicle/person annotations.")
    print(f"Across {len(coco['images'])} images.")

    print(f"Saving to {coco_out_path}...")
    with open(coco_out_path, 'w') as f:
        json.dump(coco, f)
    
    print("Success! You can now run prepare_xview.py.")

if __name__ == "__main__":
    main()