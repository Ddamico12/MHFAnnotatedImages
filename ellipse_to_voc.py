import os
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2

def create_voc_xml(image_filename, output_path, class_name, bbox, image_shape):
    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = os.path.dirname(image_filename)
    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(image_filename)
    path = ET.SubElement(root, "path")
    path.text = image_filename

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(root, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(image_shape[0])
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(image_shape[1])
    depth_elem = ET.SubElement(size, "depth")
    depth_elem.text = str(image_shape[2] if len(image_shape) > 2 else 1)

    obj = ET.SubElement(root, "object")
    name = ET.SubElement(obj, "name")
    name.text = class_name
    bndbox = ET.SubElement(obj, "bndbox")
    xmin = ET.SubElement(bndbox, "xmin")
    xmin.text = str(int(bbox[0]))
    ymin = ET.SubElement(bndbox, "ymin")
    ymin.text = str(int(bbox[1]))
    xmax = ET.SubElement(bndbox, "xmax")
    xmax.text = str(int(bbox[2]))
    ymax = ET.SubElement(bndbox, "ymax")
    ymax.text = str(int(bbox[3]))

    tree = ET.ElementTree(root)
    tree.write(output_path)

csv_path = "ellipse_parameters.csv"
image_dir = Path("Ultrasound Fetus Dataset/Data/matched_dataset")
output_dir = Path("ellipse_voc_annotations")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    filename = row["filename"]
    category = row["category"]
    center_x = row["center_x"]
    center_y = row["center_y"]
    axis_x = row["axis_x"]
    axis_y = row["axis_y"]

    # Compute bounding box
    xmin = center_x - axis_x
    ymin = center_y - axis_y
    xmax = center_x + axis_x
    ymax = center_y + axis_y

    # Get image shape
    img_path = image_dir / category / filename
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read image: {img_path}")
        continue
    h, w = img.shape[:2]
    shape = (w, h, img.shape[2] if len(img.shape) > 2 else 1)

    # Clamp bbox to image size
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    xml_path = output_dir / f"{Path(filename).stem}.xml"
    create_voc_xml(str(img_path), xml_path, category, (xmin, ymin, xmax, ymax), shape)

print(f"VOC XML files saved to {output_dir}") 