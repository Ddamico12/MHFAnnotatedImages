import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path

# Set your annotation directory
annotation_dir = Path("Ultrasound Fetus Dataset/Data/matched_dataset")
output_csv = "ellipse_parameters.csv"

rows = []

# Loop through all subfolders (categories)
for category in os.listdir(annotation_dir):
    category_path = annotation_dir / category
    if not category_path.is_dir():
        continue
    for fname in os.listdir(category_path):
        if fname.endswith("_Annotation.png"):
            img_path = category_path / fname
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Threshold to binary
            _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            # Fit ellipse to the largest contour
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) < 5:
                continue  # Need at least 5 points to fit ellipse
            ellipse = cv2.fitEllipse(cnt)
            (center_x, center_y), (axis_x, axis_y), angle = ellipse
            # Save parameters
            original_img = fname.replace("_Annotation.png", ".png")
            rows.append({
                "filename": original_img,
                "category": category,
                "center_x": center_x,
                "center_y": center_y,
                "axis_x": axis_x / 2,  # OpenCV returns full length, divide by 2 for radius
                "axis_y": axis_y / 2,
                "angle": angle
            })

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"Saved ellipse parameters to {output_csv}") 