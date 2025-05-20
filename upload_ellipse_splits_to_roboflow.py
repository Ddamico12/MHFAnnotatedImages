import os
from pathlib import Path
from roboflow import Roboflow
import time
import urllib3

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Roboflow configuration
API_KEY = "YOUR_API_KEY"
WORKSPACE = "dannys-workspace-qn4xc"
PROJECTS = {
    "train": "mhfproject-train",
    "validation": "mhfproject-validation",
    "test": "mhfproject-test"
}

base_dir = Path("ellipse_split")

rf = Roboflow(api_key=API_KEY)

for split, project_name in PROJECTS.items():
    images_dir = base_dir / split / "images"
    labels_dir = base_dir / split / "labels"
    project = rf.workspace(WORKSPACE).project(project_name)
    print(f"Uploading {split} set to project {project_name}...")

    for xml_file in labels_dir.glob("*.xml"):
        image_name = xml_file.stem + ".png"
        img_path = images_dir / image_name
        if not img_path.exists():
            print(f"Image not found for annotation: {xml_file}")
            continue
        try:
            project.upload(
                image_path=str(img_path),
                annotation_path=str(xml_file),
                batch_name=f"{split}_ellipse_upload",
                split=split,
                num_retry_uploads=3,
                tag_names=[split]
            )
            print(f"Uploaded {img_path} and {xml_file}")
        except Exception as e:
            print(f"Error uploading {img_path}: {e}")

print("All splits uploaded!") 