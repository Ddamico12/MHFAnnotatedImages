import os
from roboflow import Roboflow
import pandas as pd
from pathlib import Path

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="rf_VmC6XcaT6DU1zeghO9CIJcuqs5w1")

# Retrieve your current workspace and project name
workspace = rf.workspace()
print("Current workspace:", workspace)

# Specify the project for upload
workspaceId = 'dannys-workspace-qn4xc'  # Using the URL identifier
projectId = 'mhfproject'  # Using the lowercase project ID from the workspace info
project = rf.workspace(workspaceId).project(projectId)

# Define paths
base_path = Path('Ultrasound Fetus Dataset/Data')
csv_path = base_path / 'matched_dataset/matched_data.csv'
matched_dataset_path = base_path / 'matched_dataset'

# Read the CSV file
df = pd.read_csv(csv_path)

# Upload images with their labels
for idx, row in df.iterrows():
    image_name = row['image_filename']
    category = row['corrected_category']  # Using the corrected category from the CSV
    
    # Construct the image path
    image_path = matched_dataset_path / category / image_name
    
    if image_path.exists():
        try:
            # Upload the image with its label and metadata
            project.upload(
                image_path=str(image_path),
                batch_name="fetal_health_dataset",
                split="train",
                num_retry_uploads=3,
                tag_names=[category],
                sequence_number=idx + 1,
                sequence_size=len(df)
            )
            print(f"Successfully uploaded {image_path}")
        except Exception as e:
            print(f"Error uploading {image_path}: {str(e)}")
    else:
        print(f"Warning: Image not found at {image_path}")

print("Upload complete!") 
