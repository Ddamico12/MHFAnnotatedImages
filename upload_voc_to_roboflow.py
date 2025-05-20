from roboflow import Roboflow
import os
from tqdm import tqdm

def upload_voc_annotations():
    """Upload VOC format annotations to Roboflow."""
    # Initialize Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace().project("mhf-dataset")
    
    # Path to VOC annotations
    voc_dir = "ellipse_voc_annotations"
    
    # Get list of images
    images = [f for f in os.listdir(voc_dir) if f.endswith(('.jpg', '.png'))]
    
    # Upload each image with its VOC annotation
    for img in tqdm(images, desc="Uploading VOC annotations"):
        img_path = os.path.join(voc_dir, img)
        annotation_path = os.path.join(voc_dir, img.replace('.jpg', '.xml').replace('.png', '.xml'))
        
        if os.path.exists(annotation_path):
            project.upload(
                image_path=img_path,
                annotation_path=annotation_path,
                split="train"  # Default to train split
            )
        else:
            print(f"Warning: No VOC annotation found for {img}")

if __name__ == "__main__":
    upload_voc_annotations() 