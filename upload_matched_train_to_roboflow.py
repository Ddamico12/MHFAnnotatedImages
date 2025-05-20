from roboflow import Roboflow
import os
from tqdm import tqdm

def upload_matched_train_data():
    """Upload matched training data to Roboflow."""
    # Initialize Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace().project("mhf-dataset")
    
    # Path to matched training data
    train_dir = "matched_datasets_split/train"
    
    # Get list of images
    images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))]
    
    # Upload each image with its annotation
    for img in tqdm(images, desc="Uploading matched training images"):
        img_path = os.path.join(train_dir, img)
        annotation_path = os.path.join(train_dir, img.replace('.jpg', '.xml').replace('.png', '.xml'))
        
        if os.path.exists(annotation_path):
            project.upload(
                image_path=img_path,
                annotation_path=annotation_path,
                split="train"
            )
        else:
            print(f"Warning: No annotation found for {img}")

if __name__ == "__main__":
    upload_matched_train_data() 