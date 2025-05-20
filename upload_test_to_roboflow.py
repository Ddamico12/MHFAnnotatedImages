from roboflow import Roboflow
import os
from tqdm import tqdm

def upload_test_data():
    """Upload test data to Roboflow."""
    # Initialize Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace().project("mhf-dataset")
    
    # Path to test data
    test_dir = "ellipse_split/test"
    
    # Get list of images
    images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
    
    # Upload each image with its annotation
    for img in tqdm(images, desc="Uploading test images"):
        img_path = os.path.join(test_dir, img)
        annotation_path = os.path.join(test_dir, img.replace('.jpg', '.xml').replace('.png', '.xml'))
        
        if os.path.exists(annotation_path):
            project.upload(
                image_path=img_path,
                annotation_path=annotation_path,
                split="test"
            )
        else:
            print(f"Warning: No annotation found for {img}")

if __name__ == "__main__":
    upload_test_data() 