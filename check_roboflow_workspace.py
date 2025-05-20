from roboflow import Roboflow
import os

def check_workspace():
    """Check Roboflow workspace configuration."""
    # Initialize Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    
    try:
        # Get workspace info
        workspace = rf.workspace()
        print(f"Workspace name: {workspace.name}")
        print(f"Workspace ID: {workspace.id}")
        
        # Get project info
        project = workspace.project("mhf-dataset")
        print(f"\nProject name: {project.name}")
        print(f"Project ID: {project.id}")
        
        # Check dataset info
        dataset = project.version(1)
        print(f"\nDataset version: {dataset.version}")
        print(f"Number of images: {dataset.image_count}")
        
        return True
    except Exception as e:
        print(f"Error checking workspace: {str(e)}")
        return False

if __name__ == "__main__":
    check_workspace() 