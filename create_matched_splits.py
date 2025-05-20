import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def create_matched_splits():
    """Create matched splits for training data."""
    # Load dataset
    df = pd.read_csv('annotations.csv')
    
    # Split data into train and temp
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # Split temp into validation and test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create directories
    base_dir = Path("matched_datasets_split")
    for split in ['train', 'validation', 'test']:
        (base_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Copy files to respective directories
    for split, df_split in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        for _, row in df_split.iterrows():
            img_name = row['image_name']
            src_img = Path(f"ellipse_split/{split}/images/{img_name}")
            src_xml = Path(f"ellipse_split/{split}/labels/{img_name.replace('.png', '.xml')}")
            
            if src_img.exists() and src_xml.exists():
                shutil.copy2(src_img, base_dir / split / img_name)
                shutil.copy2(src_xml, base_dir / split / img_name.replace('.png', '.xml'))
    
    print("Created matched splits in matched_datasets_split/")

if __name__ == "__main__":
    create_matched_splits() 