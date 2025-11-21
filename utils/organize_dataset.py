import os
import shutil
import sys
from pathlib import Path

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask_config import TRAIN_DIR, VAL_DIR, TEST_DIR

# Source directories
SOURCE_WITH_MASK = r"D:\extracted_dataset\data\with_mask"
SOURCE_WITHOUT_MASK = r"D:\extracted_dataset\data\without_mask"

def organize_dataset():
    """
    Organizes the extracted dataset into train/val/test splits.
    Uses 70% train, 15% val, 15% test split.
    """
    print("Organizing dataset...")
    
    # Create directories
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(os.path.join(split_dir, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'no_mask'), exist_ok=True)
    
    # Process with_mask images
    with_mask_files = list(Path(SOURCE_WITH_MASK).glob('*.png')) + list(Path(SOURCE_WITH_MASK).glob('*.jpg'))
    total_with_mask = len(with_mask_files)
    
    train_split = int(0.7 * total_with_mask)
    val_split = int(0.85 * total_with_mask)
    
    print(f"Found {total_with_mask} images with mask")
    
    for i, img_path in enumerate(with_mask_files):
        if i < train_split:
            dest = os.path.join(TRAIN_DIR, 'mask', img_path.name)
        elif i < val_split:
            dest = os.path.join(VAL_DIR, 'mask', img_path.name)
        else:
            dest = os.path.join(TEST_DIR, 'mask', img_path.name)
        
        shutil.copy2(str(img_path), dest)
    
    print(f"Copied {total_with_mask} 'with mask' images")
    
    # Process without_mask images
    without_mask_files = list(Path(SOURCE_WITHOUT_MASK).glob('*.png')) + list(Path(SOURCE_WITHOUT_MASK).glob('*.jpg'))
    total_without_mask = len(without_mask_files)
    
    train_split = int(0.7 * total_without_mask)
    val_split = int(0.85 * total_without_mask)
    
    print(f"Found {total_without_mask} images without mask")
    
    for i, img_path in enumerate(without_mask_files):
        if i < train_split:
            dest = os.path.join(TRAIN_DIR, 'no_mask', img_path.name)
        elif i < val_split:
            dest = os.path.join(VAL_DIR, 'no_mask', img_path.name)
        else:
            dest = os.path.join(TEST_DIR, 'no_mask', img_path.name)
        
        shutil.copy2(str(img_path), dest)
    
    print(f"Copied {total_without_mask} 'without mask' images")
    print("\nDataset organization complete!")
    print(f"Train: {train_split * 2} images")
    print(f"Val: {(val_split - train_split) * 2} images")
    print(f"Test: {(total_with_mask - val_split) + (total_without_mask - val_split)} images")

if __name__ == "__main__":
    organize_dataset()
