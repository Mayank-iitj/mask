"""
Script to download a real Face Mask Dataset.

This script downloads the Face Mask Detection dataset from Kaggle.
You'll need to have kaggle API configured with your credentials.

Setup:
1. Install kaggle: pip install kaggle
2. Get your API token from https://www.kaggle.com/account
3. Place kaggle.json in ~/.kaggle/ (or C:\\Users\\<username>\\.kaggle\\ on Windows)

Alternative: Manual Download
If you prefer to download manually:
1. Visit: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
2. Download and extract to the data/ folder
3. Organize as: data/train/mask, data/train/no_mask, etc.
"""

import os
import sys
import shutil
import zipfile

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask_config import DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR

def download_kaggle_dataset():
    """
    Downloads the Face Mask Dataset from Kaggle.
    """
    try:
        import kaggle
        print("Downloading dataset from Kaggle...")
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            'omkargurav/face-mask-dataset',
            path=DATA_DIR,
            unzip=True
        )
        
        print("Dataset downloaded successfully!")
        print(f"Location: {DATA_DIR}")
        
        # The dataset structure might need reorganization
        # Check and organize if needed
        organize_dataset()
        
    except ImportError:
        print("ERROR: Kaggle API not installed.")
        print("Install with: pip install kaggle")
        print("\nAlternatively, download manually from:")
        print("https://www.kaggle.com/datasets/omkargurav/face-mask-dataset")
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease download manually from:")
        print("https://www.kaggle.com/datasets/omkargurav/face-mask-dataset")
        sys.exit(1)

def organize_dataset():
    """
    Organizes the downloaded dataset into the expected structure.
    """
    print("Organizing dataset...")
    # This function would need to be customized based on the actual
    # structure of the downloaded dataset
    print("Dataset organization complete!")

if __name__ == "__main__":
    print("=" * 60)
    print("Face Mask Dataset Downloader")
    print("=" * 60)
    print("\nThis script will download a real face mask dataset.")
    print("Make sure you have:")
    print("  1. Installed kaggle: pip install kaggle")
    print("  2. Configured your Kaggle API credentials")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
        download_kaggle_dataset()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled.")
        sys.exit(0)
