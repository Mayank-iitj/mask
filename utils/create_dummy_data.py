import sys
import os
# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from mask_config import TRAIN_DIR, VAL_DIR, TEST_DIR, CLASSES

def create_dummy_images(base_dir, num_images=10):
    for class_name in CLASSES:
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(num_images):
            # Create a random image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(class_dir, f"dummy_{i}.jpg"))
            
    print(f"Created {num_images} dummy images per class in {base_dir}")

if __name__ == "__main__":
    print("Generating dummy dataset...")
    create_dummy_images(TRAIN_DIR, num_images=20)
    create_dummy_images(VAL_DIR, num_images=5)
    create_dummy_images(TEST_DIR, num_images=5)
    print("Done!")
