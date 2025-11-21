import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Configuration
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model Configuration
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'mask_detector.h5')
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# Classes
CLASSES = ['mask', 'no_mask']
NUM_CLASSES = len(CLASSES)

# Detection Configuration
MIN_CONFIDENCE = 0.5
