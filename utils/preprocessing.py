import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from mask_config import IMAGE_SIZE

def preprocess_image(image):
    """
    Preprocesses a single image for the model.
    Args:
        image: numpy array of the image (BGR format from OpenCV)
    Returns:
        preprocessed_image: numpy array ready for model inference
    """
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image) # MobileNetV2 specific preprocessing
        return image
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

def preprocess_face(face_image):
    """
    Preprocesses a cropped face image.
    """
    return preprocess_image(face_image)
