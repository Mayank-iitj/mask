import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from mask_config import MODEL_PATH, MIN_CONFIDENCE, IMAGE_SIZE

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence)

    def detect_faces(self, image):
        """
        Detects faces in an image.
        Args:
            image: RGB image
        Returns:
            results: Mediapipe detection results
        """
        return self.face_detection.process(image)

class MaskClassifier:
    def __init__(self):
        self.model = None
        self.load_classifier()

    def load_classifier(self):
        try:
            self.model = load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict_mask(self, face_image):
        """
        Predicts mask status for a single face image.
        Args:
            face_image: Cropped face image (RGB)
        Returns:
            prediction: (mask_prob, no_mask_prob)
        """
        if self.model is None:
            return None

        try:
            face = cv2.resize(face_image, IMAGE_SIZE)
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            
            (mask, withoutMask) = self.model.predict(face)[0]
            return (mask, withoutMask)
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

def detect_and_predict_mask(frame, face_detector, mask_classifier):
    """
    Detects faces and predicts mask status for each face in the frame.
    """
    # Convert to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(frame_rgb)
    
    faces = []
    locs = []
    preds = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), 
                            int(bboxC.width * iw), int(bboxC.height * ih))

            # Ensure bounding box falls within frame dimensions
            (startX, startY) = (max(0, x), max(0, y))
            (endX, endY) = (min(iw - 1, x + w), min(ih - 1, y + h))

            # Extract the face ROI
            face = frame_rgb[startY:endY, startX:endX]
            
            if face.size == 0:
                continue

            faces.append(face)
            locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            # Predict for each face
            for face in faces:
                pred = mask_classifier.predict_mask(face)
                if pred is not None:
                    preds.append(pred)
                else:
                    # Fallback if prediction fails
                    preds.append((0.5, 0.5)) 

    return (locs, preds)
