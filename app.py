import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from utils.inference import FaceDetector, MaskClassifier, detect_and_predict_mask
from utils.visualization import draw_results
from mask_config import MIN_CONFIDENCE, MODEL_PATH

st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="wide")

@st.cache_resource
def load_models():
    """
    Load the face detector and mask classifier models.
    Cached to avoid reloading on every interaction.
    """
    face_detector = FaceDetector(min_detection_confidence=MIN_CONFIDENCE)
    mask_classifier = MaskClassifier()
    return face_detector, mask_classifier

def main():
    st.title("Face Mask Detection App 😷")
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("Detects faces and classifies them as **Mask** or **No Mask**.")

    # Load models
    with st.spinner("Loading models..."):
        face_detector, mask_classifier = load_models()

    # Check if model file exists
    if mask_classifier.model is None:
        st.error(f"Model not found at `{MODEL_PATH}`! Please train the model first.")
        st.info("Run `python training/train_model.py` to train the model.")
        return

    # Sidebar
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ["Home", "Live Webcam", "Upload Image", "Upload Video"])

    if choice == "Home":
        st.subheader("Welcome!")
        st.write("This application uses Deep Learning to detect face masks in real-time.")
        st.write("### Instructions:")
        st.write("1. **Live Webcam**: Use your camera to detect masks in real-time.")
        st.write("2. **Upload Image**: Upload an image to detect masks.")
        st.write("3. **Upload Video**: Upload a video file for processing.")
        
        st.info(f"Model loaded from: `{MODEL_PATH}`")

    elif choice == "Live Webcam":
        st.subheader("Live Webcam Detection")
        st.write("Click the checkbox below to start the webcam.")
        run = st.checkbox('Start Webcam')
        
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
            
            # Detection
            (locs, preds) = detect_and_predict_mask(frame, face_detector, mask_classifier)
            
            # Visualization
            frame = draw_results(frame, locs, preds)
            
            # Convert to RGB for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
        
        camera.release()

    elif choice == "Upload Image":
        st.subheader("Upload Image")
        image_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            image = Image.open(image_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button('Detect Masks'):
                with st.spinner("Processing..."):
                    # Convert PIL Image to BGR numpy array
                    frame = np.array(image.convert('RGB'))
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Detection
                    (locs, preds) = detect_and_predict_mask(frame_bgr, face_detector, mask_classifier)
                    
                    # Visualization
                    res_frame = draw_results(frame_bgr, locs, preds)
                    res_frame = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.image(res_frame, caption='Processed Image', use_column_width=True)
                    
                    # Stats
                    mask_count = sum([1 for p in preds if p[0] > p[1]])
                    no_mask_count = len(preds) - mask_count
                    st.success(f"Found {len(preds)} faces: {mask_count} Mask, {no_mask_count} No Mask")

    elif choice == "Upload Video":
        st.subheader("Upload Video")
        video_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
        
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(video_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            stop_button = st.button("Stop Processing")
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detection
                (locs, preds) = detect_and_predict_mask(frame, face_detector, mask_classifier)
                
                # Visualization
                frame = draw_results(frame, locs, preds)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                stframe.image(frame)
            
            cap.release()
            os.unlink(tfile.name)

if __name__ == '__main__':
    main()
