"""Streamlit web app for Face Mask Detection

This app provides a web UI for:
  - Uploading an image to check if a person is wearing a mask
  - (Optional) Live webcam detection if run locally
"""

from __future__ import annotations

import os
import sys
import tempfile
from io import BytesIO

# Core imports with error handling
try:
    import streamlit as st
except ImportError as e:
    print(f"Error: streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

try:
    import numpy as np
except ImportError as e:
    st.error("NumPy not installed. Please add 'numpy' to requirements.txt")
    sys.exit(1)

try:
    from PIL import Image
except ImportError as e:
    st.error("Pillow not installed. Please add 'pillow' to requirements.txt")
    sys.exit(1)

# Conditional imports for model inference
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    KERAS_BACKEND = "tensorflow"
except ImportError:
    try:
        from keras.models import load_model
        from keras.preprocessing.image import img_to_array
        KERAS_BACKEND = "keras"
    except ImportError as e:
        st.error("⚠️ TensorFlow/Keras not installed. Please add 'tensorflow>=2.10.0' to requirements.txt")
        st.stop()

# OpenCV import with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV not available. Some features may be limited. Install 'opencv-python-headless' if needed.")

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "mymodel.h5")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))

st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

st.title("😷 Face Mask Detector")
st.markdown(
    """
    Upload an image to check if a person is wearing a face mask.
    The model outputs a probability (> 0.5 = NO MASK, ≤ 0.5 = MASK).
    """
)


@st.cache_resource
def load_trained_model(model_path: str):
    """Load the trained Keras model once and cache it."""
    if not os.path.isfile(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please ensure 'mymodel.h5' is in the repository root or set MODEL_PATH environment variable.")
        st.stop()
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize and normalize image for model input."""
    img = img.resize((150, 150))
    arr = img_to_array(img)
    arr = np.expand_dims(arr.astype("float32") / 255.0, axis=0)
    return arr


def predict_mask(model, img: Image.Image) -> tuple[float, str]:
    """Run inference and return probability and label."""
    try:
        arr = preprocess_image(img)
        prob = float(model.predict(arr, verbose=0)[0][0])
        label = "NO MASK" if prob > THRESHOLD else "MASK"
        return prob, label
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        raise


# Load model with error handling
try:
    model = load_trained_model(MODEL_PATH)
    st.sidebar.success(f"✅ Model loaded successfully")
    st.sidebar.info(f"Backend: {KERAS_BACKEND}")
except Exception as e:
    st.error(f"Failed to initialize app: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Run prediction
        with st.spinner("Analyzing..."):
            prob, label = predict_mask(model, image)

        # Display result
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Label", label)
        with col2:
            st.metric("Probability (no-mask)", f"{prob:.4f}")

        if prob > THRESHOLD:
            st.error("⚠️ No mask detected!")
        else:
            st.success("✅ Mask detected!")
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.exception(e)

st.markdown("---")
st.markdown(
    """
    **Note:** This model was trained on a simple CNN and is for demonstration purposes.
    For production use, consider using a more robust model and additional validation.
    """
)
