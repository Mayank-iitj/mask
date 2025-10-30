"""Streamlit web app for Face Mask Detection

This app provides a web UI for:
  - Uploading an image to check if a person is wearing a mask
  - (Optional) Live webcam detection if run locally
"""

from __future__ import annotations

import os
import tempfile
from io import BytesIO

import streamlit as st
import numpy as np
from PIL import Image

# Conditional imports for model inference
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
except Exception:
    from keras.models import load_model
    from keras.preprocessing.image import img_to_array

import cv2

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
        st.error(f"Model file not found: {model_path}")
        st.stop()
    return load_model(model_path)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize and normalize image for model input."""
    img = img.resize((150, 150))
    arr = img_to_array(img)
    arr = np.expand_dims(arr.astype("float32") / 255.0, axis=0)
    return arr


def predict_mask(model, img: Image.Image) -> tuple[float, str]:
    """Run inference and return probability and label."""
    arr = preprocess_image(img)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "NO MASK" if prob > THRESHOLD else "MASK"
    return prob, label


# Load model
model = load_trained_model(MODEL_PATH)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
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

st.markdown("---")
st.markdown(
    """
    **Note:** This model was trained on a simple CNN and is for demonstration purposes.
    For production use, consider using a more robust model and additional validation.
    """
)
