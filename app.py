# app.py

import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- AI/ML Model and OpenCV Functions ---

# Load the trained CNN model
try:
    model = load_model('image_quality_model.keras')
    st.session_state['model_loaded'] = True
except (IOError, ImportError):
    st.session_state['model_loaded'] = False

# OpenCV-based analysis function (detects specific issues)
def analyze_specific_issues(image_cv):
    """Takes an OpenCV image and analyzes for specific issues."""
    grayscale_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect Blurriness
    focus_measure = cv2.Laplacian(grayscale_image, cv2.CV_64F).var()
    
    # 2. Brightness level
    brightness = np.mean(grayscale_image)
    
    # 3. Contrast level
    contrast = np.std(grayscale_image)
    
    return focus_measure, brightness, contrast

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="Image Quality Analyzer")

st.title("AI-based Image Quality Analyzer 🖼️")
st.markdown("Upload an image, and the AI model will classify it as 'Good' or 'Bad' quality.")

# Sidebar
with st.sidebar:
    st.header("How it works?")
    st.markdown("""
    This app uses a hybrid approach to analyze image quality:
    
    1.  **AI Classification:** A trained **Convolutional Neural Network (CNN)** first predicts the overall quality ('Good' or 'Bad').
    2.  **Issue Detection:** If the quality is found to be 'Bad', **OpenCV** is used to identify specific issues like blurriness, dullness, or low contrast.
    """)
    st.header("Analysis Thresholds")
    st.info("These thresholds are only used if the AI model flags an image as 'Bad'.")
    focus_threshold = st.slider("Focus Threshold (lower = more blurry)", 0, 500, 100)
    brightness_threshold = st.slider("Brightness Threshold (lower = darker)", 0, 255, 70)
    contrast_threshold = st.slider("Contrast Threshold (lower = less contrast)", 0, 255, 50)

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if not st.session_state.get('model_loaded', False):
    st.error("Error: 'image_quality_model.h5' not found. Please train the model by running the `train_model.py` script first.")

elif uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_image, caption="Your Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("Analysis Results")

        # --- AI Model Prediction ---
        with st.spinner('AI model is analyzing...'):
            # Prepare image for the model
            image_for_model = pil_image.resize((150, 150))
            img_array = img_to_array(image_for_model)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Make prediction
            prediction = model.predict(img_array)[0][0]
            is_good_quality = prediction > 0.5

        # --- Display Final Results ---
        st.subheader("Overall Quality Assessment")
        if is_good_quality:
            st.success(f"**Excellent! The AI classified this image as Good Quality.** (Confidence: {prediction*100:.2f}%)")
            st.write("No major issues were detected.")
        else:
            st.error(f"**Poor Quality! The AI classified this image as Bad Quality.** (Confidence: {(1-prediction)*100:.2f}%)")
            
            # If bad, analyze for specific issues
            st.markdown("---")
            st.subheader("Specific Issues Found (OpenCV Analysis):")
            
            image_cv = np.array(pil_image.convert('RGB'))
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            
            focus, brightness, contrast = analyze_specific_issues(image_cv)
            issues = []

            # Check and report issues
            if focus < focus_threshold:
                issues.append(f"**Blurriness:** Focus level ({focus:.2f}) is below the threshold ({focus_threshold}).")
            if brightness < brightness_threshold:
                issues.append(f"**Dark/Dull:** Brightness level ({brightness:.2f}) is below the threshold ({brightness_threshold}).")
            if contrast < contrast_threshold:
                issues.append(f"**Low Contrast:** Contrast level ({contrast:.2f}) is below the threshold ({contrast_threshold}).")

            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.info("The AI deemed this as poor quality, but the specific OpenCV checks were not below their thresholds.")

else:
    if st.session_state.get('model_loaded', False):
        st.info("Please upload an image file to begin analysis.")
