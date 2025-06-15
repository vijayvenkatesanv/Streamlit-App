import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

# ✅ Set page config FIRST
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️")

# --------------------------------------
# 🎯 Load pre-trained model
# --------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("svm_simple_pipeline.pkl")

model = load_model()

# --------------------------------------
# 🧠 App Title and Description
# --------------------------------------
st.title("✍️ MNIST Digit Classifier")
st.markdown(
    """
    Upload a **28×28 grayscale image** of a handwritten digit (0–9).  
    The model will predict the digit using an SVM classifier.
    """
)

# --------------------------------------
# 📤 Image Upload Section
# --------------------------------------
uploaded_file = st.file_uploader("📁 Upload an image (PNG or JPG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 🖼️ Image Processing
    image = Image.open(uploaded_file).convert("L")      # Grayscale
    image = ImageOps.invert(image)                      # White digit on black
    image = image.resize((28, 28))                      # Resize to MNIST size

    st.image(image, caption="🖼️ Processed Image", width=150)

    # Convert to array and flatten
    img_array = np.array(image).astype("float32").reshape(1, -1)

    # 🔍 Prediction
    prediction = model.predict(img_array)

    st.markdown(f"### ✅ Predicted Digit: **{prediction[0]}**")

# --------------------------------------
# 📝 Footer
# --------------------------------------
st.markdown("---")
st.markdown(
    "<center><small>Developed using Streamlit & Scikit-learn · Model: SVM</small></center>",
    unsafe_allow_html=True
)
