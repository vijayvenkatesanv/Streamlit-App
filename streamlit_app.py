import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

# ✅ THIS MUST BE FIRST Streamlit COMMAND
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️")

@st.cache_resource
def load_model():
    return joblib.load("svm_simple_pipeline.pkl")

model = load_model()

st.title("✍️ MNIST Digit Classifier")
st.markdown(
    "Upload a **28×28 grayscale** digit image (white digit on black background)."
)

uploaded_file = st.file_uploader("📁 Upload PNG/JPG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")   # Convert to grayscale
    image = ImageOps.invert(image)                   # Invert colors
    image = image.resize((28, 28))                   # Resize to 28x28

    st.image(image, caption="🖼️ Uploaded Image", width=150)

    img_array = np.array(image).astype("float32").reshape(1, -1)

    prediction = model.predict(img_array)

    st.markdown(f"### 🎯 Predicted Digit: **{prediction[0]}**")
