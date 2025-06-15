import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

# ✅ MUST BE THE FIRST Streamlit COMMAND
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️")

# ✅ Load the model once
@st.cache_resource
def load_model():
    return joblib.load("svm_simple_pipeline.pkl")

model = load_model()

# 🧠 App title and instructions
st.title("✍️ MNIST Digit Classifier")
st.markdown("Upload a **28×28 grayscale image** of a digit (white digit on black background).")

# 📁 File uploader
uploaded_file = st.file_uploader("Upload PNG/JPG image", type=["png", "jpg", "jpeg"], key="uploader")

# 👇 Only process if image is uploaded
if uploaded_file is not None:
    # 🖼️ Preprocessing
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    st.image(image, caption="🖼️ Uploaded Image", width=150)

    # 🔍 Predict
    img_array = np.array(image).astype("float32").reshape(1, -1)
    prediction = model.predict(img_array)
    st.markdown(f"### 🎯 Predicted Digit: **{prediction[0]}**")

    # 🔁 Reset
    if st.button("🔁 Reset and Upload Another"):
        st.experimental_rerun()
