import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

# ✅ Must be the very first Streamlit command
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️")

# ✅ Load model only once
@st.cache_resource
def load_model():
    return joblib.load("svm_simple_pipeline.pkl")

model = load_model()

# 🎯 App title
st.title("✍️ MNIST Digit Classifier")
st.markdown("Upload a **28×28 grayscale image** of a digit (white digit on black background).")

# 📁 File uploader
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"], key="uploader")

# Reset button always shown
reset_clicked = st.button("🔁 Reset App")

# 👇 Handle reset early
if reset_clicked:
    st.session_state.clear()
    st.experimental_rerun()

# 👇 Only run prediction if a file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    st.image(image, caption="🖼️ Uploaded Digit", width=150)

    img_array = np.array(image).astype("float32").reshape(1, -1)

    prediction = model.predict(img_array)
    st.markdown(f"### 🎯 Predicted Digit: **{prediction[0]}**")
