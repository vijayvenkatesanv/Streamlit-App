import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

# Must be FIRST Streamlit command
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️")

# Load the model once
@st.cache_resource
def load_model():
    return joblib.load("svm_simple_pipeline.pkl")

model = load_model()

# App Title
st.title("✍️ MNIST Digit Classifier")
st.markdown("Upload a **28×28 grayscale image** of a digit (white digit on black background).")

# Reset logic (works on Streamlit Cloud too)
if "reset" not in st.session_state:
    st.session_state.reset = False

if st.button("🔁 Reset App"):
    st.session_state.reset = True
    st.rerun()  # ✅ use st.rerun(), not experimental

# Upload
uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    st.image(image, caption="🖼️ Uploaded Digit", width=150)

    # Flatten & normalize
    img_array = np.array(image).astype("float32").reshape(1, -1)

    # Predict
    prediction = model.predict(img_array)
    st.markdown(f"### 🎯 Predicted Digit: **{prediction[0]}**")
