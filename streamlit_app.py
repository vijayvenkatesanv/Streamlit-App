import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

# Must be the FIRST Streamlit command
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("svm_simple_pipeline.pkl")

model = load_model()

# App Title
st.title("✍️ MNIST Digit Classifier")
st.markdown("Upload up to **30 grayscale images (28×28)** of digits (white digit on black background).")

# Reset logic
if "reset" not in st.session_state:
    st.session_state.reset = False

if st.button("🔁 Reset App"):
    st.session_state.reset = True
    st.rerun()

# Multiple file uploader
uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.markdown(f"### 📂 Total Images Uploaded: {len(uploaded_files)}")
    
    if len(uploaded_files) > 30:
        st.warning("⚠️ Only the first 30 images will be processed.")
        uploaded_files = uploaded_files[:30]
    
    cols = st.columns(3)  # 3 columns layout

    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx % 3]:
            try:
                image = Image.open(uploaded_file).convert("L")
                image = ImageOps.invert(image)
                image = image.resize((28, 28))
                img_array = np.array(image).astype("float32").reshape(1, -1)

                prediction = model.predict(img_array)[0]

                st.image(image, caption=f"Predicted: {prediction}", width=100)
            except Exception as e:
                st.error(f"Error with image {uploaded_file.name}: {e}")
