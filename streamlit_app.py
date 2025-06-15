import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

# ---------------------------
# Must be the FIRST command
# ---------------------------
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# ---------------------------
# Load SVM Model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("svm_simple_pipeline.pkl")

model = load_model()

# ---------------------------
# Page Header
# ---------------------------
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>✍️ MNIST Digit Classifier</h1>
    <p style='text-align: center; color: #555;'>Upload up to <b>30 grayscale images</b> (28×28 pixels) of handwritten digits.<br>Each image should be white digits on a black background.</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# Reset Button
# ---------------------------
if "reset" not in st.session_state:
    st.session_state.reset = False

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    if st.button("🔁 Reset App", use_container_width=True):
        st.session_state.reset = True
        st.rerun()

# ---------------------------
# File Uploader
# ---------------------------
uploaded_files = st.file_uploader(
    "📂 Upload your digit images (PNG/JPG/JPEG):",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ---------------------------
# Prediction Display
# ---------------------------
if uploaded_files:
    st.markdown(f"<h4 style='color: #6C63FF;'>📸 Total Uploaded: {len(uploaded_files)}</h4>", unsafe_allow_html=True)

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

                st.image(image, caption=f"🔢 Predicted: {prediction}", width=120)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px;'>Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
