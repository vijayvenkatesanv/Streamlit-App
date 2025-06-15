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
# Page Styling and Header
# ---------------------------
st.markdown("""
    <style>
        h1, h4, p {
            text-align: center;
        }
        .block-container {
            padding-top: 2rem;
        }
        .uploadedImage img {
            border-radius: 8px;
            border: 2px solid #6C63FF;
            background-color: #1E1E1E;
        }
        .stButton>button {
            background-color: #6C63FF;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #5a52cc;
        }
    </style>

    <h1 style='color: #6C63FF;'>✍️ MNIST Digit Classifier</h1>
    <p style='color: #AAAAAA;'>Upload up to <b>30 grayscale images</b> (28×28 pixels) of handwritten digits.<br>Each image should be white digits on a black background.</p>
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

    cols = st.columns(3)  # 3-column layout
    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx % 3]:
            try:
                image = Image.open(uploaded_file).convert("L")
                image = ImageOps.invert(image)
                image = image.resize((28, 28))
                img_array = np.array(image).astype("float32").reshape(1, -1)

                prediction = model.predict(img_array)[0]

                st.markdown("<div class='uploadedImage'>", unsafe_allow_html=True)
                st.image(image, caption=f"🔢 Predicted: {prediction}", width=120)
                st.markdown("</div>", unsafe_allow_html=True)
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
