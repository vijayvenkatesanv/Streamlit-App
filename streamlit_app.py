import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# ---------------------------
# Load the Trained Model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("svm_simple_pipeline.pkl")  # Make sure this exists

model = load_model()

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 85%;
            margin: auto;
        }
        .stButton>button {
            background-color: #6C63FF;
            color: white;
            font-weight: bold;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #554bcf;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Title and Description
# ---------------------------
st.markdown("""
    <h1 style='color: #6C63FF; text-align:center;'>✍️ MNIST Digit Classifier</h1>
    <p style='color: #AAAAAA; font-size: 16px; text-align:center;'>
        Upload up to <b>30 grayscale images</b> (28×28 pixels) of handwritten digits.<br>
        Each image must contain <b>white digits on a black background</b>.
    </p>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)

# ---------------------------
# Reset Button
# ---------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    if st.button("🔁 Reset App", use_container_width=True):
        st.session_state.clear()
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
# Prediction and Display
# ---------------------------
if uploaded_files:
    st.markdown(f"<h4 style='color: #6C63FF;'>📸 Total Uploaded: {len(uploaded_files)}</h4>", unsafe_allow_html=True)

    if len(uploaded_files) > 30:
        st.warning("⚠️ Only the first 30 images will be processed.")
        uploaded_files = uploaded_files[:30]

    cols = st.columns(3)
    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx % 3]:
            try:
                image = Image.open(uploaded_file).convert("L")
                image = ImageOps.invert(image)
                image = image.resize((28, 28))
                img_array = np.array(image).astype("float32").reshape(1, -1)

                prediction = model.predict(img_array)[0]

                # ✅ Only render image with caption — no st.markdown
                st.image(image, caption=f"🔢 Predicted: {prediction}", width=130)

            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 13px; color: #777;'>Built with ❤️ using <a href='https://streamlit.io' target='_blank' style='color: #6C63FF;'>Streamlit</a></p>",
    unsafe_allow_html=True
)
