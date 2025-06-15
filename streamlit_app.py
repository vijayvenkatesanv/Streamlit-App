import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import base64
from io import BytesIO

# ✅ This MUST be first Streamlit command
st.set_page_config(
    page_title="🎨 MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="auto"
)

# Inject custom CSS (AFTER set_page_config)
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: white;
    }
    .stApp {
        background-color: #1e1e1e;
    }
    h1, h4, .stMarkdown, .stButton > button, .stFileUploader {
        color: #FAD02E;
    }
    .prediction-box {
        background-color: #2c2f33;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #444;
        margin-bottom: 10px;
        text-align: center;
    }
    .stButton button {
        background-color: #3c40c6;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #575fcf;
        color: white;
    }
    .footer {
        color: #ccc;
        text-align: center;
        font-size: 12px;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# The rest of your code follows...


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
    <h1 style='text-align: center;'>🧠 MNIST Digit Classifier</h1>
    <p style='text-align: center; color: #aaa;'>Upload up to <b>30 grayscale digit images</b> (28×28 pixels, white digit on black background).</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# Reset Button
# ---------------------------
if "reset" not in st.session_state:
    st.session_state.reset = False

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    if st.button("🔁 Reset App"):
        st.session_state.reset = True
        st.rerun()

# ---------------------------
# Upload Images
# ---------------------------
uploaded_files = st.file_uploader(
    "📂 Upload up to 30 images (PNG/JPG/JPEG):",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ---------------------------
# Prediction Section
# ---------------------------
if uploaded_files:
    st.markdown(f"<h4 style='color: #F97B22;'>🖼️ {len(uploaded_files)} Image(s) Uploaded</h4>", unsafe_allow_html=True)

    if len(uploaded_files) > 30:
        st.warning("⚠️ Only the first 30 images will be processed.")
        uploaded_files = uploaded_files[:30]

    cols = st.columns(3)

    colors = ['#00B894', '#F97B22', '#6C5CE7', '#D63031', '#00CEC9', '#E84393']

    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx % 3]:
            try:
                image = Image.open(uploaded_file).convert("L")
                image = ImageOps.invert(image)
                image = image.resize((28, 28))
                img_array = np.array(image).astype("float32").reshape(1, -1)

                prediction = model.predict(img_array)[0]
                color = colors[idx % len(colors)]

                st.markdown(f"""
                    <div class='prediction-box' style='border-left: 5px solid {color};'>
                        <img src="data:image/png;base64,{image_to_base64(image)}" width="100" />
                        <p style='color: {color}; font-size: 20px;'>🔢 Prediction: <b>{prediction}</b></p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {uploaded_file.name} - {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("<div class='footer'>Made with ❤️ using Streamlit | Theme: Dark & Colorful</div>", unsafe_allow_html=True)

# ---------------------------
# Helper: Convert image to base64
# ---------------------------
import base64
from io import BytesIO

def image_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_data = buf.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str
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
