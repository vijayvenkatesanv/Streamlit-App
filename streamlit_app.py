import streamlit as st
import numpy as np
import joblib
import zipfile
from PIL import Image, ImageOps
import os

# --------------------------------------------
# Step 1: Unzip and load model from ZIP
# --------------------------------------------
model_path = "svm_simple_pipeline.pkl"
zip_path = "svm_simple_pipeline.zip"

# Unzip only if the model is not already extracted
if not os.path.exists(model_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

# Load the model
model = joblib.load(model_path)

# --------------------------------------------
# Step 2: Streamlit UI
# --------------------------------------------
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢")
st.title("🧠 MNIST Digit Classifier")
st.write("Upload a **28x28 grayscale image** of a digit (like MNIST).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to grayscale and invert (to match MNIST style)
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))  # Resize to 28x28

    st.image(image, caption='Uploaded Digit', width=150)

    # Convert image to NumPy array and normalize
    img_array = np.array(image).astype("float32")
    img_flattened = img_array.reshape(1, -1) / 255.0

    # Predict
    prediction = model.predict(img_flattened)
    st.success(f"🧾 Predicted Digit: **{prediction[0]}**")
