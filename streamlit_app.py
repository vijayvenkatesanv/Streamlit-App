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

if not os.path.exists(model_path):
    with zipfile.ZipFile("svm_simple_pipeline.zip", 'r') as zip_ref:
        zip_ref.extractall()

model = joblib.load(model_path)

# --------------------------------------------
# Step 2: Streamlit Interface
# --------------------------------------------
st.title("MNIST Digit Classifier")
st.write("Upload a **28x28 grayscale image** of a digit (like MNIST).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert image to grayscale
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)  # Invert to match MNIST-style white digit on black
    image = image.resize((28, 28))  # Resize to MNIST format

    st.image(image, caption='Uploaded Digit', use_column_width=False)

    # Preprocess the image
    img_array = np.array(image).astype("float32")
    img_flattened = img_array.reshape(1, -1) / 255.0  # Flatten and normalize

    # Predict
    prediction = model.predict(img_flattened)
    st.write(f"Predicted Digit: **{prediction[0]}**")
