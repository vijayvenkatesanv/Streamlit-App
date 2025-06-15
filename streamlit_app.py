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
]]
