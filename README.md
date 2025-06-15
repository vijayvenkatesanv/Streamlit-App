# Machine Learning App

This is an ML App

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://Streamlit-App.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

![image](https://github.com/user-attachments/assets/992a95b1-8281-4017-aff8-8eedca6a7e7f)

![image](https://github.com/user-attachments/assets/81d1a312-757e-489e-9529-2e335273c5d4)
# ✍️ MNIST Digit Classifier (Streamlit + Logistic Regression)

An interactive web application for classifying handwritten digit images (0–9) using a Logistic Regression model trained on the MNIST dataset. Built using [Streamlit](https://streamlit.io), this tool enables users to upload multiple digit images, visualize them, and obtain predictions in real-time.

---

## 📌 Overview

This project demonstrates the power of simple machine learning models like **Logistic Regression** in solving computer vision problems when paired with efficient preprocessing and a clean UI. The app is lightweight, runs locally, and supports batch image predictions (up to 30 images).

---

## 🚀 Features

- 🖼️ **Multi-image Upload**: Upload up to **30 grayscale digit images** in `.png`, `.jpg`, or `.jpeg` format
- ⚙️ **Preprocessing Included**:
  - Converts images to grayscale (if not already)
  - Inverts black-on-white digits automatically
  - Resizes to `28×28` and flattens to 784-pixel vector
- 🤖 **Model Used**: Logistic Regression (scikit-learn)
- 📊 **Live Predictions**: See model predictions with the image preview
- 🎨 **Modern Dark UI**: Custom styled interface with Streamlit and CSS
- 🧹 **Reset Button**: Easily restart the session with one click

---

## 🧠 Model Information

- **Type**: Multiclass Logistic Regression
- **Library**: `scikit-learn`
- **Dataset**: MNIST (60,000 train, 10,000 test handwritten digit samples)
- **Input Dimension**: 28×28 grayscale pixels → Flattened to 784 features
- **Output**: Digit class (0 through 9)
- **Saved Format**: `.pkl` using `joblib`

> ℹ️ **Note**: Despite the filename `svm_simple_pipeline.pkl`, the model is **Logistic Regression**, not SVM.

---

## 🛠️ Tech Stack

| Component       | Tool/Library     |
|----------------|------------------|
| Frontend        | Streamlit        |
| ML Model        | scikit-learn (Logistic Regression) |
| Image Processing| Pillow (PIL), NumPy |
| Serialization   | joblib           |
| Language        | Python 3.8+      |
| Deployment      | Localhost / Streamlit Cloud |

---

## 📁 Project Structure

mnist-digit-classifier/
│
├── app.py # Streamlit application script
├── svm_simple_pipeline.pkl # Trained Logistic Regression model
├── requirements.txt # Dependency list
├── README.md # You're reading this
└── sample_digits/ # (Optional) Sample images for testing

yaml
Copy
Edit

---

## 🧪 How to Run Locally

### 1. Clone this repo

```bash
git clone https://github.com/your-username/mnist-digit-classifier.git
cd mnist-digit-classifier
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Launch the Streamlit app
bash
Copy
Edit
streamlit run app.py
It will open at: http://localhost:8501

📥 Input Guidelines
To ensure accurate predictions:

📷 Format: PNG, JPG, or JPEG

🖤 Foreground: White digits

⚫ Background: Black (if not, the app will invert)

🔲 Size: 28×28 pixels

🌑 Grayscale: Required (or RGB convertible to grayscale)

If your digit image is black-on-white, the app auto-inverts it.

📦 requirements.txt
text
Copy
Edit
streamlit>=1.28
numpy>=1.22
Pillow>=9.0
joblib>=1.2
scikit-learn>=1.1
📸 Screenshots
🖼️ Upload Area

📊 Prediction Result

Add your screenshots to the screenshots/ folder for a complete showcase.

☁️ Deploy to Streamlit Cloud
To deploy on Streamlit Cloud:

Push code to GitHub

Go to Streamlit Cloud

Paste your GitHub repo link

Set app.py as the entry file

Done!

📝 License
This project is licensed under the MIT License. You are free to use, distribute, and modify it with proper attribution.

👤 Author
Vijay Venkatesan V
Department of Electronics and Communication Engineering
Amrita School of Engineering, Coimbatore
📧 Email: venkatvijay614@gmail.com

🌟 Acknowledgements
MNIST Dataset

Streamlit Docs

scikit-learn

yaml
Copy
Edit

---

Let me know if you'd like:
- GitHub-style badges (e.g., Python version, license, Streamlit deploy)
- A PDF version of this README
- Help packaging this as a downloadable `.zip` with model + code + README

I'm happy to assist!






You said:
https://github.com/vijayvenkatesanv/Streamlit-App

witht hsi cocorect it


ChatGPT said:
Thanks! Here's the corrected and GitHub-ready README.md with your repo link: https://github.com/vijayvenkatesanv/Streamlit-App.

markdown
Copy
Edit
# ✍️ MNIST Digit Classifier | Streamlit App

An interactive digit recognition web application using **Logistic Regression** trained on the MNIST dataset. Built with [Streamlit](https://streamlit.io), this app enables users to upload handwritten digit images and receive real-time predictions with a clean, modern interface.

👉 **Live on GitHub**: [vijayvenkatesanv/Streamlit-App](https://github.com/vijayvenkatesanv/Streamlit-App)

---

## 📌 Overview

This project showcases the application of classical machine learning (Logistic Regression) to image classification. With minimal resources, it can predict up to 30 images of handwritten digits in real-time using a trained model.

---

## 🚀 Features

- 🔢 Multi-image classification (up to 30 images at once)
- 🖼️ Upload images in PNG, JPG, or JPEG format
- 🧠 Logistic Regression model with high accuracy on MNIST
- 🖤 Automatically inverts black-on-white digit images
- 🖥️ Modern UI with responsive layout (Dark Mode)
- 🔁 Reset app state with a single click

---

## 🧠 Model Information

- **Model Type**: Logistic Regression
- **Library**: scikit-learn
- **Dataset**: MNIST (60k training, 10k test samples)
- **Input**: 28×28 grayscale digit images → flattened to 784 features
- **Saved As**: `svm_simple_pipeline.pkl` (contains Logistic Regression)

---

## 🛠️ Tech Stack

| Component       | Tool/Library     |
|----------------|------------------|
| UI & Frontend  | Streamlit        |
| ML Backend     | scikit-learn     |
| Image Processing| Pillow (PIL), NumPy |
| Model Saving   | joblib           |
| Python Version | Python 3.8+      |

---

## 📁 Project Structure

Streamlit-App/
├── app.py # Streamlit application code
├── svm_simple_pipeline.pkl # Trained Logistic Regression model
├── requirements.txt # All Python dependencies
├── README.md # Project documentation (this file)
└── sample_digits/ # (Optional) Example digit images

yaml
Copy
Edit

---

## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/vijayvenkatesanv/Streamlit-App.git
cd Streamlit-App
2. Install Required Packages
bash
Copy
Edit
pip install -r requirements.txt
3. Launch the App
bash
Copy
Edit
streamlit run app.py
It will open in your browser at: http://localhost:8501

📥 Input Requirements
File types: .png, .jpg, .jpeg

Size: Preferably 28×28 (resizing is applied)

Format: Grayscale (or convertible)

Digit should be white on black; if not, app auto-inverts

📦 requirements.txt
shell
Copy
Edit
streamlit>=1.28
numpy>=1.22
Pillow>=9.0
joblib>=1.2
scikit-learn>=1.1
📸 Screenshots
Upload Area	Prediction Results

To add your own screenshots: Create a screenshots/ folder and place PNG files inside.

☁️ Deployment (Streamlit Cloud)
To deploy this app using Streamlit Cloud:

Push this repo to your GitHub.

Sign in at share.streamlit.io.

Connect your repo and set app.py as the entry point.

Click Deploy. Done!

📄 License
This project is licensed under the MIT License — feel free to use, modify, and distribute with attribution.

👤 Author
Vijay Venkatesan V
Dept. of Electronics and Communication Engineering
Amrita School of Engineering, Coimbatore
📧 venkatvijay614@gmail.com
🔗 GitHub

🙌 Acknowledgements
Streamlit

scikit-learn

MNIST Dataset
