# Machine Learning App

This is an ML App

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://Streamlit-App.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

![image](https://github.com/user-attachments/assets/992a95b1-8281-4017-aff8-8eedca6a7e7f)


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

yaml
Copy
Edit

MNIST Dataset
