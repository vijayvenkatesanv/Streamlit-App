# Machine Learning App

This is an ML App

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://Streamlit-App.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

![image](https://github.com/user-attachments/assets/992a95b1-8281-4017-aff8-8eedca6a7e7f)


This project is a Streamlit-based web application designed for real-time handwritten digit classification using a Logistic Regression model trained on the MNIST dataset. It allows users to upload up to 30 grayscale images of handwritten digits (preferably 28×28 pixels in size), which are then preprocessed, inverted (if needed), and passed through a scikit-learn pipeline for prediction.

The machine learning model used here is a Logistic Regression classifier, trained on the classic MNIST dataset consisting of 60,000 training images and 10,000 test images of handwritten digits (0–9). Despite its simplicity compared to modern deep learning approaches, Logistic Regression performs surprisingly well on this dataset and offers the benefit of being lightweight, fast, and interpretable. The model is serialized using joblib and stored in the file svm_simple_pipeline.pkl.

The web application is implemented using Streamlit, which provides an elegant and interactive user interface. The app includes a dark-themed layout, custom CSS styling, and a responsive image upload section. When users upload digit images (in .png, .jpg, or .jpeg format), the app automatically resizes them, converts them to grayscale, and inverts the colors to match the expected MNIST format (white digits on black background). Predictions are displayed alongside each uploaded image with a minimal and clean aesthetic.

Below is a summary of the core technologies used in the project:

Component	Technology Used
Frontend	Streamlit
Image Processing	PIL (Pillow), NumPy
Model Training	Scikit-learn (Logistic Regression)
Serialization	Joblib
Python Version	Python 3.8+

To run the project locally, clone the repository and install the required dependencies listed in requirements.txt. Once installed, launch the app using the command:

arduino
Copy
Edit
streamlit run app.py
This will start a local server, and the app can be accessed in your browser at http://localhost:8501. The app also includes a reset button that allows users to clear all session states and start over, which is useful when uploading multiple sets of images.

The input expectations for the images are simple: grayscale, square format (preferably 28×28), and clear digit contours. The app handles preprocessing internally, so minor variations in image size or background color are automatically corrected. This makes it user-friendly even for non-technical users.

The entire project is lightweight, portable, and suitable for educational demonstrations or quick ML prototyping. It can be deployed easily on Streamlit Cloud by linking the GitHub repo and setting app.py as the main script. Overall, the project serves as a minimal but functional example of integrating traditional machine learning with an intuitive web-based frontend.

If you’d like me to write a one-line summary for the GitH
