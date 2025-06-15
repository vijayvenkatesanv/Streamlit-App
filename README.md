# MNSIT DIGIT CLASSIFIER 

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://Streamlit-App.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

![image](https://github.com/user-attachments/assets/992a95b1-8281-4017-aff8-8eedca6a7e7f)


This project is a machine learning web application that classifies handwritten digits using a Logistic Regression model. The model is trained on the MNIST dataset, which contains grayscale images of handwritten digits from 0 to 9. Each image is 28×28 pixels in size and represents a single digit. The dataset includes 60,000 training samples and 10,000 testing samples, making it a standard benchmark for evaluating digit recognition models. The chosen classifier, Logistic Regression, is a lightweight and interpretable algorithm that performs well on this dataset by leveraging all 784 pixel features of each image.

The application is built using Streamlit, a Python framework that simplifies the creation of interactive web interfaces for data science and machine learning applications. The user interface allows users to upload up to 30 digit images in PNG, JPG, or JPEG format. These images are automatically converted to grayscale, resized to 28×28 if necessary, and inverted to ensure white digits on a black background, which is the expected format for the MNIST-trained model. Once preprocessed, each image is passed through the trained pipeline, and the predicted digit is displayed directly above the image in a clean, card-style layout.

The frontend design uses custom CSS embedded within the Streamlit app to provide a dark theme, rounded image borders, modern button styles, and consistent spacing. This improves the user experience and makes the interface visually appealing. The application also includes a reset button that clears the session and allows the user to start a new prediction round without reloading the entire app.

The project relies on several Python libraries. Image processing is handled using Pillow (PIL) and NumPy, which support grayscale conversion, resizing, and pixel value manipulation. The machine learning pipeline is built with scikit-learn, and the model is saved using Joblib for efficient loading at runtime. Streamlit serves as the interface layer, managing state, rendering components, and hosting the app on a local or cloud server.

Overall, this project demonstrates how traditional machine learning models can be effectively combined with modern web interfaces to build an intuitive, end-to-end application for visual digit classification. It is well-suited for demonstrations, education, or lightweight digit recognition tasks without requiring complex deep learning frameworks.
![Screenshot 2025-06-15 154550](https://github.com/user-attachments/assets/684cfe40-c465-4c92-9309-58a5ec2013a0)
