# 🧠 Dementia Detection Web App

A **Flask-based web application** that predicts **Dementia** based on patient information using a **machine learning model**. The app provides an **interactive form**, **risk meter**, and **confidence percentage** for the prediction.

---

## **Features**

- User-friendly input form for patient data:
  - Age, Gender, BMI
  - Blood pressure, Cholesterol levels
  - MMSE Score
  - Lifestyle and medical history factors (Smoking, Alcohol, Physical Activity, Sleep Quality, etc.)
- Dynamic **risk meter** to visualize Dementia risk.
- **Confidence percentage** displayed for each prediction.
- **Demo buttons** for quick testing with sample Dementia / Healthy patient data.
- Clean and responsive design for easy use.

---

## **Installation**

1. **Clone the repository:**

git clone https://github.com/yourusername/DementiaDetectionApp.git
cd DementiaDetectionApp
Install dependencies:

Copy code
pip install flask numpy scikit-learn
Make sure you have Flask, numpy, scikit-learn, and any other dependencies installed.

Place the trained ML model (dementia_model.pkl) in the project root.

Usage
1)Run the Flask application:
python app.py

2)Open your browser and go to:
http://127.0.0.1:5000/

Enter patient information using sliders or numeric input.

Click Predict to see:

Dementia prediction (🧠 or ✅)

Confidence percentage

Risk meter and visual indicators

Folder Structure

DementiaDetectionApp/
│
├── templates/
│   ├── index.html         # Main input form page
│   └── result.html        # Prediction result page
│
├── static/
│   └── css/
│       └── style.css      # Styling for the web pages
│
├── dementia_model.pkl     # Trained ML model
├── app.py                 # Flask backend
└── README.md              # This file

Notes
Ensure the ML model matches the input features. Changing slider names or order may affect predictions.
