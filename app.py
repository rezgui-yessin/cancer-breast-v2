import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Breast Cancer Chatbot", layout="centered")

st.title("🩺 Breast Cancer Prediction Chatbot")
st.write("Enter the tumor features below:")

# ---- INPUTS (same order used in training) ----
radius_mean = st.number_input("Radius Mean", min_value=0.0)
texture_mean = st.number_input("Texture Mean", min_value=0.0)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0)
area_mean = st.number_input("Area Mean", min_value=0.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0)

# ---- PREDICTION ----
if st.button("Predict"):
    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Result: Malignant Tumor (Cancer Detected)")
    else:
        st.success("✅ Result: Benign Tumor (No Cancer)")
