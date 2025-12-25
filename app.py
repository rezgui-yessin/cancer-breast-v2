import streamlit as st
import numpy as np
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered"
)

# ===============================
# LOAD MODEL & SCALER
# ===============================
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    **Breast Cancer Prediction App**

    This application uses a **Logistic Regression**
    machine learning model trained on the
    *Breast Cancer Wisconsin (Diagnostic)* dataset.

    ⚠️ This tool is for **educational purposes only**  
    and **not a medical diagnosis**.
    """
)

# ===============================
# MAIN TITLE
# ===============================
st.markdown(
    "<h1 style='text-align: center;'>🩺 Breast Cancer Prediction Chatbot</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Enter tumor characteristics to predict if it is <b>Benign</b> or <b>Malignant</b>.</p>",
    unsafe_allow_html=True
)

st.divider()

# ===============================
# INPUT FORM
# ===============================
st.subheader("📥 Tumor Features")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", min_value=0.0, step=0.01)
    texture_mean = st.number_input("Texture Mean", min_value=0.0, step=0.01)
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, step=0.01)

with col2:
    area_mean = st.number_input("Area Mean", min_value=0.0, step=1.0)
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, step=0.001)

# ===============================
# PREDICTION BUTTON
# ===============================
st.divider()

if st.button("🔍 Predict", use_container_width=True):

    # Prepare input
    X = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean
    ]])

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ **Malignant Tumor Detected**")
        st.write(f"**Confidence:** {probability[0][1]*100:.2f}%")
    else:
        st.success("✅ **Benign Tumor Detected**")
        st.write(f"**Confidence:** {probability[0][0]*100:.2f}%")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size: 13px;'>"
    "Developed by Yassine Rezgui • Machine Learning Project</p>",
    unsafe_allow_html=True
)
