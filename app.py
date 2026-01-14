# ===============================
# Breast Cancer Prediction Chatbot (Top 10 Features + Visualizations)
# ===============================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ===============================
# DARK MODE CSS
# ===============================
st.markdown("""
    <style>
    body {background-color: #0e1117; color: #fafafa;}
    .stApp {background-color: #0e1117;}
    </style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data/data.csv")
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# ===============================
# CLEAN DATA (HANDLE inf/NAN)
# ===============================
X_all = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
X_all.fillna(X_all.mean(), inplace=True)

# ===============================
# TEMPORARY MODEL TO SELECT TOP 10 FEATURES
# ===============================
scaler_all = StandardScaler()
X_scaled_all = scaler_all.fit_transform(X_all)

temp_model = LogisticRegression(max_iter=1000)
temp_model.fit(X_scaled_all, y)

feature_importance = pd.DataFrame({
    'Feature': X_all.columns,
    'Coefficient': np.abs(temp_model.coef_[0])
}).sort_values(by='Coefficient', ascending=False)

top10_features = feature_importance['Feature'].head(10).tolist()
print("Top 10 Features:", top10_features)

# ===============================
# USE TOP 10 FEATURES
# ===============================
X = df[top10_features]

# Split data for evaluation
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# New scaler for top10
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train final models
log_model = LogisticRegression(max_iter=1000)
knn_model = KNeighborsClassifier(n_neighbors=5)

log_model.fit(X_train_scaled, y_train)
knn_model.fit(X_train_scaled, y_train)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🧠 Breast Cancer Chatbot")
st.sidebar.info(
    "Logistic Regression + KNN\nTop 10 Features\nEducational Use Only"
)

# ===============================
# TITLE
# ===============================
st.markdown("<h1 style='text-align:center;'>🩺 Breast Cancer Prediction Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Fill tumor features and click Predict</p>", unsafe_allow_html=True)

# ===============================
# INPUT SLIDERS
# ===============================
st.subheader("🔹 Enter Tumor Features")
input_values = {}
for feature in top10_features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    input_values[feature] = st.slider(feature, min_val, max_val, mean_val)


# ===============================
# PDF GENERATOR
# ===============================
def generate_pdf(result, confidence, features):
    file_path = "prediction_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>Breast Cancer Prediction Report</b>", styles["Title"]))
    content.append(Paragraph("<br/>", styles["Normal"]))
    content.append(Paragraph(f"<b>Result:</b> {result}", styles["Normal"]))
    content.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["Normal"]))
    content.append(Paragraph("<br/><b>Input Features:</b>", styles["Normal"]))
    for k, v in features.items():
        content.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    doc.build(content)
    return file_path


# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict", use_container_width=True):
    X_input = np.array([list(input_values.values())])
    X_input_scaled = scaler.transform(X_input)

    # Logistic Prediction
    pred_log = log_model.predict(X_input_scaled)[0]
    prob_log = log_model.predict_proba(X_input_scaled)[0]

    # KNN Prediction
    pred_knn = knn_model.predict(X_input_scaled)[0]
    prob_knn = knn_model.predict_proba(X_input_scaled)[0]

    # Show Results (Logistic)
    st.subheader("📊 Logistic Regression Prediction")
    if pred_log == 1:
        st.error("⚠️ Malignant Tumor Detected")
        st.progress(int(prob_log[1] * 100))
        confidence_log = prob_log[1] * 100
        result_log = "Malignant"
    else:
        st.success("✅ Benign Tumor Detected")
        st.progress(int(prob_log[0] * 100))
        confidence_log = prob_log[0] * 100
        result_log = "Benign"

    # Show Results (KNN)
    st.subheader("📊 KNN Prediction")
    if pred_knn == 1:
        st.error("⚠️ Malignant Tumor Detected")
        st.progress(int(prob_knn[1] * 100))
        confidence_knn = prob_knn[1] * 100
        result_knn = "Malignant"
    else:
        st.success("✅ Benign Tumor Detected")
        st.progress(int(prob_knn[0] * 100))
        confidence_knn = prob_knn[0] * 100
        result_knn = "Benign"

    # ===============================
    # PDF Download
    # ===============================
    pdf_path = generate_pdf(result_log, confidence_log, input_values)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="breast_cancer_report.pdf", mime="application/pdf")

    # ===============================
    # FEATURE IMPORTANCE BAR PLOT
    # ===============================
    st.subheader("📊 Top 10 Feature Importance (Logistic Regression)")
    fi_values = np.abs(log_model.coef_[0])
    fi_df = pd.DataFrame({'Feature': top10_features, 'Importance': fi_values})
    fi_df = fi_df.sort_values(by='Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi_df['Feature'], fi_df['Importance'], color='skyblue')
    ax.set_title("Top 10 Features Importance")
    st.pyplot(fig)

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    st.subheader("📊 Confusion Matrix (Logistic Regression)")
    y_pred_test = log_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_test)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ===============================
    # MODEL COMPARISON
    # ===============================
    st.subheader("📊 Model Comparison (Accuracy)")
    acc_log = accuracy_score(y_test, log_model.predict(X_test_scaled))
    acc_knn = accuracy_score(y_test, knn_model.predict(X_test_scaled))

    fig, ax = plt.subplots()
    sns.barplot(x=['Logistic Regression', 'KNN'], y=[acc_log, acc_knn], palette='viridis')
    ax.set_ylim(0.9, 1)
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:13px;'>Developed by Yassine Rezgui • ML Project</p>",
            unsafe_allow_html=True)
