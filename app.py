# ===============================
# Breast Cancer Prediction Chatbot (Interactive Dashboard)
# ===============================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Breast Cancer Prediction Chatbot",
    page_icon="🩺",
    layout="wide"
)

# ===============================
# DARK MODE + STYLING CSS
# ===============================
st.markdown("""
<style>
body {background-color: #0e1117; color: #fafafa; font-family: 'Arial', sans-serif;}
.stApp {background-color: #0e1117;}
.stButton>button {background-color:#4CAF50; color:white; height:3em; width:100%; border-radius:10px; font-weight:bold;}
.prediction-card {
    background-color: #1e222b; 
    padding: 15px; 
    border-radius: 15px; 
    margin-bottom: 20px;
}
.prediction-label {
    font-weight:bold; 
    margin-bottom:5px; 
    font-size:16px;
}
.bar-container {
    background-color:#e0e0e0; 
    border-radius:10px; 
    width:100%; 
    height:25px;
}
.bar-fill {
    height:25px; 
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data/data.csv")
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X_all = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
X_all.fillna(X_all.mean(), inplace=True)

# ===============================
# SELECT TOP 10 FEATURES
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

# ===============================
# PREPARE DATA
# ===============================
X = df[top10_features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# TRAIN MODELS
# ===============================
log_model = LogisticRegression(max_iter=1000)
knn_model = KNeighborsClassifier(n_neighbors=5)

log_model.fit(X_train_scaled, y_train)
knn_model.fit(X_train_scaled, y_train)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🧠 Breast Cancer Chatbot")
st.sidebar.info("Logistic Regression + KNN\nTop 10 Features\nEducational Use Only")

# ===============================
# TITLE
# ===============================
st.markdown("<h1 style='text-align:center; margin-bottom:10px;'>🩺 Breast Cancer Prediction Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; margin-bottom:20px;'>Fill tumor features and click Predict</p>", unsafe_allow_html=True)

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
# PDF GENERATOR (with top padding)
# ===============================
def generate_pdf(result, confidence, features):
    file_path = "prediction_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    # Top padding
    content.append(Spacer(1, 50))

    content.append(Paragraph("<b>Breast Cancer Prediction Report</b>", styles["Title"]))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"<b>Result:</b> {result}", styles["Normal"]))
    content.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["Normal"]))
    content.append(Spacer(1, 15))
    content.append(Paragraph("<b>Input Features:</b>", styles["Normal"]))
    for k, v in features.items():
        content.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    doc.build(content)
    return file_path

# ===============================
# GRADIENT COLOR FUNCTION
# ===============================
def gradient_color(perc):
    if perc > 55:
        return "#FF4B4B"  # red for Malignant
    else:
        intensity = int(75 + 180 * (perc/100))
        return f'rgb(75, {intensity}, 75)'  # green gradient

# ===============================
# PREDICTION DASHBOARD
# ===============================
if st.button("🔍 Predict", use_container_width=True):
    X_input = np.array([list(input_values.values())])
    X_input_scaled = scaler.transform(X_input)

    # Logistic Regression
    prob_log = log_model.predict_proba(X_input_scaled)[0]
    perc_log = prob_log[1]*100
    pred_log = 1 if perc_log > 55 else 0

    # KNN
    prob_knn = knn_model.predict_proba(X_input_scaled)[0]
    perc_knn = prob_knn[1]*100
    pred_knn = 1 if perc_knn > 55 else 0

    st.subheader("🔹 Model Predictions Dashboard")
    col1, col2 = st.columns(2)

    # Logistic Regression Card
    with col1:
        color_log = gradient_color(perc_log)
        label_log = f"Logistic Regression: {'Malignant' if pred_log==1 else 'Benign'} ({perc_log:.2f}%)"
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label" style="color:{color_log}">{label_log}</div>
            <div class="bar-container">
                <div class="bar-fill" style="width:{perc_log}%; background-color:{color_log};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # KNN Card
    with col2:
        color_knn = gradient_color(perc_knn)
        label_knn = f"KNN: {'Malignant' if pred_knn==1 else 'Benign'} ({perc_knn:.2f}%)"
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label" style="color:{color_knn}">{label_knn}</div>
            <div class="bar-container">
                <div class="bar-fill" style="width:{perc_knn}%; background-color:{color_knn};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # PDF report
    final_result = "Malignant" if max(perc_log, perc_knn) > 55 else "Benign"
    final_confidence = max(perc_log, perc_knn)
    pdf_path = generate_pdf(final_result, final_confidence, input_values)
    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="breast_cancer_report.pdf", mime="application/pdf")

    # Feature Importance
    st.subheader("📊 Top 10 Feature Importance (Logistic Regression)")
    fi_values = np.abs(log_model.coef_[0])
    fi_df = pd.DataFrame({'Feature': top10_features, 'Importance': fi_values})
    fi_df = fi_df.sort_values(by='Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi_df['Feature'], fi_df['Importance'], color='skyblue')
    ax.set_title("Top 10 Features Importance")
    st.pyplot(fig)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:13px;'>Developed by Yessine Rezgui • ML Project</p>", unsafe_allow_html=True)
