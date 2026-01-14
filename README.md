🧠 Breast Cancer Prediction Project

This project implements a machine learning system to predict whether a breast tumor is benign or malignant using the Breast Cancer Wisconsin (Diagnostic) Dataset.
The project includes data analysis, model training, evaluation, comparison, and a web-based chatbot UI built with Streamlit.

📌 Project Objectives

Perform data preprocessing and visualization

Train and evaluate Logistic Regression and K-Nearest Neighbors (KNN)

Compare models using cross-validation accuracy

Save the best trained model and scaler

Deploy a web chatbot UI for real-time predictions

📊 Dataset

Source: Kaggle – Breast Cancer Wisconsin (Diagnostic)

Samples: 569

Features: 30 numerical features

Target:

0 → Benign

1 → Malignant

🗂 Project Structure
cancer-project-v2/
│
├── data/
│   └── data.csv
│
├── models/
│   ├── logistic_model.pkl
│   └── scaler.pkl
│
├── sample.ipynb        # Jupyter notebook (EDA, training, evaluation)
├── app.py              # Streamlit web chatbot UI
├── requirements.txt
└── README.md

🧪 Machine Learning Models
✔ Logistic Regression

Simple and interpretable

Performs very well on linearly separable medical data

Selected as the best model based on cross-validation accuracy

✔ K-Nearest Neighbors (KNN)

Distance-based classifier

Good performance but more sensitive to scaling and noise

📈 Evaluation Metrics

Accuracy

Confusion Matrix

Precision, Recall, F1-score

5-Fold Cross-Validation

🏆 Model Comparison

The models were compared using mean cross-validation accuracy.

Example output:

Logistic Regression CV Accuracy ≈ 0.96
KNN CV Accuracy ≈ 0.94


➡ Logistic Regression was selected as the final model.

💾 Saved Files

After training, the following files are generated:

models/logistic_model.pkl → trained Logistic Regression model

models/scaler.pkl → fitted StandardScaler

These files are used by the chatbot UI.

🌐 Web Chatbot (Streamlit)

A professional web interface allows users to input tumor features and receive predictions instantly.

Run the chatbot:
streamlit run app.py


The app opens automatically in your browser.

🛠 Installation & Setup
1️⃣ Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

2️⃣ Install dependencies
pip install -r requirements.txt

📦 Requirements
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
streamlit
jupyter

🚀 How to Use the Project

Open and run sample.ipynb to:

Explore data

Train models

Save the model and scaler

Run app.py to launch the web chatbot

Enter feature values and get prediction:

✅ Benign

⚠️ Malignant

⚠️ Disclaimer

This project is for educational purposes only.
It is not a medical diagnostic tool and should not be used for real medical decisions.

👨‍💻 Author

Yassine Rezgui
Machine Learning / Software Engineering Student

⭐ Future Improvements

Add all 30 features to the chatbot UI

Add ROC-AUC curve

Hyperparameter tuning

Deploy online (Streamlit Cloud)