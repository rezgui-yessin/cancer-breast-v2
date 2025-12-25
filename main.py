# import joblib
# import numpy as np

# # Load model and scaler
# model = joblib.load("models/logistic_model.pkl")
# scaler = joblib.load("models/scaler.pkl")

# print("🧠 Breast Cancer Prediction Chatbot")
# print("Type 'exit' to quit\n")

# # Use 5 features (simple & safe)
# features = [
#     "radius_mean",
#     "texture_mean",
#     "perimeter_mean",
#     "area_mean",
#     "smoothness_mean"
# ]

# while True:
#     values = []
#     for feature in features:
#         val = input(f"Enter {feature}: ")
#         if val.lower() == "exit":
#             print("👋 Goodbye!")
#             exit()
#         values.append(float(val))

#     X = np.array([values])
#     X_scaled = scaler.transform(X)

#     prediction = model.predict(X)

#     if prediction[0] == 1:
#         print("⚠️ Result: MALIGNANT (Cancer detected)\n")
#     else:
#         print("✅ Result: BENIGN (No cancer)\n")
