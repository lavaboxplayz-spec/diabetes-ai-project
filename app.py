import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
scaler = joblib.load("scaler.joblib")
model = joblib.load("diabetes_logreg_model.joblib")

st.title("Diabetes Prediction App")
st.write("Enter the following details to predict diabetes risk:")

pregnancies = st.number_input("Pregnancies", min_value=0.0, step=1.0)
glucose = st.number_input("Glucose", min_value=0.0, step=1.0)
bloodpressure = st.number_input("Blood Pressure", min_value=0.0, step=1.0)
skinthickness = st.number_input("Skin Thickness", min_value=0.0, step=1.0)
insulin = st.number_input("Insulin", min_value=0.0, step=1.0)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0.0, step=1.0)

input_features = np.array([
    pregnancies, glucose, bloodpressure, skinthickness,
    insulin, bmi, dpf, age
]).reshape(1, -1)

if st.button("Predict"):
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"Prediction: Diabetic (Probability: {probability:.2f})")
    else:
        st.success(f"Prediction: Non-Diabetic (Probability: {probability:.2f})")
