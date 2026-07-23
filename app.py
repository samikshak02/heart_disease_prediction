import streamlit as st
import pandas as pd
import joblib

# Load model, scaler and columns
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler (1).pkl")
columns = joblib.load("columns (1).pkl")

st.title("Heart Disease Prediction ❤️")

Age = st.number_input("Age", 1, 100, 40)
Sex = st.selectbox("Sex", ["M", "F"])
ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
RestingBP = st.number_input("Resting Blood Pressure", 50, 250, 120)
Cholesterol = st.number_input("Cholesterol", 0, 700, 200)
FastingBS = st.selectbox("Fasting Blood Sugar", [0, 1])
RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
MaxHR = st.number_input("Maximum Heart Rate", 50, 250, 150)
ExerciseAngina = st.selectbox("Exercise Angina", ["N", "Y"])
Oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0)
ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):

    sample = pd.DataFrame([{
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope
    }])

    sample = pd.get_dummies(sample)
    sample = sample.reindex(columns=columns, fill_value=0)

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("Heart Disease: Yes")
    else:
        st.success("Heart Disease: No")