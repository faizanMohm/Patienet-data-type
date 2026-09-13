import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("risk_model.pkl")


# App title
st.title("Healthcare Risk Stratification App")

st.write("Enter patient details to predict the risk.")


# Patient inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=50
)

stayfrom = st.number_input(
    "Length of Stay (Days)",
    min_value=0,
    max_value=500,
    value=100
)

treatment_cost = st.number_input(
    "Treatment Cost",
    min_value=0.0,
    value=30000.0
)

abnormal_lab_count = st.number_input(
    "Abnormal Lab Count",
    min_value=0,
    max_value=20,
    value=0
)


# Prediction
if st.button("Predict Risk"):

    input_data = pd.DataFrame({
        "Age": [age],
        "stayfrom": [stayfrom],
        "TreatmentCost": [treatment_cost],
        "AbnormalLabCount": [abnormal_lab_count]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # Display result
    if prediction == 1:
        st.error("High Risk / Deceased Prediction")
    else:
        st.success("Low Risk / Non-Deceased Prediction")


    st.write(
        f"Risk Probability: {probability:.2%}"
    )

    
