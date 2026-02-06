import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load model
model = load_model("models/churn_model")

st.title("📊 Customer Churn Prediction App")

st.write("Enter customer details to predict churn")

# User Inputs
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Create input dataframe
input_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "PhoneService": phone_service,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method
}])

# Predict
if st.button("Predict Churn"):
    prediction = predict_model(model, data=input_data)
    result = prediction["prediction_label"][0]

    if result == "Yes":
        st.error(" Customer is likely to churn")
    else:
        st.success(" Customer is NOT likely to churn")
