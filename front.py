from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import time
from PIL import Image

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Optional banner image (local or hosted)
st.image("https://cdn.pixabay.com/photo/2020/04/23/13/57/business-5087581_1280.jpg", use_container_width=True)

# App Header
st.title("✨ Customer Churn Prediction")
st.subheader("Will the customer stay or leave? Let’s find out!")

# Load model and scaler
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

# User Inputs
st.markdown("---")
st.header("📋 Enter Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input("Credit Score", 300, 1000, 600)
    age = st.number_input("Age", 18, 150, 35)
    tenure = st.number_input("Tenure (Years)", 0, 90, 3)

with col2:
    balance = st.number_input("Balance Amount", 0.0, 2500000.0, 50000.0)
    num_products = st.number_input("Number of Products", 1, 50, 2)
    estimated_salary = st.number_input("Estimated Salary", 0.0, 2000000.0, 60000.0)

with col3:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])

# Feature engineering
features = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "EstimatedSalary": estimated_salary,
    "Geography_France": 1 if geography == "France" else 0,
    "Geography_Germany": 1 if geography == "Germany" else 0,
    "Geography_Spain": 1 if geography == "Spain" else 0,
    "Gender_Female": 1 if gender == "Female" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,
    "HasCrCard_1": 1 if has_credit_card == "Yes" else 0,
    "HasCrCard_0": 1 if has_credit_card == "No" else 0,
    "IsActiveMember_1": 1 if is_active_member == "Yes" else 0,
    "IsActiveMember_0": 1 if is_active_member == "No" else 0
}

input_df = pd.DataFrame([features])
input_scaled = input_df.copy()
input_scaled[scale_vars] = scaler.transform(input_df[scale_vars])

# Match expected feature order
expected_feature_order = model.get_booster().feature_names
input_scaled = input_scaled[expected_feature_order]

# Prediction Section
st.markdown("---")
st.header("🔍 Prediction")

if st.button("🚀 Predict Now"):
    with st.spinner("⏳ Analyzing data..."):
        time.sleep(2)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    prediction_label = "Churned" if prediction == 1 else "Retained"

    # Styled result
    st.markdown(
        f"<div style='padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; "
        f"background-color: {'#ff4b4b' if prediction == 1 else '#4CAF50'}; color: white;'>"
        f"{'⚠️' if prediction == 1 else '✅'} Predicted Status: {prediction_label}</div>",
        unsafe_allow_html=True
    )
    st.write(f"📌 **Probability of Churn:** {probabilities[1]:.2%}")
    st.write(f"📌 **Probability of Retention:** {probabilities[0]:.2%}")

    # 💡 Custom Insights
    st.markdown("### 📌 Custom Insights")
    insights = []

    if credit_score < 500:
        insights.append("• Low credit score may increase churn risk.")
    if age < 25:
        insights.append("• Younger customers are more likely to churn.")
    if tenure < 2:
        insights.append("• Short tenure indicates low brand loyalty.")
    if balance > 100000 and estimated_salary < 40000:
        insights.append("• High balance with low salary may lead to financial dissatisfaction.")
    if num_products == 1:
        insights.append("• Single-product customers are more likely to leave.")
    if is_active_member == "No":
        insights.append("• Inactive members have a higher churn rate.")

    if num_products >= 3:
        insights.append("✅ Having multiple products suggests strong engagement.")
    if is_active_member == "Yes":
        insights.append("✅ Active usage is a sign of customer satisfaction.")
    if has_credit_card == "Yes":
        insights.append("✅ Customers with credit cards show higher retention.")
    if tenure > 5:
        insights.append("✅ Long tenure indicates high loyalty.")

    if insights:
        for insight in insights:
            st.write(insight)
    else:
        st.write("🔍 No strong churn indicators found. Customer profile appears stable.")

# FAQ Section
with st.expander("💡 What influences churn?"):
    st.write("""
    - **Credit Score**: Lower scores increase churn risk.
    - **Age & Tenure**: Younger customers with short tenure are more likely to leave.
    - **Balance & Salary**: High balance with low salary can be risky.
    - **Number of Products**: More products = more loyalty.
    - **Activity**: Active members are less likely to churn.
    """)
