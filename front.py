import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model (replace with your actual model file)
# with open("churn_model.pkl", "rb") as file:
#     model = pickle.load(file)

# For demo purposes, let's simulate prediction
def fake_predict(data):
    return int(sum(data) % 2 == 0)  # Dummy prediction logic

# App Title
st.set_page_config(page_title="Customer Churn Predictor", page_icon="💼", layout="centered")

st.title("💼 Customer Churn Prediction App")
st.subheader("🔍 Predict whether a customer will leave the bank or not")

# Sidebar for user input
st.sidebar.header("📝 Input Customer Details")

def user_input_features():
    credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
    geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 18, 92, 35)
    tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)
    balance = st.sidebar.slider("Balance", 0.0, 250000.0, 50000.0)
    num_of_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.sidebar.selectbox("Active Member?", ["Yes", "No"])
    estimated_salary = st.sidebar.slider("Estimated Salary", 10000.0, 200000.0, 50000.0)

    # Encode categorical
    geography_map = {"France": 0, "Germany": 1, "Spain": 2}
    gender_map = {"Male": 1, "Female": 0}
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0

    features = [
        credit_score,
        geography_map[geography],
        gender_map[gender],
        age,
        tenure,
        balance,
        num_of_products,
        has_cr_card,
        is_active_member,
        estimated_salary
    ]
    return np.array(features).reshape(1, -1)

input_data = user_input_features()

# Submit button
if st.button("🔮 Predict Churn"):
    # prediction = model.predict(input_data)
    prediction = fake_predict(input_data)

    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **likely to stay**.")
else:
    st.info("👉 Enter the customer details in the sidebar and click **Predict Churn**.")

# Footer
st.markdown("""
---
🔗 Built with Streamlit • Made by [Your Name]  
""")
