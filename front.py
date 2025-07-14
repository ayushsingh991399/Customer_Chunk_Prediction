from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import time

# Set Streamlit layout
st.set_page_config(layout="wide", page_title="Customer Churn Prediction")

# Custom CSS styles
st.markdown(
    """
    <style>
        .stButton>button {
            background: linear-gradient(90deg, #6a11cb, #2575fc);
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
            cursor: pointer;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #2575fc, #6a11cb);
            transform: scale(1.05);
        }
        .prediction-result {
            font-size: 22px;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
        }
        .churned {
            background-color: #ff4b4b;
            color: white;
        }
        .retained {
            background-color: #4CAF50;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

# UI Header
st.title("✨ Customer Churn Prediction")
st.subheader("Will the customer stay or leave? Let’s find out!")

st.markdown("---")
st.header("📋 Enter Customer Information")

# Layout columns for better input display
col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)

with col2:
    balance = st.number_input("Balance Amount", min_value=0.0, max_value=250000.0, value=50000.0)
    num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
    estimated_salary = st.number_input("Estimated Salary", min_value=10000.0, max_value=200000.0, value=60000.0)

with col3:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])

# One-hot encode categorical features
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

# Convert to DataFrame
input_df = pd.DataFrame([features])
input_scaled = input_df.copy()
input_scaled[scale_vars] = scaler.transform(input_df[scale_vars])

# Ensure feature order matches model
expected_feature_order = model.get_booster().feature_names
input_scaled = input_scaled[expected_feature_order]

# Predict Button
st.markdown("---")
st.header("🔍 Prediction")
if st.button("🚀 Predict Now"):
    with st.spinner("⏳ Analyzing data..."):
        time.sleep(2)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    prediction_label = "Churned" if prediction == 1 else "Retained"

    st.markdown(
        f"<div class='prediction-result {'churned' if prediction == 1 else 'retained'}'>"
        f"{'⚠️' if prediction == 1 else '✅'} <b>Predicted Status:</b> {prediction_label}</div>",
        unsafe_allow_html=True
    )
    st.write(f"📌 **Probability of Churn:** {probabilities[1]:.2%}")
    st.write(f"📌 **Probability of Retention:** {probabilities[0]:.2%}")

# Optional Tips
with st.expander("💡 What influences churn?"):
    st.write("""
    - **Credit Score**: Lower scores increase churn risk.
    - **Age & Tenure**: Younger customers with short tenure are more likely to leave.
    - **Balance & Salary**: High balance with low salary can be risky.
    - **Number of Products**: More products = more loyalty.
    - **Activity**: Active members are less likely to churn.
    """)
