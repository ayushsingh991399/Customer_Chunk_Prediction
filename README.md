# 💼 Customer Churn Prediction App

This is an interactive **Streamlit web application** that predicts whether a bank customer is likely to churn (leave the service) based on various financial and demographic features.

<img src="https://cdn.pixabay.com/photo/2019/04/11/18/13/banner-4113732_1280.jpg" width="100%" alt="Banner"/>

---

## 🚀 Features

- 🎨 Attractive, responsive UI with banner image
- 📋 Input form for key customer data
- 🔮 Real-time churn prediction using a trained machine learning model
- 📊 Custom insights based on user inputs
- 📈 Probability of churn and retention
- 💡 Helpful FAQ and feature explanations

---

## 📦 Files Included

- `churn_app.py` – The main Streamlit application
- `best_model.pkl` – Pre-trained churn prediction model
- `scaler.pkl` – MinMaxScaler used for input scaling
- `requirements.txt` – Dependencies list (optional for deployment)
- `README.md` – You're reading it 😊

---

## 🧠 Model Info

The model was trained on a historical customer churn dataset using features such as:

- Credit Score  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Has Credit Card  
- Is Active Member  
- Estimated Salary

Model type: `XGBoostClassifier`  
Scaler: `MinMaxScaler`

---

## ▶️ How to Run Locally

1. 🔽 Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-churn-app.git
   cd customer-churn-app
```
📦 Install dependencies:

```bash

pip install -r requirements.txt
```
🚀 Run the Streamlit app:
```bash

streamlit run churn_app.py
```
