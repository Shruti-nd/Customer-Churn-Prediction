import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.markdown("""
    <style>
    /* Background and app layout */
    .stApp {
        background: linear-gradient(120deg, #fffde7, #fff9c4);
        padding: 2rem;
    }

    /* Title and headers */
    h1, h2, h3 {
        color: #795548; /* brown for contrast */
        font-family: 'Segoe UI', sans-serif;
    }

    /* Text and body */
    body {
        color: #4e342e;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Input widgets */
    .stNumberInput input, .stSelectbox, .stTextInput input {
        background-color: #fffde7;
        border: 1px solid #d7ccc8;
        border-radius: 10px;
        color: #4e342e;
    }

    /* Buttons */
    .stButton>button {
        background-color: #fbc02d;
        color: black;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #fdd835;
        color: black;
    }

    /* Metric and success messages */
    .stMetric, .stAlert-success {
        background-color: #fffde7 !important;
        border-left: 5px solid #fbc02d;
    }

    /* Expander (for input summary) */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        color: #6d4c41;
    }

    /* Divider color */
    hr {
        border-top: 1px solid #ffe082;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.markdown("""
This tool helps you estimate whether a customer is likely to **churn** or **stay** with the service based on a few key inputs.

🔍 **Churn** means the customer has **stopped using** the service.  
✅ **Not Churn** means the customer is **still active**.

""")

st.subheader("📋 Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=130, value=10)

with col2:
    monthlycharge = st.number_input("Monthly Charges (₹)", min_value=30, max_value=150)
    gender = st.selectbox("Gender", ["Male", "Female"])

st.divider()

st.markdown("### 📝 Confirm Your Details")

with st.expander("Click to view entered details"):
    st.write(f"- **Age:** {age}")
    st.write(f"- **Gender:** {gender}")
    st.write(f"- **Tenure:** {tenure} months")
    st.write(f"- **Monthly Charges:** ₹{monthlycharge}")

if st.button("🔮 Predict"):
    gender_selected = 1 if gender == "Female" else 0
    input_data = [age, gender_selected, tenure, monthlycharge]
    input_array = scaler.transform([np.array(input_data)])

    prediction = model.predict(input_array)[0]
    predicted = "Churn ❌" if prediction == 1 else "Not Churn ✅"

    st.success(f"**Prediction Result:** {predicted}")
    st.info("🧠 Note: This prediction is based on historical data and is probabilistic in nature.")
else:
    st.caption("⬅️ Fill in the details and click the Predict button to see results.")

    st.divider()
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📘 About")
    st.markdown(
        """
        <div style='font-size: 0.85rem; line-height: 1.4; color: #4e342e;'>
        This app predicts whether a customer will churn (leave) based on service details like age, gender, tenure, and monthly charges.  
        Churn prediction helps businesses improve customer retention.
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown("#### 📈 Model Info")
    st.markdown(
        """
        <div style='font-size: 0.85rem; line-height: 1.4; color: #4e342e;'>
        - Model: Logistic Regression or Random Forest  
        - Accuracy: ~87% on test data  
        - Scaler: StandardScaler  
        - Data: 1,000 telecom customer records
        </div>
        """,
        unsafe_allow_html=True
    )


