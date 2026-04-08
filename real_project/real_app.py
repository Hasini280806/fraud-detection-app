import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# ================= LOAD MODEL =================
model = load_model("real_project/model/fraud_model.h5")
scaler = joblib.load("real_project/model/scaler.pkl")
pca = joblib.load("real_project/model/pca.pkl")

# ================= PAGE CONFIG =================
st.set_page_config(layout="wide")

# ================= STYLE =================
st.markdown("""
<style>
h1 {text-align: center;}

.stButton > button {
    display: block;
    margin: 20px auto;
    background: linear-gradient(90deg,#00C9FF,#92FE9D);
    color: black;
    font-size: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("💳 Real-Time Fraud Detection System")

# ================= METRICS =================
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", "10,000+")
col2.metric("Fraud Cases", "250")
col3.metric("Detection Accuracy", "98%")

st.markdown("---")

# ================= INPUT SECTION =================
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Amount", value=500.0)
    oldbalanceOrg = st.number_input("Old Balance (Sender)", value=1000.0)
    newbalanceOrig = st.number_input("New Balance (Sender)", value=500.0)

with col2:
    oldbalanceDest = st.number_input("Old Balance (Receiver)", value=0.0)
    newbalanceDest = st.number_input("New Balance (Receiver)", value=500.0)
    transaction_type = st.selectbox("Transaction Type", [0, 1])  # 0=normal,1=suspicious

time = st.number_input("Transaction Time", value=10000)

# ================= PREDICT BUTTON =================
if st.button("🔍 Check Fraud"):

    # Create input array
    input_data = np.array([[time, transaction_type, amount,
                            oldbalanceOrg, newbalanceOrig,
                            oldbalanceDest, newbalanceDest]])

    # Scale
    scaled_data = scaler.transform(input_data)

    # PCA
    pca_data = pca.transform(scaled_data)

    # Predict
    prediction = model.predict(pca_data)
    result = int(prediction[0][0] > 0.5)

    # ================= RESULT =================
    if result == 1:
        st.error("🚨 Fraud Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    # ================= GAUGE =================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction[0][0] * 100,
        title={'text': "Fraud Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if result == 1 else "green"},
        }
    ))

    st.plotly_chart(fig)