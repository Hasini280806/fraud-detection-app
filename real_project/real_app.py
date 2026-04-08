 import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ================== LOAD MODEL ==================
model = joblib.load("real_project/model/fraud_model.pkl")
scaler = joblib.load("real_project/model/scaler.pkl")
pca = joblib.load("real_project/model/pca.pkl")

# ================== PAGE CONFIG ==================
st.set_page_config(layout="wide")

# ================== STYLE ==================
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

# ================== TITLE ==================
st.title("💳 Real-Time Fraud Detection System")

# ================== INPUT ==================
st.sidebar.header("Transaction Input")

amount = st.sidebar.number_input("Amount", value=100.0)
time = st.sidebar.number_input("Time (seconds)", value=10000.0)
transaction_type = st.sidebar.selectbox("Transaction Type", [0, 1, 2, 3, 4])

location_risk = st.sidebar.slider("Location Risk", 0, 5, 2)
device_risk = st.sidebar.slider("Device Risk", 0, 5, 1)
past_transactions = st.sidebar.slider("Past Transactions", 0, 10, 3)

# ================== PREPARE INPUT ==================
input_data = np.array([[amount, time, transaction_type,
                        location_risk, device_risk, past_transactions]])

# Scale + PCA
input_scaled = scaler.transform(input_data)
input_pca = pca.transform(input_scaled)

# ================== BUTTON ==================
detect = st.button("🚀 Detect Fraud")

# ================== PREDICTION ==================
if detect:
    prediction = model.predict(input_pca)
    prob = float(prediction[0])

    if prob > 0.5:
        st.error(f"🚨 Fraud Detected | Confidence: {prob:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction | Confidence: {1-prob:.2f}")

    # ================== GAUGE ==================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Fraud Probability"},
        gauge={'axis': {'range': [0, 1]}}
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================== FEATURE IMPORTANCE ==================
    st.subheader("📊 Feature Importance")

    features = ["Amount", "Time", "Type", "Location", "Device", "History"]
    values = input_data[0]

    fig_bar = go.Figure([go.Bar(
        x=values,
        y=features,
        orientation='h'
    )])

    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig_bar, use_container_width=True)   



