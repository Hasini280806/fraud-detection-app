import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# LOAD MODEL
model = joblib.load("real_project/model/fraud_model.pkl")
scaler = joblib.load("real_project/model/scaler.pkl")

st.set_page_config(layout="wide")

st.title("💳 Fraud Detection System")

# Inputs (MATCH TRAIN DATA)
step = st.number_input("Step", value=1)
type_val = st.selectbox("Transaction Type", [0, 1, 2, 3, 4])  # encoded
amount = st.number_input("Amount", value=1000.0)
oldbalanceOrg = st.number_input("Old Balance Sender", value=1000.0)
newbalanceOrig = st.number_input("New Balance Sender", value=500.0)
oldbalanceDest = st.number_input("Old Balance Receiver", value=0.0)
newbalanceDest = st.number_input("New Balance Receiver", value=500.0)

if st.button("Check Fraud"):

    data = np.array([[step, type_val, amount,
                      oldbalanceOrg, newbalanceOrig,
                      oldbalanceDest, newbalanceDest]])

    scaled = scaler.transform(data)

    pred = model.predict(scaled)[0]

    if pred == 1:
        st.error("🚨 Fraud Detected")
    else:
        st.success("✅ Legit Transaction")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=int(pred)*100,
        title={'text': "Fraud Result"},
        gauge={'axis': {'range': [0, 100]}}
    ))

    st.plotly_chart(fig)

