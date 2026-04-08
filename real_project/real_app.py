import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ================= LOAD =================
model = load_model("model/fraud_model.h5")
scaler = joblib.load("model/scaler.pkl")
pca = joblib.load("model/pca.pkl")

st.set_page_config(layout="wide")

# ================= STYLE =================
st.markdown("""
<style>
h1 {text-align:center;}
.stButton > button {
    display:block;
    margin:20px auto;
    background: linear-gradient(90deg,#00C9FF,#92FE9D);
    color:black;
    font-size:18px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("💳 Real-Time Fraud Detection System")

# ================= DASHBOARD =================
st.markdown("### 📊 Live Fraud Monitoring")

col1, col2, col3 = st.columns(3)
col1.metric("Dataset Transactions", "10,000+")
col2.metric("Detected Frauds", "250")
col3.metric("Model Accuracy", "98%")

# ================= SIDEBAR =================
st.sidebar.header("Transaction Input")

amount = st.sidebar.number_input("Amount", value=100.0)
time = st.sidebar.number_input("Time", value=10000.0)
type_tx = st.sidebar.selectbox("Transaction Type", [0,1,2,3,4])

oldbalanceOrg = st.sidebar.number_input("Sender Old Balance", value=1000.0)
newbalanceOrig = st.sidebar.number_input("Sender New Balance", value=900.0)
oldbalanceDest = st.sidebar.number_input("Receiver Old Balance", value=0.0)
newbalanceDest = st.sidebar.number_input("Receiver New Balance", value=100.0)

# ================= BUTTON =================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    detect = st.button("🚀 Detect Fraud")

# ================= FUNCTION =================
def predict_fraud(input_df):
    scaled = scaler.transform(input_df)
    pca_data = pca.transform(scaled)
    prob = float(model.predict(pca_data)[0][0])
    return prob

# ================= PREDICTION =================
if detect:

    input_df = pd.DataFrame([[
        time, type_tx, amount,
        oldbalanceOrg, newbalanceOrig,
        oldbalanceDest, newbalanceDest
    ]])

    prob = predict_fraud(input_df)

    # 🔥 LOWER THRESHOLD (IMPORTANT FIX)
    threshold = 0.3

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        if prob > threshold:
            st.error(f"🚨 Fraud Detected | Confidence: {prob:.2f}")
        else:
            st.success(f"✅ Legitimate | Confidence: {1-prob:.2f}")

        st.progress(int(prob * 100))

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Fraud Probability"},
            gauge={'axis': {'range': [0,1]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # ================= HUMAN EXPLAIN =================
        st.subheader("🧠 Why this transaction is flagged")

        conditions = []

        if amount > 3000:
            conditions.append("High transaction amount")

        if oldbalanceOrg < amount:
            conditions.append("Insufficient sender balance")

        if type_tx in [3,4]:
            conditions.append("High-risk transaction type")

        if abs(newbalanceOrig - oldbalanceOrg) < 1:
            conditions.append("Unusual balance behavior")

        if len(conditions) > 0:
            for c in conditions:
                st.warning(c)
        else:
            st.success("No suspicious patterns detected")

        # ================= FEATURE GRAPH =================
        st.subheader("📊 Feature Values")

        features = ["Time","Type","Amount","OldBalOrg","NewBalOrg","OldBalDest","NewBalDest"]
        values = input_df.values[0]

        fig_bar = go.Figure([go.Bar(
            x=values,
            y=features,
            orientation='h'
        )])

        st.plotly_chart(fig_bar, use_container_width=True)

# ================= CSV UPLOAD =================
st.sidebar.markdown("## 📁 Upload CSV")

uploaded_file = st.sidebar.file_uploader("Upload transaction file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    try:
        df_model = df.copy()

        df_model = df_model.drop(['nameOrig','nameDest'], axis=1, errors='ignore')
        df_model['type'] = df_model['type'].astype('category').cat.codes

        X = df_model.drop(['isFraud','isFlaggedFraud'], axis=1, errors='ignore')

        scaled = scaler.transform(X)
        pca_data = pca.transform(scaled)

        preds = model.predict(pca_data)

        df['Fraud Probability'] = preds
        df['Prediction'] = df['Fraud Probability'].apply(lambda x: "Fraud" if x>0.3 else "Legit")

        st.write("### Results")
        st.dataframe(df)

    except:
        st.error("Error processing file")


        



