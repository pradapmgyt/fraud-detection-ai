import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('models/fraud_model.pkl')

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 AI Fraud Detection System (Real Data)")
st.markdown("### Upload transaction data or select a real record")

# -------- FILE UPLOAD -------- #
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/creditcard.csv")

# Show dataset
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------- SELECT ROW -------- #
st.subheader("🔍 Select Transaction Row")

row_index = st.number_input("Enter Row Index", min_value=0, max_value=len(df)-1, value=0)

selected_row = df.iloc[row_index]

st.write("### Selected Transaction")
st.dataframe(selected_row)

# -------- PREPARE INPUT -------- #
X = selected_row.drop("Class").values.reshape(1, -1)

# -------- PREDICT -------- #
if st.button("🚀 Check Transaction"):
    prediction = model.predict(X)

    if prediction[0] == 1:
        st.error("🚨 Fraud Transaction Detected!")
    else:
        st.success("✅ Safe Transaction")

# -------- EXTRA INFO -------- #
st.info("Note: This uses real dataset features (V1–V28, Time, Amount)")