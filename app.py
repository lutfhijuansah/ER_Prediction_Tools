import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("er_rf_model.joblib")

st.set_page_config(page_title="Engagement Rate Predictor", layout="centered")

# Title
st.title("📊 Engagement Rate Prediction App")

st.markdown("""
Aplikasi ini memprediksi **Engagement Rate (ER)** berdasarkan input metrik sosial media:
- Likes  
- Comments  
- Shares  
- Saves  
- Reach  
- Views  
""")

# Input form
with st.form("input_form"):
    likes = st.number_input("Likes", min_value=0, step=1)
    comments = st.number_input("Comments", min_value=0, step=1)
    shares = st.number_input("Shares", min_value=0, step=1)
    saves = st.number_input("Saves", min_value=0, step=1)
    reach = st.number_input("Reach", min_value=1, step=1)  # reach jangan nol biar tidak error
    views = st.number_input("Views", min_value=0, step=1)

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    input_data = pd.DataFrame([{
        "Likes": likes,
        "Comments": comments,
        "Share": shares,
        "Saved": saves,
        "Reach": reach,
        "Views": views
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("🔮 Predicted Engagement Rate (ER)")
    st.success(f"{prediction:.2%}")  # ditampilkan dalam persen
