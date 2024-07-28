import streamlit as st
import requests
import json

st.title("Model Predictions and Interpretability")

features = st.text_input("Enter features (comma-separated):")

if st.button("Predict"):
    features_list = [float(x) for x in features.split(",")]
    response = requests.post("http://127.0.0.1:5000/", json={'features': features_list})
    predictions = response.json()
    st.write(f"Logistic Regression Prediction: {predictions['log_reg_prediction']}")
    st.write(f"SVM Prediction: {predictions['svm_prediction']}")

st.write("Dataset Analysis and Interpretability coming soon!")
