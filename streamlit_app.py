import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('model.pkl')
explainer = shap.Explainer(model.named_steps['classifier'], model.named_steps['preprocessor'].transform)

# Title
st.title('Model Prediction and SHAP Visualization')

# File upload
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    if st.button('Predict'):
        prediction = model.predict(data)
        st.write('Predictions:', prediction)

        shap_values = explainer(data)
        shap.summary_plot(shap_values, data)
        st.pyplot(bbox_inches='tight')
