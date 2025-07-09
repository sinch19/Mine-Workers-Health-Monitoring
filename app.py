import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
try:
    model = load_model("model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading scaler: {e}")

# Label maps
alcohol_map = {'Never': 0, 'Occasionally': 1, 'Frequently': 2}
label_map = {0: 'Limited', 1: 'Moderate', 2: 'Best'}

# Page settings
st.set_page_config(page_title="Worker Capability Prediction", layout="centered")

# Title and description
st.title("Worker Capability Prediction 🏗️")
st.markdown("""
This app predicts the *Worker Capability* based on the following health and workplace parameters:
- **Age**
- **BMI**
- **SpO₂ Level**
- **Alcohol Consumption**
- **Job Satisfaction**
""")

# Input Section
with st.expander("Enter Worker Details", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
        bmi = st.number_input("BMI", min_value=16.5, max_value=35.0, value=23.0, step=0.1)
        spo2 = st.number_input("SpO₂ Level", min_value=85.0, max_value=100.0, value=95.0, step=0.1)
    with col2:
        alcohol = st.selectbox("Alcohol Consumption", list(alcohol_map.keys()))
        job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=10, value=5)

# Buttons
col1, col2 = st.columns(2)
with col1:
    predict_button = st.button("Predict Worker Capability", use_container_width=True)
with col2:
    reset_button = st.button("Reset Form", use_container_width=True)

# Initialize session state
if "predicted_label" not in st.session_state:
    st.session_state.predicted_label = None

# Prediction logic
if predict_button:
    if model and scaler:
        with st.spinner("Making prediction..."):
            input_data = np.array([[age, bmi, spo2, alcohol_map[alcohol], job_satisfaction]])
            input_scaled = scaler.transform(input_data)
            predictions = model.predict(input_scaled)
            predicted_label = label_map[np.argmax(predictions)]
            st.session_state.predicted_label = predicted_label

        # Output result
        st.success(f"### Predicted Worker Capability: **{st.session_state.predicted_label}**")

        # Additional feedback
        if predicted_label == "Best":
            st.balloons()
            st.info("This worker is performing excellently! 🎉 Keep up the great work!")
        elif predicted_label == "Moderate":
            st.warning("This worker shows average capability. Consider providing additional support.")
        else:
            st.error("This worker might need intervention. Please review their condition.")
    else:
        st.error("Model or scaler is not loaded.")

# Show Health Insights
if st.session_state.predicted_label and st.button("Show Health Insights 📋"):
    if st.session_state.predicted_label == "Best":
        st.markdown("""
        - **SpO₂**: Healthy oxygen levels  
        - **BMI**: Within optimal range  
        - **Job Satisfaction**: High  
        - **Alcohol Consumption**: Minimal or none  
        """)
    elif st.session_state.predicted_label == "Moderate":
        st.markdown("""
        - **SpO₂**: Slightly low, monitor regularly  
        - **BMI**: Slightly above average  
        - **Job Satisfaction**: Moderate  
        - **Alcohol Consumption**: Occasional  
        """)
    else:
        st.markdown("""
        - **SpO₂**: Low, medical checkup advised  
        - **BMI**: High, could pose health risks  
        - **Job Satisfaction**: Low  
        - **Alcohol Consumption**: Frequent, may impact performance  
        """)

# Reset logic
if reset_button:
    st.session_state.predicted_label = None
    st.experimental_rerun()

# About section
st.markdown("---")
st.header("About This App")
st.markdown("""
This app is designed to assist supervisors in mining operations by predicting worker capabilities using health checkup data. It supports early intervention, task reassignment, and helps maintain a safe and productive workforce.
""")
