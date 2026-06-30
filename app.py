import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Web Page Configuration Layout
st.set_page_config(page_title="AI Disease Predictor", layout="centered")
st.title("🩺 Medical Disease Prediction System")
st.write("Input patient clinical data below to calculate diabetes risk using Machine Learning.")

# 2. Cache Data Loading & Model Training for Instant Speeds
@st.cache_resource
def train_production_model():
    file_name = 'diabetes.csv'
    
    # Safety Check: Stop execution gracefully if data file is missing
    if not os.path.exists(file_name):
        st.error(f"❌ **File Not Found:** '{file_name}' was not found in your current working directory!")
        st.info(f"💡 **Current Working Directory:** {os.getcwd()}")
        st.warning("Please make sure you have downloaded 'diabetes.csv' and placed it in this exact folder.")
        st.stop()
        
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
    df = pd.read_csv(file_name, names=columns, skiprows=1)
    
    # Clean file arrays just like project.py does
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df['Outcome'] = df['Outcome'].astype(int)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train our robust production classifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# Initialize our engine
model, scaler = train_production_model()

# 3. Build UI Input Blocks (Split cleanly into 2 columns)
st.subheader("📊 Patient Clinical Metrics")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 110)
    bp = st.slider("Blood Pressure (mm Hg)", 0, 150, 70)
    skin = st.slider("Skin Thickness (mm)", 0, 99, 20)

with col2:
    insulin = st.slider("Insulin Level (mu U/ml)", 0, 846, 79)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=67.0, value=32.0, step=0.1)
    pedigree = st.number_input("Diabetes Pedigree Function Score", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)

# 4. Handle Calculations and Predictions on Click
if st.button("🔮 Calculate Risk Diagnostics", type="primary"):
    # Pack parameters into a 2D matrix layout shape
    user_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
    
    # Transform coordinates using original data scale parameters
    user_data_scaled = scaler.transform(user_data)
    
    # Run the model inferences
    prediction = model.predict(user_data_scaled)
    probability = model.predict_proba(user_data_scaled)[0][1]
    
    st.markdown("---")
    st.subheader("📋 Diagnostic Assessment Results")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk Warning:** The system indicates a **Diabetes Positive** prediction pattern.")
        st.write(f"The model has computed a risk coefficient probability of **{probability * 100:.2f}%**.")
    else:
        st.success(f"🎉 **Low Risk Profile:** The system indicates a **Healthy / Negative** prediction pattern.")
        st.write(f"The model indicates a risk coefficient probability of only **{probability * 100:.2f}%**.")
        