import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
from data_preprocess import preprocess_input

# Load model and preprocessing artifacts
model = tf.keras.models.load_model('models/churn_model.h5')
scaler = joblib.load('models/scaler.joblib')
encoders = joblib.load('models/encoders.joblib')

# Streamlit UI
st.title('Customer Churn Prediction')

# Input form
gender = st.selectbox('Gender', ['Female', 'Male'])
senior = st.selectbox('Senior Citizen', [0, 1])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.number_input('Tenure (months)', min_value=0)
phone = st.selectbox('Phone Service', ['Yes', 'No'])
multiple = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly = st.number_input('Monthly Charges', min_value=0.0)
total = st.number_input('Total Charges', min_value=0.0)

input_data = {
    'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
    'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet,
    'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
    'TechSupport': support, 'StreamingTV': tv, 'StreamingMovies': movies,
    'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
    'MonthlyCharges': monthly, 'TotalCharges': total
}

if st.button('Predict Churn'):
    processed = preprocess_input(input_data, scaler, encoders)
    prediction = model.predict(processed)[0][0]
    st.write(f'Churn Probability: {prediction:.2f}')
    if prediction > 0.5:
        st.write('Likely to Churn (Yes)')
    else:
        st.write('Unlikely to Churn (No)')