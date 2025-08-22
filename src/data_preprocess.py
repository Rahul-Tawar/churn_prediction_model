import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess_data(df, save_artifacts=True, artifacts_dir='models'):
    """
    Preprocess the Telco Customer Churn dataset for training.
    
    Args:
        df (pd.DataFrame): Input dataset.
        save_artifacts (bool): Whether to save scaler and encoders.
        artifacts_dir (str): Directory to save preprocessing artifacts.
    
    Returns:
        X (pd.DataFrame): Preprocessed features.
        y (pd.Series): Target variable.
        scaler (StandardScaler): Fitted scaler.
        encoders (dict): Dictionary of fitted LabelEncoders.
    """
    # Drop customerID
    df = df.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric, drop NaNs
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    
    # Encode categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns.drop('Churn')
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Scale numerical features
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Save preprocessing artifacts
    if save_artifacts:
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.joblib'))
        joblib.dump(encoders, os.path.join(artifacts_dir, 'encoders.joblib'))
    
    return X, y, scaler, encoders

def preprocess_input(data, scaler, encoders):
    """
    Preprocess a single input row for prediction.
    
    Args:
        data (dict): Input dictionary with feature values.
        scaler (StandardScaler): Fitted scaler from training.
        encoders (dict): Dictionary of fitted LabelEncoders.
    
    Returns:
        np.ndarray: Preprocessed input array.
    """
    df = pd.DataFrame([data])
    
    # Encode categoricals
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    for col in cat_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
    
    # Scale numerical
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.transform(df[num_cols])
    
    return df.values