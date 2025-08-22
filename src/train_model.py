import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from data_preprocess import preprocess_data

def train_and_evaluate():
    """
    Train a neural network model on the Telco Customer Churn dataset and evaluate it.
    Saves the trained model to models/churn_model.h5.
    """
    # Load data
    df = pd.read_csv('data/telco_churn.csv')
    
    # Preprocess data
    X, y, scaler, encoders = preprocess_data(df, save_artifacts=True, artifacts_dir='models')
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Save model
    model.save('models/churn_model.h5')
    
    return model, history

if __name__ == "__main__":
    model, history = train_and_evaluate()