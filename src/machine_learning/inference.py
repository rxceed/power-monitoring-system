import pandas as pd
import numpy as np
import os
import pickle
from catboost import CatBoostClassifier
import tensorflow as tf

# ==========================================
# CONFIGURATION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CATBOOST_MODEL_PATH = os.path.join(CURRENT_DIR, 'catboost_model.cbm')
RNN_MODEL_PATH = os.path.join(CURRENT_DIR, 'rnn_model.keras')
CLASS_PATH = os.path.join(CURRENT_DIR, 'class_names.npy')
SCALER_PATH = os.path.join(CURRENT_DIR, 'scaler.pkl')

def load_model_catboost():
    """Loads both the model and the label encoder."""
    # 1. Load Model
    if os.path.exists(CATBOOST_MODEL_PATH):
        #print(f"Loading model from {MODEL_PATH}...")
        model = CatBoostClassifier()
        model.load_model(CATBOOST_MODEL_PATH)
        
    else:
        #print(f"Error: Model file '{MODEL_PATH}' not found.")
        #print("Please run the training script first to generate the model.")
        pass
    return model

def load_model_RNN():
    model = tf.keras.models.load_model(RNN_MODEL_PATH)

    # Load the class names
    class_names = np.load(CLASS_PATH, allow_pickle=True)

    # Load the scaler (Must be the same one used in training!)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, class_names, scaler

def RNN_input_preprocess(data_sequence, scaler):
    feature_cols = ['current', 'power_factor', 'power']

    # Convert to DataFrame
    df_input = pd.DataFrame(data_sequence, columns=feature_cols)
    input_tensor = tf.convert_to_tensor(df_input)
    # Scale the data using the loaded scaler
    input_scaled = scaler.transform(df_input)

    # Reshape to (1, 10, 5) -> (Batch_Size, Time_Steps, Features)
    input_reshaped = tf.reshape(input_scaled, [1,20,3])
    #input_reshaped = tf.reshape(input_tensor, [1,20,3])

    return input_reshaped

def predict_RNN(model, class_names, data_sequence):
    prediction_probs = model.predict(data_sequence)
    predicted_index = np.argmax(prediction_probs)
    predicted_label = class_names[predicted_index]
    confidence = prediction_probs[0][predicted_index]
    return predicted_label, confidence

def predict_single(model, data_dict):
    """
    Takes a single dictionary of data, processes it, and predicts churn.
    """
    if model is None:
        return

    # 1. Convert single dictionary to 1-row DataFrame
    new_data_df = pd.DataFrame([data_dict])

    # 2. Preprocess New Data
    # One-Hot Encode categorical variables
    processed_df = pd.get_dummies(new_data_df)

    # 3. Align Columns (CRITICAL STEP)
    # Get features model expects
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
    else:
        model_features = processed_df.columns

    # Add missing columns with 0
    for col in model_features:
        if col not in processed_df.columns:
            processed_df[col] = 0

    # Reorder columns to match model
    processed_df = processed_df[model_features]

    # 4. Make Prediction
    prediction_label = model.predict(processed_df)[0]
    probabilities = model.predict_proba(processed_df)[0]
    
    # Get probability dynamically by finding the index of the predicted class in model.classes_
    # This works whether 'prediction' is an integer (0) or a string ("Yes")
    class_index = np.where(model.classes_ == prediction_label)[0][0]
    probability = probabilities[class_index]

    return prediction_label, probability

def inference_data_catboost(data_record):
    model = load_model_catboost()
    #pred, prob = predict_single(model, data_record)
    return predict_single(model, data_record)

def inference_data_RNN(data_sequence):
    model, class_names, scaler = load_model_RNN()
    preprocessed_data = RNN_input_preprocess(data_sequence, scaler)
    return predict_RNN(model, class_names, preprocessed_data)
    
if __name__ == "__main__":
    # 1. Load the model once
    model = load_model_catboost()

    if model:
        print("\n--- Starting Single Record Inference ---")
        
        # 2. Define individual records to feed one by one
        record_1 = {'current': 0.2, 'voltage': 226.6, 'power_factor': 0.9, 'power': 40.4, 'energy': 2.76}

        # 3. Feed them one at a time
        print(predict_single(model, record_1))