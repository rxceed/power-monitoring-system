import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, 'rtos_data.csv')
MODEL_PATH = os.path.join(CURRENT_DIR, 'lstm_model.keras')
CLASS_PATH = os.path.join(CURRENT_DIR, 'class_names_lstm.npy')
SCALER_PATH = os.path.join(CURRENT_DIR, 'scaler.pkl')

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
print("Loading and processing data...")
df = pd.read_csv(CSV_PATH)

# Sort by timestamp to ensure time sequence is correct
df = df.sort_values('timestamp')

# Define features
features = ['current', 'power_factor', 'power']
target = 'label'

# Normalize features (Scale to 0-1 range)
# We must save this scaler to use it later for inference
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved to 'scaler.pkl'")

# Encode Labels (String -> Integer)
encoder = LabelEncoder()
df['label_encoded'] = encoder.fit_transform(df[target])

# Save the class names so we can decode predictions later
np.save(CLASS_PATH, encoder.classes_)
print("Class names saved to 'classes.npy'")

# Create Sequences (Sliding Window)
def create_sequences(data, labels, window_size=10):
    X = []
    y = []
    for i in range(len(data) - window_size):
        # Input: window_size steps
        X.append(data[i : i + window_size])
        # Target: The label of the step immediately following the window
        y.append(labels[i + window_size - 1])
    return np.array(X), np.array(y)

WINDOW_SIZE = 20
X, y = create_sequences(df[features].values, df['label_encoded'].values, WINDOW_SIZE)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. BUILD AND TRAIN MODEL
# ==========================================
print(f"Building model with input shape: {X_train.shape[1:]}")

model = Sequential([
    # SimpleRNN Layer
    LSTM(128, input_shape=(WINDOW_SIZE, len(features)), activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    # Output Layer
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, y_train, 
                    epochs=25, 
                    batch_size=32, 
                    validation_data=(X_test, y_test))

# ==========================================
# 3. SAVE MODEL
# ==========================================
model.save(MODEL_PATH)
print("Model saved successfully to 'simple_rnn_model.keras'")