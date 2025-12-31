import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, ConfusionMatrixDisplay
import tensorflow as tf
import os
import pickle
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, 'rtos_data.csv')
MODEL_PATH = os.path.join(CURRENT_DIR, 'rnn_model.keras')
CLASS_PATH = os.path.join(CURRENT_DIR, 'class_names.npy')
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

WINDOW_SIZE = 10
X, y = create_sequences(df[features].values, df['label_encoded'].values, WINDOW_SIZE)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 1. GENERATE WANDB INIT AND CONFIG
wandb.init(
    project="rtos-classification_simple-rnn",
    config={
        "learning_rate": 0.001,
        "epochs": 25,
        "batch_size": 32,
        "window_size": WINDOW_SIZE,
        "architecture": "SimpleRNN",
        "hidden_units": 128
    }
)
config = wandb.config

# ==========================================
# 2. BUILD AND TRAIN MODEL
# ==========================================
model = Sequential([
    SimpleRNN(config.hidden_units, input_shape=(config.window_size, len(features)), activation='tanh', return_sequences=False),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 4. UPLOAD MODEL TO WANDB (Using Callback)
# This automatically uploads the best model to wandb artifacts
wandb_callbacks = [
    WandbMetricsLogger(),
    WandbModelCheckpoint(filepath="model_checkpoint.keras", save_best_only=True)
]

history = model.fit(
    X_train, y_train, 
    epochs=config.epochs, 
    batch_size=config.batch_size, 
    validation_data=(X_test, y_test),
    callbacks=wandb_callbacks
)

# ==========================================
# 3. EVALUATION (Confusion Matrix, Precision, Recall, F1)
# ==========================================
print("\nEvaluating model...")
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print("\n" + "="*50)
print("SKLEARN CONFUSION MATRIX")
print("="*50)

# 3. Create the Confusion Matrix
# IMPORTANT: Pass the integer range as 'labels' to match y_test/y_pred
# We use np.unique(y_test) to get the actual integer IDs present
cm = confusion_matrix(y_test, y_pred)

# --- VISUALIZATION ---
fig, ax = plt.subplots(figsize=(12, 10))

# Use encoder.classes_ (the strings) ONLY for the display_labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')

plt.title("Confusion Matrix (SimpleRNN)")
plt.tight_layout()
plt.savefig("confusion_matrix_rnn.png")
plt.show() # This will display the plot if you are in a GUI/Notebook environment

print("\nConfusion Matrix saved to 'confusion_matrix_rnn.png'")

# --- DISPLAY SCORES ---
print("\n--- CLASSIFICATION REPORT ---")
report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
# 2 & 3. UPLOAD TO WANDB
# Uploading individual summary metrics
wandb.run.summary["test_precision"] = precision
wandb.run.summary["test_recall"] = recall
wandb.run.summary["test_f1"] = f1


# Table data for W&B
metrics_data = []

print("\n" + "="*85)
print(f"{'Device Label':<60} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
print("-" * 85)

for label, metrics in report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        p = metrics['precision']
        r = metrics['recall']
        f1 = metrics['f1-score']
        
        print(f"{label:<60} | {p:.4f} | {r:.4f} | {f1:.4f}")
        
        # Store in list for W&B Table
        metrics_data.append([label, p, r, f1])

# Upload Metrics to W&B
# Create one Table with all metrics
metrics_table = wandb.Table(
    data=metrics_data, 
    columns=["Device", "Precision", "Recall", "F1 Score"]
)

# Uploading Confusion Matrix Plot
wandb.log({
    "performance_metrics_table": metrics_table,
    "precision_chart": wandb.plot.bar(metrics_table, "Device", "Precision", title="Precision per Device"),
    "recall_chart": wandb.plot.bar(metrics_table, "Device", "Recall", title="Recall per Device"),
    "f1_score_chart": wandb.plot.bar(metrics_table, "Device", "F1 Score", title="F1 Score per Device"),
    "macro_f1": report['macro avg']['f1-score'],
    "macro_precision": report['macro avg']['precision'],
    "macro_recall": report['macro avg']['recall'],
    "conf_mat": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test, 
        preds=y_pred,
        class_names=list(encoder.classes_)
    )
})

# Save and upload the final model as an artifact explicitly if needed
model.save(MODEL_PATH)
artifact = wandb.Artifact('simple_rnn_model', type='model')
artifact.add_file(MODEL_PATH)
wandb.log_artifact(artifact)

wandb.finish()
print("Model saved successfully to 'simple_rnn_model.keras'")