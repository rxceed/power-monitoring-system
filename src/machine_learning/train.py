import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import os

# ==========================================
# CONFIGURATION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, 'rtos_data.csv')
TARGET_COLUMN = 'label'  # The column you want to predict
test_size = 0.2          # 20% of data used for testing
random_state = 42        # Ensures reproducibility

def create_dummy_csv():
    """
    Creates a dummy dataset if one doesn't exist, just so this script 
    is runnable right out of the box.
    """
    if not os.path.exists(CSV_PATH):
        print(f"Creating dummy dataset: {CSV_PATH}...")
        data = {
            'Age': np.random.randint(18, 70, 100),
            'Income': np.random.randint(30000, 90000, 100),
            'CreditScore': np.random.randint(300, 850, 100),
            'SubscriptionType': np.random.choice(['Basic', 'Standard', 'Premium'], 100),
            'Churn': np.random.choice([0, 1], 100) # 0 = Stayed, 1 = Left
        }
        df = pd.DataFrame(data)
        df.to_csv(CSV_PATH, index=False)
        print("Dummy dataset created successfully.\n")

def load_and_preprocess_data(filepath, target_col):
    """
    Loads data, handles missing values, and encodes categorical variables.
    Returns X, y, and the label encoder (if used).
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # 1. Separate Features (X) and Target (y)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")
    
    X = df.drop(columns=[target_col, "timestamp"])
    y = df[target_col]

    # 4. Handle Missing Values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X, y, #le

def train_model():
    # 1. Setup Data
    create_dummy_csv()
    
    try:
        # Now capturing 'le' (the encoder) from the function
        #X, y, le = load_and_preprocess_data(CSV_PATH, TARGET_COLUMN)
        X, y = load_and_preprocess_data(CSV_PATH, TARGET_COLUMN)
    except FileNotFoundError:
        print(f"Error: File {CSV_PATH} not found.")
        return

    # 2. Split Data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

        # 3. Initialize CatBoost
    # verbose=0 silences the iteration logs
    cb_model = CatBoostClassifier(
        iterations=100, 
        learning_rate=0.1, 
        depth=6, 
        random_state=random_state, 
        verbose=1
    )

    # 4. Train the Model
    print("Training CatBoost model...")
    cb_model.fit(X_train, y_train)

    # 5. Make Predictions
    y_pred = cb_model.predict(X_test)

    # 6. Evaluate
    print("\n" + "="*30)
    print("MODEL EVALUATION")
    print("="*30)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save the Model AND the Encoder
    model_filename = 'catboost_model.pkl' # Changed filename
    
    #joblib.dump(cb_model, model_filename)
    cb_model.save_model("catboost_model.cbm", 'cbm')
    #print(f"\nModel saved to {model_filename}")
    
    #if le:
    #    joblib.dump(le, encoder_filename)
    #    print(f"Label Encoder saved to {encoder_filename}")
    #else:
    #    print("No Label Encoder to save (target was already numeric).")

if __name__ == "__main__":
    train_model()