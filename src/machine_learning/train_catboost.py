import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os
import wandb
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
config = {
    "iterations": 200,
    "learning_rate": 0.01,
    "depth": 6,
    "test_size": 0.2,
    "random_state": 42,
}

wandb.init(project="rtos-classification_catboost", config=config)
cfg = wandb.config 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, 'rtos_data.csv')
MODEL_PATH = os.path.join(CURRENT_DIR, 'catboost_model.cbm')
TARGET_COLUMN = 'label' 

def create_dummy_csv():
    if not os.path.exists(CSV_PATH):
        data = {
            'Age': np.random.randint(18, 70, 100),
            'Income': np.random.randint(30000, 90000, 100),
            'CreditScore': np.random.randint(300, 850, 100),
            'Churn': np.random.choice([0, 1], 100) 
        }
        pd.DataFrame(data).to_csv(CSV_PATH, index=False)

def load_and_preprocess_data(filepath, target_col):
    df = pd.read_csv(filepath)
    # Dropping columns mentioned in your original script
    cols_to_drop = [target_col, "timestamp", "voltage", "energy"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df[target_col]
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    return X, y

def train_model():
    create_dummy_csv()
    X, y = load_and_preprocess_data(CSV_PATH, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    cb_model = CatBoostClassifier(
        iterations=cfg.iterations, 
        learning_rate=cfg.learning_rate, 
        depth=cfg.depth, 
        random_state=cfg.random_state, 
        logging_level='Verbose' ,
        eval_metric='Accuracy',      # Metric used for best model selection
        loss_function='MultiClass',   # Explicitly set multiclass loss
    )

    print("Training CatBoost model...")
    cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    eval_metrics = cb_model.get_evals_result()
    
    # We check if the key is 'Logloss' or 'MultiClass' dynamically
    loss_key = 'MultiClass' if 'MultiClass' in eval_metrics['learn'] else 'Logloss'
    
    for i in range(len(eval_metrics['learn']['Accuracy'])):
        wandb.log({
            "iteration": i,
            "train_loss": eval_metrics['learn'][loss_key][i],
            "train_accuracy": eval_metrics['learn']['Accuracy'][i],
            "val_loss": eval_metrics['validation'][loss_key][i],
            "val_accuracy": eval_metrics['validation']['Accuracy'][i]
        })

    # 1. Get raw predictions
    y_pred = cb_model.predict(X_test).flatten()

    le = LabelEncoder()
    # Fit on all possible labels seen in the data to ensure we have a complete map
    all_labels = pd.concat([y_train, y_test, pd.Series(y_pred)]).astype(str).str.strip().unique()
    le.fit(all_labels)

    y_true_int = le.transform(y_test.astype(str).str.strip())
    y_pred_int = le.transform(pd.Series(y_pred).astype(str).str.strip())
    class_names = le.classes_.tolist()

    # ==========================================
    # 5. SKLEARN CONFUSION MATRIX (Local Display)
    # ==========================================
    print("\n" + "="*50)
    print("SKLEARN CONFUSION MATRIX")
    print("="*50)
    
    # Calculate matrix
    cm = confusion_matrix(y_test, y_pred, labels=cb_model.classes_)
    
    # Visual display using matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cb_model.classes_)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
    plt.title("Confusion Matrix (Catboost)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_catboost.png")
    plt.show() # This will display the plot if you are in a GUI/Notebook environment
    
    print("\nConfusion Matrix saved to 'confusion_matrix_catboost.png'")

    # ==========================================
    # 6. F1 SCORE EVALUATION (Console + W&B)
    # ==========================================
    report = classification_report(y_test, y_pred, output_dict=True)
    
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
    # Log the chart and the final summary metric
    wandb.log({
        "performance_metrics_table": metrics_table,
        "precision_chart": wandb.plot.bar(metrics_table, "Device", "Precision", title="Precision per Device"),
        "recall_chart": wandb.plot.bar(metrics_table, "Device", "Recall", title="Recall per Device"),
        "f1_score_chart": wandb.plot.bar(metrics_table, "Device", "F1 Score", title="F1 Score per Device"),
        "macro_f1": report['macro avg']['f1-score'],
        "macro_precision": report['macro avg']['precision'],
        "macro_recall": report['macro avg']['recall'],
        "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true_int,
        preds=y_pred_int,
        class_names=class_names
        )
    })

    print(f"\nOverall Macro F1: {report['macro avg']['f1-score']:.4f}")
    print("F1 scores uploaded to W&B dashboard.")

    # 7. Finalize
    wandb.finish()

if __name__ == "__main__":
    train_model()