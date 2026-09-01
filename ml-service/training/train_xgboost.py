import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, hamming_loss, classification_report
)

# Add parent directory to path to import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import prepare_data_splits, TARGET_CONDITIONS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

def train_and_evaluate_xgboost(random_state=42):
    print("=== Training XGBoost Multi-Label Baseline Classifier ===")
    
    splits = prepare_data_splits(random_state=random_state)
    X_train, Y_train = splits["X_train"], splits["Y_train"]
    X_val, Y_val = splits["X_val"], splits["Y_val"]
    X_test, Y_test = splits["X_test"], splits["Y_test"]
    
    # Initialize multi-output XGBoost classifier
    base_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss"
    )
    
    model = MultiOutputClassifier(base_xgb, n_jobs=-1)
    
    print("Fitting XGBoost on training set (69,999 samples)...")
    model.fit(X_train[ALL_FEATURES], Y_train)
    print("XGBoost training completed!")
    
    # Evaluate on Test Set (15,000 samples)
    y_pred_probs = np.array([est.predict_proba(X_test[ALL_FEATURES])[:, 1] for est in model.estimators_]).T
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)
    
    # Calculate metrics
    exact_match_acc = accuracy_score(Y_test, y_pred_binary)
    h_loss = hamming_loss(Y_test, y_pred_binary)
    
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(Y_test, y_pred_binary, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(Y_test, y_pred_binary, average="weighted", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(Y_test, y_pred_binary, average="micro", zero_division=0)
    
    try:
        roc_auc_macro = roc_auc_score(Y_test, y_pred_probs, average="macro")
    except Exception:
        roc_auc_macro = 0.0

    per_class_metrics = {}
    confusion_matrices = {}
    
    for idx, cond in enumerate(TARGET_CONDITIONS):
        y_true_c = Y_test.iloc[:, idx].values
        y_pred_c = y_pred_binary[:, idx]
        y_prob_c = y_pred_probs[:, idx]
        
        p, r, f1, _ = precision_recall_fscore_support(y_true_c, y_pred_c, average="binary", zero_division=0)
        acc_c = accuracy_score(y_true_c, y_pred_c)
        try:
            auc_c = float(roc_auc_score(y_true_c, y_prob_c))
        except Exception:
            auc_c = 0.0
            
        cm = confusion_matrix(y_true_c, y_pred_c).tolist()
        confusion_matrices[cond] = cm
        
        per_class_metrics[cond] = {
            "accuracy": round(float(acc_c), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(auc_c, 4),
            "positive_count": int(y_true_c.sum()),
            "total_count": len(y_true_c)
        }

    overall_metrics = {
        "model_name": "XGBoost Multi-Label Classifier",
        "exact_match_accuracy": round(float(exact_match_acc), 4),
        "hamming_loss": round(float(h_loss), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "micro_f1": round(float(f1_micro), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_roc_auc": round(float(roc_auc_macro), 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrices": confusion_matrices
    }
    
    print("\n--- XGBoost Overall Test Evaluation ---")
    print(f"Exact Match Accuracy: {overall_metrics['exact_match_accuracy']*100:.2f}%")
    print(f"Hamming Loss:         {overall_metrics['hamming_loss']:.4f}")
    print(f"Macro F1 Score:       {overall_metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score:    {overall_metrics['weighted_f1']:.4f}")
    print(f"Macro ROC-AUC:        {overall_metrics['macro_roc_auc']:.4f}")
    
    print("\nPer-Class Breakdown:")
    for cond, m in per_class_metrics.items():
        print(f" - {cond:25s}: Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1_score']:.4f}, AUC={m['roc_auc']:.4f}")
        
    # Save Model Artifacts
    model_dir = os.path.join(r"e:\Project\SEM7\Malnutrtion", "ml-service", "models", "xgboost")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "xgboost_multilabel.pkl")
    metrics_path = os.path.join(model_dir, "xgboost_metrics.json")
    
    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(overall_metrics, f, indent=2)
        
    print(f"\nSaved XGBoost model to {model_path}")
    print(f"Saved XGBoost metrics to {metrics_path}")
    
    return model, overall_metrics

if __name__ == "__main__":
    train_and_evaluate_xgboost()
