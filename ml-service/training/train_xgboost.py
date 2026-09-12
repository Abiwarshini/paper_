import os
import sys
import time
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, hamming_loss, classification_report
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import (
    prepare_data_splits, TARGET_CONDITIONS, NUTRITION_CONDITIONS,
    DISEASE_CONDITIONS, ALL_FEATURES
)


def tune_thresholds_on_val(model, X_val, Y_val):
    """
    Select optimal probability threshold per target condition on Validation set.
    Evaluates candidate thresholds [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    to maximize validation F1-score while preserving specificity.
    """
    val_probs = np.array([est.predict_proba(X_val[ALL_FEATURES])[:, 1] for est in model.estimators_]).T
    thresholds = {}
    candidate_th = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    for idx, cond in enumerate(TARGET_CONDITIONS):
        y_true = Y_val.iloc[:, idx].values
        probs = val_probs[:, idx]

        best_th = 0.50
        best_f1 = -1.0

        for th in candidate_th:
            pred_b = (probs >= th).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true, pred_b, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        thresholds[cond] = {
            "selected_threshold": round(float(best_th), 2),
            "val_f1": round(float(best_f1), 4),
            "reason": f"Optimized on validation set to maximize F1-score ({best_f1:.4f})"
        }

    return thresholds, val_probs


def train_and_evaluate_xgboost(random_state=42):
    print("=== Training XGBoost Multi-Label Baseline Classifier (10 Targets) ===")
    start_time = time.time()

    splits = prepare_data_splits(random_state=random_state)
    X_train, Y_train = splits["X_train"], splits["Y_train"]
    X_val, Y_val = splits["X_val"], splits["Y_val"]
    X_test, Y_test = splits["X_test"], splits["Y_test"]

    # Base XGBoost with balanced hyperparameters
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

    print(f"Fitting XGBoost across {len(TARGET_CONDITIONS)} conditions on train set ({len(X_train)} samples)...")
    train_start = time.time()
    model.fit(X_train[ALL_FEATURES], Y_train)
    training_time_sec = round(time.time() - train_start, 2)
    print(f"XGBoost training completed in {training_time_sec} seconds!")

    # Threshold selection on validation set
    print("Calibrating validation thresholds for each condition...")
    val_threshold_info, _ = tune_thresholds_on_val(model, X_val, Y_val)

    # Test Set Inference Timing
    test_start = time.time()
    y_pred_probs = np.array([est.predict_proba(X_test[ALL_FEATURES])[:, 1] for est in model.estimators_]).T
    inference_time_sec = round(time.time() - test_start, 4)
    per_sample_inference_ms = round((inference_time_sec / len(X_test)) * 1000, 4)

    # Apply tuned thresholds for final binary classification
    y_pred_binary = np.zeros_like(y_pred_probs, dtype=int)
    for idx, cond in enumerate(TARGET_CONDITIONS):
        th = val_threshold_info[cond]["selected_threshold"]
        y_pred_binary[:, idx] = (y_pred_probs[:, idx] >= th).astype(int)

    # Overall Metrics
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

        cm = confusion_matrix(y_true_c, y_pred_c)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        confusion_matrices[cond] = cm.tolist()

        per_class_metrics[cond] = {
            "accuracy": round(float(acc_c), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(auc_c, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "selected_threshold": val_threshold_info[cond]["selected_threshold"],
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
        "training_time_seconds": training_time_sec,
        "inference_time_seconds": inference_time_sec,
        "per_sample_inference_ms": per_sample_inference_ms,
        "threshold_info": val_threshold_info,
        "per_class_metrics": per_class_metrics,
        "confusion_matrices": confusion_matrices
    }

    print("\n================ XGBoost Test Set Benchmark ================")
    print(f"Exact Match Accuracy:     {overall_metrics['exact_match_accuracy']*100:.2f}%")
    print(f"Hamming Loss:             {overall_metrics['hamming_loss']:.4f}")
    print(f"Macro F1 Score:           {overall_metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score:        {overall_metrics['weighted_f1']:.4f}")
    print(f"Macro ROC-AUC:            {overall_metrics['macro_roc_auc']:.4f}")
    print(f"Training Time:            {training_time_sec}s")
    print(f"Inference Time (15k):     {inference_time_sec}s ({per_sample_inference_ms:.4f} ms/sample)")

    print("\n--- Growth & Nutrition Assessment Metrics ---")
    for cond in NUTRITION_CONDITIONS:
        m = per_class_metrics[cond]
        print(f" - {cond:32s}: Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1_score']:.4f}, AUC={m['roc_auc']:.4f}, Th={m['selected_threshold']}")

    print("\n--- Pediatric Disease & Health-Condition Risk Metrics ---")
    for cond in DISEASE_CONDITIONS:
        m = per_class_metrics[cond]
        print(f" - {cond:32s}: Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1_score']:.4f}, AUC={m['roc_auc']:.4f}, Th={m['selected_threshold']}")

    # Save Model Artifacts
    target_dirs = [
        Path(__file__).resolve().parent.parent / "models" / "xgboost",
        Path(r"e:\Project\SEM7\Malnutrtion") / "ml-service" / "models" / "xgboost"
    ]

    for m_dir in target_dirs:
        try:
            m_dir.mkdir(parents=True, exist_ok=True)
            model_path = m_dir / "xgboost_multilabel.pkl"
            metrics_path = m_dir / "xgboost_metrics.json"
            joblib.dump(model, str(model_path))
            with open(str(metrics_path), "w") as f:
                json.dump(overall_metrics, f, indent=2)
            print(f"Saved XGBoost model & metrics to {m_dir}")
        except Exception as e:
            print(f"Notice: skipped saving to {m_dir} ({e})")

    return model, overall_metrics


if __name__ == "__main__":
    train_and_evaluate_xgboost()
