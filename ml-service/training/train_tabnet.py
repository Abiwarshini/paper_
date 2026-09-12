import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, hamming_loss
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import (
    prepare_data_splits, TARGET_CONDITIONS, ALL_FEATURES
)


def tune_thresholds_on_val(model, X_val, Y_val):
    """
    Select optimal probability threshold per target condition on Validation set.
    """
    val_probs_raw = model.predict_proba(X_val)
    val_probs = np.column_stack([p[:, 1] for p in val_probs_raw])

    thresholds = {}
    candidate_th = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    for idx, cond in enumerate(TARGET_CONDITIONS):
        y_true = Y_val[:, idx]
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


def train_and_evaluate_tabnet(
    n_d=16,
    n_a=16,
    n_steps=4,
    gamma=1.3,
    lambda_sparse=1e-4,
    learning_rate=2e-2,
    batch_size=1024,
    virtual_batch_size=128,
    max_epochs=15,
    patience=5,
    random_state=42
):
    print("=== Training TabNet Multi-Task Classifier on DHS Dataset ===")
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    start_time = time.time()
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), preprocessor = prepare_data_splits(random_state=random_state)

    X_train_np = X_train[ALL_FEATURES].values.astype(np.float32)
    Y_train_np = Y_train[TARGET_CONDITIONS].values.astype(int)

    X_val_np = X_val[ALL_FEATURES].values.astype(np.float32)
    Y_val_np = Y_val[TARGET_CONDITIONS].values.astype(int)

    X_test_np = X_test[ALL_FEATURES].values.astype(np.float32)
    Y_test_np = Y_test[TARGET_CONDITIONS].values.astype(int)

    model = TabNetMultiTaskClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=learning_rate, weight_decay=1e-4),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=4, gamma=0.5),
        seed=random_state,
        verbose=1
    )

    print(f"\nFitting TabNet (Train: {len(X_train_np):,}, Val: {len(X_val_np):,})...")
    train_start = time.time()

    model.fit(
        X_train=X_train_np,
        y_train=Y_train_np,
        eval_set=[(X_val_np, Y_val_np)],
        eval_name=["val"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        drop_last=False
    )

    training_time_sec = round(time.time() - train_start, 2)
    print(f"TabNet training completed in {training_time_sec} seconds!")

    # Threshold calibration on validation set
    val_threshold_info, _ = tune_thresholds_on_val(model, X_val_np, Y_val_np)

    # Test Set Inference
    test_start = time.time()
    test_probs_raw = model.predict_proba(X_test_np)
    test_probs = np.column_stack([p[:, 1] for p in test_probs_raw])
    inference_time_sec = round(time.time() - test_start, 4)
    per_sample_ms = round((inference_time_sec / len(X_test_np)) * 1000, 4)

    # Apply tuned thresholds
    test_pred_binary = np.zeros_like(test_probs, dtype=int)
    for idx, cond in enumerate(TARGET_CONDITIONS):
        th = val_threshold_info[cond]["selected_threshold"]
        test_pred_binary[:, idx] = (test_probs[:, idx] >= th).astype(int)

    # Overall Metrics
    exact_match_acc = accuracy_score(Y_test_np, test_pred_binary)
    h_loss = hamming_loss(Y_test_np, test_pred_binary)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(Y_test_np, test_pred_binary, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(Y_test_np, test_pred_binary, average="weighted", zero_division=0)

    try:
        roc_auc_macro = roc_auc_score(Y_test_np, test_probs, average="macro")
    except Exception:
        roc_auc_macro = 0.0

    per_class_metrics = {}
    confusion_matrices = {}

    for idx, cond in enumerate(TARGET_CONDITIONS):
        y_true_c = Y_test_np[:, idx]
        y_pred_c = test_pred_binary[:, idx]
        y_prob_c = test_probs[:, idx]

        p, r, f1, _ = precision_recall_fscore_support(y_true_c, y_pred_c, average="binary", zero_division=0)
        acc_c = accuracy_score(y_true_c, y_pred_c)

        try:
            auc_c = float(roc_auc_score(y_true_c, y_prob_c))
        except Exception:
            auc_c = 0.0

        cm = confusion_matrix(y_true_c, y_pred_c)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        confusion_matrices[cond] = cm.tolist()
        per_class_metrics[cond] = {
            "accuracy": round(float(acc_c), 4),
            "precision": round(float(p), 4),
            "recall_sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc_c), 4),
            "optimal_threshold": val_threshold_info[cond]["selected_threshold"],
            "support_positive": int(np.sum(y_true_c)),
            "support_total": int(len(y_true_c))
        }

    # Save TabNet Model
    output_dir = Path(__file__).resolve().parent.parent / "models" / "tabnet"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = output_dir / "tabnet_model"
    saved_filepath = model.save_model(str(model_save_path))

    metrics_payload = {
        "model_name": "TabNet Multi-Task Classifier",
        "architecture": f"TabNet(n_d={n_d}, n_a={n_a}, n_steps={n_steps}, gamma={gamma}, lambda_sparse={lambda_sparse})",
        "dataset": "NFHS-5 dhs_clean.parquet (198,849 records)",
        "dataset_split": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test)
        },
        "training_time_seconds": training_time_sec,
        "inference_latency": {
            "total_test_seconds": inference_time_sec,
            "per_sample_ms": per_sample_ms
        },
        "overall_metrics": {
            "exact_match_accuracy": round(float(exact_match_acc), 4),
            "hamming_loss": round(float(h_loss), 4),
            "macro_f1": round(float(f1_macro), 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "macro_roc_auc": round(float(roc_auc_macro), 4)
        },
        "threshold_calibration": val_threshold_info,
        "per_condition_metrics": per_class_metrics,
        "confusion_matrices": confusion_matrices,
        "features_used": ALL_FEATURES,
        "target_conditions": TARGET_CONDITIONS
    }

    metrics_path = output_dir / "tabnet_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nSaved TabNet model to: {saved_filepath}")
    print(f"Saved TabNet metrics to: {metrics_path}")

    print("\n--- TabNet Test Set Performance Summary ---")
    print(f"Exact Match Accuracy: {exact_match_acc * 100:.2f}%")
    print(f"Macro F1-Score:       {f1_macro:.4f}")
    print(f"Weighted F1-Score:    {f1_weighted:.4f}")
    print(f"Macro ROC-AUC:        {roc_auc_macro:.4f}")
    print(f"Hamming Loss:         {h_loss:.4f}")

    for cond, m in per_class_metrics.items():
        print(f"  {cond:22s} | Acc: {m['accuracy']*100:.2f}% | F1: {m['f1_score']:.4f} | AUC: {m['roc_auc']:.4f} | Thresh: {m['optimal_threshold']}")

    return model, metrics_payload


if __name__ == "__main__":
    train_and_evaluate_tabnet()
