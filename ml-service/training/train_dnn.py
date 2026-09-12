import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, hamming_loss
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import (
    prepare_data_splits, TARGET_CONDITIONS, NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES, ALL_FEATURES
)


class TabularDataset(Dataset):
    def __init__(self, X_df, Y_df=None):
        self.X = torch.tensor(X_df[ALL_FEATURES].values.astype(np.float32), dtype=torch.float32)
        if Y_df is not None:
            self.Y = torch.tensor(Y_df[TARGET_CONDITIONS].values.astype(np.float32), dtype=torch.float32)
        else:
            self.Y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X[idx], self.Y[idx]
        return self.X[idx]


class TabularDNN(nn.Module):
    """
    Deep Neural Network for Tabular Malnutrition Prediction:
    Input (18) -> Dense(256) -> BatchNorm -> ReLU -> Dropout(0.2)
               -> Dense(128) -> BatchNorm -> ReLU -> Dropout(0.2)
               -> Dense(64) -> ReLU
               -> Output Head (3 targets: Stunting, Wasting, Malnutrition)
    """

    def __init__(self, input_dim=len(ALL_FEATURES), num_targets=len(TARGET_CONDITIONS), dropout=0.2):
        super(TabularDNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.output_head = nn.Linear(64, num_targets)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.output_head(x)
        return logits


def train_and_evaluate_dnn(
    epochs=15,
    batch_size=512,
    learning_rate=1e-3,
    dropout=0.2,
    patience=4,
    random_state=42
):
    print("=== Training Tabular Deep Neural Network (DNN) on DHS Dataset ===")
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    start_time = time.time()
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), preprocessor = prepare_data_splits(random_state=random_state)

    train_dataset = TabularDataset(X_train, Y_train)
    val_dataset = TabularDataset(X_val, Y_val)
    test_dataset = TabularDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TabularDNN(input_dim=len(ALL_FEATURES), num_targets=len(TARGET_CONDITIONS), dropout=dropout).to(device)

    # Calculate class positive weights for BCEWithLogitsLoss
    pos_counts = Y_train.sum(axis=0).values
    neg_counts = len(Y_train) - pos_counts
    pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-5), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0

    print(f"\nBeginning DNN Training across {epochs} epochs (Train: {len(X_train):,}, Val: {len(X_val):,})...")
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(x_batch)

        train_loss = total_train_loss / len(train_dataset)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_preds_list = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                total_val_loss += loss.item() * len(x_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds_list.append(probs)

        val_loss = total_val_loss / len(val_dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    training_time_sec = round(time.time() - train_start, 2)
    print(f"DNN training completed in {training_time_sec}s. Restoring best weights...")
    model.load_state_dict(best_weights)

    # Validation Threshold Tuning
    model.eval()
    val_probs_list = []
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(device)
            probs = torch.sigmoid(model(x_batch)).cpu().numpy()
            val_probs_list.append(probs)
    val_probs = np.concatenate(val_probs_list, axis=0)

    val_threshold_info = {}
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

        val_threshold_info[cond] = {
            "selected_threshold": round(float(best_th), 2),
            "val_f1": round(float(best_f1), 4),
            "reason": f"Optimized on validation set to maximize F1-score ({best_f1:.4f})"
        }

    # Test Set Inference
    test_start = time.time()
    test_probs_list = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            probs = torch.sigmoid(model(x_batch)).cpu().numpy()
            test_probs_list.append(probs)
    test_probs = np.concatenate(test_probs_list, axis=0)

    inference_time_sec = round(time.time() - test_start, 4)
    per_sample_ms = round((inference_time_sec / len(X_test)) * 1000, 4)

    # Apply tuned thresholds
    test_pred_binary = np.zeros_like(test_probs, dtype=int)
    for idx, cond in enumerate(TARGET_CONDITIONS):
        th = val_threshold_info[cond]["selected_threshold"]
        test_pred_binary[:, idx] = (test_probs[:, idx] >= th).astype(int)

    # Metrics
    exact_match_acc = accuracy_score(Y_test, test_pred_binary)
    h_loss = hamming_loss(Y_test, test_pred_binary)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(Y_test, test_pred_binary, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(Y_test, test_pred_binary, average="weighted", zero_division=0)

    try:
        roc_auc_macro = roc_auc_score(Y_test, test_probs, average="macro")
    except Exception:
        roc_auc_macro = 0.0

    per_class_metrics = {}
    confusion_matrices = {}

    for idx, cond in enumerate(TARGET_CONDITIONS):
        y_true_c = Y_test.iloc[:, idx].values
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

    # Save Model Weights & Metrics
    output_dir = Path(__file__).resolve().parent.parent / "models" / "dnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_dnn.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": len(ALL_FEATURES),
        "num_targets": len(TARGET_CONDITIONS),
        "dropout": dropout,
        "features": ALL_FEATURES,
        "targets": TARGET_CONDITIONS
    }, model_path)

    metrics_payload = {
        "model_name": "Tabular Deep Neural Network (DNN)",
        "architecture": "Dense(256)->BN->ReLU->Drop(0.2)->Dense(128)->BN->ReLU->Drop(0.2)->Dense(64)->ReLU->Linear(3)",
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

    metrics_path = output_dir / "dnn_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nSaved DNN model to: {model_path}")
    print(f"Saved DNN metrics to: {metrics_path}")

    print("\n--- DNN Test Set Performance Summary ---")
    print(f"Exact Match Accuracy: {exact_match_acc * 100:.2f}%")
    print(f"Macro F1-Score:       {f1_macro:.4f}")
    print(f"Weighted F1-Score:    {f1_weighted:.4f}")
    print(f"Macro ROC-AUC:        {roc_auc_macro:.4f}")
    print(f"Hamming Loss:         {h_loss:.4f}")

    for cond, m in per_class_metrics.items():
        print(f"  {cond:22s} | Acc: {m['accuracy']*100:.2f}% | F1: {m['f1_score']:.4f} | AUC: {m['roc_auc']:.4f} | Thresh: {m['optimal_threshold']}")

    return model, metrics_payload


if __name__ == "__main__":
    train_and_evaluate_dnn()
