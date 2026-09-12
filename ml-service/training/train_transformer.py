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

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import (
    prepare_data_splits, TARGET_CONDITIONS, NUTRITION_CONDITIONS,
    DISEASE_CONDITIONS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
)


class FTTransformer(nn.Module):
    """
    FT-Transformer (Feature Tokenizer Transformer) for Tabular Multi-Target Prediction.
    - Feature Tokenizer: maps continuous & categorical features into token embeddings (d_token)
    - Prepend learned [CLS] token
    - Transformer Encoder Stack with multi-head self-attention
    - Multi-target prediction head across 10 conditions (5 Growth & Nutrition + 5 Disease Risk)
    """

    def __init__(
        self,
        num_numerical=len(NUMERICAL_FEATURES),
        categorical_cardinalities=[2, 3],  # gender (2), breastfeeding_status (3)
        d_token=64,
        n_layers=3,
        n_heads=4,
        d_ffn=128,
        dropout=0.1,
        n_targets=len(TARGET_CONDITIONS)
    ):
        super(FTTransformer, self).__init__()

        self.num_numerical = num_numerical
        self.d_token = d_token
        self.n_targets = n_targets

        # Numerical Feature Tokenizer weights & biases
        self.num_weights = nn.Parameter(torch.randn(num_numerical, d_token) * 0.01)
        self.num_biases = nn.Parameter(torch.zeros(num_numerical, d_token))

        # Categorical Feature Tokenizers
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_token) for cardinality in categorical_cardinalities
        ])

        # [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.01)

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Multi-task classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_ffn // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn // 2, n_targets)
        )

    def forward(self, x_num, x_cat):
        """
        Forward pass.
        x_num: (batch_size, num_numerical)
        x_cat: (batch_size, num_categorical)
        """
        batch_size = x_num.size(0)

        # 1. Tokenize Numerical Features: (batch_size, num_num, d_token)
        x_num_expanded = x_num.unsqueeze(-1)
        num_tokens = x_num_expanded * self.num_weights.unsqueeze(0) + self.num_biases.unsqueeze(0)

        # 2. Tokenize Categorical Features: list of (batch_size, 1, d_token)
        cat_tokens = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_tokens.append(emb(x_cat[:, i]).unsqueeze(1))

        if cat_tokens:
            cat_tokens = torch.cat(cat_tokens, dim=1)
            feature_tokens = torch.cat([num_tokens, cat_tokens], dim=1)
        else:
            feature_tokens = num_tokens

        # 3. Prepend [CLS] token: (batch_size, 1 + num_features, d_token)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, feature_tokens], dim=1)

        # 4. Pass through Transformer Encoder
        transformed = self.transformer(tokens)

        # 5. Extract [CLS] output representation
        cls_out = transformed[:, 0, :]

        # 6. Classification Head (logits)
        logits = self.head(cls_out)
        return logits


class TabularDataset(Dataset):
    def __init__(self, df_x, df_y=None):
        self.x_num = torch.tensor(df_x[NUMERICAL_FEATURES].values, dtype=torch.float32)
        self.x_cat = torch.tensor(df_x[CATEGORICAL_FEATURES].values, dtype=torch.long)
        if df_y is not None:
            self.y = torch.tensor(df_y.values, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_num[idx], self.x_cat[idx], self.y[idx]
        return self.x_num[idx], self.x_cat[idx]


def tune_transformer_thresholds(val_preds, val_targets):
    """Calibrate optimal decision threshold per condition on validation set."""
    thresholds = {}
    candidate_th = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    for idx, cond in enumerate(TARGET_CONDITIONS):
        y_true = val_targets[:, idx]
        probs = val_preds[:, idx]

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
            "reason": f"Validation F1-score optimization ({best_f1:.4f})"
        }

    return thresholds


def train_and_evaluate_transformer(epochs=10, batch_size=256, lr=1e-3, random_state=42):
    print("=== Training FT-Transformer Multi-Target Model (10 Conditions) ===")
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    start_time = time.time()

    splits = prepare_data_splits(random_state=random_state)
    X_train, Y_train = splits["X_train"], splits["Y_train"]
    X_val, Y_val = splits["X_val"], splits["Y_val"]
    X_test, Y_test = splits["X_test"], splits["Y_test"]

    # Datasets & DataLoaders
    train_dataset = TabularDataset(X_train, Y_train)
    val_dataset = TabularDataset(X_val, Y_val)
    test_dataset = TabularDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    model = FTTransformer(
        num_numerical=len(NUMERICAL_FEATURES),
        categorical_cardinalities=[2, 3],
        d_token=64,
        n_targets=len(TARGET_CONDITIONS)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_f1 = -1.0
    best_val_loss = float("inf")

    model_dir = Path(__file__).resolve().parent.parent / "models" / "transformer"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_transformer.pt"

    train_start = time.time()
    best_val_preds = None
    best_val_targets = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x_num, x_cat)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                logits = model(x_num, x_cat)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(y.cpu().numpy())

        val_loss /= len(val_dataset)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_binary = (val_preds >= 0.5).astype(int)
        val_exact = accuracy_score(val_targets, val_binary)
        val_f1_macro = precision_recall_fscore_support(val_targets, val_binary, average="macro", zero_division=0)[2]

        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Exact Acc: {val_exact*100:.2f}% | Val Macro F1: {val_f1_macro:.4f}")

        if val_f1_macro > best_val_f1 or (val_f1_macro == best_val_f1 and val_loss < best_val_loss):
            best_val_f1 = val_f1_macro
            best_val_loss = val_loss
            best_val_preds = val_preds
            best_val_targets = val_targets

            torch.save({
                "model_state_dict": model.state_dict(),
                "d_token": model.d_token,
                "n_targets": model.n_targets,
                "num_numerical": len(NUMERICAL_FEATURES),
                "categorical_cardinalities": [2, 3]
            }, str(best_model_path))
            print(f" -> Checkpoint saved (Val Macro F1: {best_val_f1:.4f})")

    training_time_sec = round(time.time() - train_start, 2)
    print(f"\nFT-Transformer training finished in {training_time_sec}s!")

    # Threshold calibration
    val_threshold_info = tune_transformer_thresholds(best_val_preds, best_val_targets)

    # Test Set Evaluation
    print("Evaluating FT-Transformer on held-out test set (15,000 samples)...")
    checkpoint = torch.load(str(best_model_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_start = time.time()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x_num, x_cat, y in test_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            logits = model(x_num, x_cat)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds.append(probs)
            test_targets.append(y.numpy())

    inference_time_sec = round(time.time() - test_start, 4)
    per_sample_inference_ms = round((inference_time_sec / len(test_dataset)) * 1000, 4)

    y_pred_probs = np.vstack(test_preds)
    Y_test_arr = np.vstack(test_targets)

    # Apply calibrated validation thresholds
    y_pred_binary = np.zeros_like(y_pred_probs, dtype=int)
    for idx, cond in enumerate(TARGET_CONDITIONS):
        th = val_threshold_info[cond]["selected_threshold"]
        y_pred_binary[:, idx] = (y_pred_probs[:, idx] >= th).astype(int)

    # Metrics
    exact_match_acc = accuracy_score(Y_test_arr, y_pred_binary)
    h_loss = hamming_loss(Y_test_arr, y_pred_binary)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(Y_test_arr, y_pred_binary, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(Y_test_arr, y_pred_binary, average="weighted", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(Y_test_arr, y_pred_binary, average="micro", zero_division=0)

    try:
        roc_auc_macro = roc_auc_score(Y_test_arr, y_pred_probs, average="macro")
    except Exception:
        roc_auc_macro = 0.0

    per_class_metrics = {}
    confusion_matrices = {}

    for idx, cond in enumerate(TARGET_CONDITIONS):
        y_true_c = Y_test_arr[:, idx]
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
        "model_name": "FT-Transformer Tabular Multi-Target Model",
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

    print("\n================ FT-Transformer Test Set Benchmark ================")
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

    # Save to model directories
    target_dirs = [
        model_dir,
        Path(r"e:\Project\SEM7\Malnutrtion") / "ml-service" / "models" / "transformer"
    ]

    for m_dir in target_dirs:
        try:
            m_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = m_dir / "transformer_metrics.json"
            model_save_path = m_dir / "best_transformer.pt"
            torch.save(checkpoint, str(model_save_path))
            with open(str(metrics_path), "w") as f:
                json.dump(overall_metrics, f, indent=2)
            print(f"Saved FT-Transformer model & metrics to {m_dir}")
        except Exception as e:
            print(f"Notice: skipped saving to {m_dir} ({e})")

    return model, overall_metrics


if __name__ == "__main__":
    train_and_evaluate_transformer(epochs=10, batch_size=256, lr=1e-3)
