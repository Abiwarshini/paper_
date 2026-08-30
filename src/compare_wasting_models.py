"""Create wasting-model and TabNet target comparison tables from saved metrics."""

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_row(model: str, target: str, metrics: dict[str, float]) -> str:
    return (
        f"{model:<14} {target:<10} {metrics['accuracy']:>8.2%}  "
        f"{metrics['precision']:>9.2%}  {metrics['recall']:>7.2%}  "
        f"{metrics['f1']:>7.2%}  {metrics['roc_auc']:>8.2%}"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    tabnet = load_json(results / "tabnet_wasting_metrics.json")
    dnn = load_json(results / "dnn_wasting_metrics.json")["test_metrics"]
    xgb = load_json(results / "wasting" / "enhanced" / "results_wasting_full.json")["XGBoost"]
    stunting = load_json(results / "tabnet_stunting_metrics.json")["test_metrics"]

    wasting_header = "Model          Target     Accuracy  Precision  Recall    F1        ROC-AUC"
    separator = "-" * len(wasting_header)
    wasting_metrics = {
        "accuracy": xgb["accuracy"], "precision": xgb["precision"],
        "recall": xgb["recall"], "f1": xgb["f1"], "roc_auc": xgb["auc_roc"],
    }
    dnn_metrics = {
        "accuracy": dnn["accuracy"], "precision": dnn["precision"],
        "recall": dnn["recall"], "f1": dnn["f1_score"], "roc_auc": dnn["roc_auc"],
    }
    tabnet_metrics = {
        "accuracy": tabnet["accuracy"], "precision": tabnet["precision"],
        "recall": tabnet["recall"], "f1": tabnet["f1_score"], "roc_auc": tabnet["roc_auc"],
    }
    lines = [
        "WASTING MODEL COMPARISON",
        "=" * len(wasting_header),
        wasting_header,
        separator,
        format_row("XGBoost", "Wasting", wasting_metrics),
        format_row("DNN", "Wasting", dnn_metrics),
        format_row("TabNet", "Wasting", tabnet_metrics),
        "=" * len(wasting_header),
        "",
        "TABNET TARGET COMPARISON",
        "=" * len(wasting_header),
        wasting_header,
        separator,
        format_row("TabNet Stunting", "Stunting", {
            "accuracy": stunting["accuracy"], "precision": stunting["precision"],
            "recall": stunting["recall"], "f1": stunting["f1_score"], "roc_auc": stunting["roc_auc"],
        }),
        format_row("TabNet Wasting", "Wasting", tabnet_metrics),
        "=" * len(wasting_header),
        "",
        "Note: XGBoost values are the existing enhanced-results values, evaluated at its saved optimal threshold.",
        "DNN and TabNet values use the saved baseline metrics at threshold 0.5.",
    ]
    output_path = results / "wasting_model_comparison_table.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
