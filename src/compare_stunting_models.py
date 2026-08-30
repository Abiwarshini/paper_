"""Create a comparison table for the stunting baseline models."""

from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "results" / "stunting_model_comparison_table.txt"
    rows = [
        ("XGBoost", "Stunting", 0.6960, 0.6389, 0.3658, 0.4652, 0.7251),
        ("DNN", "Stunting", 0.6349, 0.4959, 0.6044, 0.5448, 0.6776),
        ("TabNet", "Stunting", 0.6155, 0.4764, 0.6428, 0.5473, 0.6676),
    ]
    header = "Model       Target     Accuracy  Precision  Recall    F1        ROC-AUC"
    separator = "-" * len(header)
    lines = [
        "STUNTING MODEL COMPARISON",
        "=" * len(header),
        header,
        separator,
    ]
    lines.extend(
        f"{model:<11} {target:<10} {accuracy:>8.2%}  {precision:>9.2%}  {recall:>7.2%}  {f1:>7.2%}  {roc_auc:>8.2%}"
        for model, target, accuracy, precision, recall, f1, roc_auc in rows
    )
    lines.append("=" * len(header))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
