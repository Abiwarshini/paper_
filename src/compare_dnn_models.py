"""Generate comparison table between DNN Stunting and DNN Wasting models."""

import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
results_dir = project_root / "results"

# Load metrics
with open(results_dir / "dnn_stunting_metrics.json", "r") as f:
    stunting_metrics = json.load(f)

with open(results_dir / "dnn_wasting_metrics.json", "r") as f:
    wasting_metrics = json.load(f)

# Extract test metrics
stunting_test = stunting_metrics["test_metrics"]
wasting_test = wasting_metrics["test_metrics"]

# Create comparison table
print("\n" + "=" * 95)
print("DNN MODEL COMPARISON: STUNTING vs WASTING")
print("=" * 95)
print(f"{'Model':<20} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'ROC-AUC':<15}")
print("-" * 95)
print(f"{'DNN Stunting':<20} {stunting_test['accuracy']:<15.4f} {stunting_test['precision']:<15.4f} {stunting_test['recall']:<15.4f} {stunting_test['f1_score']:<15.4f} {stunting_test['roc_auc']:<15.4f}")
print(f"{'DNN Wasting':<20} {wasting_test['accuracy']:<15.4f} {wasting_test['precision']:<15.4f} {wasting_test['recall']:<15.4f} {wasting_test['f1_score']:<15.4f} {wasting_test['roc_auc']:<15.4f}")
print("=" * 95)

# Save to file
output_file = results_dir / "dnn_comparison_table.txt"
with open(output_file, "w") as f:
    f.write("=" * 95 + "\n")
    f.write("DNN MODEL COMPARISON: STUNTING vs WASTING\n")
    f.write("=" * 95 + "\n")
    f.write(f"{'Model':<20} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'ROC-AUC':<15}\n")
    f.write("-" * 95 + "\n")
    f.write(f"{'DNN Stunting':<20} {stunting_test['accuracy']:<15.4f} {stunting_test['precision']:<15.4f} {stunting_test['recall']:<15.4f} {stunting_test['f1_score']:<15.4f} {stunting_test['roc_auc']:<15.4f}\n")
    f.write(f"{'DNN Wasting':<20} {wasting_test['accuracy']:<15.4f} {wasting_test['precision']:<15.4f} {wasting_test['recall']:<15.4f} {wasting_test['f1_score']:<15.4f} {wasting_test['roc_auc']:<15.4f}\n")
    f.write("=" * 95 + "\n")

print(f"\nComparison table saved to: {output_file}")

# Additional summary
print("\nMODEL SUMMARY:")
print("-" * 95)
print(f"Stunting Best Epoch: {stunting_metrics['best_epoch']}")
print(f"Wasting Best Epoch: {wasting_metrics['best_epoch']}")
print("-" * 95)
print(f"Stunting ROC-AUC:  {stunting_test['roc_auc']:.4f}")
print(f"Wasting ROC-AUC:   {wasting_test['roc_auc']:.4f}")
print(f"Difference:        {abs(stunting_test['roc_auc'] - wasting_test['roc_auc']):.4f}")
print("-" * 95)
