import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocessing import prepare_data_splits
from training.train_xgboost import train_and_evaluate_xgboost
from training.train_transformer import train_and_evaluate_transformer
from training.train_dnn import train_and_evaluate_dnn
from training.train_tabnet import train_and_evaluate_tabnet
from training.evaluate_models import generate_four_model_report


def main():
    print("=" * 80)
    print("      4-MODEL AI MALNUTRITION PREDICTION TRAINING & BENCHMARK SUITE")
    print("      Architectures: XGBoost, FT-Transformer, Deep Neural Network (DNN), TabNet")
    print("      Dataset: NFHS-5 dhs_clean.parquet (198,849 records)")
    print("=" * 80)

    total_start = time.time()

    # Step 1: Preprocessing & Data Splits
    print("\n>>> STEP 1/5: Loading & Preprocessing Dataset Splits...")
    prepare_data_splits(random_state=42)

    # Step 2: XGBoost Multi-Output Baseline
    print("\n>>> STEP 2/5: Training XGBoost Baseline...")
    train_and_evaluate_xgboost(random_state=42)

    # Step 3: Deep Neural Network (DNN)
    print("\n>>> STEP 3/5: Training Tabular Deep Neural Network (DNN)...")
    train_and_evaluate_dnn(epochs=12, batch_size=512, random_state=42)

    # Step 4: TabNet Multi-Task Classifier
    print("\n>>> STEP 4/5: Training TabNet Multi-Task Classifier...")
    train_and_evaluate_tabnet(max_epochs=12, batch_size=1024, random_state=42)

    # Step 5: FT-Transformer
    print("\n>>> STEP 5/5: Training FT-Transformer...")
    train_and_evaluate_transformer(epochs=10, batch_size=512, random_state=42)

    # Final Benchmark Comparison
    print("\n>>> COMPILING FINAL 4-MODEL EVALUATION REPORT...")
    generate_four_model_report()

    total_time = round(time.time() - total_start, 2)
    print(f"\nAll 4 models successfully trained and benchmarked in {total_time}s ({total_time/60:.2f} mins)!")


if __name__ == "__main__":
    main()
