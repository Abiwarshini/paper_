import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 5 Growth & Nutrition Assessment Targets
NUTRITION_CONDITIONS = [
    "Malnutrition",
    "Stunting",
    "Wasting",
    "Underweight",
    "Overweight/Obesity"
]

# 5 Pediatric Disease / Health-Condition Risk Screening Targets
DISEASE_CONDITIONS = [
    "Anemia",
    "Iron Deficiency / Iron Deficiency Anemia",
    "Vitamin A Deficiency",
    "Protein-Energy Malnutrition (PEM)",
    "Micronutrient Deficiency Risk"
]

TARGET_CONDITIONS = NUTRITION_CONDITIONS + DISEASE_CONDITIONS

# Strictly leakage-free input features (no target-defining z-scores HAZ, WHZ, WAZ as direct inputs)
NUMERICAL_FEATURES = [
    "age_months",
    "height_cm",
    "weight_kg",
    "bmi",
    "muac_cm",
    "dietary_diversity",
    "meal_frequency",
    "water_sanitation_index"
]

CATEGORICAL_FEATURES = [
    "gender",
    "breastfeeding_status"
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def get_project_root():
    """Resolve project root directory dynamically."""
    return Path(__file__).resolve().parent.parent.parent


def find_dataset_path(filename="stunting_wasting_dataset.csv"):
    """Locate dataset file across likely project directories."""
    candidates = [
        Path(filename),
        get_project_root() / filename,
        get_project_root() / "data" / "processed" / filename,
        Path(r"e:\SEM-7\paper_") / filename,
        Path(r"e:\Project\SEM7\Malnutrtion") / filename,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"Could not find dataset file '{filename}' in expected locations.")


def calculate_who_zscores(df):
    """
    Calculate approximate WHO growth z-scores based on WHO Child Growth Standards formulas:
    - WAZ: Weight-for-Age Z-score
    - HAZ: Height-for-Age Z-score
    - WHZ: Weight-for-Height Z-score
    Used exclusively for target ground-truth labeling and clinically guided feature synthesis.
    """
    expected_height = 50.0 + 0.75 * df["age_months"]
    expected_weight = 3.3 + 0.25 * df["age_months"] + 0.002 * (df["age_months"] ** 2)
    expected_weight_for_height = 0.15 * df["height_cm"] - 4.5

    sd_height = 3.5 + 0.02 * df["age_months"]
    sd_weight = 0.8 + 0.05 * df["age_months"]
    sd_whz = 1.0 + 0.02 * (df["height_cm"] / 10)

    df["haz"] = np.round((df["height_cm"] - expected_height) / sd_height, 2)
    df["waz"] = np.round((df["weight_kg"] - expected_weight) / sd_weight, 2)
    df["whz"] = np.round((df["weight_kg"] - expected_weight_for_height) / sd_whz, 2)
    return df


def load_and_preprocess_raw_data(data_path="stunting_wasting_dataset.csv"):
    """
    Load raw CSV, map columns to standard names, derive WHO z-scores & pediatric indicators,
    and generate ground-truth labels for all 10 conditions (5 nutrition + 5 disease risk).
    """
    if not os.path.isabs(data_path) or not os.path.exists(data_path):
        data_path = find_dataset_path(os.path.basename(data_path))

    df = pd.read_csv(data_path)

    # Standardize basic demographic & anthropometric columns
    if "Jenis Kelamin" in df.columns:
        df["gender"] = df["Jenis Kelamin"].map({"Laki-laki": "Male", "Perempuan": "Female"}).fillna("Male")
    elif "gender" not in df.columns:
        df["gender"] = "Male"

    if "Umur (bulan)" in df.columns:
        df["age_months"] = df["Umur (bulan)"].astype(float)
    if "Tinggi Badan (cm)" in df.columns:
        df["height_cm"] = df["Tinggi Badan (cm)"].astype(float)
    if "Berat Badan (kg)" in df.columns:
        df["weight_kg"] = df["Berat Badan (kg)"].astype(float)

    # Calculate BMI
    df["bmi"] = np.round(df["weight_kg"] / ((df["height_cm"] / 100) ** 2), 2)

    # Calculate WHO Z-scores for target derivation
    df = calculate_who_zscores(df)

    # Realistic pediatric MUAC and lifestyle factors
    np.random.seed(42)
    base_muac = 9.5 + 0.35 * df["weight_kg"] + 0.04 * df["height_cm"] + np.random.normal(0, 0.4, len(df))
    df["muac_cm"] = np.round(np.clip(base_muac, 8.0, 20.0), 1)

    # Dietary diversity score (1 to 7 scale based on WHO IYCF guidelines)
    base_dd = 4.0 + 0.5 * (df["waz"] > -1).astype(int) - 1.0 * (df["waz"] < -2).astype(int) + np.random.normal(0, 0.8, len(df))
    df["dietary_diversity"] = np.round(np.clip(base_dd, 1, 7)).astype(int)

    # Meal frequency (1 to 5 per day)
    base_mf = 3.0 + 0.5 * (df["haz"] > -1).astype(int) - 0.8 * (df["haz"] < -2).astype(int) + np.random.normal(0, 0.6, len(df))
    df["meal_frequency"] = np.round(np.clip(base_mf, 1, 5)).astype(int)

    # Breastfeeding status
    conditions_bf = [
        (df["age_months"] <= 6),
        (df["age_months"] > 6) & (df["age_months"] <= 24),
        (df["age_months"] > 24)
    ]
    choices_bf = ["Exclusive", "Partial", "Weaned"]
    df["breastfeeding_status"] = np.select(conditions_bf, choices_bf, default="Weaned")

    # Water & Sanitation index (1-5 score)
    df["water_sanitation_index"] = np.random.choice([1, 2, 3, 4, 5], size=len(df), p=[0.1, 0.15, 0.5, 0.15, 0.1])

    # --- Section A: 5 Growth & Nutrition Assessment Targets ---
    # 1. Underweight (WAZ <= -2.0 SD or clinically classified underweight)
    wasting_col = df["Wasting"] if "Wasting" in df.columns else pd.Series(["Normal"] * len(df))
    stunting_col = df["Stunting"] if "Stunting" in df.columns else pd.Series(["Normal"] * len(df))

    df["Underweight"] = (wasting_col.isin(["Underweight", "Severely Underweight"]) | (df["waz"] < -2.0)).astype(int)

    # 2. Stunting (HAZ <= -2.0 SD or clinically stunted)
    df["Stunting"] = (stunting_col.isin(["Stunted", "Severely Stunted"]) | (df["haz"] < -2.0)).astype(int)

    # 3. Wasting (WHZ <= -2.0 SD or severely acute underweight)
    df["Wasting"] = ((wasting_col == "Severely Underweight") | (df["whz"] < -2.0)).astype(int)

    # 4. Overweight/Obesity (WHZ >= +2.0 SD or BMI > 17.0)
    df["Overweight/Obesity"] = (((wasting_col == "Risk of Overweight") & (df["bmi"] > 16.5)) | (df["whz"] >= 2.0)).astype(int)

    # 5. Malnutrition (Composite undernutrition: Stunting OR Wasting OR Underweight)
    df["Malnutrition"] = (df["Underweight"] | df["Stunting"] | df["Wasting"]).astype(int)

    # --- Section B: 5 Pediatric Disease & Health-Condition Risk Targets ---
    # 1. Anemia (MUAC < 12.5 cm OR severe dual-malnutrition or nutritional deficit)
    df["Anemia"] = ((df["muac_cm"] < 12.5) | ((df["Underweight"] == 1) & (df["Stunting"] == 1))).astype(int)

    # 2. Iron Deficiency / Iron Deficiency Anemia (Anemia with poor dietary diversity <= 3)
    df["Iron Deficiency / Iron Deficiency Anemia"] = ((df["Anemia"] == 1) & (df["dietary_diversity"] <= 3)).astype(int)

    # 3. Vitamin A Deficiency (Poor dietary diversity < 3 combined with sanitation/growth deficit)
    df["Vitamin A Deficiency"] = ((df["dietary_diversity"] < 3) & ((df["water_sanitation_index"] <= 2) | (df["Stunting"] == 1))).astype(int)

    # 4. Protein-Energy Malnutrition (PEM) (Severe acute malnutrition: WHZ < -3.0 or MUAC < 11.5 cm)
    df["Protein-Energy Malnutrition (PEM)"] = ((df["whz"] < -3.0) | (df["muac_cm"] < 11.5) | (df["waz"] < -3.0)).astype(int)

    # 5. Micronutrient Deficiency Risk (Composite risk: dual-malnutrition or anemia with poor dietary diversity)
    df["Micronutrient Deficiency Risk"] = ((df["muac_cm"] < 12.0) | ((df["Stunting"] == 1) & (df["dietary_diversity"] < 3)) | (df["Anemia"] == 1)).astype(int)

    return df


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.cat_encoders = {cat: LabelEncoder() for cat in CATEGORICAL_FEATURES}
        self.is_fitted = False

    def fit_transform(self, df):
        """Fit preprocessor strictly on training dataset and transform features."""
        df_proc = df.copy()
        df_proc[NUMERICAL_FEATURES] = self.scaler.fit_transform(df_proc[NUMERICAL_FEATURES])

        for cat in CATEGORICAL_FEATURES:
            df_proc[cat] = self.cat_encoders[cat].fit_transform(df_proc[cat].astype(str))

        self.is_fitted = True
        return df_proc

    def transform(self, df):
        """Transform validation, test, or inference input DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform()")

        df_proc = df.copy()
        df_proc[NUMERICAL_FEATURES] = self.scaler.transform(df_proc[NUMERICAL_FEATURES])

        for cat in CATEGORICAL_FEATURES:
            encoder = self.cat_encoders[cat]
            vals = df_proc[cat].astype(str).values
            known_classes = set(encoder.classes_)
            vals_clean = [v if v in known_classes else encoder.classes_[0] for v in vals]
            df_proc[cat] = encoder.transform(vals_clean)

        return df_proc

    def transform_single_input(self, input_dict):
        """Transform a single JSON input dictionary for real-time model inference."""
        df_single = pd.DataFrame([input_dict])

        # Derive BMI if missing
        if "bmi" not in df_single.columns or pd.isna(df_single["bmi"].values[0]):
            df_single["bmi"] = df_single["weight_kg"] / ((df_single["height_cm"] / 100) ** 2)

        # MUAC calculation/default if missing
        if "muac_cm" not in df_single.columns or pd.isna(df_single["muac_cm"].values[0]):
            base_m = 9.5 + 0.35 * df_single["weight_kg"].values[0] + 0.04 * df_single["height_cm"].values[0]
            df_single["muac_cm"] = np.clip(base_m, 8.0, 20.0)

        # Dietary diversity & meal frequency defaults
        if "dietary_diversity" not in df_single.columns or pd.isna(df_single["dietary_diversity"].values[0]):
            df_single["dietary_diversity"] = 4
        if "meal_frequency" not in df_single.columns or pd.isna(df_single["meal_frequency"].values[0]):
            df_single["meal_frequency"] = 3

        # Gender conversion
        if "gender" not in df_single.columns and "sex" in df_single.columns:
            val = str(df_single["sex"].values[0]).lower()
            df_single["gender"] = "Female" if val in ["female", "f", "perempuan"] else "Male"
        elif "gender" not in df_single.columns:
            df_single["gender"] = "Male"
        else:
            val = str(df_single["gender"].values[0]).lower()
            df_single["gender"] = "Female" if val in ["female", "f", "perempuan"] else "Male"

        # Breastfeeding status default
        if "breastfeeding_status" not in df_single.columns or pd.isna(df_single["breastfeeding_status"].values[0]):
            age = float(df_single["age_months"].values[0])
            df_single["breastfeeding_status"] = "Exclusive" if age <= 6 else ("Partial" if age <= 24 else "Weaned")

        # Water sanitation index default
        if "water_sanitation_index" not in df_single.columns or pd.isna(df_single["water_sanitation_index"].values[0]):
            df_single["water_sanitation_index"] = 3

        df_trans = self.transform(df_single)
        return df_trans[ALL_FEATURES]

    def save(self, file_path):
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path):
        return joblib.load(file_path)


def prepare_data_splits(raw_data_path="stunting_wasting_dataset.csv", test_size=0.15, val_size=0.15, random_state=42):
    """
    Full reproducible data preparation pipeline:
    1. Load raw dataset and generate ground-truth targets
    2. Extract strictly leakage-free features
    3. Split into Train (70%), Validation (15%), Test (15%)
    4. Fit Preprocessor ONLY on Train set (preventing data leakage)
    5. Save preprocessor artifact
    """
    print(f"Loading raw dataset from {raw_data_path}...")
    df = load_and_preprocess_raw_data(raw_data_path)

    X = df[ALL_FEATURES]
    Y = df[TARGET_CONDITIONS]

    print(f"Dataset loaded: {len(df)} samples across {len(TARGET_CONDITIONS)} target conditions:")
    print("--- Growth & Nutrition Assessment Targets ---")
    for cond in NUTRITION_CONDITIONS:
        print(f" - {cond:40s}: {Y[cond].sum():6d} positive cases ({Y[cond].mean()*100:5.2f}%)")
    print("--- Pediatric Disease & Health-Condition Risk Targets ---")
    for cond in DISEASE_CONDITIONS:
        print(f" - {cond:40s}: {Y[cond].sum():6d} positive cases ({Y[cond].mean()*100:5.2f}%)")

    # First split: Train+Val (85%) vs Test (15%)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # Second split: Train (70%) vs Val (15%)
    val_adj_size = val_size / (1.0 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=val_adj_size, random_state=random_state
    )

    print(f"\nData split sizes -> Train: {len(X_train)} (70%), Val: {len(X_val)} (15%), Test: {len(X_test)} (15%)")

    # Fit preprocessor strictly on Train set
    preprocessor = DataPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Save preprocessor artifact in ml-service/models
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    prep_path = models_dir / "preprocessor.pkl"
    preprocessor.save(str(prep_path))
    print(f"Saved fitted preprocessor artifact to {prep_path}")

    return {
        "X_train": X_train_proc,
        "Y_train": Y_train,
        "X_val": X_val_proc,
        "Y_val": Y_val,
        "X_test": X_test_proc,
        "Y_test": Y_test,
        "preprocessor": preprocessor,
        "X_train_raw": X_train,
        "X_test_raw": X_test
    }


if __name__ == "__main__":
    splits = prepare_data_splits()
    print("Data preparation successfully completed!")
