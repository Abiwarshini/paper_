import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

TARGET_CONDITIONS = [
    "Underweight",
    "Stunting",
    "Wasting",
    "Overweight",
    "Obesity",
    "Anemia Risk",
    "Micronutrient Deficiency"
]

NUMERICAL_FEATURES = [
    "age_months",
    "height_cm",
    "weight_kg",
    "bmi",
    "waz",
    "haz",
    "whz",
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


def calculate_who_zscores(df):
    """
    Calculate approximate WHO growth z-scores based on WHO Child Growth Standards formulas:
    - WAZ: Weight-for-Age Z-score
    - HAZ: Height-for-Age Z-score
    - WHZ: Weight-for-Height Z-score
    """
    # Expected mean height & weight for age (0-60 months) from WHO standards
    expected_height = 50.0 + 0.75 * df["age_months"]
    expected_weight = 3.3 + 0.25 * df["age_months"] + 0.002 * (df["age_months"] ** 2)
    expected_weight_for_height = 0.15 * df["height_cm"] - 4.5

    # Standard deviations
    sd_height = 3.5 + 0.02 * df["age_months"]
    sd_weight = 0.8 + 0.05 * df["age_months"]
    sd_whz = 1.0 + 0.02 * (df["height_cm"] / 10)

    df["haz"] = np.round((df["height_cm"] - expected_height) / sd_height, 2)
    df["waz"] = np.round((df["weight_kg"] - expected_weight) / sd_weight, 2)
    df["whz"] = np.round((df["weight_kg"] - expected_weight_for_height) / sd_whz, 2)
    return df


def load_and_preprocess_raw_data(data_path="stunting_wasting_dataset.csv"):
    """
    Load raw CSV, map columns to standard names, derive WHO z-scores & additional child health indicators,
    and generate true multilabel binary targets for 7 nutrition conditions.
    """
    if not os.path.isabs(data_path):
        data_path = os.path.join(r"e:\Project\SEM7\Malnutrtion", data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    df = pd.read_csv(data_path)

    # Standardize basic demographic & anthropometric columns
    df["gender"] = df["Jenis Kelamin"].map({"Laki-laki": "Male", "Perempuan": "Female"}).fillna("Male")
    df["age_months"] = df["Umur (bulan)"].astype(float)
    df["height_cm"] = df["Tinggi Badan (cm)"].astype(float)
    df["weight_kg"] = df["Berat Badan (kg)"].astype(float)

    # Calculate BMI
    df["bmi"] = np.round(df["weight_kg"] / ((df["height_cm"] / 100) ** 2), 2)

    # Calculate WHO Z-scores
    df = calculate_who_zscores(df)

    # Generate synthetic realistic MUAC and lifestyle factors linked to anthropometrics for dataset richness
    np.random.seed(42)
    base_muac = 9.5 + 0.35 * df["weight_kg"] + 0.04 * df["height_cm"] + np.random.normal(0, 0.4, len(df))
    df["muac_cm"] = np.round(np.clip(base_muac, 8.0, 20.0), 1)

    # Dietary diversity score (1 to 7 scale based on WHO IYCF guidelines)
    base_dd = 4.0 + 0.5 * (df["waz"] > -1).astype(int) - 1.0 * (df["waz"] < -2).astype(int) + np.random.normal(0, 0.8, len(df))
    df["dietary_diversity"] = np.round(np.clip(base_dd, 1, 7)).astype(int)

    # Meal frequency (1 to 5 per day)
    base_mf = 3.0 + 0.5 * (df["haz"] > -1).astype(int) - 0.8 * (df["haz"] < -2).astype(int) + np.random.normal(0, 0.6, len(df))
    df["meal_frequency"] = np.round(np.clip(base_mf, 1, 5)).astype(int)

    # Breastfeeding status (Exclusive for <6m, Weaned for >24m, Partial for middle)
    conditions_bf = [
        (df["age_months"] <= 6),
        (df["age_months"] > 6) & (df["age_months"] <= 24),
        (df["age_months"] > 24)
    ]
    choices_bf = ["Exclusive", "Partial", "Weaned"]
    df["breastfeeding_status"] = np.select(conditions_bf, choices_bf, default="Weaned")

    # Water & Sanitation index (1-5 score)
    df["water_sanitation_index"] = np.random.choice([1, 2, 3, 4, 5], size=len(df), p=[0.1, 0.15, 0.5, 0.15, 0.1])

    # --- Target Labels (7 Nutrition Conditions) ---
    # 1. Underweight: Wasting status is Underweight/Severely Underweight OR WAZ < -2.0
    df["Underweight"] = (df["Wasting"].isin(["Underweight", "Severely Underweight"]) | (df["waz"] < -2.0)).astype(int)

    # 2. Stunting: Stunting status is Stunted/Severely Stunted OR HAZ < -2.0
    df["Stunting"] = (df["Stunting"].isin(["Stunted", "Severely Stunted"]) | (df["haz"] < -2.0)).astype(int)

    # 3. Wasting: Severely Underweight OR WHZ < -2.0
    df["Wasting"] = ((df["Wasting"] == "Severely Underweight") | (df["whz"] < -2.0)).astype(int)

    # 4. Overweight: Risk of Overweight with moderate BMI (15.5 < BMI <= 17.5) or WHZ between +2 and +3
    df["Overweight"] = (((df["Wasting"] == "Risk of Overweight") & (df["bmi"] <= 17.5)) | ((df["whz"] >= 2.0) & (df["whz"] < 3.0))).astype(int)

    # 5. Obesity: Risk of Overweight with high BMI (> 17.5) or WHZ >= 3.0
    df["Obesity"] = (((df["Wasting"] == "Risk of Overweight") & (df["bmi"] > 17.5)) | (df["whz"] >= 3.0)).astype(int)

    # 6. Anemia Risk: MUAC < 12.5 cm OR severe dual-malnutrition (Stunting & Underweight combined)
    df["Anemia Risk"] = ((df["muac_cm"] < 12.5) | ((df["Underweight"] == 1) & (df["Stunting"] == 1))).astype(int)

    # 7. Micronutrient Deficiency: MUAC < 12.0 cm OR Stunting combined with low dietary diversity (< 3)
    df["Micronutrient Deficiency"] = ((df["muac_cm"] < 12.0) | ((df["Stunting"] == 1) & (df["dietary_diversity"] < 3))).astype(int)

    return df


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.cat_encoders = {cat: LabelEncoder() for cat in CATEGORICAL_FEATURES}
        self.is_fitted = False

    def fit_transform(self, df):
        """Fit preprocessor on train dataset and transform numerical & categorical features."""
        df_proc = df.copy()

        # Handle numerical scaling
        df_proc[NUMERICAL_FEATURES] = self.scaler.fit_transform(df_proc[NUMERICAL_FEATURES])

        # Handle categorical label encoding
        for cat in CATEGORICAL_FEATURES:
            df_proc[cat] = self.cat_encoders[cat].fit_transform(df_proc[cat].astype(str))

        self.is_fitted = True
        return df_proc

    def transform(self, df):
        """Transform validation, test, or single inference input DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform()")

        df_proc = df.copy()
        df_proc[NUMERICAL_FEATURES] = self.scaler.transform(df_proc[NUMERICAL_FEATURES])

        for cat in CATEGORICAL_FEATURES:
            encoder = self.cat_encoders[cat]
            # Handle unseen categories gracefully
            vals = df_proc[cat].astype(str).values
            known_classes = set(encoder.classes_)
            vals_clean = [v if v in known_classes else encoder.classes_[0] for v in vals]
            df_proc[cat] = encoder.transform(vals_clean)

        return df_proc

    def transform_single_input(self, input_dict):
        """Transform a single JSON input dict for real-time model inference."""
        df_single = pd.DataFrame([input_dict])

        # Fill default values if optional inputs are missing
        if "bmi" not in df_single.columns or pd.isna(df_single["bmi"].values[0]):
            df_single["bmi"] = df_single["weight_kg"] / ((df_single["height_cm"] / 100) ** 2)

        # Derive Z-scores if missing
        if "waz" not in df_single.columns or "haz" not in df_single.columns or "whz" not in df_single.columns:
            df_single = calculate_who_zscores(df_single)

        if "muac_cm" not in df_single.columns:
            base_m = 9.5 + 0.35 * df_single["weight_kg"].values[0] + 0.04 * df_single["height_cm"].values[0]
            df_single["muac_cm"] = np.clip(base_m, 8.0, 20.0)

        if "dietary_diversity" not in df_single.columns:
            df_single["dietary_diversity"] = 4
        if "meal_frequency" not in df_single.columns:
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

        if "breastfeeding_status" not in df_single.columns:
            age = df_single["age_months"].values[0]
            df_single["breastfeeding_status"] = "Exclusive" if age <= 6 else ("Partial" if age <= 24 else "Weaned")

        if "water_sanitation_index" not in df_single.columns:
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
    1. Load raw dataset
    2. Extract targets & features
    3. Split into Train (70%), Validation (15%), Test (15%)
    4. Fit Preprocessor ONLY on Train set (preventing data leakage)
    5. Save preprocessor artifact
    """
    print(f"Loading raw dataset from {raw_data_path}...")
    df = load_and_preprocess_raw_data(raw_data_path)

    X = df[ALL_FEATURES]
    Y = df[TARGET_CONDITIONS]

    print(f"Dataset loaded: {len(df)} samples across 7 target conditions:")
    for cond in TARGET_CONDITIONS:
        print(f" - {cond:25s}: {Y[cond].sum()} positive cases ({Y[cond].mean()*100:.2f}%)")

    # First split: Train+Val vs Test
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # Second split: Train vs Val
    val_adj_size = val_size / (1.0 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=val_adj_size, random_state=random_state
    )

    print(f"\nData split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Fit preprocessor strictly on Train set
    preprocessor = DataPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Save preprocessor
    prep_path = os.path.join(r"e:\Project\SEM7\Malnutrtion", "ml-service", "models", "preprocessor.pkl")
    os.makedirs(os.path.dirname(prep_path), exist_ok=True)
    preprocessor.save(prep_path)
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
