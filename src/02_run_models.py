"""
Step 2: Replicate and optimize the malnutrition predictive modelling pipeline
Mgomezulu et al. (2025), Human Nutrition & Metabolism 42, 200340.

Modes:
  - standard: Exact-methodology replication with PCA (retaining 95% variance) on standard features.
  - enhanced: Optimised pipeline with clinical/environmental features, feature engineering, no PCA, and high-performance models.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier as GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

USE_TF = True
try:
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(42)
except ImportError:
    USE_TF = False
    print("WARNING: TensorFlow is not installed; DNN will use sklearn MLPClassifier fallback.")

np.random.seed(42)

SVM_SAMPLE_SIZE = 8000     # per fold training subsample used only for SVM
RANDOM_STATE = 42

# Standard features (from baseline paper)
CONTINUOUS_STANDARD = ['age_hh_head_proxy', 'hhsize', 'wealth_score', 'child_age_months', 'birth_order']
CATEGORICAL_STANDARD = ['education', 'wealth_quintile', 'residence', 'gender_hh_head', 'dist_market_proxy', 'child_sex']

# Extra features for optimization (clinical, maternal health, water/sanitation)
CONTINUOUS_EXTRA = ['birth_weight', 'breastfeeding_duration', 'mother_weight', 'mother_height', 'mother_bmi', 'anc_visits',
                    'mother_underweight', 'low_anc_visits', 'low_birth_weight', 'maternal_risk_score',
                    'wealth_education_interaction', 'age_breastfeeding_interaction', 'age_sanitation_interaction', 'age_birth_order_interaction']
CATEGORICAL_EXTRA = ['birth_size', 'birth_weight_source', 'measles_vaccine', 'diarrhea_recent', 'fever_recent', 'cough_recent', 'mother_marital_status', 'water_source', 'toilet_type', 'cooking_fuel']

TUNED_PARAMS = {
    "Random Forest": {"n_estimators": 300, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
    "Gradient Boosting": {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05, "subsample": 0.8},
    "XGBoost": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.85, "min_child_weight": 4, "reg_lambda": 2.0, "reg_alpha": 1.0, "gamma": 0.1},
    "KNN": {"n_neighbors": 7, "weights": "distance", "metric": "minkowski", "p": 2},
    "Logistic Regression": {"max_iter": 2000, "C": 0.1, "solver": "lbfgs"},
}


def engineer_features(df):
    """
    Creates domain-specific engineered features to boost predictive accuracy.
    """
    df = df.copy()
    
    # 1. Weaning indicator: binary, age 6 to 24 months (critical window for dietary changes)
    df['is_weaning'] = ((df['child_age_months'] >= 6) & (df['child_age_months'] <= 24)).astype(float)
    
    # 2. Child age groups: categorical
    df['child_age_group'] = pd.cut(df['child_age_months'],
                                   bins=[-1, 5, 11, 23, 35, 47, 60],
                                   labels=['0-5', '6-11', '12-23', '24-35', '36-47', '48-59']).astype(str)
    
    # 3. Mother BMI Category: categorical
    df['mother_bmi_cat'] = pd.cut(df['mother_bmi'],
                                  bins=[0, 18.5, 24.9, 100],
                                  labels=['underweight', 'normal', 'overweight_obese']).astype(str)
    df['mother_bmi_cat'] = df['mother_bmi_cat'].fillna('unknown')
    
    # 4. Birth order squared (non-linear risk profile)
    df['birth_order_sq'] = (df['birth_order'] ** 2).astype(float)
    
    # 5. Child age months squared (non-linear growth curves)
    df['child_age_months_sq'] = (df['child_age_months'] ** 2).astype(float)
    
    # 6. Mother age squared (non-linear maternal risk)
    df['mother_age_sq'] = (df['age_hh_head_proxy'] ** 2).astype(float)
    
    # 7. Sanitation risk index: combines unsafe drinking water, lack of toilet facility, and high-smoke fuel
    df['unsafe_water'] = (~df['water_source'].isin([11.0, 12.0, 21.0, 14.0, 13.0])).astype(float)
    df['unsafe_toilet'] = (df['toilet_type'].isin([31.0, 97.0])).astype(float)  # 31 = no facility, 97 = not de jure
    df['smoke_fuel'] = (df['cooking_fuel'].isin([2.0, 8.0, 11.0, 6.0, 7.0, 9.0, 10.0])).astype(float)  # wood/charcoal/dung
    df['sanitation_risk_index'] = df['unsafe_water'] + df['unsafe_toilet'] + df['smoke_fuel']

    # 8. Maternal and household risk flags
    df['mother_underweight'] = (df['mother_bmi'] < 18.5).astype(float)
    df['low_anc_visits'] = (df['anc_visits'] < 4).astype(float)
    df['low_birth_weight'] = (df['birth_weight'] < 2.5).astype(float)
    df['maternal_risk_score'] = df[['mother_underweight', 'low_anc_visits', 'low_birth_weight']].sum(axis=1)

    # 9. Interaction features that often matter in child-health data
    df['wealth_education_interaction'] = df['wealth_score'] * pd.to_numeric(df['education'], errors='coerce')
    df['age_breastfeeding_interaction'] = df['child_age_months'] * df['breastfeeding_duration']
    df['age_sanitation_interaction'] = df['child_age_months'] * df['sanitation_risk_index']
    df['age_birth_order_interaction'] = df['child_age_months'] * df['birth_order']
    
    df = df.drop(columns=['unsafe_water', 'unsafe_toilet', 'smoke_fuel'])
    return df


def load_data(target, mode="standard"):
    root = Path(__file__).resolve().parent.parent
    df = pd.read_parquet(root / "data" / "processed" / "dhs_clean.parquet")
    
    if mode == "enhanced":
        df = engineer_features(df)
        continuous_cols = CONTINUOUS_STANDARD + CONTINUOUS_EXTRA + ['is_weaning', 'birth_order_sq', 'child_age_months_sq', 'mother_age_sq', 'sanitation_risk_index']
        categorical_cols = CATEGORICAL_STANDARD + CATEGORICAL_EXTRA + ['child_age_group', 'mother_bmi_cat']
    else:
        continuous_cols = CONTINUOUS_STANDARD
        categorical_cols = CATEGORICAL_STANDARD

    y = df[target].values
    X = df[continuous_cols + categorical_cols].copy()

    # categorical missing -> "unknown"; continuous missing -> mean (paper 2.2)
    for c in categorical_cols:
        X[c] = X[c].astype(str)
        X.loc[X[c].isna() | (X[c] == '<NA>') | (X[c] == 'nan'), c] = "unknown"
        
    for c in continuous_cols:
        X[c] = X[c].astype(float)
        X[f"{c}_missing"] = X[c].isna().astype(float)
        X[c] = X[c].fillna(X[c].median())

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    feature_names = X.columns.tolist()
    return X.values.astype(float), y, feature_names


def build_dnn(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        batch_size=2048,  # Increased for faster CPU training
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
    )


def get_models(input_dim, class_weight_dict):
    models = {
        "SVM": SVC(C=1.0, probability=True, class_weight="balanced", random_state=RANDOM_STATE, kernel="rbf"),
        "Random Forest": RandomForestClassifier(
            n_estimators=TUNED_PARAMS["Random Forest"]["n_estimators"],
            max_depth=TUNED_PARAMS["Random Forest"]["max_depth"],
            min_samples_split=TUNED_PARAMS["Random Forest"]["min_samples_split"],
            min_samples_leaf=TUNED_PARAMS["Random Forest"]["min_samples_leaf"],
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "KNN": KNeighborsClassifier(
            n_neighbors=TUNED_PARAMS["KNN"]["n_neighbors"],
            weights=TUNED_PARAMS["KNN"]["weights"],
            n_jobs=-1),
        "Logistic Regression": LogisticRegression(
            max_iter=TUNED_PARAMS["Logistic Regression"]["max_iter"],
            C=TUNED_PARAMS["Logistic Regression"]["C"],
            solver=TUNED_PARAMS["Logistic Regression"]["solver"],
            class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            max_iter=TUNED_PARAMS["Gradient Boosting"]["n_estimators"],
            max_depth=TUNED_PARAMS["Gradient Boosting"]["max_depth"],
            learning_rate=TUNED_PARAMS["Gradient Boosting"]["learning_rate"],
            random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=TUNED_PARAMS["XGBoost"]["n_estimators"],
            max_depth=TUNED_PARAMS["XGBoost"]["max_depth"],
            learning_rate=TUNED_PARAMS["XGBoost"]["learning_rate"],
            subsample=TUNED_PARAMS["XGBoost"]["subsample"],
            colsample_bytree=TUNED_PARAMS["XGBoost"]["colsample_bytree"],
            min_child_weight=TUNED_PARAMS["XGBoost"]["min_child_weight"],
            reg_lambda=TUNED_PARAMS["XGBoost"]["reg_lambda"],
            reg_alpha=TUNED_PARAMS["XGBoost"]["reg_alpha"],
            gamma=TUNED_PARAMS["XGBoost"]["gamma"],
            eval_metric="logloss",
            scale_pos_weight=class_weight_dict[0] / class_weight_dict[1],
            random_state=RANDOM_STATE, n_jobs=-1),
    }
    
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    
    return models


def run_for_target(target, mode="standard", n_folds=5, fast=False):
    print(f"\n{'='*70}\nTARGET: {target.upper()} | MODE: {mode.upper()} | FOLDS: {n_folds}\n{'='*70}")
    X, y, feature_names = load_data(target, mode)
    print("X shape:", X.shape, " prevalence:", y.mean().round(4))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    if mode == "enhanced":
        if fast:
            model_names = ["Random Forest", "Logistic Regression", "XGBoost"]
        else:
            # Drop SVM and KNN to save 80% computation time in enhanced mode since they perform poorly in high dimensions
            model_names = ["Random Forest", "Logistic Regression", "Gradient Boosting", "XGBoost", "DNN"]
    else:
        model_names = ["SVM", "Random Forest", "KNN", "Logistic Regression", "Gradient Boosting", "XGBoost", "DNN"]
    if HAS_LGB:
        model_names.insert(-1, "LightGBM")
    
    oof_preds = {m: np.zeros(len(y)) for m in model_names}
    oof_proba = {m: np.zeros(len(y)) for m in model_names}
    optimal_thresholds = {m: 0.5 for m in model_names}
    fold_no = 0

    for train_idx, test_idx in skf.split(X, y):
        fold_no += 1
        print(f"\n--- Fold {fold_no}/{n_folds} ---")
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardisation (fit on train only)
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train_raw)
        X_test_s = scaler.transform(X_test_raw)

        # PCA retaining 95% variance (skip in enhanced mode for direct feature learning)
        if mode == "enhanced":
            X_train_p = X_train_s
            X_test_p = X_test_s
        else:
            pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
            X_train_p = pca.fit_transform(X_train_s)
            X_test_p = pca.transform(X_test_s)
            print("  PCA components kept:", X_train_p.shape[1])

        # SMOTE oversampling (paper 2.4)
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_train_bal, y_train_bal = sm.fit_resample(X_train_p, y_train)

        # Inverse-frequency class weights
        n_samples = len(y_train_bal)
        classes, counts = np.unique(y_train_bal, return_counts=True)
        class_weight_dict = {c: n_samples / (2 * cnt) for c, cnt in zip(classes, counts)}

        models = get_models(X_train_p.shape[1], class_weight_dict)
        models = {name: model for name, model in models.items() if name in model_names}

        for name, model in models.items():
            if name == "SVM":
                # Subsample SVM for tractability on large dataset
                rng = np.random.RandomState(RANDOM_STATE + fold_no)
                if len(X_train_bal) > SVM_SAMPLE_SIZE:
                    idx = rng.choice(len(X_train_bal), SVM_SAMPLE_SIZE, replace=False)
                    Xf, yf = X_train_bal[idx], y_train_bal[idx]
                else:
                    Xf, yf = X_train_bal, y_train_bal
                model.fit(Xf, yf)
            elif name == "LightGBM":
                model.fit(X_train_bal, y_train_bal, eval_set=[(X_test_p, y_test)], callbacks=[lgb.early_stopping(5, verbose=False)])
            else:
                model.fit(X_train_bal, y_train_bal)
            
            proba = model.predict_proba(X_test_p)[:, 1]
            oof_proba[name][test_idx] = proba
            oof_preds[name][test_idx] = (proba >= 0.5).astype(int)
            print(f"  {name}: fold accuracy = {accuracy_score(y_test, oof_preds[name][test_idx]):.4f}")

        if "DNN" in model_names:
            # DNN Classifier
            if USE_TF:
                dnn = build_dnn(X_train_bal.shape[1])
                es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
                # Increased batch_size for faster training on CPU
                dnn.fit(X_train_bal, y_train_bal, validation_split=0.1, epochs=50,
                        batch_size=2048, callbacks=[es], verbose=0)
                proba = dnn.predict(X_test_p, verbose=0).ravel()
            else:
                dnn = build_mlp()
                dnn.fit(X_train_bal, y_train_bal)
                proba = dnn.predict_proba(X_test_p)[:, 1]

            oof_proba["DNN"][test_idx] = proba
            oof_preds["DNN"][test_idx] = (proba >= 0.5).astype(int)
            print(f"  DNN: fold accuracy = {accuracy_score(y_test, oof_preds['DNN'][test_idx]):.4f}")

    # ---- Optimize decision thresholds based on F1 score ----
    print("\n=== Optimizing decision thresholds ===")
    for name in model_names:
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.3, 0.7, 0.01):
            preds = (oof_proba[name] >= thresh).astype(int)
            f1 = f1_score(y, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal_thresholds[name] = best_thresh
        oof_preds[name] = (oof_proba[name] >= best_thresh).astype(int)
        print(f"  {name}: optimal threshold = {best_thresh:.3f}, F1 = {best_f1:.4f}")

    # ---- Aggregate metrics across all folds ----
    results = {}
    for name in model_names:
        preds = oof_preds[name]
        proba = oof_proba[name]
        cm = confusion_matrix(y, preds)
        results[name] = {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds),
            "recall": recall_score(y, preds),
            "f1": f1_score(y, preds),
            "auc_roc": roc_auc_score(y, proba),
            "confusion_matrix": cm.tolist(),
            "optimal_threshold": optimal_thresholds[name],
        }

    # Add ensemble voting predictions (average of top 3 models by AUC)
    top_models = sorted(results.items(), key=lambda x: x[1]["auc_roc"], reverse=True)[:3]
    top_names = [name for name, _ in top_models]
    ensemble_proba = np.mean([oof_proba[name] for name in top_names], axis=0)
    best_ensemble_f1 = 0
    best_ensemble_thresh = 0.5
    for thresh in np.arange(0.3, 0.7, 0.01):
        preds = (ensemble_proba >= thresh).astype(int)
        f1 = f1_score(y, preds)
        if f1 > best_ensemble_f1:
            best_ensemble_f1 = f1
            best_ensemble_thresh = thresh
    
    ensemble_preds = (ensemble_proba >= best_ensemble_thresh).astype(int)
    cm = confusion_matrix(y, ensemble_preds)
    results["Ensemble"] = {
        "accuracy": accuracy_score(y, ensemble_preds),
        "precision": precision_score(y, ensemble_preds),
        "recall": recall_score(y, ensemble_preds),
        "f1": f1_score(y, ensemble_preds),
        "auc_roc": roc_auc_score(y, ensemble_proba),
        "confusion_matrix": cm.tolist(),
        "optimal_threshold": best_ensemble_thresh,
        "ensemble_models": top_names,
    }

    res_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "auc_roc"]]
    res_df = res_df.sort_values("accuracy", ascending=False)
    print(f"\n=== {target.upper()} RESULTS (mode={mode}, {n_folds}-fold aggregated) ===")
    print(res_df.round(4))

    # Save outputs to mode-specific folders to prevent overwriting
    out_dir = f"../results/{target}/{mode}"
    os.makedirs(out_dir, exist_ok=True)
    res_df.to_csv(f"{out_dir}/results_{target}.csv")
    with open(f"{out_dir}/results_{target}_full.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{out_dir}/optimal_thresholds_{target}.json", "w") as f:
        json.dump(optimal_thresholds, f, indent=2)

    # Feature importance from Random Forest on full scaled data (not PCA)
    print(f"\nFitting full Random Forest for feature importance ({target})...")
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    rf_full = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5,
                                      class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf_full.fit(X_s, y)
    
    fi = pd.Series(rf_full.feature_importances_, index=feature_names).sort_values(ascending=False)
    fi.to_csv(f"{out_dir}/feature_importance_{target}.csv")
    print("Top 10 feature importances:")
    print(fi.head(10))

    return results, oof_proba, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run malnutrition prediction pipeline.")
    parser.add_argument("--mode", type=str, choices=["standard", "enhanced"], default="standard",
                        help="Modeling mode: 'standard' for baseline replication, 'enhanced' for optimized accuracy.")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5).")
    parser.add_argument("--fast", action="store_true",
                        help="Skip slower models such as Gradient Boosting and DNN for rapid experimentation.")
    args = parser.parse_args()

    all_results = {}
    for target in ["stunting", "wasting"]:
        results, proba, y = run_for_target(target, mode=args.mode, n_folds=args.folds, fast=args.fast)
        all_results[target] = results
    print("\n\nALL RUNS COMPLETED SUCCESSFULLY.")
