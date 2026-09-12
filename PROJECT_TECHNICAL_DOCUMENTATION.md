# 4-Model AI Malnutrition Prediction & Clinical Growth Screening System
### XGBoost, FT-Transformer, Deep Neural Network (DNN), and TabNet on NFHS-5 Survey Records

---

## 1. System Architecture Overview

This project implements an end-to-end clinical decision-support and screening pipeline for pediatric malnutrition monitoring under the **Integrated Child Development Services (ICDS)** and **WHO Child Growth Standards**.

It deploys a multi-model consensus architecture spanning **four distinct machine learning and deep learning families**:
1. **XGBoost Multi-Output Baseline** (Gradient Boosted Decision Trees)
2. **FT-Transformer** (Tabular Neural Self-Attention with Continuous Feature Tokenization)
3. **Tabular Deep Neural Network (DNN)** (Dense Feed-Forward with Batch Normalization and Dropout)
4. **TabNet Multi-Task Classifier** (Sequential Attentive Tabular Transformer with Feature Selection Masks)

```text
                                Patient Intake / Survey Data
                                             ↓
                               Data Validation & Sanitization
                                             ↓
                               Preprocessing & Scaling Shield
                             (StandardScaler, Entity Encoders)
                                             ↓
        ┌───────────────────┬────────────────────┬──────────────────┬──────────────────┐
        ↓                   ↓                    ↓                  ↓
  XGBoost Baseline    FT-Transformer            DNN               TabNet
 (Gradient Trees)    (Self-Attention)      (BatchNorm+ReLU)  (Sequential Attn)
        ↓                   ↓                    ↓                  ↓
   SHAP Values      Gradient Attribution  Input Gradients    Attentive Masks
        ↓                   ↓                    ↓                  ↓
        └───────────────────┴────────────────────┴──────────────────┘
                                             ↓
                               Consensus Risk Evaluation
                         (Macro F1 / Dynamic Calibration)
                                             ↓
                           4-Model AI Dashboard & API Suite
              ┌──────────────────────────────┴──────────────────────────────┐
              ↓                                                             ↓
   Malnutrition Risk Screening                                 4-Model Benchmark Evaluation
   (Stunting, Wasting, Overall Risk)                           (Real Test Metrics Comparison)
```

---

## 2. Microservice Topology & Port Allocations

- **Frontend Application**: React 18 + Vite (Vanilla CSS, Lucide Icons, Glassmorphic UI) on `http://localhost:5173`
- **Backend API Gateway**: Node.js Express Proxy Gateway on `http://localhost:3000`
- **Machine Learning Engine**: Python Flask Service on `http://localhost:5005`

---

## 3. Dataset Audit: NFHS-5 (`dhs_clean.parquet`)

The system trains, evaluates, and infers exclusively on the official **Demographic and Health Survey (DHS Phase 7 / NFHS-5)** dataset from India:
- **Total Records**: **198,849 real survey cases**
- **Data Splits (Fixed Seed: 42)**:
  - **70% Training Set**: 139,194 records
  - **15% Validation Set**: 29,827 records (used strictly for early stopping and threshold tuning)
  - **15% Held-Out Test Set**: 29,828 records (used strictly for unbiased benchmark reporting)

### Primary Targets:
1. **Stunting**: Height-for-age z-score (HAZ) $\le -2.00$ SD (chronic linear growth failure). Prevalence: 36.15% (71,888 cases).
2. **Wasting**: Weight-for-height z-score (WHZ) $\le -2.00$ SD (acute wasting). Prevalence: 18.70% (37,193 cases).
3. **Composite Malnutrition**: Child suffering from Stunting OR Wasting ($\text{Stunting} \lor \text{Wasting}$). Prevalence: 49.92% (99,273 cases).

### Input Features (18 Predictors):
- **Numerical Features (9)**: `child_age_months`, `birth_weight`, `breastfeeding_duration`, `birth_order`, `mother_bmi`, `anc_visits`, `hhsize`, `sanitation_risk_index`, `maternal_risk_score`.
- **Categorical Features (9)**: `child_sex`, `education`, `wealth_quintile`, `residence`, `birth_size`, `diarrhea_recent`, `fever_recent`, `cough_recent`, `measles_vaccine`.

---

## 4. Model Architectures & Hyperparameters

### Model 1 — XGBoost Multi-Output Baseline
- **Architecture**: `MultiOutputClassifier(XGBClassifier)`
- **Estimators**: 150
- **Max Depth**: 6
- **Learning Rate**: 0.08
- **Subsample Ratio**: 0.85
- **Column Sample by Tree**: 0.85
- **Explainability**: SHAP (TreeExplainer) & Gini Feature Importance

### Model 2 — FT-Transformer (Feature Tokenizer Transformer)
- **Continuous Tokenizer**: Learned projection weights and biases into $d_{\text{token}} = 64$
- **Categorical Embeddings**: Entity embedding tables for all 9 categorical variables
- **Encoder Layers**: 2 Transformer encoder layers with norm-first architecture
- **Self-Attention Heads**: 4 heads, GELU activations, dropout = 0.10
- **Optimizer**: AdamW ($\text{lr} = 2 \times 10^{-3}$, $\text{weight\_decay} = 10^{-4}$)
- **Explainability**: Input Gradient Attribution

### Model 3 — Deep Neural Network (DNN)
- **Architecture**:
  - `Linear(18 -> 256) -> BatchNorm1d -> ReLU -> Dropout(0.2)`
  - `Linear(256 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.2)`
  - `Linear(128 -> 64) -> BatchNorm1d -> ReLU`
  - `Linear(64 -> 3)` (Multi-task output head)
- **Loss**: `nn.BCEWithLogitsLoss` with class positive weighting
- **Optimizer**: AdamW ($\text{lr} = 10^{-3}$) with `ReduceLROnPlateau` scheduler
- **Explainability**: Integrated Input Gradients

### Model 4 — TabNet (`pytorch-tabnet`)
- **Architecture**: `TabNetMultiTaskClassifier`
- **Decision Steps ($N_{\text{steps}}$)**: 4
- **Feature Dimension ($N_d$)**: 16
- **Attention Dimension ($N_a$)**: 16
- **Gamma ($\gamma$)**: 1.3
- **Sparse Regularization ($\lambda_{\text{sparse}}$)**: $10^{-4}$
- **Batch Size**: 1024 (Virtual Batch Size: 128)
- **Explainability**: TabNet Sequential Attention Selection Masks (`model.explain(X)`)

---

## 5. Held-Out Test Benchmark Comparison (29,828 Unseen Records)

All models were evaluated on the exact same 29,828 test cases. No mock data, no synthetic values:

| Evaluation Metric | XGBoost Baseline | FT-Transformer | DNN (Deep Neural Net) | TabNet | Benchmark Leader |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Exact Match Accuracy** | 24.38% | 20.96% | 20.59% | **26.57%** | **TabNet** |
| **Macro F1-Score** | 0.5191 | 0.5221 | **0.5235** | 0.5141 | **DNN** |
| **Weighted F1-Score** | 0.5712 | 0.5726 | **0.5736** | 0.5671 | **DNN** |
| **Macro ROC-AUC** | **0.6378** | 0.6325 | 0.6342 | 0.6283 | **XGBoost** |
| **Hamming Loss** | 0.4251 | 0.4746 | 0.4889 | **0.4242** | **TabNet** (Lowest) |
| **Inference Latency** | **0.0055 ms** | 0.1221 ms | 0.0090 ms | 0.0262 ms | **XGBoost** (Fastest) |
| **Training Time** | **14.7s** | 156.8s | 56.7s | 207.5s | **XGBoost** (Fastest) |

### Per-Condition Test Performance (F1 / ROC-AUC)

| Condition | XGBoost F1 / AUC | Transformer F1 / AUC | DNN F1 / AUC | TabNet F1 / AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Stunting** | **0.5576** / **0.6596** | 0.5555 / 0.6535 | 0.5567 / 0.6555 | 0.5534 / 0.6486 |
| **Wasting** | 0.3257 / **0.6050** | 0.3365 / 0.5995 | **0.3393** / 0.6006 | 0.3173 / 0.5961 |
| **Malnutrition** | 0.6740 / **0.6489** | 0.6744 / 0.6446 | **0.6746** / 0.6465 | 0.6716 / 0.6402 |

---

## 6. Model Explainability Mechanisms

- **XGBoost**: Tree SHAP values (`shap.TreeExplainer`) isolating exact additive contributions per survey feature.
- **FT-Transformer**: Input Gradient Attribution ($\left| \frac{\partial \text{Logit}}{\partial x_i} \right|$) through the self-attention stack.
- **DNN**: Backpropagated Input Gradient Attribution measuring activation sensitivity to each feature.
- **TabNet**: True Attention Selection Masks ($M_b$) per decision step via `model.explain(X)`.

---

## 7. API Specification

### Endpoint: `POST /predict` (Consensus across all 4 models)

**Request Body**:
```json
{
  "childName": "Priya",
  "child_age_months": 18,
  "child_sex": "Female",
  "birth_weight": 2.1,
  "birth_size": "Smaller than Average",
  "breastfeeding_duration": 8,
  "birth_order": 4,
  "mother_bmi": 17.2,
  "education": "No Education",
  "anc_visits": 1,
  "wealth_quintile": "Poorest",
  "residence": "Rural",
  "hhsize": 7,
  "sanitation_risk_index": 3,
  "diarrhea_recent": "Yes",
  "fever_recent": "Yes",
  "cough_recent": "Yes",
  "measles_vaccine": "No"
}
```

**Response**:
```json
{
  "status": "success",
  "best_model": "DNN",
  "best_model_reason": "Highest test Macro F1-score (0.5235) across all target conditions on 29,828 unseen test records.",
  "xgboost": {
    "prediction": "Malnourished (Moderate Risk)",
    "probability": 0.644,
    "overall_risk": "MODERATE",
    "top_factors": [...]
  },
  "transformer": {
    "prediction": "Malnourished (High Risk)",
    "probability": 0.666,
    "overall_risk": "HIGH",
    "top_factors": [...]
  },
  "dnn": {
    "prediction": "Malnourished (Moderate Risk)",
    "probability": 0.625,
    "overall_risk": "MODERATE",
    "top_factors": [...]
  },
  "tabnet": {
    "prediction": "Malnourished (High Risk)",
    "probability": 0.667,
    "overall_risk": "HIGH",
    "top_factors": [...]
  },
  "comparison": [...]
}
```

---

## 8. Service Execution Guide

### 1. Start Python Flask ML Service
```bash
cd ml-service
python app.py
# Listening on http://127.0.0.1:5005
```

### 2. Start Node.js Express Gateway
```bash
cd backend
node server.js
# Listening on http://localhost:3000 (proxies ML traffic to port 5005)
```

### 3. Start React Frontend
```bash
cd frontend
npm run dev
# Running on http://localhost:5173
```

---

## 9. Medical Safety & Clinical Disclaimer

> [!WARNING]
> **Clinical Research Disclaimer**: This system provides AI-based malnutrition risk screening for academic and research purposes and is **not a medical diagnosis**. Any child identified as Moderate or High risk should immediately be referred to a qualified pediatrician or public health worker (ANM/ICDS) for clinical anthropometric assessment, clinical examination, and nutritional intervention.
