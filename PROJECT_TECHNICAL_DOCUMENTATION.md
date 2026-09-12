# Dual-Model AI Prediction System: XGBoost & FT-Transformer for Pediatric Nutrition & Disease Risk Screening

## 1. System Architecture Overview

This project implements an end-to-end clinical decision-support and screening pipeline for pediatric health monitoring under the Integrated Child Development Services (ICDS) framework. It combines classical tree-based machine learning (**XGBoost**) and state-of-the-art tabular deep learning (**FT-Transformer**) to screen children across 10 health conditions divided into two core sections:
1. **Growth & Nutrition Assessment** (5 indicators)
2. **Pediatric Disease & Health-Condition Risk Screening** (5 targets)

```text
                             User Input (Child Profile)
                                         ↓
                               Data Validation & Sanitization
                                         ↓
                                   Preprocessing
                    (Scaling, Categorical Encoding, Leakage Shield)
                                         ↓
                       ┌─────────────────┴─────────────────┐
                       ↓                                   ↓
               XGBoost Baseline                      FT-Transformer
           (Multi-Output Gradient Trees)     (Tabular Neural Self-Attention)
                       ↓                                   ↓
                   SHAP Values                    Input Gradient Attributions
                       ↓                                   ↓
                       └─────────────────┬─────────────────┘
                                         ↓
                             Consensus Risk Evaluation
                               (Macro F1 Selection)
                                         ↓
                             Dual-Section UI Dashboard
                ┌────────────────────────┴────────────────────────┐
                ↓                                                 ↓
   Section 1: Growth & Nutrition                  Section 2: Pediatric Disease Risk
 (Malnutrition, Stunting, Wasting,              (Anemia, Iron Deficiency Anemia,
    Underweight, Overweight)                     Vitamin A, PEM, Micronutrients)
```

---

## 2. Microservice Topology & Port Allocations

- **Frontend**: React 18 + Vite (Tailwind/Vanilla CSS, Lucide Icons, Glassmorphic UI) on `http://localhost:5173`
- **Backend API Gateway**: Node.js Express Proxy on `http://localhost:3000`
- **Machine Learning Service**: Python Flask Service on `http://localhost:5005`

---

## 3. Evaluated Targets & Clinical Definitions

The system screens for 10 pediatric targets across two clinical categories:

### Section A: Growth & Nutrition Assessment (5 Indicators)
1. **Malnutrition (Composite Undernutrition)**: Child exhibiting any severe growth deficit (Stunting OR Wasting OR Underweight).
2. **Stunting (Height-for-Age Deficit)**: Height-for-age z-score (HAZ) $\le -2.0$ SD according to WHO Child Growth Standards. Indicates chronic malnutrition and impaired linear growth.
3. **Wasting (Weight-for-Height Deficit)**: Weight-for-height z-score (WHZ) $\le -2.0$ SD. Indicates acute nutritional deficit requiring urgent feeding intervention.
4. **Underweight (Weight-for-Age Deficit)**: Weight-for-age z-score (WAZ) $\le -2.0$ SD. Composite measure reflecting both acute and chronic undernutrition.
5. **Overweight/Obesity**: Weight-for-height z-score (WHZ) $\ge +2.0$ SD or BMI-for-age $\ge +2.0$ SD. Identifies children at risk of pediatric adiposity and metabolic conditions.

### Section B: Pediatric Disease & Deficiency Risk (5 Targets)
1. **Anemia**: Pediatric hemoglobin level $< 11.0$ g/dL (NFHS-5 clinical biomarker `hw57`: severe, moderate, or mild). Strongly associated with MUAC $<12.5$ cm and dual growth faltering.
2. **Iron Deficiency / Iron Deficiency Anemia (IDA)**: Microcytic hypochromic nutritional deficiency characterized by anemia co-occurring with low dietary diversity ($\le 3$ food groups).
3. **Vitamin A Deficiency Risk**: Validated nutritional risk based on lack of vitamin A-rich foods (low dietary diversity $< 3$) combined with poor water/sanitation access.
4. **Protein-Energy Malnutrition (PEM)**: Clinical Severe Acute Malnutrition (SAM) defined by WHZ $<-3.0$ SD or MUAC $<11.5$ cm according to WHO/ICDS guidelines.
5. **Micronutrient Deficiency Risk**: Multi-deficiency index combining dual-undernutrition with dietary diversity $< 4$ and poor sanitation.

---

## 4. Strict Data Leakage Prevention

In accordance with strict clinical AI guidelines:
- **No Target Z-Scores in Predictors**: Direct z-scores (`haz`, `whz`, `waz`) are **strictly excluded** from the input feature vector `ALL_FEATURES`.
- Only primary physiological observations (`age_months`, `height_cm`, `weight_kg`, `bmi`, `muac_cm`, `dietary_diversity`, `meal_frequency`, `water_sanitation_index`, `gender`, `breastfeeding_status`) are provided to the models.
- Preprocessing transformations (StandardScaler, LabelEncoders) are fit **exclusively on the 70% Training set** and applied without re-fitting to the validation and test splits.

---

## 5. Model Architectures & Hyperparameters

### Model 1 — XGBoost Multi-Output Baseline
- **Architecture**: `MultiOutputClassifier(XGBClassifier)`
- **Trees per Head**: 100
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **Subsample Ratio**: 0.8
- **Column Sample By Tree**: 0.8
- **Evaluation Metric**: `logloss`
- **Parallel Workers**: All available CPU cores (`n_jobs=-1`)

### Model 2 — FT-Transformer (Feature Tokenizer Transformer)
- **Architecture**: Multi-Task Tabular Transformer with learned CLS token and parallel classification head
- **Continuous Feature Tokenizer**: Learned linear projection with per-feature weight and bias vectors
- **Categorical Embeddings**: Entity embedding layers for `gender` and `breastfeeding_status`
- **Token Embedding Dimension ($d_{\text{token}}$)**: 64
- **Self-Attention Heads ($n_{\text{heads}}$)**: 4
- **Feed-Forward Dimension ($d_{\text{ffn}}$)**: 128
- **Transformer Encoder Layers ($n_{\text{layers}}$)**: 3
- **Activation Function**: GELU
- **Dropout Rate**: 0.10
- **Loss Function**: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`)
- **Optimizer**: AdamW ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$)
- **Learning Rate Scheduler**: `ReduceLROnPlateau(patience=2, factor=0.5)`

---

## 6. Reproducible Test Benchmark Comparison (15,000 Unseen Samples)

Both models were trained on 69,999 records and benchmarked on 15,000 held-out test records:

| Evaluation Metric | XGBoost Baseline | FT-Transformer | Difference ($\Delta$) | Leading Architecture |
| :--- | :---: | :---: | :---: | :--- |
| **Exact Match Accuracy** | **98.50%** | 94.05% | $-4.45\%$ | **XGBoost** |
| **Macro F1-Score** | **0.9812** | 0.8396 | $-0.1416$ | **XGBoost** |
| **Weighted F1-Score** | **0.9933** | 0.9730 | $-0.0203$ | **XGBoost** |
| **Macro ROC-AUC** | **0.9999** | 0.9990 | $-0.0009$ | **XGBoost** |
| **Hamming Loss** | **0.0022** | 0.0089 | $+0.0067$ | **XGBoost** |
| **Training Time** | **21.39s** | 537.92s | $+516.53\text{s}$ | **XGBoost** (25x faster) |
| **Inference Latency** | **0.0153 ms/case** | 0.1203 ms/case | $+0.1050\text{ ms}$ | **XGBoost** (7.8x faster) |

### Per-Condition Benchmark Breakdown

| Condition Target | XGBoost Acc | XGBoost F1 | XGBoost ROC-AUC | Transformer Acc | Transformer F1 | Transformer ROC-AUC | Selected Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Malnutrition** | 99.51% | 0.9939 | 0.9998 | 97.39% | 0.9672 | 0.9979 | 0.50 / 0.40 |
| **Stunting** | 99.60% | 0.9909 | 0.9999 | 98.10% | 0.9573 | 0.9983 | 0.45 / 0.35 |
| **Wasting** | 99.90% | 0.9959 | 1.0000 | 99.09% | 0.9636 | 0.9995 | 0.50 / 0.40 |
| **Underweight** | 99.81% | 0.9958 | 1.0000 | 99.07% | 0.9795 | 0.9996 | 0.60 / 0.60 |
| **Overweight/Obesity** | 99.87% | 0.9988 | 1.0000 | 99.23% | 0.9931 | 0.9998 | 0.50 / 0.60 |
| **Anemia** | 99.69% | 0.9767 | 0.9998 | 99.40% | 0.9558 | 0.9995 | 0.45 / 0.40 |
| **Iron Deficiency (IDA)** | 99.85% | 0.9370 | 0.9998 | 99.91% | 0.9613 | 0.9999 | 0.40 / 0.45 |
| **Vitamin A Deficiency** | 99.97% | 0.9541 | 0.9999 | 99.72% | 0.4878 | 0.9973 | 0.30 / 0.50 |
| **PEM** | 100.00% | 1.0000 | 1.0000 | 99.88% | 0.1818 | 0.9988 | 0.20 / 0.20 |
| **Micronutrient Deficiency**| 99.57% | 0.9687 | 0.9997 | 99.31% | 0.9490 | 0.9995 | 0.45 / 0.45 |

### Analysis & Best-Model Selection
- **Primary Metric**: **Macro F1-Score** is used as the primary selection criterion because conditions like PEM (0.11% prevalence) and Vitamin A Deficiency (0.36% prevalence) represent severe clinical conditions with high class imbalance.
- **Winner**: **XGBoost Baseline** achieves superior performance (Macro F1 = 0.9812 vs 0.8396) due to better sample efficiency on extreme minority classes, while FT-Transformer exhibits excellent performance on high-prevalence continuous targets.

---

## 7. Model Explainability

### XGBoost Explainability (SHAP & Gini Importance)
- Computes mean tree feature importance across all estimators for each prediction.
- Identifies the top 5 contributing features (e.g., Weight, Height, MUAC, Dietary Diversity).
- Labels impact levels: *High Impact* ($\ge 15\%$), *Moderate Impact* ($\ge 8\%$), or *Minor Influence*.

### FT-Transformer Explainability (Input Gradient Attributions)
- Computes input gradient attributions:
  $$\text{Attribution}_i = \left| \frac{\partial \sum \text{Logits}}{\partial x_{\text{num}, i}} \right|$$
- Identifies which continuous features actively caused the largest activation swing in the self-attention stack.

---

## 8. API Specification

### Endpoint: `POST /predict` (Dual-Model Consensus)
**Request Body**:
```json
{
  "age_months": 24,
  "height_cm": 85.0,
  "weight_kg": 11.5,
  "gender": "Female",
  "muac_cm": 14.0,
  "dietary_diversity": 4,
  "meal_frequency": 3,
  "breastfeeding_status": "Partial",
  "water_sanitation_index": 4
}
```

**Response**:
```json
{
  "status": "success",
  "best_model": "XGBoost",
  "best_model_reason": "Selected based on benchmark test Macro F1-score (XGBoost: 0.9812, FT-Transformer: 0.8396)",
  "xgboost": {
    "model": "XGBoost",
    "overall_risk": "LOW",
    "top_prediction": "Normal",
    "nutrition_assessment": [...],
    "disease_screening": [...],
    "top_factors": [...]
  },
  "transformer": {
    "model": "FT-Transformer",
    "overall_risk": "LOW",
    "nutrition_assessment": [...],
    "disease_screening": [...]
  },
  "comparison": [
    {
      "condition": "Stunting",
      "category": "Growth & Nutrition",
      "xgboost_percentage": 1.2,
      "transformer_percentage": 2.4,
      "delta": 1.2
    }
  ],
  "medical_disclaimer": "AI-based risk screening only — this result is not a medical diagnosis."
}
```

---

## 9. Medical Safety & Disclaimer

> [!WARNING]
> **Clinical Disclaimer**: This AI system is designed for public health risk screening, academic research, and field-worker prioritization under ICDS. It **does NOT constitute a medical diagnosis** or prescribe treatment. Any child identified as High Risk should immediately be referred to a qualified healthcare professional (ANM, Medical Officer, or Pediatrician) for clinical examination, lab testing, and therapeutic management.

---

## 10. Service Execution Guide

### 1. Start Python Flask ML Service
```bash
cd ml-service
python app.py
# Runs on http://127.0.0.1:5005
```

### 2. Start Node.js Express Gateway
```bash
cd backend
node server.js
# Runs on http://localhost:3000 (proxies to port 5005)
```

### 3. Start React Frontend
```bash
cd frontend
npm run dev
# Runs on http://localhost:5173
```
