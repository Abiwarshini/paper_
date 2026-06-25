# Child Malnutrition Risk Prediction & Growth Monitoring System

An end-to-end AI/ML pipeline that goes beyond simple malnutrition classification to provide **early risk prediction, explainability, growth forecasting, and actionable nutrition decision support**.


## 1. Motivation & Research Gap

Most existing literature on malnutrition prediction stops at:

```
INPUT  →  PREDICTION  →  RESULT
```

**Reference Paper 1 (PLOS ONE — Bangladesh DHS study):**
- Dataset: Bangladesh Demographic and Health Survey (DHS)
- Predicts: Stunting, Wasting, Underweight
- Models: Logistic Regression, ANN, Random Forest, XGBoost
- Techniques: Boruta feature selection, SHAP explainability
- Best result: XGBoost, ~76–81% accuracy
- **Gaps:** No deployed system, no recommendations, no growth tracking, no decision support

**Reference Paper 2 (ScienceDirect / recent ML comparison papers):**
- Compares ML models (XGBoost, RF, CatBoost, etc.) mainly for accuracy gains
- **Gaps:** Still prediction-only; no decision-support layer, no deployment

**The gap this project addresses:** none of the existing work answers *"what should be done next?"* — what likely deficiency is driving the risk, what the child's growth trajectory looks like, what nutrition action to recommend, or how to prioritize cases for limited field-worker attention.


## 2. System Overview

This project extends prediction into a **full decision-support pipeline**:

| Stage | Description |
|---|---|
| 1. Data Collection | Age, weight, height, MUAC, area, (optional) nutrition history |
| 2. Malnutrition Prediction | ML classification → Low / Medium / High risk |
| 3. Growth Forecasting | LSTM-based prediction of future weight/height/growth trend (if longitudinal data available) |
| 4. Explainable AI | SHAP-based explanation of *why* a case is flagged high-risk |
| 5. Risk Interpretation | Probabilistic grouping into protein-energy / iron-deficiency-risk / vitamin-deficiency-risk / weak-immunity-risk indicators (not diagnoses) |
| 6. Nutrition Recommendation | Rule-based + ML-informed dietary suggestions |
| 7. Priority Ranking | Sorts cases High → Medium → Low for limited field-worker bandwidth |
| 8. Dashboard | Case list, risk levels, growth charts, alerts, priority queue |

### Comparison with existing work

| Component | Existing Papers | This Project |
|---|---|---|
| Prediction | ✅ | ✅ |
| Explainability | Partial | ✅ |
| Growth Forecasting | ❌ | ✅ (LSTM) |
| Nutrition Recommendation | ❌ | ✅ |
| Priority Ranking | ❌ | ✅ |
| Dashboard | ❌ | ✅ |
| Deficiency Risk Mapping | ❌ | Partial (risk-based only) |

---

## 3. Tech Stack

> Defaults below — adjust in `requirements.txt` / `package.json` to match your actual implementation.

**ML / DL**
- Python 3.10+
- scikit-learn, XGBoost
- TensorFlow / Keras or PyTorch (for LSTM, optional TabNet/DNN)
- SHAP (explainability)

**Backend / API**
- FastAPI (or Flask) for model-serving endpoints

**Dashboard**
- Streamlit (fastest for an academic prototype) **or** React + Recharts (if a full frontend is preferred)

**Data**
- pandas, numpy
- CSV / SQLite for local data; PostgreSQL optional for multi-user deployment

---

## 4. Project Structure

```
malnutrition-risk-prediction/
├── data/
│   ├── raw/                # original/source datasets
│   ├── processed/          # cleaned, feature-engineered data
│   └── sample/             # small sample for quick testing
├── notebooks/               # EDA, experimentation
├── src/
│   ├── data_preprocessing.py
│   ├── train_classifier.py # XGBoost / RF / ANN training
│   ├── train_lstm.py       # growth forecasting model
│   ├── explainability.py   # SHAP analysis
│   ├── recommendation.py   # nutrition recommendation logic
│   ├── priority_ranking.py
│   └── utils.py
├── models/                  # saved model artifacts (.pkl / .h5)
├── dashboard/                # Streamlit app or React frontend
│   └── app.py
├── api/
│   └── main.py              # FastAPI app
├── tests/
├── .env.example
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 5. Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip / conda
- Git
- (Optional) Node.js 18+ if using a React dashboard

### Clone the repository
```bash
git clone https://github.com/<your-username>/malnutrition-risk-prediction.git
cd malnutrition-risk-prediction
```

### Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt` (suggested baseline):
```
pandas
numpy
scikit-learn
xgboost
tensorflow        # or torch, if using PyTorch
shap
matplotlib
seaborn
streamlit          # if using Streamlit dashboard
fastapi
uvicorn
python-dotenv
```

---

## 6. Configuration

Copy the example environment file and fill in your own values:

```bash
cp .env.example .env
```

`.env.example`:
```env
# --- Paths ---
DATA_DIR=./data/processed
MODEL_DIR=./models

# --- Model selection ---
CLASSIFIER_MODEL=xgboost          # options: xgboost | random_forest | ann
GROWTH_MODEL=lstm

# --- Risk thresholds (tune based on validation results) ---
RISK_THRESHOLD_LOW=0.33
RISK_THRESHOLD_HIGH=0.66

# --- API ---
API_HOST=0.0.0.0
API_PORT=8000

# --- Dashboard ---
DASHBOARD_PORT=8501

# --- Database (optional, for multi-user deployment) ---
DATABASE_URL=sqlite:///./data/app.db
```

> Never commit your real `.env` file — keep it in `.gitignore`.

---

## 7. Usage

### Train the classification model
```bash
python src/train_classifier.py --config configs/classifier_config.yaml
```

### Train the growth-forecasting model (if time-series data available)
```bash
python src/train_lstm.py --data data/processed/growth_timeseries.csv
```

### Generate SHAP explanations
```bash
python src/explainability.py --model models/xgboost_model.pkl
```

### Run the API server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Run the dashboard (Streamlit example)
```bash
streamlit run dashboard/app.py
```

---

## 8. Dataset

This project is designed to work with anthropometric survey data containing, at minimum:

- Age (months)
- Weight (kg)
- Height/Length (cm)
- MUAC (cm)
- Area/region (categorical)
- (Optional) longitudinal records for growth forecasting

Place raw files in `data/raw/` and run `src/data_preprocessing.py` to generate cleaned features in `data/processed/`.

---

## 9. Model Outputs

| Output | Type | Notes |
|---|---|---|
| Malnutrition risk | Classification (Low/Medium/High) | Primary output |
| Growth trend forecast | Regression (future weight/height) | Requires longitudinal data |
| SHAP feature importance | Explanation | Per-case and global |
| Deficiency-risk indicators | Probabilistic grouping | **Risk signal, not diagnosis** |
| Nutrition recommendation | Text/structured suggestion | Diet-category based |
| Priority rank | Ordinal | For triaging limited field-worker time |

---

## 10. Future Work

- Expand longitudinal data collection for more robust growth forecasting
- Validate deficiency-risk groupings against clinical ground truth where available
- Add multi-language support for the dashboard
- Explore federated/edge deployment for low-connectivity field use

---

## 11. References

- PLOS ONE — Bangladesh DHS-based malnutrition prediction study (Boruta + SHAP + ML ensemble)
- ScienceDirect — comparative ML benchmarking studies for malnutrition prediction (XGBoost, RF, CatBoost)
- *(Add your full base-paper citation and any additional literature here in your preferred citation style.)*

---

## 12. License

This project is released under the [MIT License](LICENSE) — update as appropriate for your institution's IP policy.

## 13. Contributing

This is currently an academic research project. Issues and suggestions are welcome via GitHub Issues; pull requests should target the `dev` branch.
