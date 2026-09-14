# Telco Customer Churn — End-to-End ML System

![tests](https://github.com/cheonu/ml-engineering-lab/actions/workflows/tests.yml/badge.svg)

An end-to-end machine learning project that predicts customer churn for a telecom
company: from raw data cleaning through model training, experiment tracking, a
versioned model registry, a live prediction API, drift monitoring, and an
automated retraining loop.

The focus is on ML **engineering** — reproducibility, testing, and serving — not
just model accuracy.

---

## The Problem

Churn is when a customer stops using a service. It matters because acquiring a new
customer costs far more than retaining an existing one, so a business wants to know
*which customers are about to leave* in time to intervene with a retention offer.

This project predicts, for each customer, the probability that they will churn,
using account and service attributes (contract type, tenure, monthly charges, which
services they subscribe to, and so on).

---

## The Data

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— 7,043 customers, 19 features, binary target (`Churn`: Yes/No).

Key findings from exploratory analysis (`notebooks/01_eda.ipynb`):

- **Class imbalance.** Only 26.5% of customers churned. This means accuracy is a
  misleading metric — a model that always predicts "won't churn" scores 73.5% while
  being useless.
- **Contract type is the strongest signal.** Month-to-month customers churn at
  **42.7%**, versus **2.8%** for two-year contracts — a ~15x difference.
- **A data-quality trap.** `TotalCharges` is stored as text because 11 brand-new
  customers (tenure = 0) have a blank value instead of a number. These are filled
  with 0 during cleaning.

---

## Approach

Preprocessing is packaged **inside the model** as a scikit-learn `Pipeline`
(`ColumnTransformer` + `OneHotEncoder` + `RandomForestClassifier`). This is a
deliberate design choice: because the encoder is fitted during training and saved
with the model, the exact same transformation is applied at serving time. This
eliminates **training/serving skew** — a common production bug where the encoding at
inference silently drifts from the encoding at training.

```
raw CSV
  │
  ▼
clean (drop ID, fix TotalCharges, map target)      src/data.py
  │
  ▼
Pipeline ─ ColumnTransformer (encode categoricals) ─ RandomForest   src/train.py
  │
  ▼
MLflow  ─ track params/metrics + register model      src/experiment.py
  │
  ▼
FastAPI ─ load champion from registry, serve          src/serve.py
```

The full lifecycle then closes into a loop:

```
monitor drift ──► (if drift) retrain ──► compare new vs champion f1
                                              │
                        promote to @champion  │  keep current champion
                        only if better  ◄─────┘
                                │
                                ▼
                serving loads models:/churn-model@champion
```

---

## Results

Experiments were tracked in MLflow. The comparison below is the core insight of the
project:

| Model                     | Accuracy | Churn recall | Precision |
| ------------------------- | :------: | :----------: | :-------: |
| Random Forest, depth 3    |  0.779   |    0.334     |   0.665   |
| Random Forest, depth 8    | **0.800**|    0.503     |   0.662   |
| Random Forest, depth 12   |  0.789   |    0.497     |   0.631   |
| Depth 12 + class weighting |  0.771  |  **0.709**   |   0.554   |

**The best model has the lowest accuracy.** The depth-8 model has the highest
accuracy (0.800) but catches only ~50% of customers who actually churn. The
class-weighted model has the *lowest* accuracy (0.771) yet catches **71%** of
churners.

For this problem, **recall on the churn class is the metric that matters**: missing
a customer who leaves is far more costly than sending a retention offer to someone
who would have stayed. Optimising for accuracy would have selected the wrong model.
Class weighting (`class_weight="balanced"`) tells the model to treat the rare churn
class as more important, trading precision for the recall the business needs.

---

## Engineering

- **Config-driven training.** All hyperparameters, data schema, and experiment
  settings live in `configs/churn.yaml`. Running a new experiment means editing a
  config, not the code.
- **Experiment tracking & model registry** via MLflow (local SQLite backend, no
  cloud cost). The model is registered as `churn-model` and versioned automatically
  on each run.
- **Data versioning** via DVC — the raw dataset is tracked by DVC (not committed to
  git) so runs are reproducible.
- **Tests** (`tests/test_data.py`) verify the data pipeline: ID dropping, numeric
  conversion, missing-value handling, and stratification of the train/test split.
- **Serving** via FastAPI with a `/predict` endpoint (returns label + churn
  probability) and a `/health` endpoint (reports model load status).
- **Drift monitoring** (`src/monitor.py`) with Evidently. It compares a reference
  dataset against current data and flags feature drift — the signal that live data
  has diverged from what the model was trained on, and that retraining may be needed.
- **Automated retraining loop** (`src/retrain.py`) — see below.
- **Containerized** with Docker; the image bundles the code, dependencies, and the
  model registry, and serves the API on port 8000.
- **CI** via GitHub Actions runs the test suite on every push and pull request.

---

## Retraining Strategy

Models decay as the world changes. `src/retrain.py` implements a closed loop:

1. **Detect drift** — compare recent data against a reference set (`check_drift`).
2. **Retrain only if drifted** — no drift means no retrain; the loop exits.
3. **Gated promotion** — after retraining, compare the new model's churn f1 against
   the current production model (the `@champion` alias in the registry). The new
   model is promoted **only if it is better**; otherwise the existing champion is
   kept.
4. **Serving follows the champion** — `serve.py` loads `models:/churn-model@champion`,
   so only a vetted model ever reaches production.

The key design decision: **drift is a signal to investigate and retrain, not an
automatic reason to deploy.** A new model is never shipped just because it was
retrained — it must beat the incumbent on the business metric first. This guards
against deploying a worse model when drift is caused by, say, a broken data feed
rather than a genuine shift.

---

## How to Run

Install dependencies (uses [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
```

Get the data (download the CSV from the Kaggle link above into
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`, or `dvc pull` if a remote is
configured).

**Train** (reads the config, trains the pipeline, logs and registers the model):

```bash
uv run python src/train.py --config configs/churn.yaml
```

**View experiments** in the MLflow UI (http://127.0.0.1:5000):

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Serve** the model (http://127.0.0.1:8000, interactive docs at `/docs`):

```bash
uv run uvicorn serve:app --app-dir src --reload
```

**Predict** with a request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 70.35
  }'
# -> {"churn": 1, "churn_probability": 0.8967}
```

**Check for data drift** (prints the share of drifted columns for a no-drift
baseline and a simulated-drift example):

```bash
uv run python src/monitor.py
```

**Run the retraining loop** (checks drift, retrains if drifted, promotes only if the
new model beats the current champion):

```bash
uv run python src/retrain.py
```

**Run the tests:**

```bash
uv run pytest
```

**Run in Docker** (builds an image with the app and model, serves on port 8000):

```bash
docker build -t churn-api:v1 .
docker run -p 8000:8000 churn-api:v1
```

---

## Project Structure

```
├── configs/
│   └── churn.yaml          # experiment + data + model configuration
├── data/raw/               # raw dataset (DVC-tracked, not in git)
├── notebooks/
│   └── 01_eda.ipynb        # exploratory data analysis
├── src/
│   ├── data.py             # load + clean + split
│   ├── train.py            # config-driven entry point, builds the pipeline
│   ├── experiment.py       # shared MLflow training/logging logic
│   ├── evaluate.py         # metric computation
│   ├── serve.py            # FastAPI inference service
│   ├── monitor.py          # Evidently drift detection
│   └── retrain.py          # drift-triggered retraining + gated promotion
├── tests/
│   └── test_data.py        # data pipeline tests
├── .github/workflows/
│   └── tests.yml           # CI: run pytest on push/PR
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Notes & Next Steps

- The tracking store is local SQLite; a team setup would move this to a shared
  Postgres + artifact store.
- The drift check in `monitor.py` uses a simulated shift (all-month-to-month
  contracts, a $30 charge increase) to demonstrate detection. A production version
  would compare against real incoming batches and run on a schedule. Note the
  simulation shifts `MonthlyCharges` but not `TotalCharges`; in reality a price
  change would move both.
- The retraining loop uses churn f1 as the promotion criterion. A production system
  would ideally gate on a measured drop in performance against ground-truth outcomes
  (which arrive later), not on input drift alone.
- Further extensions: schedule the retraining loop (cron / GitHub Actions
  `on: schedule`), and move the tracking store to a shared Postgres + artifact store
  for team use.
