# 🛡️ FraudSentinel — End-to-End Credit Card Fraud Detection Pipeline & Dashboard

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat&logo=docker)](https://www.docker.com/)

> **Production-Grade Real-Time Fraud Monitoring Platform**: Leakage-free SMOTE preprocessing → Supervised & Unsupervised Models (Random Forest, Optuna XGBoost, Logistic Regression, Isolation Forest) → Threshold Optimization (0.6554) → SHAP Explainability → Dual FastAPI & Streamlit Visual Dashboards.

---

## 📌 Executive Summary & Key Highlights

* **Leakage-Free Pipeline**: Strict train/test splitting before applying SMOTE resampling to eliminate synthetic data leakage into test sets.
* **Optimal Model Performance**: Random Forest achieves **0.9808 ROC-AUC**, **0.8680 PR-AUC**, and **0.8542 F1-Score** at tuned decision threshold `0.6554` (Test Set Confusion Matrix: **56,852 TN, 12 FP, 16 FN, 82 TP**).
* **Explainable AI (XAI)**: Integrated `SHAP (TreeExplainer)` engine providing real-time top-3 feature contribution explanations and directional impact per transaction.
* **Dual Monitoring Applications**:
  * **FastAPI Backend & Embedded Dark Client**: High-throughput REST API (`/predict`, `/metrics`, `/health`, `/random`) with built-in interactive dashboard served directly at `http://localhost:8000/`.
  * **Streamlit Pro Dashboard**: Interactive multi-tab UI for live transaction inspection, batch uploads, and performance analytics.
* **Cloud & Container Ready**: Containerized via Docker (`Dockerfile`), VS Code DevContainers, and Render cloud deploy specification (`render.yaml`).

---

## 📊 Model Benchmark & Comparison

Evaluation conducted on a held-out test dataset (**56,962 transactions**).

| Model | ROC-AUC | PR-AUC | F1 @ 0.5 Threshold | Best F1 Score | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **🌲 Random Forest (Best Overall)** | **0.9808** | **0.8680** | **0.8155** | **0.8542** | `0.6554` |
| **⚡ XGBoost (Optuna-Tuned)** | 0.9777 | 0.8766 | 0.7644 | 0.8743 | `0.9928` |
| **📈 Logistic Regression** | 0.9704 | 0.7212 | 0.1119 | 0.7558 | `0.9999` |
| **🔍 Isolation Forest (Anomaly)** | 0.9552 | 0.1725 | 0.1304 | 0.2944 | `0.7666` |

### 🎯 Confusion Matrix (Random Forest @ Threshold = 0.6554)

```
                       Predicted Legitimate (0)   Predicted Fraud (1)
 Actual Legitimate (0)        56,852 (TN)               12 (FP)
 Actual Fraud (1)                16 (FN)                82 (TP)
```

---

## 🗂️ Project Directory Structure

```
.
├── app/
│   ├── api.py               # FastAPI REST backend & embedded HTML/Plotly dashboard UI
│   └── streamlit_app.py     # Interactive Streamlit dashboard (Transaction Check & Model Metrics)
├── data/
│   └── creditcard.csv       # Kaggle Credit Card Fraud dataset (284,807 rows × 31 columns)
├── models/
│   ├── config.json          # Best hyperparameters, thresholds, and performance metrics
│   ├── model.pkl            # Production Random Forest pipeline artifact
│   ├── xgb_model.pkl        # Optuna-tuned XGBoost model artifact
│   ├── lr_model.pkl         # Logistic Regression model artifact
│   ├── if_model.pkl         # Isolation Forest model artifact
│   ├── scaler.pkl           # Fitted RobustScaler artifact
│   └── test_sample.csv      # Extracted sample dataset for UI testing & random sampling
├── reports/                 # Saved evaluation plots (ROC, PR curves, Confusion Matrices)
├── src/
│   ├── eda.py               # Exploratory Data Analysis script
│   ├── preprocess.py        # Scaler fitting, train/test split, and SMOTE resampling
│   ├── train.py             # Multi-model training and Optuna tuning engine
│   ├── evaluate.py          # Metric computation and evaluation visualizer
│   ├── predict.py           # Core inference engine with SHAP explainability
│   └── smoke_test.py        # System smoke test script
├── .devcontainer/           # VS Code Container environment
├── Dockerfile               # Production container definition
├── render.yaml              # Render Cloud deployment config
├── requirements.txt         # Project dependency manifest
└── README.md                # System documentation
```

---

## ⚙️ Architecture & Pipeline Workflow

```
[ Raw Data: creditcard.csv ]
         │
         ▼
 [ Split Train (80%) / Test (20%) ]  ◄── Prevents Data Leakage!
         │
         ├───► [ Fit RobustScaler on Train ]
         │
         ├───► [ Apply SMOTE Resampling (train ratio: 0.15) ]
         │
         ▼
 [ Model Training & Optimization ] ──► (Random Forest, Optuna XGBoost, LR, Isolation Forest)
         │
         ▼
 [ Threshold Tuning ] ─────────────► Maximize F1 Score on Validation (Best = 0.6554)
         │
         ▼
 [ Inference Engine (predict.py) ] ──► Computes Probabilities & SHAP Feature Explanations
         │
         ├───► [ FastAPI REST Endpoint & UI (`/predict`, `/docs`, `/`) ]
         └───► [ Streamlit Interactive Application (`app/streamlit_app.py`) ]
```

---

## ⚡ Quick Start & Usage

### 1. Prerequisites & Environment Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/Credit-Card-Fraud.git
cd Credit-Card-Fraud

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Dataset Setup & Training

Download `creditcard.csv` from Kaggle and place it in the `data/` directory.

```bash
# Optional: Run Exploratory Data Analysis
python src/eda.py

# Train models & generate artifacts (with Optuna hyperparameter optimization)
python src/train.py

# Optional: Run quick training without Optuna tuning
python src/train.py --no-tune
```

### 3. Launch Applications

#### Option A: Launch FastAPI Backend & Dashboard UI
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```
* **Embedded Interactive UI**: [http://localhost:8000/](http://localhost:8000/)
* **Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Option B: Launch Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
Access the dashboard in your web browser at `http://localhost:8501`.

---

## 🔌 API Endpoints Summary

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` or `/dashboard` | `GET` | HTML/Plotly Interactive Fraud Sentinel UI |
| `/health` | `GET` | System status, model loading status, active threshold |
| `/metrics` | `GET` | Retrieves full JSON benchmarking & config metrics |
| `/predict` | `POST` | Scores transaction JSON & returns fraud verdict + top-3 SHAP reasons |
| `/random` | `GET` | Fetches a sample transaction (supports `?target=fraud` or `?target=legit`) |

### Example Request (`POST /predict`):
```json
{
  "Time": 80000.0,
  "Amount": 150.0,
  "V1": -1.3598,
  "V2": -0.0728,
  "V14": -0.3111
}
```

### Example Response:
```json
{
  "probability": 0.892415,
  "is_fraud": true,
  "verdict": "FRAUD",
  "confidence": "HIGH",
  "threshold": 0.6554,
  "shap_top3": [
    {
      "feature": "V14",
      "raw_value": -4.2105,
      "shap_value": 0.4812,
      "direction": "↑ fraud"
    },
    {
      "feature": "V17",
      "raw_value": -2.8912,
      "shap_value": 0.3510,
      "direction": "↑ fraud"
    },
    {
      "feature": "V12",
      "raw_value": -3.1042,
      "shap_value": 0.2981,
      "direction": "↑ fraud"
    }
  ]
}
```

---

## 🐳 Docker & Cloud Deployment

### Run with Docker

```bash
# Build Docker image
docker build -t fraud-sentinel .

# Run Docker container
docker run -p 8000:8000 fraud-sentinel
```

### Deploy on Render

The repository includes a ready-to-use `render.yaml` for automatic deployment on Render:
1. Connect your GitHub repository to Render.
2. Render will automatically detect `render.yaml` and configure the Web Service with Python 3.11 environment.

---

## 📝 License

Distributed under the MIT License.

