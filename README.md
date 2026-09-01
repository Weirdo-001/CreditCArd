# Credit Card Fraud Detection

> **End-to-end ML pipeline**: EDA → Preprocessing → SMOTE → Model Training (4 models) → Optuna tuning → Threshold optimization → SHAP explainability → Streamlit dashboard

---

## 🗂️ Project Structure

```
/data           creditcard.csv (284,807 rows × 31 cols)
/src
  eda.py        EDA plots (class dist, amount dist, correlation heatmap)
  preprocess.py Split → Scale (RobustScaler) → SMOTE (train only)
  train.py      Trains LR, RF, XGBoost (Optuna), Isolation Forest
  evaluate.py   ROC, PR curve, confusion matrix, SHAP plots
  predict.py    Inference engine with SHAP explanations
/app
  streamlit_app.py   Two-tab dashboard (Transaction Check + Model Dashboard)
  api.py             FastAPI /predict endpoint
/models         model.pkl, scaler.pkl, config.json (auto-generated)
/reports        All evaluation plots (auto-generated)
requirements.txt
README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Move dataset
```bash
# Place creditcard.csv in /data/
move creditcard.csv data/
```

### 3. Run EDA (optional)
```bash
python src/eda.py
```

### 4. Train all models
```bash
# Full run with Optuna tuning (recommended, ~15-30 min)
python src/train.py

# Quick run without Optuna (uses default XGBoost params, ~3-5 min)
python src/train.py --no-tune
```

### 5. Launch Streamlit app
```bash
streamlit run app/streamlit_app.py
```

### 6. (Optional) Launch FastAPI backend
```bash
cd app && uvicorn api:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

---

## 🧠 Pipeline Design

### Why split BEFORE SMOTE?

```
❌ WRONG (data leakage):
   full_data → SMOTE → split → train/test
   synthetic fraud patterns bleed into test → inflated metrics

✅ CORRECT (this project):
   full_data → split → SMOTE(train only) → train model → evaluate on raw test
```

### Models Trained

| Model | Type | Key Config |
|---|---|---|
| Logistic Regression | Supervised, baseline | `class_weight=balanced`, saga solver |
| Random Forest | Supervised, ensemble | 200 trees, balanced weights |
| XGBoost | Supervised, boosting | Optuna-tuned, `scale_pos_weight` |
| Isolation Forest | **Unsupervised** anomaly | `contamination=fraud_rate` |

### Metrics Prioritized

- ✅ **PR-AUC** and **F1** (correct for imbalanced data)
- ❌ Accuracy (meaningless — 99.83% by predicting all zeros)

### Threshold Tuning

Default 0.5 threshold underperforms on imbalanced data.  
We compute F1 at every threshold and pick the maximum:

```python
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1])
best_threshold = thresholds[np.argmax(f1s)]
```

This alone moves F1 from ~0.65 → **~0.83+**.

---

## 🎯 Expected Metrics (XGBoost)

| Metric | Expected Range |
|---|---|
| ROC-AUC | 0.97 – 0.98 |
| PR-AUC | 0.85 – 0.90 |
| F1 @ 0.5 threshold | 0.65 – 0.75 |
| F1 @ optimal threshold | **0.80 – 0.87** |

---

## 🚀 Deploy to Streamlit Community Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Set main file: `app/streamlit_app.py`
4. Add `creditcard.csv` via Git LFS or a download script
5. Deploy — free tier, no credit card needed

---

## 🔍 SHAP Explainability

The `/predict` endpoint and Streamlit app both return **top-3 SHAP feature reasons** per transaction:

```json
{
  "probability": 0.923,
  "verdict": "FRAUD",
  "shap_top3": [
    {"feature": "V14", "shap_value": -0.812, "direction": "↑ fraud"},
    {"feature": "V4",  "shap_value":  0.531, "direction": "↓ fraud"},
    {"feature": "V12", "shap_value": -0.487, "direction": "↑ fraud"}
  ]
}
```

---

## 📦 Saved Artifacts

| File | Description |
|---|---|
| `models/model.pkl` | Best XGBoost model |
| `models/scaler.pkl` | RobustScaler fitted on train only |
| `models/config.json` | Threshold, params, metrics |
| `models/y_test.npy` | Test labels for dashboard |
| `models/y_prob_*.npy` | Test probabilities per model |
