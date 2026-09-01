"""
smoke_test.py
─────────────
End-to-end inference verification using real fraud rows from the test set.

Verifies the full path that would run in production/demo:
  raw CSV row → scaler (from scaler.pkl) → predict.py → verdict

Run from the src/ directory:
    python smoke_test.py

What this tests that evaluate.py does NOT:
  - That the saved threshold in config.json is actually usable (< 1.0)
  - That predict.py's scaler application matches training
  - That the ImbPipeline's predict_proba skips SMOTE at inference time
  - That SHAP extraction from named_steps['clf'] works end-to-end
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import json

# ── load raw data to get real fraud rows ──────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
CFG_PATH  = os.path.join(BASE_DIR, "models", "config.json")

print("=" * 60)
print("  FRAUD SENTINEL — END-TO-END SMOKE TEST")
print("=" * 60)

# ── 1. Verify config.json threshold is sane ───────────────────────────────────
print("\n[1] Checking saved threshold in config.json ...")
with open(CFG_PATH) as f:
    config = json.load(f)

thr = config["best_threshold"]
assert thr < 1.0, (
    f"FAIL: threshold={thr} >= 1.0 — prob >= 1.0 is always False at inference! "
    "Re-run train.py with the fixed best_threshold_f1()."
)
assert thr > 0.0, f"FAIL: threshold={thr} <= 0.0 — something went badly wrong."
print(f"  ✓ threshold={thr:.6f}  (sane — between 0 and 1)")

# ── 2. Replicate the train/test split to get the SAME test rows ───────────────
print("\n[2] Loading data and reproducing train/test split ...")
from sklearn.model_selection import train_test_split

df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=["Class"])
y  = df["Class"]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  Test set: {X_test.shape[0]:,} rows | Fraud: {y_test.sum()} rows")

fraud_idx  = y_test[y_test == 1].index
legit_idx  = y_test[y_test == 0].index
print(f"  Using {min(5, len(fraud_idx))} fraud + {min(3, len(legit_idx))} legit rows")

# ── 3. Load predictor (tests scaler + pipeline + SHAP all at once) ────────────
print("\n[3] Loading FraudPredictor ...")
from predict import FraudPredictor
predictor = FraudPredictor(os.path.join(BASE_DIR, "models"))
print(f"  ✓ Predictor loaded | threshold={predictor.threshold:.6f}")

assert predictor.threshold < 1.0, (
    f"FAIL: predictor.threshold={predictor.threshold} >= 1.0"
)

# ── 4. Score known fraud rows ─────────────────────────────────────────────────
print("\n[4] Scoring known FRAUD rows ...")
fraud_rows = X_test.loc[fraud_idx[:5]]

n_correct = 0
for row_idx, (df_idx, row) in enumerate(fraud_rows.iterrows()):
    tx = row.to_dict()
    result = predictor.predict(tx)

    flag = "✓ FLAGGED" if result["is_fraud"] else "✗ MISSED "
    if result["is_fraud"]:
        n_correct += 1

    print(f"  Row {row_idx+1}: prob={result['probability']:.4f} | "
          f"verdict={result['verdict']:11s} | {flag} | "
          f"top_reason={result['shap_top3'][0]['feature']} "
          f"({result['shap_top3'][0]['direction']})")

print(f"\n  Fraud detection: {n_correct}/{min(5, len(fraud_idx))} correctly flagged")

# ── 5. Score known legit rows ─────────────────────────────────────────────────
print("\n[5] Scoring known LEGIT rows ...")
legit_rows = X_test.loc[legit_idx[:3]]

n_fp = 0
for row_idx, (df_idx, row) in enumerate(legit_rows.iterrows()):
    tx = row.to_dict()
    result = predictor.predict(tx)

    flag = "✓ CORRECT " if not result["is_fraud"] else "✗ FALSE POS"
    if result["is_fraud"]:
        n_fp += 1

    print(f"  Row {row_idx+1}: prob={result['probability']:.4f} | "
          f"verdict={result['verdict']:11s} | {flag}")

print(f"\n  False positives: {n_fp}/{min(3, len(legit_idx))}")

# ── 6. Assert minimum expectations ───────────────────────────────────────────
print("\n[6] Assertions ...")

assert n_correct > 0, (
    "FAIL: Zero fraud rows were flagged — the inference threshold or predict() "
    "pipeline is broken. Check that threshold < 1.0 and scaler is applied correctly."
)
print(f"  ✓ At least one fraud row flagged ({n_correct} caught)")

# SHAP sanity: all top3 features should be real feature names
feat_set = set(predictor.feature_names)
sample_result = predictor.predict(fraud_rows.iloc[0].to_dict())
for s in sample_result["shap_top3"]:
    assert s["feature"] in feat_set, f"FAIL: Unknown SHAP feature '{s['feature']}'"
    assert s["direction"] in ("↑ fraud", "↓ fraud"), \
        f"FAIL: Unexpected direction '{s['direction']}'"
print("  ✓ SHAP top-3 features are valid and directional")

# Threshold consistency: predictor.threshold matches config.json
assert abs(predictor.threshold - config["best_threshold"]) < 1e-9, \
    "FAIL: predictor.threshold != config['best_threshold']"
print(f"  ✓ predictor.threshold == config.json threshold ({predictor.threshold:.6f})")

print("\n" + "=" * 60)
print("  ALL CHECKS PASSED — inference path is verified")
print("=" * 60)
