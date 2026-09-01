"""
predict.py
──────────
Inference module: loads model + scaler + threshold, scores a new transaction.
Returns fraud probability, verdict, and top-3 SHAP feature reasons.
Used by both the FastAPI backend and the Streamlit app directly.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import shap
from typing import Dict, Any, List, Tuple

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class FraudPredictor:
    """Stateful predictor — load once, score many times."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self._load()

    def _load(self):
        model_path  = os.path.join(self.models_dir, "model.pkl")
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        config_path = os.path.join(self.models_dir, "config.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"model.pkl not found at {model_path}. "
                "Run `python src/train.py` first."
            )

        self.pipeline = joblib.load(model_path)   # ImbPipeline(smote, clf)
        self.scaler   = joblib.load(scaler_path)

        with open(config_path) as f:
            config = json.load(f)

        self.threshold     = config["best_threshold"]
        self.feature_names = config["feature_names"]

        # TreeExplainer needs the raw classifier, not the pipeline wrapper
        clf = self.pipeline.named_steps["clf"]
        self.explainer = shap.TreeExplainer(clf)
        print(f"[predict] Loaded pipeline | threshold={self.threshold:.4f}")

    def _prepare(self, transaction: Dict[str, float]) -> pd.DataFrame:
        """Build feature DataFrame, apply the SAME scaler fitted on training data."""
        df = pd.DataFrame([transaction])[self.feature_names]
        df[["Amount", "Time"]] = self.scaler.transform(df[["Amount", "Time"]])
        return df

    def predict(self, transaction: Dict[str, float]) -> Dict[str, Any]:
        """
        Score a single transaction.
        Returns:
          - probability: float [0, 1]
          - is_fraud: bool
          - verdict: "FRAUD" | "LEGITIMATE"
          - confidence: "HIGH" | "MEDIUM" | "LOW"
          - shap_top3: list of {feature, value, shap_value, direction}
        """
        X = self._prepare(transaction)

        prob   = float(self.pipeline.predict_proba(X)[:, 1][0])
        is_fraud = prob >= self.threshold

        # confidence bucket
        if prob > 0.8 or prob < 0.2:
            confidence = "HIGH"
        elif prob > 0.6 or prob < 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # SHAP — extract raw values from classifier step (not pipeline)
        X_values = X.values if hasattr(X, "values") else X
        shap_vals = self.explainer.shap_values(X_values)[0]
        top3 = self._top3_shap(X.iloc[0], shap_vals)

        return {
            "probability": round(prob, 6),
            "is_fraud":    bool(is_fraud),
            "verdict":     "FRAUD" if is_fraud else "LEGITIMATE",
            "confidence":  confidence,
            "threshold":   round(self.threshold, 4),
            "shap_top3":   top3,
        }

    def predict_batch(self, transactions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in transactions]

    def _top3_shap(
        self,
        row: pd.Series,
        shap_values: np.ndarray,
        n: int = 3
    ) -> List[Dict[str, Any]]:
        """Return top-n features by |SHAP| with direction and raw value."""
        abs_shap = np.abs(shap_values)
        top_idx  = np.argsort(abs_shap)[-n:][::-1]
        result = []
        for idx in top_idx:
            result.append({
                "feature":    self.feature_names[idx],
                "raw_value":  round(float(row.iloc[idx]), 4),
                "shap_value": round(float(shap_values[idx]), 4),
                "direction":  "↑ fraud" if shap_values[idx] > 0 else "↓ fraud",
            })
        return result

    def get_shap_values(self, transaction: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        """Return full SHAP value array + feature names for custom plots."""
        X = self._prepare(transaction)
        shap_vals = self.explainer.shap_values(X.values)[0]
        return shap_vals, self.feature_names


# ── Singleton instance for Streamlit (avoids reloading on every rerun) ─────────
_predictor: FraudPredictor | None = None

def get_predictor() -> FraudPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor()
    return _predictor


if __name__ == "__main__":
    # Quick smoke test with a synthetic transaction
    predictor = get_predictor()
    sample = {f: 0.0 for f in predictor.feature_names}
    sample["Amount"] = 150.0
    sample["Time"]   = 80000.0
    result = predictor.predict(sample)
    print(result)
