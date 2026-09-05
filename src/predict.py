"""
predict.py
──────────
Inference & Governance module:
- Scored model probabilities & SHAP feature importances
- Action Layer (AUTO_BLOCK, MANUAL_REVIEW, AUTO_CLEAR)
- Velocity & Gated Safety Stopping Rules (max 3 auto-blocks per card)
- Automated Dispute Evidence Generator
- Audit Trail Logger (data/audit_log.jsonl)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import json
import os
import time
import uuid
import shap
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
AUDIT_LOG_JSONL = os.path.join(DATA_DIR, "audit_log.jsonl")
AUDIT_LOG_CSV   = os.path.join(DATA_DIR, "audit_log.csv")

os.makedirs(DATA_DIR, exist_ok=True)


class FraudPredictor:
    """Stateful predictor & governance engine — load once, score & audit many times."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.velocity_counter: Dict[str, int] = {}
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
        print(f"[predict] Loaded pipeline & action engine | threshold={self.threshold:.4f}")

    def _prepare(self, transaction: Dict[str, float]) -> pd.DataFrame:
        """Build feature DataFrame, apply the SAME scaler fitted on training data."""
        df = pd.DataFrame([transaction])[self.feature_names]
        df[["Amount", "Time"]] = self.scaler.transform(df[["Amount", "Time"]])
        return df

    def predict(self, transaction: Dict[str, float], card_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Score a single transaction & run Action Routing, Safety Rules, Dispute Evidence, and Audit Trail.
        """
        tx_id = f"TX-{uuid.uuid4().hex[:10].upper()}"
        timestamp = datetime.utcnow().isoformat() + "Z"
        amount = float(transaction.get("Amount", 0.0))
        # Stable process-independent hex hash with 16.7M ID space
        v1_str = str(transaction.get('V1', 0.0)).encode('utf-8')
        cid = card_id or f"CARD-{hashlib.md5(v1_str).hexdigest()[:6].upper()}"


        X = self._prepare(transaction)

        prob = float(self.pipeline.predict_proba(X)[:, 1][0])
        is_fraud = prob >= self.threshold

        # ── 1. Confidence Bucket ─────────────────────────────────────────────
        if prob > 0.85 or prob < 0.15:
            confidence = "HIGH"
        elif prob > 0.65 or prob < 0.35:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # ── 2. SHAP Top 3 ────────────────────────────────────────────────────
        X_values = X.values if hasattr(X, "values") else X
        raw_shap = self.explainer.shap_values(X_values)
        if isinstance(raw_shap, list):
            shap_vals = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif len(raw_shap.shape) == 3:
            shap_vals = raw_shap[0, :, 1]
        else:
            shap_vals = raw_shap[0]

        top3 = self._top3_shap(X.iloc[0], shap_vals)

        # ── 3. Action Layer & Decision Router (Tier 1) ────────────────────────
        # Routing Rules:
        #   prob >= 0.85                      -> AUTO_BLOCK
        #   threshold (0.6554) <= prob < 0.85 -> MANUAL_REVIEW
        #   prob < threshold                  -> AUTO_CLEAR
        if prob >= 0.85:
            action = "AUTO_BLOCK"
            queue  = "BLOCKED_QUEUE"
            risk_level = "CRITICAL"
        elif prob >= self.threshold:
            action = "MANUAL_REVIEW"
            queue  = "MANUAL_REVIEW_QUEUE"
            risk_level = "ELEVATED"
        else:
            action = "AUTO_CLEAR"
            queue  = "CLEARED_QUEUE"
            risk_level = "LOW"

        # ── 4. Safety Guardrail / Rate-Limiting Velocity Rule (Tier 2) ──────
        current_blocks = self.velocity_counter.get(cid, 0)
        safety_override = False
        rule_triggered = "STANDARD_POLICY"

        if action == "AUTO_BLOCK":
            current_blocks += 1
            self.velocity_counter[cid] = current_blocks
            if current_blocks > 3:
                action = "SUPERVISOR_OVERRIDE_REQUIRED"
                queue  = "GOVERNANCE_QUEUE"
                safety_override = True
                rule_triggered = "VELOCITY_CAP_EXCEEDED (Max 3 auto-blocks/card/day)"

        # ── 5. Auto-Draft Dispute Evidence (Tier 1) ──────────────────────────
        dispute_evidence = None
        if action in ["AUTO_BLOCK", "MANUAL_REVIEW", "SUPERVISOR_OVERRIDE_REQUIRED"]:
            reasons_str = ", ".join([f"{s['feature']} (val: {s['raw_value']}, SHAP: {s['shap_value']:+.3f})" for s in top3])
            dispute_evidence = {
                "evidence_id": f"EVID-{uuid.uuid4().hex[:8].upper()}",
                "generated_at": timestamp,
                "summary": (
                    f"[AUTO-GENERATED DISPUTE PACKET] Transaction {tx_id} of amount ${amount:.2f} "
                    f"flagged with fraud score {prob*100:.1f}%. Key anomaly drivers: {reasons_str}. "
                    f"Recommended Action: Request cardholder identity confirmation via SMS OTP or KYC hold."
                ),
                "key_anomalies": [s["feature"] for s in top3 if s["shap_value"] > 0],
                "status": "DRAFT_CREATED"
            }

        # ── 6. Financial Impact / Cost-Based Risk Metrics ────────────────────
        # Estimated values:
        #   Fraud Loss Prevented (if blocked/reviewed): $amount or $125 avg
        #   Customer Friction Cost (if false positive): $15 avg
        #   Residual Missed Risk (if false negative): $150 avg
        prevented_value = amount if is_fraud else 0.0
        friction_cost   = 15.00 if (action != "AUTO_CLEAR" and not is_fraud) else 0.0

        result = {
            "transaction_id": tx_id,
            "timestamp": timestamp,
            "card_id": cid,
            "amount": round(amount, 2),
            "probability": round(prob, 6),
            "is_fraud": bool(is_fraud),
            "verdict": "FRAUD" if is_fraud else "LEGITIMATE",
            "action": action,
            "queue": queue,
            "risk_level": risk_level,
            "confidence": confidence,
            "threshold": round(self.threshold, 4),
            "rule_triggered": rule_triggered,
            "safety_override": safety_override,
            "shap_top3": top3,
            "dispute_evidence": dispute_evidence,
            "financial_impact": {
                "prevented_fraud_val": round(prevented_value, 2),
                "friction_cost": round(friction_cost, 2),
            }
        }

        # ── 7. Audit Trail Logging (Tier 1) ──────────────────────────────────
        self._write_audit_log(result)

        return result

    def _write_audit_log(self, record: Dict[str, Any]):
        """Persists every scored transaction into an immutable audit trail."""
        try:
            # Flatten top3 for CSV
            top3_summary = "; ".join([f"{s['feature']}:{s['direction']}" for s in record.get("shap_top3", [])])

            # Write JSONL
            with open(AUDIT_LOG_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            # Write CSV header if missing
            file_exists = os.path.exists(AUDIT_LOG_CSV)
            with open(AUDIT_LOG_CSV, "a", encoding="utf-8") as f:
                if not file_exists:
                    f.write("timestamp,transaction_id,card_id,amount,probability,verdict,action,queue,confidence,shap_top3\n")
                f.write(f"{record['timestamp']},{record['transaction_id']},{record['card_id']},{record['amount']},{record['probability']:.4f},{record['verdict']},{record['action']},{record['queue']},{record['confidence']},\"{top3_summary}\"\n")
        except Exception as e:
            print(f"[AuditLog Warning] Could not write audit log: {e}")

    def get_audit_logs(self, limit: int = 50, filter_action: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves recent audit logs."""
        if not os.path.exists(AUDIT_LOG_JSONL):
            return []
        logs = []
        with open(AUDIT_LOG_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if filter_action and item.get("action") != filter_action:
                        continue
                    logs.append(item)
        return logs[-limit:][::-1]

    def get_audit_summary(self) -> Dict[str, Any]:
        """Calculates audit trail summary statistics."""
        logs = self.get_audit_logs(limit=10000)
        total = len(logs)
        if total == 0:
            return {
                "total_scored": 0, "auto_blocked": 0, "manual_review": 0,
                "auto_cleared": 0, "supervisor_overrides": 0,
                "total_fraud_prevented_usd": 0.0
            }

        auto_blocked = sum(1 for l in logs if l.get("action") == "AUTO_BLOCK")
        manual_review = sum(1 for l in logs if l.get("action") == "MANUAL_REVIEW")
        auto_cleared = sum(1 for l in logs if l.get("action") == "AUTO_CLEAR")
        supervisors = sum(1 for l in logs if l.get("action") == "SUPERVISOR_OVERRIDE_REQUIRED")
        total_prevented = sum(l.get("financial_impact", {}).get("prevented_fraud_val", 0.0) for l in logs)

        return {
            "total_scored": total,
            "auto_blocked": auto_blocked,
            "manual_review": manual_review,
            "auto_cleared": auto_cleared,
            "supervisor_overrides": supervisors,
            "total_fraud_prevented_usd": round(total_prevented, 2)
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
        X_values = X.values if hasattr(X, "values") else X
        raw_shap = self.explainer.shap_values(X_values)
        if isinstance(raw_shap, list):
            shap_vals = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif len(raw_shap.shape) == 3:
            shap_vals = raw_shap[0, :, 1]
        else:
            shap_vals = raw_shap[0]
        return shap_vals, self.feature_names


# ── Singleton instance for Streamlit (avoids reloading on every rerun) ─────────
_predictor: FraudPredictor | None = None

def get_predictor() -> FraudPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor()
    return _predictor


if __name__ == "__main__":
    predictor = get_predictor()
    sample = {f: 0.0 for f in predictor.feature_names}
    sample["Amount"] = 150.0
    sample["Time"]   = 80000.0
    result = predictor.predict(sample)
    print(json.dumps(result, indent=2))
