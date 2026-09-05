"""
stream_simulator.py
───────────────────
Simulates real-time transaction streaming into FraudSentinel inference engine.
Reads sample test rows and streams them sequentially with configurable delay.
Outputs live colorized terminal logs & logs entries into audit trail.
"""

import sys
import os
import time
import json
import random
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, SRC_DIR)

from predict import get_predictor

def run_stream_simulation(delay_seconds: float = 0.8, max_events: int = 30):
    sample_path = os.path.join(MODELS_DIR, "test_sample.csv")
    if not os.path.exists(sample_path):
        print(f"❌ Error: {sample_path} not found. Run `python src/train.py` first.")
        return

    df = pd.read_csv(sample_path)
    predictor = get_predictor()

    print("\n" + "="*75)
    print(" 🛡️ FRAUDSENTINEL REAL-TIME STREAMING INFERENCE SIMULATOR")
    print(" ="*75)
    print(f" Streaming {max_events} transactions (interval: {delay_seconds}s)...\n")

    fraud_rows = df[df["Class"] == 1]
    legit_rows = df[df["Class"] == 0]

    count = 0
    while count < max_events:
        count += 1
        # Interleave fraud rows for demo impact
        if random.random() < 0.35 and len(fraud_rows) > 0:
            row = fraud_rows.sample(1).iloc[0].to_dict()
        else:
            row = legit_rows.sample(1).iloc[0].to_dict()

        gt = int(row.pop("Class", 0))
        tx_data = {k: float(v) for k, v in row.items()}
        cid = f"CARD-{random.randint(100, 105):04d}"  # Interleaved card IDs to trigger velocity rules

        res = predictor.predict(tx_data, card_id=cid)

        tx_id = res["transaction_id"]
        amt   = res["amount"]
        prob  = res["probability"] * 100
        act   = res["action"]

        if act == "AUTO_BLOCK":
            badge = "⛔ [AUTO_BLOCK]"
        elif act == "MANUAL_REVIEW":
            badge = "⚠️ [MANUAL_REVIEW]"
        elif act == "SUPERVISOR_OVERRIDE_REQUIRED":
            badge = "🛡️ [SAFETY_OVERRIDE]"
        else:
            badge = "✅ [AUTO_CLEAR]"

        reasons = ", ".join([f"{s['feature']}:{s['direction']}" for s in res.get("shap_top3", [])])

        print(f"[{res['timestamp'][11:19]}] {tx_id} | {cid} | Amt: ${amt:>7.2f} | Score: {prob:>5.1f}% | {badge:<22} | {reasons}")

        time.sleep(delay_seconds)

    print("\n" + "="*75)
    print(f" ✅ Streaming simulation complete. Scored {count} transactions.")
    print("    Audit trail persisted to `data/audit_log.jsonl`")
    print("="*75 + "\n")

def test_velocity_escalation():
    """Explicitly tests 4 consecutive high-fraud transactions (prob >= 0.85) for CARD-VELOCITY-DEMO."""
    sample_path = os.path.join(MODELS_DIR, "test_sample.csv")
    if not os.path.exists(sample_path):
        return
    df = pd.read_csv(sample_path)
    fraud_rows = df[df["Class"] == 1]

    predictor = get_predictor()
    cid = "CARD-VELOCITY-DEMO"

    # Pre-filter for rows that score >= 0.85 (guaranteed AUTO_BLOCK candidates)
    autoblock_rows = []
    for idx, row_series in fraud_rows.iterrows():
        r = row_series.to_dict()
        r.pop("Class", None)
        X = predictor._prepare(r)
        p_val = float(predictor.pipeline.predict_proba(X)[:, 1][0])
        if p_val >= 0.85:
            autoblock_rows.append(r)

    print("\n" + "="*75)
    print(" 🛡️ TESTING VELOCITY ESCALATION RULE FOR CARD-VELOCITY-DEMO (MAX 3 AUTO-BLOCKS)")
    print(" ="*75)

    for i in range(4):
        r = autoblock_rows[i % len(autoblock_rows)].copy() if autoblock_rows else fraud_rows.iloc[i].to_dict()
        res = predictor.predict(r, card_id=cid)

        tx_id = res["transaction_id"]
        prob  = res["probability"] * 100
        act   = res["action"]
        rule  = res["rule_triggered"]

        if act == "SUPERVISOR_OVERRIDE_REQUIRED":
            badge = "🛡️ [SUPERVISOR_OVERRIDE_REQUIRED]"
        else:
            badge = f"⛔ [{act}]"

        print(f"Attempt #{i+1} | {tx_id} | {cid} | Score: {prob:>5.1f}% | {badge:<35} | Rule: {rule}")
        time.sleep(0.4)
    print("="*75 + "\n")



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-velocity":
        test_velocity_escalation()
    else:
        test_velocity_escalation()
        run_stream_simulation()

