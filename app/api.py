"""
api.py — FraudSentinel FastAPI Backend
Production REST API for Credit Card Fraud Detection
"""

import sys, os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure src/ is on Python path
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR    = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, SRC_DIR)

from predict import FraudPredictor, get_predictor

app = FastAPI(
    title="FraudSentinel API",
    description="Production REST API for Real-Time Credit Card Fraud Detection using Random Forest + SHAP Explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin web client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class TransactionInput(BaseModel):
    Time: float = Field(..., example=80000.0, description="Elapsed seconds since start")
    Amount: float = Field(..., example=124.50, description="Transaction amount in USD")
    V1: float = Field(0.0, example=1.902)
    V2: float = Field(0.0, example=-1.302)
    V3: float = Field(0.0, example=-1.305)
    V4: float = Field(0.0, example=-0.810)
    V5: float = Field(0.0, example=0.051)
    V6: float = Field(0.0, example=1.652)
    V7: float = Field(0.0, example=-1.053)
    V8: float = Field(0.0, example=0.539)
    V9: float = Field(0.0, example=-0.199)
    V10: float = Field(0.0, example=0.895)
    V11: float = Field(0.0, example=0.389)
    V12: float = Field(0.0, example=-0.045)
    V13: float = Field(0.0, example=-1.028)
    V14: float = Field(0.0, example=0.414)
    V15: float = Field(0.0, example=0.473)
    V16: float = Field(0.0, example=-1.744)
    V17: float = Field(0.0, example=0.239)
    V18: float = Field(0.0, example=0.556)
    V19: float = Field(0.0, example=-1.360)
    V20: float = Field(0.0, example=-0.592)
    V21: float = Field(0.0, example=-0.260)
    V22: float = Field(0.0, example=-0.300)
    V23: float = Field(0.0, example=0.281)
    V24: float = Field(0.0, example=-0.982)
    V25: float = Field(0.0, example=-0.580)
    V26: float = Field(0.0, example=0.638)
    V27: float = Field(0.0, example=-0.013)
    V28: float = Field(0.0, example=-0.064)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """System health check endpoint."""
    try:
        predictor = get_predictor()
        return {
            "status": "healthy",
            "model_loaded": True,
            "threshold": predictor.threshold,
            "feature_count": len(predictor.feature_names)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})


@app.get("/metrics", tags=["Model Information"])
def get_metrics():
    """Retrieve model performance metrics and parameters."""
    cfg_path = os.path.join(MODELS_DIR, "config.json")
    if not os.path.exists(cfg_path):
        raise HTTPException(status_code=444, detail="Config not found.")
    with open(cfg_path) as f:
        config = json.load(f)
    return config


@app.post("/predict", tags=["Inference"])
def predict_transaction(tx: TransactionInput):
    """
    Score a credit card transaction in real time.
    Returns fraud probability, binary verdict, confidence bucket, and SHAP top-3 explanations.
    """
    try:
        predictor = get_predictor()
        payload = tx.model_dump()
        result = predictor.predict(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/random", tags=["Testing"])
def get_random_sample(target: Optional[str] = None):
    """Fetch a random test transaction (supports target='fraud' or target='legit')."""
    sample_path = os.path.join(MODELS_DIR, "test_sample.csv")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="test_sample.csv not found.")

    df = pd.read_csv(sample_path)
    if target == "fraud":
        sub = df[df["Class"] == 1]
    elif target == "legit":
        sub = df[df["Class"] == 0]
    else:
        sub = df[df["Class"] == 1] if np.random.random() < 0.5 else df[df["Class"] == 0]

    row = sub.sample(1).iloc[0].to_dict()
    ground_truth = int(row.pop("Class"))
    return {
        "ground_truth": "FRAUD" if ground_truth == 1 else "LEGITIMATE",
        "ground_truth_code": ground_truth,
        "transaction": row
    }


@app.get("/", response_class=HTMLResponse, tags=["Dashboard Client"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard Client"])
def render_dashboard():
    """Serve full-width Streamlit-matching production dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FraudSentinel — Production REST Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background: #0e1117;
            color: #fafafa;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            line-height: 1.5;
            min-height: 100vh;
        }

        .topbar {
            background: #161b22;
            border-bottom: 1px solid #30363d;
            padding: 0 2rem;
            height: 54px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .topbar-brand { display: flex; flex-direction: column; }
        .topbar-logo { font-size: 1rem; font-weight: 700; color: #fff; }
        .topbar-sub { font-size: 0.72rem; color: #8b949e; }
        .topbar-right { display: flex; gap: 0.5rem; }
        .nav-btn {
            background: transparent;
            border: 1px solid #30363d;
            color: #c9d1d9;
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }
        .nav-btn:hover { background: #21262d; border-color: #58a6ff; color: #fff; }

        .tabs {
            background: #0e1117;
            border-bottom: 1px solid #21262d;
            padding: 0 2rem;
            display: flex;
        }
        .tab-btn {
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            color: #8b949e;
            padding: 0.8rem 1.2rem;
            font-size: 0.88rem;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .tab-btn.active { color: #ff4b4b; border-bottom-color: #ff4b4b; }

        .main { padding: 1.5rem 2rem; }

        .sample-row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 0.8rem;
            margin-bottom: 0.8rem;
        }
        .sample-btn {
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 0.55rem 1rem;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
            font-family: inherit;
        }
        .sample-btn-rand  { background: #161b22; color: #c9d1d9; }
        .sample-btn-rand:hover  { background: #21262d; border-color: #58a6ff; }
        .sample-btn-fraud { background: #1c0a0a; color: #fca5a5; border-color: #7f1d1d; }
        .sample-btn-fraud:hover { background: #3b0f0f; border-color: #ef4444; }
        .sample-btn-legit { background: #0a1a0e; color: #6ee7b7; border-color: #065f46; }
        .sample-btn-legit:hover { background: #0f2a18; border-color: #10b981; }

        .gt-label {
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.25rem 0.8rem;
            border-radius: 4px;
            display: inline-block;
            margin-bottom: 0.8rem;
        }
        .gt-fraud { background: #3f1212; color: #fca5a5; border: 1px solid #7f1d1d; }
        .gt-legit { background: #064e3b; color: #6ee7b7; border: 1px solid #065f46; }

        .section {
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 0.72rem;
            font-weight: 700;
            color: #58a6ff;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border-bottom: 1px solid #21262d;
            padding-bottom: 0.4rem;
            margin-bottom: 0.8rem;
        }
        .section-hint { font-size: 0.77rem; color: #8b949e; margin-bottom: 0.8rem; }

        .input-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 0.8rem; }
        .input-grid-7 { display: grid; grid-template-columns: repeat(7, 1fr); gap: 0.5rem; margin-bottom: 0.8rem; }

        .field label {
            display: block;
            font-size: 0.76rem;
            font-weight: 600;
            color: #8b949e;
            margin-bottom: 0.3rem;
        }
        .field input {
            width: 100%;
            background: #0d1117;
            border: 1px solid #30363d;
            color: #f0f6fc;
            padding: 0.45rem 0.6rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-family: inherit;
        }
        .field input:focus { outline: none; border-color: #58a6ff; }
        .field input:disabled { background: #161b22; color: #e6edf3; border-color: #21262d; }

        .analyze-btn {
            width: 100%;
            background: #1f6feb;
            border: none;
            color: #fff;
            padding: 0.75rem;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 0.8rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            font-family: inherit;
        }
        .analyze-btn:hover { background: #388bfd; }

        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1.4fr 1fr;
            gap: 1rem;
            align-items: center;
        }

        .verdict-box {
            border: 2px solid;
            border-radius: 8px;
            padding: 1.5rem 1rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 180px;
        }
        .verdict-fraud { background: #1a0a0a; border-color: #ef4444; color: #ef4444; }
        .verdict-legit { background: #0a1a0e; border-color: #10b981; color: #10b981; }
        .verdict-title { font-size: 1.5rem; font-weight: 700; }
        .verdict-sub { font-size: 0.78rem; color: #8b949e; margin-top: 0.5rem; }

        .shap-title { font-size: 0.72rem; font-weight: 700; color: #c9d1d9; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem; }
        .shap-item {
            background: #0d1117;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 0.45rem 0.7rem;
            margin-bottom: 0.3rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
        }
        .shap-feat { font-weight: 600; color: #58a6ff; }
        .shap-up   { color: #ef4444; font-weight: 600; }
        .shap-down { color: #10b981; font-weight: 600; }

        .hidden { display: none !important; }
    </style>
</head>
<body>

<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">🛡️ FraudSentinel</div>
        <div class="topbar-sub">Credit Card Fraud Detection · Random Forest + SHAP · End-to-end verified pipeline</div>
    </div>
    <div class="topbar-right">
        <a href="/docs"    target="_blank" class="nav-btn">⚡ Swagger Docs</a>
        <a href="/health"  target="_blank" class="nav-btn">💚 Health</a>
        <a href="/metrics" target="_blank" class="nav-btn">📊 Metrics</a>
    </div>
</div>

<div class="tabs">
    <button class="tab-btn active">🔍 Transaction Check</button>
</div>

<div class="main">

    <div class="sample-row">
        <button class="sample-btn sample-btn-rand"  onclick="loadRandom()">🎲 Random transaction</button>
        <button class="sample-btn sample-btn-fraud" onclick="loadRandom('fraud')">🚨 Random fraud row</button>
        <button class="sample-btn sample-btn-legit" onclick="loadRandom('legit')">✅ Random legit row</button>
    </div>
    <div id="gtBadge" class="hidden"></div>

    <form id="apiForm">
        <div class="section">
            <div class="section-title">Transaction Details</div>
            <div class="section-hint">Amount &amp; Time are editable. V1–V28 are auto-loaded from the selected transaction (read-only).</div>

            <div class="input-grid-2">
                <div class="field">
                    <label>Time (seconds)</label>
                    <input type="number" id="Time" value="80000" step="any">
                </div>
                <div class="field">
                    <label>Amount (USD)</label>
                    <input type="number" id="Amount" value="74.20" step="any">
                </div>
            </div>

            <div class="section-title" style="margin-top:0.6rem">
                PCA Components — V1 to V14
                <span style="color:#444;font-size:.68rem;font-weight:400"> (auto-populated · read-only)</span>
            </div>
            <div class="input-grid-7" id="v_a"></div>

            <div class="section-title">
                PCA Components — V15 to V28
                <span style="color:#444;font-size:.68rem;font-weight:400"> (auto-populated · read-only)</span>
            </div>
            <div class="input-grid-7" id="v_b"></div>

            <button type="submit" class="analyze-btn">🔍 Analyze Transaction</button>
        </div>
    </form>

    <div id="resultsSection" class="section hidden">
        <div class="section-title">Fraud Probability</div>
        <div class="results-grid">
            <div id="gaugePlot"></div>
            <div id="verdictBox" class="verdict-box">
                <div id="verdictTitle" class="verdict-title"></div>
                <div id="verdictSub"   class="verdict-sub"></div>
            </div>
            <div id="shapBox"></div>
        </div>
        <div style="margin-top:1rem">
            <div class="section-title">SHAP Feature Impact</div>
            <div id="shapBarPlot"></div>
        </div>
    </div>

</div>

<script>
    // Build V1–V28 inputs
    const va = document.getElementById('v_a');
    const vb = document.getElementById('v_b');
    for (let i = 1;  i <= 14; i++) va.innerHTML += `<div class="field"><label>V${i}</label><input type="number" id="V${i}"  value="0.000" step="any" disabled></div>`;
    for (let i = 15; i <= 28; i++) vb.innerHTML += `<div class="field"><label>V${i}</label><input type="number" id="V${i}" value="0.000" step="any" disabled></div>`;

    async function loadRandom(target = null) {
        try {
            let url = '/random';
            if (target) url += `?target=${target}`;
            const res  = await fetch(url);
            const data = await res.json();
            const tx   = data.transaction;
            for (const key in tx) {
                const el = document.getElementById(key);
                if (el) el.value = parseFloat(tx[key]).toFixed(3);
            }
            const badge = document.getElementById('gtBadge');
            if (data.ground_truth === 'FRAUD') {
                badge.className   = 'gt-label gt-fraud';
                badge.innerText   = '🚨 Fraud row loaded — ground truth label: FRAUD (Class = 1)';
            } else {
                badge.className   = 'gt-label gt-legit';
                badge.innerText   = '✅ Legit row loaded — ground truth label: LEGITIMATE (Class = 0)';
            }
        } catch(e) { console.error(e); }
    }

    document.getElementById('apiForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
            Time:   parseFloat(document.getElementById('Time').value),
            Amount: parseFloat(document.getElementById('Amount').value)
        };
        for (let i = 1; i <= 28; i++) payload[`V${i}`] = parseFloat(document.getElementById(`V${i}`).value || 0);

        const res  = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        document.getElementById('resultsSection').classList.remove('hidden');

        const mainColor = data.is_fraud ? '#ef4444' : '#10b981';

        // Verdict box
        const vbox = document.getElementById('verdictBox');
        vbox.className = `verdict-box ${data.is_fraud ? 'verdict-fraud' : 'verdict-legit'}`;
        document.getElementById('verdictTitle').innerText = data.is_fraud ? '🚨 FRAUD DETECTED' : '✅ LEGITIMATE';
        document.getElementById('verdictSub').innerText   =
            `Fraud Score: ${(data.probability * 100).toFixed(2)}% | Threshold: ${data.threshold} | Confidence: ${data.confidence}`;

        // Top SHAP cards
        let sh = `<div class="shap-title">Top SHAP Features</div>`;
        data.shap_top3.forEach(item => {
            const cls = item.shap_value > 0 ? 'shap-up' : 'shap-down';
            sh += `<div class="shap-item">
                <span class="shap-feat">${item.feature} <span style="color:#8b949e;font-weight:400">(${item.raw_value})</span></span>
                <span class="${cls}">${item.direction} (${item.shap_value > 0 ? '+' : ''}${item.shap_value})</span>
            </div>`;
        });
        document.getElementById('shapBox').innerHTML = sh;

        // Plotly Gauge
        Plotly.newPlot('gaugePlot', [{
            type: 'indicator', mode: 'gauge+number',
            value: data.probability * 100,
            number: { suffix: '%', font: { size: 30, color: mainColor } },
            gauge: {
                axis: { range: [0, 100], tickcolor: '#444', tickfont: { color: '#8b949e', size: 10 } },
                bar:  { color: mainColor, thickness: 0.25 },
                bgcolor: '#0d1117',
                steps: [
                    { range: [0,  50], color: '#0a1a0e' },
                    { range: [50, 100], color: '#1a0a0a' }
                ]
            }
        }], {
            height: 220, margin: { t: 10, b: 10, l: 20, r: 20 },
            paper_bgcolor: '#161b22',
            font: { color: '#f0f6fc', family: 'Inter' }
        });

        // Plotly SHAP Bar
        const sf = data.shap_top3.map(r => r.feature).reverse();
        const sv = data.shap_top3.map(r => r.shap_value).reverse();
        Plotly.newPlot('shapBarPlot', [{
            type: 'bar', x: sv, y: sf, orientation: 'h',
            marker: { color: sv.map(v => v > 0 ? '#ef4444' : '#10b981') },
            text: sv.map(v => (v > 0 ? '+' : '') + v.toFixed(4)),
            textposition: 'outside',
            textfont: { color: '#e6edf3', size: 12 }
        }], {
            height: 180, margin: { t: 10, b: 10, l: 60, r: 80 },
            paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
            xaxis: {
                gridcolor: '#21262d', zerolinecolor: '#444', color: '#8b949e',
                title: { text: 'SHAP value (impact on fraud score)', font: { color: '#8b949e', size: 11 } }
            },
            yaxis: { color: '#58a6ff' }
        });
    });

    // Load a random sample on page load
    loadRandom();
</script>
</body>
</html>
"""
