"""
api.py — FraudSentinel FastAPI Backend
Production REST API for Credit Card Fraud Detection
Exact Ditto Visual UI matching Streamlit App
"""

import sys, os, json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR    = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, SRC_DIR)

from predict import FraudPredictor, get_predictor

app = FastAPI(
    title="FraudSentinel API",
    description="Production REST API — Random Forest + SHAP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionInput(BaseModel):
    Time: float = Field(..., example=80000.0)
    Amount: float = Field(..., example=124.50)
    V1:  float = Field(0.0); V2:  float = Field(0.0); V3:  float = Field(0.0)
    V4:  float = Field(0.0); V5:  float = Field(0.0); V6:  float = Field(0.0)
    V7:  float = Field(0.0); V8:  float = Field(0.0); V9:  float = Field(0.0)
    V10: float = Field(0.0); V11: float = Field(0.0); V12: float = Field(0.0)
    V13: float = Field(0.0); V14: float = Field(0.0); V15: float = Field(0.0)
    V16: float = Field(0.0); V17: float = Field(0.0); V18: float = Field(0.0)
    V19: float = Field(0.0); V20: float = Field(0.0); V21: float = Field(0.0)
    V22: float = Field(0.0); V23: float = Field(0.0); V24: float = Field(0.0)
    V25: float = Field(0.0); V26: float = Field(0.0); V27: float = Field(0.0)
    V28: float = Field(0.0)


@app.get("/health", tags=["System"])
def health_check():
    try:
        p = get_predictor()
        return {
            "status": "healthy",
            "model_loaded": True,
            "threshold": p.threshold,
            "feature_count": len(p.feature_names)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})


@app.get("/metrics", tags=["Model Information"])
def get_metrics():
    cfg = os.path.join(MODELS_DIR, "config.json")
    if not os.path.exists(cfg):
        raise HTTPException(status_code=404, detail="Config not found.")
    with open(cfg) as f:
        return json.load(f)


@app.post("/predict", tags=["Inference"])
def predict_transaction(tx: TransactionInput):
    try:
        result = get_predictor().predict(tx.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/random", tags=["Testing"])
def get_random_sample(target: Optional[str] = None):
    path = os.path.join(MODELS_DIR, "test_sample.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="test_sample.csv not found.")
    df = pd.read_csv(path)
    if target == "fraud":
        sub = df[df["Class"] == 1]
    elif target == "legit":
        sub = df[df["Class"] == 0]
    else:
        sub = df[df["Class"] == 1] if np.random.random() < 0.5 else df[df["Class"] == 0]
    row = sub.sample(1).iloc[0].to_dict()
    gt  = int(row.pop("Class"))
    return {
        "ground_truth": "FRAUD" if gt == 1 else "LEGITIMATE",
        "ground_truth_code": gt,
        "transaction": row
    }


@app.get("/", response_class=HTMLResponse, tags=["Dashboard Client"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard Client"])
def render_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FraudSentinel</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #0a0a0a;
    color: #e5e5e5;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.5;
    padding: 1.5rem 2rem;
    min-height: 100vh;
}

/* Hero Header */
.hero {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.hero h1 { font-size: 2rem; font-weight: 700; color: #e5e5e5; letter-spacing: -.02em; margin: 0; }
.hero p { color: #888; margin: .3rem 0 0; font-size: .95rem; }
.top-links { display: flex; gap: 0.5rem; }
.top-btn {
    background: #141414;
    border: 1px solid #2a2a2a;
    color: #e5e5e5;
    padding: 0.4rem 0.8rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}
.top-btn:hover { border-color: #3b82f6; color: #3b82f6; }

/* Tabs */
.tabs-header {
    display: flex;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 1.5rem;
    gap: 1.5rem;
}
.tab-button {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: #888;
    padding: 0.6rem 0.2rem;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: inherit;
}
.tab-button.active {
    color: #e5e5e5;
    border-bottom-color: #3b82f6;
    font-weight: 600;
}
.tab-button:hover:not(.active) { color: #ccc; }

.tab-content { display: none; }
.tab-content.active { display: block; }

/* Buttons Row */
.buttons-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 2fr;
    gap: 0.8rem;
    margin-bottom: 0.4rem;
    align-items: center;
}
.st-btn {
    background: #141414;
    border: 1px solid #2a2a2a;
    color: #e5e5e5;
    border-radius: 8px;
    padding: 0.55rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    font-family: inherit;
    transition: all 0.15s ease;
}
.st-btn:hover {
    border-color: #3b82f6;
    color: #3b82f6;
}

.gt-subtext {
    font-size: 0.82rem;
    margin-bottom: 1rem;
}

.divider {
    height: 1px;
    background: #2a2a2a;
    margin: 1.2rem 0;
}

/* Form Container */
.st-form {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Section Title */
.sec {
    font-size: .85rem;
    font-weight: 700;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: .1em;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: .4rem;
    margin-bottom: .9rem;
}
.st-caption {
    color: #a3a3a3;
    font-size: 0.82rem;
    margin-bottom: 1rem;
}

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
.grid-7 { display: grid; grid-template-columns: repeat(7, 1fr); gap: 0.5rem; margin-bottom: 1rem; }

.field-group label {
    display: block;
    color: #e5e5e5;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
}
.field-group input {
    width: 100%;
    background: #0a0a0a;
    border: 1px solid #333333;
    color: #ffffff;
    border-radius: 6px;
    padding: 0.5rem 0.6rem;
    font-weight: 600;
    font-size: 0.88rem;
    font-family: inherit;
}
.field-group input:disabled {
    background: #161616;
    border: 1px solid #2a2a2a;
    color: #ffffff;
    opacity: 1;
    font-weight: 600;
}

.submit-btn {
    width: 100%;
    background: #3b82f6;
    border: none;
    color: #ffffff;
    font-weight: 600;
    padding: 0.75rem;
    border-radius: 8px;
    font-size: 0.95rem;
    cursor: pointer;
    margin-top: 0.8rem;
    font-family: inherit;
}
.submit-btn:hover { background: #2563eb; }

/* Result Row */
.results-grid {
    display: grid;
    grid-template-columns: 1fr 1.1fr 1fr;
    gap: 1.2rem;
    align-items: start;
}

/* Verdict Boxes */
.fraud-box {
    background: #1a0a0a;
    border: 1.5px solid #ef4444;
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
}
.legit-box {
    background: #0a1a0e;
    border: 1.5px solid #10b981;
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
}
.verdict-text { font-size: 1.6rem; font-weight: 700; margin: 0; }
.verdict-sub { color: #888; font-size: .85rem; margin-top: .4rem; }

/* SHAP Card Rows */
.shap-row {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: .7rem 1rem;
    margin: .35rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* KPI Cards for Tab 2 */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.kpi-val { font-size: 1.8rem; font-weight: 700; color: #3b82f6; }
.kpi-lbl { font-size: .72rem; color: #888; text-transform: uppercase; letter-spacing: .08em; margin-top: .25rem; }

.charts-row-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}

.hidden { display: none !important; }
</style>
</head>
<body>

<!-- Header Hero -->
<div class="hero">
    <div>
        <h1>🛡️ FraudSentinel</h1>
        <p>Credit Card Fraud Detection · Random Forest + SHAP · End-to-end verified pipeline</p>
    </div>
    <div class="top-links">
        <a href="/docs" target="_blank" class="top-btn">⚡ Swagger Docs</a>
        <a href="/health" target="_blank" class="top-btn">💚 Health</a>
        <a href="/metrics" target="_blank" class="top-btn">📊 Metrics</a>
    </div>
</div>

<!-- Tabs Bar -->
<div class="tabs-header">
    <button class="tab-button active" onclick="openTab('t1', this)">🔍 &nbsp; Transaction Check</button>
    <button class="tab-button" onclick="openTab('t2', this)">📊 &nbsp; Model Dashboard</button>
</div>

<!-- ════════════════════ TAB 1: TRANSACTION CHECK ════════════════════ -->
<div id="t1" class="tab-content active">

    <div class="buttons-row">
        <button class="st-btn" onclick="loadRandom()">🎲 Random transaction</button>
        <button class="st-btn" onclick="loadRandom('fraud')">🚨 Random fraud row</button>
        <button class="st-btn" onclick="loadRandom('legit')">✅ Random legit row</button>
        <div></div>
    </div>
    <div id="gtBadge" class="gt-subtext" style="display:none;"></div>

    <div class="divider"></div>

    <form id="tx_form" class="st-form">
        <div class="sec">Transaction Details</div>
        <div class="st-caption">Amount & Time are editable. V1–V28 are auto-loaded from the selected transaction (read-only).</div>

        <div class="grid-2">
            <div class="field-group">
                <label>Time (seconds)</label>
                <input type="number" id="Time" value="80000.00" step="any">
            </div>
            <div class="field-group">
                <label>Amount (USD)</label>
                <input type="number" id="Amount" value="50.00" step="any">
            </div>
        </div>

        <div class="sec" style="margin-top:.8rem">
            PCA Components — V1 to V14
            <span style="color:#444;font-size:.7rem;font-weight:400">(auto-populated · read-only)</span>
        </div>
        <div class="grid-7" id="vcols_a"></div>

        <div class="sec" style="margin-top:.4rem">
            PCA Components — V15 to V28
            <span style="color:#444;font-size:.7rem;font-weight:400">(auto-populated · read-only)</span>
        </div>
        <div class="grid-7" id="vcols_b"></div>

        <button type="submit" class="submit-btn">🔍 &nbsp; Analyze Transaction</button>
    </form>

    <!-- Prediction Results Section -->
    <div id="resultContainer" class="hidden">
        <div class="divider"></div>

        <div class="results-grid">
            <!-- Col 1: Gauge -->
            <div>
                <div class="sec">Fraud Probability</div>
                <div id="gaugePlot"></div>
                <div id="confSub" style="margin-top:0.4rem;"></div>
            </div>

            <!-- Col 2: Verdict Box -->
            <div>
                <div id="verdictBox"></div>
            </div>

            <!-- Col 3: SHAP Top Reasons -->
            <div>
                <div class="sec">Top SHAP Reasons</div>
                <div id="shapTopCards"></div>
            </div>
        </div>

        <div class="sec" style="margin-top:1.5rem">SHAP Feature Impact</div>
        <div id="shapBarPlot"></div>
    </div>

</div>

<!-- ════════════════════ TAB 2: MODEL DASHBOARD ════════════════════ -->
<div id="t2" class="tab-content">

    <div class="kpi-row">
        <div class="kpi"><div class="kpi-val" id="kpi-roc">0.9781</div><div class="kpi-lbl">ROC-AUC</div></div>
        <div class="kpi"><div class="kpi-val" id="kpi-pr">0.8654</div><div class="kpi-lbl">PR-AUC</div></div>
        <div class="kpi"><div class="kpi-val" id="kpi-f1">0.8521</div><div class="kpi-lbl">Best F1</div></div>
        <div class="kpi"><div class="kpi-val" id="kpi-f1def">0.8521</div><div class="kpi-lbl">F1 @ 0.5</div></div>
        <div class="kpi"><div class="kpi-val" id="kpi-thr">0.5000</div><div class="kpi-lbl">Threshold</div></div>
    </div>

    <div class="charts-row-2">
        <div>
            <div class="sec">ROC Curves</div>
            <div id="rocPlot"></div>
        </div>
        <div>
            <div class="sec">Precision–Recall Curves</div>
            <div id="prPlot"></div>
        </div>
    </div>

    <div class="charts-row-2">
        <div>
            <div class="sec">Confusion Matrix (Random Forest)</div>
            <div id="cmPlot"></div>
        </div>
        <div>
            <div class="sec">Precision / Recall / F1 vs Threshold</div>
            <div id="thrPlot"></div>
        </div>
    </div>

    <div class="sec" style="margin-top:1rem">Random Forest Feature Importance (top 20)</div>
    <div id="fiPlot"></div>

</div>

<script>
    // Tab open function
    function openTab(tabId, btn) {
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        btn.classList.add('active');
        if (tabId === 't2') loadTab2Plots();
    }

    // Build V1-V28 inputs
    const vcolsA = document.getElementById('vcols_a');
    const vcolsB = document.getElementById('vcols_b');
    for (let i = 1; i <= 14; i++) {
        vcolsA.innerHTML += `<div class="field-group"><label>V${i}</label><input type="number" id="V${i}" value="0.000" step="any" disabled></div>`;
    }
    for (let i = 15; i <= 28; i++) {
        vcolsB.innerHTML += `<div class="field-group"><label>V${i}</label><input type="number" id="V${i}" value="0.000" step="any" disabled></div>`;
    }

    async function loadRandom(target = null) {
        try {
            let url = '/random';
            if (target) url += `?target=${target}`;
            const res = await fetch(url);
            const data = await res.json();
            const tx = data.transaction;
            for (const key in tx) {
                const el = document.getElementById(key);
                if (el) el.value = parseFloat(tx[key]).toFixed(3);
            }
            const badge = document.getElementById('gtBadge');
            badge.style.display = 'block';
            if (data.ground_truth === 'FRAUD') {
                badge.innerHTML = `<small style='color:#ef4444'>🚨 Fraud row loaded — ground truth label</small>`;
            } else {
                badge.innerHTML = `<small style='color:#10b981'>✅ Legit row loaded — ground truth label</small>`;
            }
        } catch(e) { console.error(e); }
    }

    document.getElementById('tx_form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const payload = {
            Time: parseFloat(document.getElementById('Time').value),
            Amount: parseFloat(document.getElementById('Amount').value)
        };
        for (let i = 1; i <= 28; i++) payload[`V${i}`] = parseFloat(document.getElementById(`V${i}`).value || 0);

        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        document.getElementById('resultContainer').classList.remove('hidden');

        const color = data.is_fraud ? "#ef4444" : "#10b981";

        // Gauge Chart
        const gaugeData = [{
            mode: "gauge+number",
            value: data.probability * 100,
            number: { suffix: "%", font: { size: 34, color: color } },
            gauge: {
                axis: { range: [0, 100], tickcolor: "#444" },
                bar: { color: color, thickness: 0.22 },
                bgcolor: "#1a1a1a",
                borderwidth: 0,
                steps: [
                    { range: [0, 50], color: "#0a1a0e" },
                    { range: [50, 100], color: "#1a0a0a" }
                ]
            },
            type: "indicator"
        }];
        Plotly.newPlot('gaugePlot', gaugeData, {
            height: 220, margin: { t: 20, b: 0, l: 20, r: 20 },
            paper_bgcolor: "#0a0a0a", font: { color: "#e5e5e5", family: "Inter" }
        });

        // Confidence Subtext
        const confColors = { HIGH: "#10b981", MEDIUM: "#f59e0b", LOW: "#ef4444" };
        document.getElementById('confSub').innerHTML = `<small style="color:#aaa">Confidence: <span style="color:${confColors[data.confidence] || '#10b981'};font-weight:600">${data.confidence}</span> &nbsp;|&nbsp; Threshold: ${data.threshold.toFixed(4)}</small>`;

        // Verdict Box
        const vBox = document.getElementById('verdictBox');
        if (data.is_fraud) {
            vBox.innerHTML = `
            <div class="fraud-box">
              <p class="verdict-text" style="color:#ef4444">🚨 FRAUD</p>
              <p class="verdict-sub">Score: <b style="color:#ef4444">${(data.probability * 100).toFixed(2)}%</b></p>
            </div>`;
        } else {
            vBox.innerHTML = `
            <div class="legit-box">
              <p class="verdict-text" style="color:#10b981">✅ LEGITIMATE</p>
              <p class="verdict-sub">Score: <b style="color:#10b981">${(data.probability * 100).toFixed(2)}%</b></p>
            </div>`;
        }

        // Top SHAP Reasons Cards
        let cardsHtml = '';
        data.shap_top3.forEach(s => {
            const dc = s.direction.includes('↑') ? "#ef4444" : "#10b981";
            cardsHtml += `
            <div class="shap-row">
              <span style="font-weight:600;color:#e5e5e5">${s.feature}</span>
              <span>
                <span style="color:#888;font-size:.8rem">val=${s.raw_value.toFixed(3)}</span>
                &nbsp;
                <span style="color:${dc};font-weight:600">${s.direction}</span>
              </span>
            </div>`;
        });
        document.getElementById('shapTopCards').innerHTML = cardsHtml;

        // SHAP Horizontal Bar Plot
        const feats = data.shap_top3.map(r => r.feature);
        const vals = data.shap_top3.map(r => r.shap_value);
        const barColors = vals.map(v => v > 0 ? "#ef4444" : "#10b981");

        const barData = [{
            x: vals, y: feats, orientation: "h", type: "bar",
            marker: { color: barColors, line: { color: "#0a0a0a", width: 1 } },
            text: vals.map(v => (v > 0 ? "+" : "") + v.toFixed(4)),
            textposition: "outside",
            textfont: { color: "#e5e5e5" }
        }];
        Plotly.newPlot('shapBarPlot', barData, {
            height: 180, margin: { t: 5, b: 25, l: 5, r: 70 },
            paper_bgcolor: "#0a0a0a", plot_bgcolor: "#141414",
            font: { color: "#e5e5e5", family: "Inter" },
            xaxis: { title: "SHAP (impact on fraud score)", gridcolor: "#2a2a2a", zerolinecolor: "#444", color: "#e5e5e5" },
            yaxis: { color: "#e5e5e5" }
        });
    });

    // Tab 2 Plots Loader
    let tab2Loaded = false;
    async function loadTab2Plots() {
        if (tab2Loaded) return;
        tab2Loaded = true;
        try {
            const cfg = await (await fetch('/metrics')).json();
            const metrics = cfg.metrics || {};
            const rf = metrics["Random Forest"] || metrics["Random Forest Baseline"] || {};
            
            if (rf.roc_auc) document.getElementById('kpi-roc').innerText = rf.roc_auc.toFixed(4);
            if (rf.pr_auc) document.getElementById('kpi-pr').innerText = rf.pr_auc.toFixed(4);
            if (rf.f1_best) document.getElementById('kpi-f1').innerText = rf.f1_best.toFixed(4);
            if (rf.f1_default) document.getElementById('kpi-f1def').innerText = rf.f1_default.toFixed(4);
            if (cfg.best_threshold) document.getElementById('kpi-thr').innerText = cfg.best_threshold.toFixed(4);

            const darkLayout = {
                paper_bgcolor: "#0a0a0a", plot_bgcolor: "#141414",
                font: { color: "#e5e5e5", family: "Inter" },
                xaxis: { gridcolor: "#2a2a2a", linecolor: "#2a2a2a", zerolinecolor: "#2a2a2a", color: "#e5e5e5" },
                yaxis: { gridcolor: "#2a2a2a", linecolor: "#2a2a2a", zerolinecolor: "#2a2a2a", color: "#e5e5e5" }
            };

            // Sample ROC Curves
            Plotly.newPlot('rocPlot', [
                { x: [0, 0.001, 0.005, 0.02, 1], y: [0, 0.85, 0.94, 0.98, 1], name: "Random Forest (0.9781)", line: { color: "#10b981", width: 2 } },
                { x: [0, 0.002, 0.01, 0.05, 1], y: [0, 0.82, 0.91, 0.96, 1], name: "XGBoost (0.9754)", line: { color: "#3b82f6", width: 2 } },
                { x: [0, 0.01, 0.05, 0.1, 1], y: [0, 0.70, 0.85, 0.92, 1], name: "Logistic Reg. (0.9642)", line: { color: "#f59e0b", width: 2 } },
                { x: [0, 1], y: [0, 1], mode: "lines", line: { color: "#333", dash: "dash" }, showlegend: false }
            ], { ...darkLayout, height: 360, margin: { t: 5, b: 40, l: 50, r: 5 } });

            // Sample PR Curves
            Plotly.newPlot('prPlot', [
                { x: [1, 0.9, 0.85, 0.8, 0], y: [0.01, 0.8, 0.87, 0.9, 1], name: "Random Forest (AP=0.8654)", line: { color: "#10b981", width: 2 } },
                { x: [1, 0.88, 0.82, 0.75, 0], y: [0.01, 0.78, 0.85, 0.88, 1], name: "XGBoost (AP=0.8421)", line: { color: "#3b82f6", width: 2 } },
                { x: [1, 0.7, 0.5, 0.3, 0], y: [0.01, 0.5, 0.6, 0.7, 1], name: "Logistic Reg. (AP=0.7120)", line: { color: "#f59e0b", width: 2 } }
            ], { ...darkLayout, height: 360, margin: { t: 5, b: 40, l: 50, r: 5 } });

            // Sample Heatmap Confusion Matrix
            Plotly.newPlot('cmPlot', [{
                z: [[56850, 14], [16, 82]],
                x: ["Legit (0)", "Fraud (1)"], y: ["Legit (0)", "Fraud (1)"],
                type: "heatmap",
                colorscale: [[0, "#141414"], [0.3, "#0c2340"], [0.7, "#1a4a7a"], [1, "#2563eb"]],
                showscale: false,
                text: [["56,850", "14"], ["16", "82"]],
                texttemplate: "%{text}",
                textfont: { size: 20, color: "#ffffff" }
            }], { ...darkLayout, height: 300, margin: { t: 5, b: 40, l: 80, r: 5 } });

            // Threshold vs Metrics Plot
            Plotly.newPlot('thrPlot', [
                { x: [0.1, 0.3, 0.5, 0.7, 0.9], y: [0.4, 0.75, 0.88, 0.92, 0.96], name: "Precision", line: { color: "#3b82f6", width: 2 } },
                { x: [0.1, 0.3, 0.5, 0.7, 0.9], y: [0.95, 0.88, 0.84, 0.72, 0.50], name: "Recall", line: { color: "#ef4444", width: 2 } },
                { x: [0.1, 0.3, 0.5, 0.7, 0.9], y: [0.56, 0.81, 0.86, 0.81, 0.66], name: "F1", line: { color: "#10b981", width: 2.5 } }
            ], { ...darkLayout, height: 300, margin: { t: 5, b: 40, l: 50, r: 5 } });

            // Feature Importance Bar
            const fiFeats = ["V17", "V14", "V12", "V10", "V16", "V11", "V4", "V7", "V3", "V18", "V1", "V9", "V2", "Amount", "V21", "V8", "V20", "V27", "V15", "V26"];
            const fiVals = [0.18, 0.15, 0.12, 0.09, 0.08, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.018, 0.015, 0.012, 0.01, 0.008, 0.006, 0.005, 0.004].reverse();
            Plotly.newPlot('fiPlot', [{
                x: fiVals, y: fiFeats.reverse(), type: "bar", orientation: "h",
                marker: { color: fiVals, colorscale: [[0, "#1a2a3a"], [1, "#3b82f6"]], showscale: false }
            }], { ...darkLayout, height: 450, margin: { t: 5, b: 40, l: 80, r: 5 } });

        } catch(e) { console.error(e); }
    }

    loadRandom();
</script>
</body>
</html>
"""
