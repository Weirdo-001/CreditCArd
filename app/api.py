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
    description="Production REST API for Real-Time Credit Card Fraud Detection using XGBoost + SHAP Explainability",
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
    """Serve rich embedded high-performance REST dashboard UI matching Streamlit features."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FraudSentinel — Production REST Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0a0a0a; color: #e5e5e5; font-family: 'Inter', sans-serif; padding: 1.5rem; }
        .container { max-width: 1200px; margin: 0 auto; }
        .hero { background: #141414; border: 1px solid #2a2a2a; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.2rem; display: flex; justify-content: space-between; align-items: center; }
        .hero h1 { font-size: 1.7rem; font-weight: 700; color: #fff; }
        .hero p { color: #888; font-size: 0.85rem; margin-top: 0.2rem; }
        .btn { background: #1f2937; color: #fff; border: 1px solid #374151; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.85rem; text-decoration: none; display: inline-flex; align-items: center; gap: 0.4rem; }
        .btn:hover { background: #2563eb; border-color: #2563eb; }
        .btn-fraud { background: #3f1212; border-color: #7f1d1d; color: #fca5a5; }
        .btn-fraud:hover { background: #991b1b; border-color: #ef4444; color: #fff; }
        .btn-legit { background: #064e3b; border-color: #065f46; color: #6ee7b7; }
        .btn-legit:hover { background: #047857; border-color: #10b981; color: #fff; }
        .btn-primary { background: #3b82f6; border-color: #3b82f6; width: 100%; padding: 0.8rem; font-size: 1rem; margin-top: 1rem; }
        .card { background: #141414; border: 1px solid #2a2a2a; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.2rem; }
        .sec-title { font-size: 0.8rem; font-weight: 700; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid #2a2a2a; padding-bottom: 0.4rem; margin-bottom: 0.8rem; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .grid-7 { display: grid; grid-template-columns: repeat(7, 1fr); gap: 0.5rem; margin-bottom: 0.8rem; }
        .grid-3 { display: grid; grid-template-columns: 1fr 1.2fr 1fr; gap: 1rem; align-items: center; }
        .input-group { margin-bottom: 0.6rem; }
        .input-group label { display: block; font-size: 0.72rem; color: #a3a3a3; font-weight: 600; margin-bottom: 0.2rem; }
        .input-group input { width: 100%; background: #0a0a0a; border: 1px solid #333; color: #fff; padding: 0.45rem; border-radius: 6px; font-size: 0.82rem; }
        .input-group input:disabled { background: #181818; color: #e5e5e5; border-color: #2a2a2a; }
        .result-box { display: none; padding: 1.5rem; border-radius: 12px; text-align: center; }
        .result-fraud { background: #1a0a0a; border: 2px solid #ef4444; color: #ef4444; }
        .result-legit { background: #0a1a0e; border: 2px solid #10b981; color: #10b981; }
        .result-title { font-size: 1.7rem; font-weight: 700; }
        .shap-item { background: #181818; border: 1px solid #2a2a2a; padding: 0.5rem 0.8rem; border-radius: 6px; margin-top: 0.4rem; display: flex; justify-content: space-between; font-size: 0.8rem; }
        .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.8rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <div>
                <h1>🛡️ FraudSentinel REST Portal</h1>
                <p>Production Microservice Engine · FastAPI + Random Forest + SHAP</p>
            </div>
            <div style="display:flex; gap:0.6rem;">
                <a href="/docs" target="_blank" class="btn">⚡ Swagger Docs (/docs)</a>
            </div>
        </div>

        <div class="card">
            <div class="sec-title">Quick Payload Generators</div>
            <div style="display: flex; gap: 0.8rem; margin-bottom: 1rem; align-items:center;">
                <button class="btn" onclick="loadRandom()">🎲 Random transaction</button>
                <button class="btn btn-fraud" onclick="loadRandom('fraud')">🚨 Random fraud row</button>
                <button class="btn btn-legit" onclick="loadRandom('legit')">✅ Random legit row</button>
                <span id="gtBadge" class="badge" style="display:none; background:#222; color:#fff;"></span>
            </div>

            <form id="apiForm">
                <div class="grid-2">
                    <div class="input-group"><label>Time (seconds)</label><input type="number" id="Time" value="80000" step="any"></div>
                    <div class="input-group"><label>Amount (USD)</label><input type="number" id="Amount" value="74.20" step="any"></div>
                </div>

                <div class="sec-title" style="margin-top:0.6rem">PCA Components (V1–V14)</div>
                <div class="grid-7" id="v_container_a"></div>

                <div class="sec-title">PCA Components (V15–V28)</div>
                <div class="grid-7" id="v_container_b"></div>

                <button type="submit" class="btn btn-primary">🚀 Send POST Request to /predict</button>
            </form>
        </div>

        <div id="outputSection" class="card" style="display:none;">
            <div class="sec-title">Inference Results & Explainability</div>
            <div class="grid-3">
                <div>
                    <div id="gaugePlot"></div>
                </div>
                <div>
                    <div id="resultBox" class="result-box">
                        <div id="resVerdict" class="result-title"></div>
                        <div id="resSub" style="margin-top:0.4rem; font-size:0.85rem; color:#aaa;"></div>
                    </div>
                </div>
                <div>
                    <div id="shapBox"></div>
                </div>
            </div>
            <div style="margin-top:1.2rem;">
                <div id="shapBarPlot"></div>
            </div>
        </div>
    </div>

    <script>
        const vContainerA = document.getElementById('v_container_a');
        const vContainerB = document.getElementById('v_container_b');

        for(let i=1; i<=14; i++) {
            vContainerA.innerHTML += `<div class="input-group"><label>V${i}</label><input type="number" id="V${i}" value="0.000" step="any" disabled></div>`;
        }
        for(let i=15; i<=28; i++) {
            vContainerB.innerHTML += `<div class="input-group"><label>V${i}</label><input type="number" id="V${i}" value="0.000" step="any" disabled></div>`;
        }

        async function loadRandom(target=null) {
            try {
                let url = '/random';
                if(target) url += `?target=${target}`;
                const res = await fetch(url);
                const data = await res.json();
                const tx = data.transaction;
                for (const key in tx) {
                    const el = document.getElementById(key);
                    if (el) el.value = parseFloat(tx[key]).toFixed(3);
                }
                const badge = document.getElementById('gtBadge');
                badge.style.display = 'inline-block';
                if(data.ground_truth === 'FRAUD') {
                    badge.style.background = '#3f1212';
                    badge.style.color = '#fca5a5';
                    badge.innerText = 'Ground Truth: FRAUD (Class=1)';
                } else {
                    badge.style.background = '#064e3b';
                    badge.style.color = '#6ee7b7';
                    badge.innerText = 'Ground Truth: LEGIT (Class=0)';
                }
            } catch(e) { console.error(e); }
        }

        document.getElementById('apiForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {};
            payload['Time'] = parseFloat(document.getElementById('Time').value);
            payload['Amount'] = parseFloat(document.getElementById('Amount').value);
            for(let i=1; i<=28; i++) {
                payload[`V${i}`] = parseFloat(document.getElementById(`V${i}`).value || 0);
            }

            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            document.getElementById('outputSection').style.display = 'block';
            const box = document.getElementById('resultBox');
            box.style.display = 'block';
            const mainColor = data.is_fraud ? '#ef4444' : '#10b981';

            if (data.is_fraud) {
                box.className = 'result-box result-fraud';
                document.getElementById('resVerdict').innerText = '🚨 FRAUD DETECTED';
            } else {
                box.className = 'result-box result-legit';
                document.getElementById('resVerdict').innerText = '✅ LEGITIMATE TRANSACTION';
            }
            document.getElementById('resSub').innerText = `Fraud Score: ${(data.probability * 100).toFixed(2)}% | Threshold: ${data.threshold} | Confidence: ${data.confidence}`;

            // Top SHAP Box
            let shapHtml = '<div style="font-weight:700; color:#fff; margin-bottom:0.4rem; font-size:0.8rem;">TOP SHAP RISK REASONS:</div>';
            data.shap_top3.forEach(item => {
                shapHtml += `<div class="shap-item"><span><strong>${item.feature}</strong> (${item.raw_value})</span><span style="color:${item.shap_value > 0 ? '#ef4444' : '#10b981'}">${item.direction} (${item.shap_value > 0 ? '+' : ''}${item.shap_value})</span></div>`;
            });
            document.getElementById('shapBox').innerHTML = shapHtml;

            // Plotly Gauge Chart
            const gaugeData = [{
                type: "indicator",
                mode: "gauge+number",
                value: data.probability * 100,
                number: { suffix: "%", font: { size: 30, color: mainColor } },
                gauge: {
                    axis: { range: [0, 100], tickcolor: "#444" },
                    bar: { color: mainColor, thickness: 0.25 },
                    bgcolor: "#141414",
                    steps: [
                        { range: [0, 50], color: "#0a1a0e" },
                        { range: [50, 100], color: "#1a0a0a" }
                    ]
                }
            }];
            Plotly.newPlot('gaugePlot', gaugeData, {
                height: 200, margin: { t: 10, b: 10, l: 20, r: 20 },
                paper_bgcolor: "#141414", font: { color: "#e5e5e5", family: "Inter" }
            });

            // Plotly SHAP Horizontal Bar Chart
            const shapFeats = data.shap_top3.map(r => r.feature).reverse();
            const shapVals = data.shap_top3.map(r => r.shap_value).reverse();
            const barColors = shapVals.map(v => v > 0 ? '#ef4444' : '#10b981');
            const shapBarData = [{
                type: 'bar',
                x: shapVals,
                y: shapFeats,
                orientation: 'h',
                marker: { color: barColors },
                text: shapVals.map(v => (v > 0 ? '+' : '') + v.toFixed(4)),
                textposition: 'outside',
                textfont: { color: '#e5e5e5' }
            }];
            Plotly.newPlot('shapBarPlot', shapBarData, {
                height: 180, margin: { t: 20, b: 20, l: 60, r: 60 },
                title: { text: 'SHAP Feature Impact (Pushing Score higher or lower)', font: { size: 12, color: '#aaa' } },
                paper_bgcolor: "#141414", plot_bgcolor: "#141414",
                xaxis: { gridcolor: "#2a2a2a", zerolinecolor: "#555", color: "#e5e5e5" },
                yaxis: { color: "#e5e5e5" }
            });
        });

        loadRandom();
    </script>
</body>
</html>
"""
