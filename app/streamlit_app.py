"""
streamlit_app.py — FraudSentinel
Black/neutral theme | No purple | Red=fraud, Green=legit, Blue=accent
"""

import sys, os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score, confusion_matrix, f1_score)

st.set_page_config(
    page_title="FraudSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — black/neutral, no purple ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; }

.stApp { background:#0a0a0a; color:#e5e5e5; }

/* header */
.hero {
    background:#141414; border:1px solid #2a2a2a;
    border-radius:12px; padding:1.8rem 2rem; margin-bottom:1.5rem;
}
.hero h1 { font-size:2rem; font-weight:700; margin:0; color:#e5e5e5; letter-spacing:-.02em; }
.hero p  { color:#888; margin:.3rem 0 0; font-size:.95rem; }

/* metric cards */
.kpi {
    background:#141414; border:1px solid #2a2a2a;
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
}
.kpi-val { font-size:1.8rem; font-weight:700; color:#3b82f6; }
.kpi-lbl { font-size:.72rem; color:#888; text-transform:uppercase;
           letter-spacing:.08em; margin-top:.25rem; }

/* verdict boxes */
.fraud-box {
    background:#1a0a0a; border:1.5px solid #ef4444;
    border-radius:12px; padding:1.4rem; text-align:center;
}
.legit-box {
    background:#0a1a0e; border:1.5px solid #10b981;
    border-radius:12px; padding:1.4rem; text-align:center;
}
.verdict-text { font-size:1.6rem; font-weight:700; margin:0; }
.sub { color:#888; font-size:.85rem; margin-top:.4rem; }

/* SHAP cards */
.shap-row {
    background:#141414; border:1px solid #2a2a2a;
    border-radius:8px; padding:.7rem 1rem; margin:.35rem 0;
    display:flex; justify-content:space-between; align-items:center;
}

/* section heading */
.sec { font-size:.85rem; font-weight:700; color:#60a5fa !important;
       text-transform:uppercase; letter-spacing:.1em;
       border-bottom:1px solid #2a2a2a; padding-bottom:.4rem;
       margin-bottom:.9rem; }

/* Force widget label visibility (V1..V28, Time, Amount) */
label[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] p,
label p {
    color: #e5e5e5 !important;
    -webkit-text-fill-color: #e5e5e5 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    opacity: 1 !important;
}

/* Captions and subtext */
div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] p,
.stCaption {
    color: #a3a3a3 !important;
    font-size: 0.82rem !important;
}

/* form container */
div[data-testid="stForm"] {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
}

/* form inputs */
div[data-testid="stNumberInput"] input {
    background:#0a0a0a !important; border:1px solid #333333 !important;
    color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;
    border-radius:6px !important; font-weight:600 !important;
}
div[data-testid="stNumberInput"] input:disabled {
    background:#161616 !important; border:1px solid #2a2a2a !important;
    color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;
    opacity:1 !important; font-weight:600 !important;
}

/* Stepper buttons */
div[data-testid="stNumberInput"] button {
    background:#222222 !important; color:#ffffff !important;
    border:1px solid #333333 !important;
}

div[data-testid="stTabs"] button { color:#888; font-weight:500; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#e5e5e5; }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#3b82f6 !important; }

/* buttons */
.stButton > button {
    background:#141414 !important; border:1px solid #2a2a2a !important;
    color:#e5e5e5 !important; border-radius:8px !important;
}
.stButton > button:hover {
    border-color:#3b82f6 !important; color:#3b82f6 !important;
}
div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
    background:#3b82f6 !important; border:none !important;
    color:#fff !important; font-weight:600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")

DARK = dict(
    paper_bgcolor="#0a0a0a",
    plot_bgcolor="#141414",
    font=dict(color="#e5e5e5", family="Inter"),
    xaxis=dict(gridcolor="#2a2a2a", linecolor="#2a2a2a", zerolinecolor="#2a2a2a"),
    yaxis=dict(gridcolor="#2a2a2a", linecolor="#2a2a2a", zerolinecolor="#2a2a2a"),
)

MODEL_COLORS = {
    "XGBoost":          "#3b82f6",
    "Logistic Reg.":    "#f59e0b",
    "Random Forest":    "#10b981",
    "Isolation Forest": "#888888",
}


# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_predictor():
    try:
        from predict import FraudPredictor
        return FraudPredictor(MODELS_DIR)
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def load_eval():
    try:
        y_test     = np.load(os.path.join(MODELS_DIR, "y_test.npy"))
        y_xgb      = np.load(os.path.join(MODELS_DIR, "y_prob_xgb.npy"))
        y_lr       = np.load(os.path.join(MODELS_DIR, "y_prob_lr.npy"))
        y_rf       = np.load(os.path.join(MODELS_DIR, "y_prob_rf.npy"))
        y_if       = np.load(os.path.join(MODELS_DIR, "y_prob_if.npy"))
        with open(os.path.join(MODELS_DIR, "config.json")) as f:
            cfg = json.load(f)
        return y_test, y_xgb, y_lr, y_rf, y_if, cfg
    except:
        return None

@st.cache_data(show_spinner="Loading test transactions…")
def load_test_rows():
    """Load sample test rows (supports both cloud deployment via test_sample.csv and local full dataset)."""
    sample_path = os.path.join(MODELS_DIR, "test_sample.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        return df.drop(columns=["Class"]).reset_index(drop=True), df["Class"].reset_index(drop=True)
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        X  = df.drop(columns=["Class"])
        y  = df["Class"]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        return X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    return None, None


# ── helper charts ─────────────────────────────────────────────────────────────
def gauge(prob, is_fraud=False):
    color = "#ef4444" if is_fraud else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 34, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#444"},
            "bar":  {"color": color, "thickness": 0.22},
            "bgcolor": "#1a1a1a",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  50],  "color": "#0a1a0e"},
                {"range": [50, 100], "color": "#1a0a0a"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=220, margin=dict(t=20, b=0, l=20, r=20),
        paper_bgcolor="#0a0a0a", font=dict(color="#e5e5e5", family="Inter"),
    )
    return fig

def shap_bar(top3):
    feats  = [r["feature"] for r in top3]
    vals   = [r["shap_value"] for r in top3]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(color=colors, line=dict(color="#0a0a0a", width=1)),
        text=[f"{v:+.4f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#e5e5e5"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=180, margin=dict(t=5, b=5, l=5, r=70),
        xaxis_title="SHAP (impact on fraud score)",
        xaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#444",
                   zerolinewidth=1.5, color="#e5e5e5"),
        yaxis=dict(color="#e5e5e5"),
        **{k: v for k, v in DARK.items() if k in ("paper_bgcolor", "plot_bgcolor", "font")},
    )
    return fig


# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛡️ FraudSentinel</h1>
  <p>Credit Card Fraud Detection · Random Forest + SHAP · End-to-end verified pipeline</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍  Transaction Check", "📊  Model Dashboard"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Transaction Check
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    predictor = load_predictor()
    if predictor is None:
        st.warning("⚠️ Model not found. Run `python src/train.py` first.")
        st.stop()

    feat_names = predictor.feature_names

    # ── Load random transaction helper ────────────────────────────────────────
    X_test_df, y_test_series = load_test_rows()

    def set_loaded_row(row, label):
        for fn in feat_names:
            st.session_state[f"fi_{fn}"] = float(row[fn])
        st.session_state["_loaded_label"] = int(label)

    # ── Auto-initialize with a real test transaction on first visit ─────────
    if "_initialized" not in st.session_state:
        if X_test_df is not None and len(X_test_df) > 0:
            set_loaded_row(X_test_df.iloc[0], y_test_series.iloc[0])
        else:
            defaults = {"Time": 80000.0, "Amount": 50.0}
            for fn in feat_names:
                st.session_state[f"fi_{fn}"] = defaults.get(fn, 0.0)
        st.session_state["_initialized"] = True

    col_load1, col_load2, col_load3, _ = st.columns([1, 1, 1, 3])
    with col_load1:
        if st.button("🎲 Random transaction", use_container_width=True):
            if X_test_df is not None:
                # 50/50 so demos aren't always legit (99.8% of raw data is legit)
                if np.random.random() < 0.5:
                    pool = y_test_series[y_test_series == 1].index.tolist()
                    lbl  = 1
                else:
                    pool = y_test_series[y_test_series == 0].index.tolist()
                    lbl  = 0
                set_loaded_row(X_test_df.iloc[np.random.choice(pool)], lbl)
    with col_load2:
        if st.button("🚨 Random fraud row", use_container_width=True):
            if X_test_df is not None:
                fraud_idx = y_test_series[y_test_series == 1].index.tolist()
                idx = np.random.choice(fraud_idx)
                set_loaded_row(X_test_df.iloc[idx], 1)
    with col_load3:
        if st.button("✅ Random legit row", use_container_width=True):
            if X_test_df is not None:
                legit_idx = y_test_series[y_test_series == 0].index.tolist()
                idx = np.random.choice(legit_idx)
                set_loaded_row(X_test_df.iloc[idx], 0)

    if "_loaded_label" in st.session_state:
        lbl = st.session_state["_loaded_label"]
        tag = "🚨 Fraud row loaded" if lbl == 1 else "✅ Legit row loaded"
        clr = "#ef4444" if lbl == 1 else "#10b981"
        st.markdown(
            f"<small style='color:{clr}'>{tag} — ground truth label</small>",
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("tx_form"):
        st.markdown('<div class="sec">Transaction Details</div>', unsafe_allow_html=True)
        st.caption("Amount & Time are editable. V1–V28 are auto-loaded from the selected transaction (read-only).")

        ca, cb = st.columns(2)
        with ca:
            time_val   = st.number_input("Time (seconds)", key="fi_Time",
                                          min_value=0.0, format="%.2f")
        with cb:
            amount_val = st.number_input("Amount (USD)",   key="fi_Amount",
                                          min_value=0.0, format="%.4f")

        # V1–V28: read-only display, values come from loaded row / session state
        st.markdown('<div class="sec" style="margin-top:.8rem">PCA Components — V1 to V14 <span style="color:#444;font-size:.7rem;font-weight:400">(auto-populated · read-only)</span></div>', unsafe_allow_html=True)
        vcols_a = st.columns(7)
        for i in range(1, 15):
            vcols_a[(i-1) % 7].number_input(
                f"V{i}", key=f"fi_V{i}", format="%.3f",
                disabled=True, label_visibility="visible")

        st.markdown('<div class="sec" style="margin-top:.4rem">PCA Components — V15 to V28 <span style="color:#444;font-size:.7rem;font-weight:400">(auto-populated · read-only)</span></div>', unsafe_allow_html=True)
        vcols_b = st.columns(7)
        for i in range(15, 29):
            vcols_b[(i-15) % 7].number_input(
                f"V{i}", key=f"fi_V{i}", format="%.3f",
                disabled=True, label_visibility="visible")

        submitted = st.form_submit_button(
            "🔍  Analyze Transaction", use_container_width=True, type="primary")

    # ── Result ────────────────────────────────────────────────────────────────
    if submitted:
        v_vals = {f"V{i}": st.session_state.get(f"fi_V{i}", 0.0) for i in range(1, 29)}
        tx = {"Time": time_val, "Amount": amount_val, **v_vals}
        with st.spinner("Scoring…"):
            res = predictor.predict(tx)

        st.markdown("---")
        r1, r2, r3 = st.columns([1, 1.1, 1])

        with r1:
            st.markdown('<div class="sec">Fraud Probability</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(gauge(res["probability"], res["is_fraud"]), use_container_width=True)
            conf_c = {"HIGH": "#10b981", "MEDIUM": "#f59e0b", "LOW": "#ef4444"}
            ck = res["confidence"]
            st.markdown(
                f"<small>Confidence: "
                f"<span style='color:{conf_c[ck]};font-weight:600'>{ck}</span>"
                f" &nbsp;|&nbsp; Threshold: {res['threshold']:.4f}</small>",
                unsafe_allow_html=True)

        with r2:
            if res["is_fraud"]:
                st.markdown(f"""
                <div class="fraud-box">
                  <p class="verdict-text" style="color:#ef4444">🚨 FRAUD</p>
                  <p class="sub">Score: <b style="color:#ef4444">{res['probability']*100:.2f}%</b></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-box">
                  <p class="verdict-text" style="color:#10b981">✅ LEGITIMATE</p>
                  <p class="sub">Score: <b style="color:#10b981">{res['probability']*100:.2f}%</b></p>
                </div>""", unsafe_allow_html=True)

        with r3:
            st.markdown('<div class="sec">Top SHAP Reasons</div>',
                        unsafe_allow_html=True)
            for s in res["shap_top3"]:
                dc = "#ef4444" if "↑" in s["direction"] else "#10b981"
                st.markdown(f"""
                <div class="shap-row">
                  <span style="font-weight:600;color:#e5e5e5">{s['feature']}</span>
                  <span>
                    <span style="color:#888;font-size:.8rem">val={s['raw_value']:.3f}</span>
                    &nbsp;
                    <span style="color:{dc};font-weight:600">{s['direction']}</span>
                  </span>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec" style="margin-top:1rem">SHAP Feature Impact</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(shap_bar(res["shap_top3"]), use_container_width=True)

        # ── Read-only V1–V28 panel ────────────────────────────────────────────
        with st.expander("📋 Show raw feature values used (V1–V28)", expanded=False):
            v_items = [f"<b>{k}</b>: {v:.4f}" for k, v in v_vals.items()]
            c1, c2, c3, c4 = st.columns(4)
            cols = [c1, c2, c3, c4]
            for idx, item in enumerate(v_items):
                cols[idx % 4].markdown(f"<small style='color:#aaa'>{item}</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    data = load_eval()
    if data is None:
        st.warning("⚠️ Run `python src/train.py` first to generate model artifacts.")
        st.stop()

    y_test, y_xgb, y_lr, y_rf, y_if, cfg = data
    threshold = cfg["best_threshold"]
    metrics   = cfg.get("metrics", {})
    rf_m      = metrics.get("Random Forest", metrics.get("Random Forest Baseline", {}))

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    for col, val, lbl in [
        (k1, rf_m.get("roc_auc","—"), "ROC-AUC"),
        (k2, rf_m.get("pr_auc","—"),  "PR-AUC"),
        (k3, rf_m.get("f1_best","—"), "Best F1"),
        (k4, rf_m.get("f1_default","—"), "F1 @ 0.5"),
        (k5, f"{threshold:.4f}",        "Threshold"),
    ]:
        col.markdown(f"""
        <div class="kpi">
          <div class="kpi-val">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC + PR ──────────────────────────────────────────────────────────────
    d1, d2 = st.columns(2)
    mp = [("Random Forest", y_rf), ("XGBoost", y_xgb), ("Logistic Reg.", y_lr), ("Isolation Forest", y_if)]

    with d1:
        st.markdown('<div class="sec">ROC Curves</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for name, yp in mp:
            fpr, tpr, _ = roc_curve(y_test, yp)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} ({auc(fpr,tpr):.4f})",
                line=dict(color=MODEL_COLORS[name], width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            line=dict(color="#333", dash="dash", width=1), showlegend=False))
        fig.update_layout(xaxis_title="FPR", yaxis_title="TPR",
            legend=dict(bgcolor="rgba(0,0,0,0)", x=.45, y=.05),
            height=360, margin=dict(t=5,b=40,l=50,r=5), **DARK)
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        st.markdown('<div class="sec">Precision–Recall Curves</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for name, yp in mp:
            p, r, _ = precision_recall_curve(y_test, yp)
            ap = average_precision_score(y_test, yp)
            fig2.add_trace(go.Scatter(
                x=r, y=p, mode="lines", name=f"{name} (AP={ap:.4f})",
                line=dict(color=MODEL_COLORS[name], width=2)))
        fig2.add_hline(y=float(y_test.mean()), line_dash="dash",
                       line_color="#444", line_width=1)
        fig2.update_layout(xaxis_title="Recall", yaxis_title="Precision",
            legend=dict(bgcolor="rgba(0,0,0,0)", x=.3, y=.95),
            height=360, margin=dict(t=5,b=40,l=50,r=5), **DARK)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Confusion Matrix + Threshold curve ───────────────────────────────────
    d3, d4 = st.columns(2)
    with d3:
        st.markdown('<div class="sec">Confusion Matrix (Random Forest)</div>',
                    unsafe_allow_html=True)
        y_pred = (y_rf >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Legit (0)", "Fraud (1)"]

        cm_norm = cm / cm.max()
        fig3 = go.Figure(go.Heatmap(
            z=cm_norm,
            x=labels, y=labels,
            colorscale=[
                [0.0, "#141414"],
                [0.3, "#0c2340"],
                [0.7, "#1a4a7a"],
                [1.0, "#2563eb"],
            ],
            showscale=False,
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=22, color="#ffffff"),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
        ))
        fig3.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            height=300, margin=dict(t=5,b=40,l=80,r=5), **DARK)
        st.plotly_chart(fig3, use_container_width=True)

    with d4:
        st.markdown('<div class="sec">Precision / Recall / F1 vs Threshold</div>',
                    unsafe_allow_html=True)
        prec, rec, thrs = precision_recall_curve(y_test, y_rf)
        f1s = 2*prec[:-1]*rec[:-1] / (prec[:-1]+rec[:-1]+1e-9)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=thrs, y=prec[:-1], mode="lines",
            name="Precision", line=dict(color="#3b82f6", width=2)))
        fig4.add_trace(go.Scatter(x=thrs, y=rec[:-1], mode="lines",
            name="Recall",    line=dict(color="#ef4444", width=2)))
        fig4.add_trace(go.Scatter(x=thrs, y=f1s, mode="lines",
            name="F1",        line=dict(color="#10b981", width=2.5)))
        fig4.add_vline(x=threshold, line_dash="dash", line_color="#888",
                       line_width=1.5,
                       annotation_text=f"thr={threshold:.4f}",
                       annotation_font_color="#888")
        fig4.update_layout(xaxis_title="Threshold",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=300, margin=dict(t=5,b=40,l=50,r=5), **DARK)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown('<div class="sec">Random Forest Feature Importance (top 20)</div>',
                unsafe_allow_html=True)
    try:
        pipe = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
        clf  = pipe.named_steps["clf"]
        imps = clf.feature_importances_
        feat_names = cfg["feature_names"]
        fi = pd.DataFrame({"f": feat_names, "i": imps}).sort_values("i").tail(20)
        fig5 = go.Figure(go.Bar(
            x=fi["i"], y=fi["f"], orientation="h",
            marker=dict(
                color=fi["i"],
                colorscale=[[0,"#1a2a3a"],[1,"#3b82f6"]],
                showscale=False,
                line=dict(color="#0a0a0a", width=0.5),
            ),
            hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
        ))
        fig5.update_layout(xaxis_title="Importance",
            height=480, margin=dict(t=5,b=40,l=80,r=5), **DARK)
        st.plotly_chart(fig5, use_container_width=True)
    except Exception as e:
        st.info(f"Feature importance unavailable: {e}")

    # ── Model comparison table ────────────────────────────────────────────────
    st.markdown('<div class="sec">Model Comparison</div>', unsafe_allow_html=True)
    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model":           name,
            "ROC-AUC":         m.get("roc_auc","—"),
            "PR-AUC":          m.get("pr_auc","—"),
            "F1 @ 0.5":        m.get("f1_default","—"),
            "F1 @ Best Thr":   m.get("f1_best","—"),
            "Best Threshold":  m.get("best_threshold","—"),
        })
    if rows:
        df_cmp = pd.DataFrame(rows).set_index("Model")
        st.dataframe(
            df_cmp.style.highlight_max(
                subset=["ROC-AUC","PR-AUC","F1 @ Best Thr"],
                color="#0c2340"),
            use_container_width=True)
