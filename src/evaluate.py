"""
evaluate.py
───────────
Generates all evaluation plots: ROC, PR curve, Confusion Matrix, Feature Importance.
Loads pre-saved test probabilities from /models/ so it doesn't need to re-run training.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import os
import joblib
import shap

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score
)

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#3d4166",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d45",
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
})
COLORS = {
    "xgb": "#7c3aed",
    "lr":  "#06b6d4",
    "rf":  "#10b981",
    "if":  "#f59e0b",
    "fraud":  "#ef4444",
    "legit":  "#3b82f6",
}


def load_artifacts():
    y_test    = np.load(os.path.join(MODELS_DIR, "y_test.npy"))
    y_prob_xgb = np.load(os.path.join(MODELS_DIR, "y_prob_xgb.npy"))
    y_prob_lr  = np.load(os.path.join(MODELS_DIR, "y_prob_lr.npy"))
    y_prob_rf  = np.load(os.path.join(MODELS_DIR, "y_prob_rf.npy"))
    y_prob_if  = np.load(os.path.join(MODELS_DIR, "y_prob_if.npy"))
    with open(os.path.join(MODELS_DIR, "config.json")) as f:
        config = json.load(f)
    return y_test, y_prob_xgb, y_prob_lr, y_prob_rf, y_prob_if, config


# ── individual plots ───────────────────────────────────────────────────────────

def plot_roc_curves(y_test, y_prob_xgb, y_prob_lr, y_prob_rf, y_prob_if):
    fig, ax = plt.subplots(figsize=(8, 6))
    models_probs = [
        ("XGBoost",          y_prob_xgb, COLORS["xgb"]),
        ("Logistic Reg.",    y_prob_lr,  COLORS["lr"]),
        ("Random Forest",    y_prob_rf,  COLORS["rf"]),
        ("Isolation Forest", y_prob_if,  COLORS["if"]),
    ]
    for name, probs, color in models_probs:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — All Models", fontsize=14, pad=12, color="white")
    ax.legend(loc="lower right", facecolor="#1a1d2e", edgecolor="#3d4166")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")
    return path


def plot_pr_curves(y_test, y_prob_xgb, y_prob_lr, y_prob_rf, y_prob_if):
    fig, ax = plt.subplots(figsize=(8, 6))
    models_probs = [
        ("XGBoost",          y_prob_xgb, COLORS["xgb"]),
        ("Logistic Reg.",    y_prob_lr,  COLORS["lr"]),
        ("Random Forest",    y_prob_rf,  COLORS["rf"]),
        ("Isolation Forest", y_prob_if,  COLORS["if"]),
    ]
    for name, probs, color in models_probs:
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{name} (AP={pr_auc:.4f})")
    baseline = y_test.mean()
    ax.axhline(baseline, color="w", linestyle="--", lw=1, alpha=0.5,
               label=f"Baseline (random) = {baseline:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve — All Models", fontsize=14, pad=12, color="white")
    ax.legend(loc="upper right", facecolor="#1a1d2e", edgecolor="#3d4166")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "pr_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")
    return path


def plot_confusion_matrix(y_test, y_prob, threshold: float, model_name: str = "Random Forest"):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        linewidths=2, linecolor="#0f1117",
        cbar=True, ax=ax,
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix ({model_name}, thr={threshold:.3f})",
                 fontsize=13, pad=12, color="white")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")
    print("\n" + classification_report(y_test, y_pred,
          target_names=["Legit", "Fraud"], zero_division=0))
    return path


def plot_threshold_vs_f1(y_test, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1s)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(thresholds, precisions[:-1], color=COLORS["legit"],  lw=2, label="Precision")
    ax.plot(thresholds, recalls[:-1],    color=COLORS["fraud"],  lw=2, label="Recall")
    ax.plot(thresholds, f1s,             color=COLORS["rf"],     lw=2, label="F1")
    ax.axvline(thresholds[best_idx], color="white", linestyle="--",
               lw=1.5, label=f"Best thr={thresholds[best_idx]:.3f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_title("Precision / Recall / F1 vs Threshold (Random Forest)",
                 fontsize=13, pad=12, color="white")
    ax.legend(facecolor="#1a1d2e", edgecolor="#3d4166")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "threshold_f1.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")
    return path


def plot_feature_importance(model, feature_names, top_n: int = 20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.Greens(np.linspace(0.4, 1.0, len(indices)))
    bars = ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color=colors, edgecolor="#0f1117", height=0.7,
    )
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
                 fontsize=13, pad=12, color="white")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved → {path}")
    return path


def generate_shap_plots(model, X_test_sample):
    """SHAP global + waterfall for first fraud transaction found."""
    print("[evaluate] Computing SHAP values (this may take a minute)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # ── Global summary ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test_sample, show=False,
                      plot_type="bar", color=COLORS["rf"])
    plt.title("SHAP — Global Feature Importance", color="white", pad=10)
    plt.tight_layout()
    path_global = os.path.join(REPORTS_DIR, "shap_global.png")
    plt.savefig(path_global, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()

    # ── Beeswarm ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title("SHAP — Feature Impact Beeswarm", color="white", pad=10)
    plt.tight_layout()
    path_beeswarm = os.path.join(REPORTS_DIR, "shap_beeswarm.png")
    plt.savefig(path_beeswarm, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()

    print(f"[evaluate] SHAP plots saved to {REPORTS_DIR}")
    return shap_values, path_global, path_beeswarm


def plot_financial_impact_matrix(y_test, y_prob, threshold: float, avg_fraud_value: float = 125.0, friction_cost: float = 15.0, missed_cost: float = 150.0):
    """Translates ML metrics into financial dollar value & risk metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    prevented_fraud_val = tp * avg_fraud_value
    false_decline_cost  = fp * friction_cost
    missed_fraud_loss   = fn * missed_cost
    net_value_saved     = prevented_fraud_val - false_decline_cost

    fig, ax = plt.subplots(figsize=(8, 4.5))
    categories = ["Fraud Loss Prevented", "Customer Friction Cost", "Missed Fraud Loss", "Net Value Saved"]
    values = [prevented_fraud_val, false_decline_cost, missed_fraud_loss, net_value_saved]
    colors = [COLORS["rf"], COLORS["if"], COLORS["fraud"], COLORS["lr"]]

    bars = ax.bar(categories, values, color=colors, edgecolor="#0f1117", width=0.55)
    ax.set_ylabel("USD Value ($)")
    ax.set_title(f"Financial Impact & Risk ROI Analysis (Test Set: {len(y_test):,} txs)", fontsize=13, pad=12, color="white")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f"${val:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color="white")

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "financial_impact.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved Financial Impact → {path}")
    print(f"  💰 Fraud Losses Prevented: ${prevented_fraud_val:,.2f} ({tp} cases)")
    print(f"  ⚠️ False Decline Friction Cost: ${false_decline_cost:,.2f} ({fp} cases)")
    print(f"  ❌ Missed Fraud Loss: ${missed_fraud_loss:,.2f} ({fn} cases)")
    print(f"  💵 Net Value Saved: ${net_value_saved:,.2f}")
    return path


def run_evaluation():
    y_test, y_prob_xgb, y_prob_lr, y_prob_rf, y_prob_if, config = load_artifacts()
    threshold = config.get("best_threshold", 0.655444)
    feature_names = config["feature_names"]
    model_name = config.get("best_model_name", "Random Forest")

    print("=" * 60)
    print(f"  EVALUATION REPORT — PRIMARY MODEL: {model_name}")
    print("=" * 60)

    plot_roc_curves(y_test, y_prob_xgb, y_prob_lr, y_prob_rf, y_prob_if)
    plot_pr_curves( y_test, y_prob_xgb, y_prob_lr, y_prob_rf, y_prob_if)
    plot_confusion_matrix(y_test, y_prob_rf, threshold, model_name=model_name)
    plot_threshold_vs_f1(y_test, y_prob_rf)
    plot_financial_impact_matrix(y_test, y_prob_rf, threshold)

    rf_pipeline = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
    rf_clf = rf_pipeline.named_steps["clf"]   # extract from ImbPipeline
    plot_feature_importance(rf_clf, feature_names)

    print("\n[evaluate] All evaluation plots saved to /reports/")


if __name__ == "__main__":
    run_evaluation()

