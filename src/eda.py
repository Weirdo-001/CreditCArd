"""
eda.py — Run this standalone to generate EDA plots before training.
Outputs all charts to /reports/eda/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "reports", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d2e",
    "axes.edgecolor": "#3d4166", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0", "grid.color": "#2a2d45",
})

print("[EDA] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Fraud rate: {df['Class'].mean()*100:.4f}%")
print(f"  Missing values: {df.isnull().sum().sum()}")


# ── 1. Class Distribution ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
counts = df["Class"].value_counts()
labels = ["Legitimate", "Fraud"]
colors = ["#3b82f6", "#ef4444"]

axes[0].bar(labels, counts.values, color=colors, width=0.4, edgecolor="#0f1117")
for i, (lbl, cnt) in enumerate(zip(labels, counts.values)):
    axes[0].text(i, cnt + 1000, f"{cnt:,}\n({cnt/len(df)*100:.2f}%)",
                 ha="center", fontsize=10, color="white")
axes[0].set_title("Class Distribution (Absolute)", fontsize=13, color="white")
axes[0].set_ylabel("Count")
axes[0].grid(axis="y", alpha=0.3)

# Log-scale version to see imbalance clearly
axes[1].bar(labels, counts.values, color=colors, width=0.4, edgecolor="#0f1117", log=True)
axes[1].set_title("Class Distribution (Log Scale)", fontsize=13, color="white")
axes[1].set_ylabel("Count (log)")
axes[1].grid(axis="y", alpha=0.3)
plt.suptitle("Severe Class Imbalance: 0.172% Fraud Rate", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("[EDA] class_distribution.png saved")


# ── 2. Amount Distribution by Class ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fraud = df[df["Class"] == 1]["Amount"]
legit = df[df["Class"] == 0]["Amount"]

axes[0].hist(legit.clip(upper=2000), bins=80, color="#3b82f6", alpha=0.7,
             label=f"Legit (n={len(legit):,})", density=True)
axes[0].hist(fraud.clip(upper=2000), bins=80, color="#ef4444", alpha=0.8,
             label=f"Fraud (n={len(fraud):,})", density=True)
axes[0].set_title("Amount Distribution by Class (clipped at $2000)", fontsize=12)
axes[0].set_xlabel("Amount (USD)"); axes[0].legend(); axes[0].grid(alpha=0.3)

# Box plot
bp = axes[1].boxplot(
    [legit.clip(upper=500), fraud.clip(upper=500)],
    labels=["Legitimate", "Fraud"],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
)
bp["boxes"][0].set_facecolor("#3b82f660")
bp["boxes"][1].set_facecolor("#ef444460")
axes[1].set_title("Amount Boxplot by Class (clipped at $500)", fontsize=12)
axes[1].set_ylabel("Amount (USD)"); axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("Fraud Transactions Cluster at Lower Amounts", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "amount_distribution.png"), dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("[EDA] amount_distribution.png saved")


# ── 3. Correlation Heatmap (V1–V28 + Amount vs Class) ─────────────────────────
v_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
corr_with_class = df[v_cols + ["Class"]].corr()["Class"].drop("Class").sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
colors_corr = ["#ef4444" if v < 0 else "#3b82f6" for v in corr_with_class.values]
bars = ax.barh(corr_with_class.index, corr_with_class.values,
               color=colors_corr, edgecolor="#0f1117", height=0.7)
ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
ax.set_title("Feature Correlation with Class (Fraud=1)", fontsize=13, color="white")
ax.set_xlabel("Pearson Correlation Coefficient")
ax.grid(axis="x", alpha=0.3)
# Annotate top 5 positive and negative
for feat, val in list(corr_with_class.items())[:3] + list(corr_with_class.items())[-3:]:
    ax.text(val + (0.002 if val >= 0 else -0.002),
            list(corr_with_class.index).index(feat),
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=8, color="white")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_with_class.png"), dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("[EDA] correlation_with_class.png saved")


# ── 4. Time distribution by class ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.hist(df[df["Class"]==0]["Time"]/3600, bins=100, color="#3b82f6",
        alpha=0.6, label="Legit", density=True)
ax.hist(df[df["Class"]==1]["Time"]/3600, bins=40, color="#ef4444",
        alpha=0.8, label="Fraud", density=True)
ax.set_title("Transaction Time Distribution by Class (hours)", fontsize=13)
ax.set_xlabel("Time (hours)"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "time_distribution.png"), dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("[EDA] time_distribution.png saved")

print(f"\n[EDA] All plots saved to {OUTPUT_DIR}")
print("\nKey stats:")
print(f"  Legit: {(df.Class==0).sum():,} | Fraud: {(df.Class==1).sum():,}")
print(f"  Fraud amount: mean=${fraud.mean():.2f}, median=${fraud.median():.2f}")
print(f"  Legit amount: mean=${legit.mean():.2f}, median=${legit.median():.2f}")
