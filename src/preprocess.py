"""
preprocess.py
─────────────
Handles data loading, scaling, and splitting ONLY.

SMOTE is intentionally NOT applied here.
It lives inside an ImbPipeline in train.py so it refits fresh on each
CV fold's training portion — preventing synthetic samples from leaking
into CV validation folds and inflating PR-AUC to near-1.0.

Correct order: Load → Split (stratified) → Scale (fit on X_train only)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import os

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw creditcard CSV."""
    print(f"[preprocess] Loading data from {path}")
    df = pd.read_csv(path)
    print(f"[preprocess] Shape: {df.shape} | Fraud rate: {df['Class'].mean()*100:.4f}%")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split BEFORE any scaling or resampling.
    Stratify on Class so both splits preserve the 0.17% fraud rate.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"[preprocess] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[preprocess] Train fraud: {y_train.sum()} | Test fraud: {y_test.sum()}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, save: bool = True):
    """
    Fit RobustScaler on X_train only (robust to outliers in Amount/Time).
    V1-V28 are already PCA-scaled — we only scale Amount and Time.
    Transform both train and test using the SAME fitted scaler.
    """
    cols_to_scale = ["Amount", "Time"]
    scaler = RobustScaler()

    X_train = X_train.copy()
    X_test  = X_test.copy()

    # Fit on train, transform both
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        print(f"[preprocess] Scaler saved to {MODELS_DIR}/scaler.pkl")

    return X_train, X_test, scaler


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "smote",
    sampling_strategy: float = 0.15,
    random_state: int = 42
):
    """
    Apply SMOTE or SMOTETomek to TRAINING DATA ONLY.
    sampling_strategy=0.15 → minority class becomes 15% of majority class count.
    We never touch X_test / y_test here.
    """
    if strategy == "smotetomek":
        resampler = SMOTETomek(
            smote=SMOTE(sampling_strategy=sampling_strategy, random_state=random_state),
            random_state=random_state
        )
        label = "SMOTETomek"
    else:
        resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        label = "SMOTE"

    print(f"[preprocess] Applying {label} (strategy={sampling_strategy}) to train set...")
    print(f"[preprocess] Before: {dict(y_train.value_counts())}")
    X_res, y_res = resampler.fit_resample(X_train, y_train)
    print(f"[preprocess] After:  {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


def get_preprocessed_data(data_path: str = DATA_PATH):
    """
    Load → Split (stratified) → Scale (fit on X_train only).

    Returns raw (un-resampled) scaled train/test splits.
    SMOTE is NOT applied here — it belongs inside ImbPipeline in train.py
    so that each CV fold resamples only its own training portion.

    Returns: X_train_sc, X_test_sc, y_train, y_test, scaler
    """
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_sc, X_test_sc, scaler    = scale_features(X_train, X_test)
    return X_train_sc, X_test_sc, y_train, y_test, scaler


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, sc = get_preprocessed_data()
    print("Preprocessing complete (no SMOTE — use ImbPipeline in train.py).")
    print(f"  X_train: {X_tr.shape}, X_test: {X_te.shape}")
