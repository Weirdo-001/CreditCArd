"""
train.py
────────
Trains all models: Logistic Regression, Random Forest, XGBoost, Isolation Forest.

Key fix: SMOTE lives inside ImbPipeline, not applied once before CV.
This means every cross_val_score / Optuna trial refits SMOTE on the fold's
training portion only — no synthetic samples bleed into the validation fold.

CV flow per fold:
  fold_train → SMOTE.fit_resample(fold_train) → clf.fit(oversampled)
  fold_val   → clf.predict_proba(fold_val)   ← raw, untouched

Final model: pipeline.fit(X_train_full, y_train) → saved as model.pkl
Test set:    pipeline.predict_proba(X_test)  ← never touched SMOTE
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics         import (average_precision_score, f1_score,
                                     precision_recall_curve, roc_auc_score)
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ImbPipeline instead of sklearn Pipeline — handles resamplers correctly in CV
from imblearn.pipeline      import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from preprocess import get_preprocessed_data

MODELS_DIR       = os.path.join(os.path.dirname(__file__), "..", "models")
SMOTE_STRATEGY   = 0.15   # minority becomes 15% of majority — avoids overfitting to synthetics
RANDOM_STATE     = 42
os.makedirs(MODELS_DIR, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def make_smote():
    return SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=RANDOM_STATE)


def best_threshold_f1(y_true, y_prob):
    """
    Return (best_threshold, best_f1) over the aligned precision/recall region.

    Uses [:-1] slicing so f1s and thresholds arrays are the same length —
    the phantom point (precision=1, recall=0) at index -1 is excluded.

    Threshold is capped at 0.9999 because prob >= 1.0 is always False at
    inference time; values that round to 1.0000 at 4dp are real thresholds
    very close to 1 (e.g. 0.99995) but would silently classify nothing as fraud.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # [:-1] aligns arrays: f1s[i] <-> thresholds[i], no phantom index
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1s)
    thr = float(thresholds[best_idx])
    thr = min(thr, 0.9999)   # safety cap — prob >= 1.0 is always False
    return thr, float(f1s[best_idx])


def evaluate_model(name, model, X_test, y_test, is_unsupervised=False):
    """Print & return a metrics dict for a fitted model/pipeline."""
    if is_unsupervised:
        raw_scores = -model.score_samples(X_test)
        y_prob = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    roc   = roc_auc_score(y_test, y_prob)
    prauc = average_precision_score(y_test, y_prob)
    thr, best_f1 = best_threshold_f1(y_test, y_prob)

    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_best    = (y_prob >= thr).astype(int)

    f1_default = f1_score(y_test, y_pred_default, zero_division=0)
    f1_best    = f1_score(y_test, y_pred_best,    zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  ROC-AUC       : {roc:.4f}")
    print(f"  PR-AUC        : {prauc:.4f}")
    print(f"  F1 @ 0.5 thr  : {f1_default:.4f}")
    print(f"  F1 @ best thr : {f1_best:.4f}  (threshold={thr:.6f})")

    return {
        "name": name,
        "roc_auc":        round(roc, 4),
        "pr_auc":         round(prauc, 4),
        "f1_default":     round(f1_default, 4),
        "f1_best":        round(f1_best, 4),
        "best_threshold": round(thr, 6),   
        "y_prob":         y_prob.tolist(),
    }


# ── model training ─────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, X_test, y_test, cv):
    print("\n[train] Logistic Regression ...")
    # SMOTE inside pipeline: each CV fold resamples its own train portion
    pipe = ImbPipeline([
        ("smote", make_smote()),
        ("clf",   LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="saga",
            random_state=RANDOM_STATE
        )),
    ])
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                cv=cv, scoring="average_precision", n_jobs=-1)
    print(f"  CV PR-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    pipe.fit(X_train, y_train)
    return pipe, evaluate_model("Logistic Regression", pipe, X_test, y_test)


def train_random_forest(X_train, y_train, X_test, y_test, cv):
    print("\n[train] Random Forest ...")
    pipe = ImbPipeline([
        ("smote", make_smote()),
        ("clf",   RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE
        )),
    ])
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                cv=cv, scoring="average_precision", n_jobs=-1)
    print(f"  CV PR-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    pipe.fit(X_train, y_train)
    return pipe, evaluate_model("Random Forest", pipe, X_test, y_test)


def tune_xgboost_optuna(X_train, y_train, cv, n_trials: int = 50):
    """
    Optuna search — each trial wraps XGBoost in ImbPipeline with SMOTE.
    cross_val_score receives the pipeline, so SMOTE is re-fit per fold.
    """
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    default_spw = neg_count / pos_count

    def objective(trial):
        clf = xgb.XGBClassifier(
            n_estimators      = trial.suggest_int("n_estimators", 200, 800),
            max_depth         = trial.suggest_int("max_depth", 3, 10),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample         = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
            gamma             = trial.suggest_float("gamma", 0, 5),
            scale_pos_weight  = trial.suggest_float(
                "scale_pos_weight", default_spw * 0.5, default_spw * 2.0),
            eval_metric="aucpr", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        # Pipeline ensures SMOTE re-fits on each fold's train portion
        pipe = ImbPipeline([("smote", make_smote()), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="average_precision", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  Best Optuna CV PR-AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def train_xgboost(X_train, y_train, X_test, y_test, cv,
                  tune: bool = True, n_trials: int = 50):
    print("\n[train] XGBoost ...")
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())

    if tune:
        best_params = tune_xgboost_optuna(X_train, y_train, cv, n_trials)
        best_params.update({
            "eval_metric": "aucpr", "tree_method": "hist",
            "random_state": RANDOM_STATE, "n_jobs": -1,
        })
    else:
        best_params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
            "scale_pos_weight": neg_count / pos_count,
            "eval_metric": "aucpr", "tree_method": "hist",
            "random_state": RANDOM_STATE, "n_jobs": -1,
        }

    pipe = ImbPipeline([
        ("smote", make_smote()),
        ("clf",   xgb.XGBClassifier(**best_params)),
    ])
    pipe.fit(X_train, y_train)
    return pipe, evaluate_model("XGBoost (Optuna-tuned)", pipe, X_test, y_test), best_params


def train_isolation_forest(X_train, y_train, X_test, y_test):
    """Unsupervised — no SMOTE, no labels used in fit."""
    print("\n[train] Isolation Forest (unsupervised) ...")
    model = IsolationForest(
        n_estimators=200,
        contamination=float(y_train.mean()),
        max_features=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train)
    return model, evaluate_model("Isolation Forest", model, X_test, y_test,
                                 is_unsupervised=True)


# ── main pipeline ──────────────────────────────────────────────────────────────

def run_training(tune_xgb: bool = True, optuna_trials: int = 50):
    print("=" * 60)
    print("  CREDIT CARD FRAUD DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # Load → Split (stratified) → Scale (fit on X_train only). No SMOTE yet.
    X_train, X_test, y_train, y_test, scaler = get_preprocessed_data()
    print(f"\n[train] Train class balance: {dict(y_train.value_counts())}")
    print(f"[train] SMOTE will run INSIDE each CV fold (ImbPipeline)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    models  = {}

    lr_pipe,  lr_res  = train_logistic_regression(X_train, y_train, X_test, y_test, cv)
    results.append(lr_res);  models["logistic_regression"] = lr_pipe

    rf_pipe,  rf_res  = train_random_forest(X_train, y_train, X_test, y_test, cv)
    results.append(rf_res);  models["random_forest"] = rf_pipe

    xgb_pipe, xgb_res, xgb_params = train_xgboost(
        X_train, y_train, X_test, y_test, cv,
        tune=tune_xgb, n_trials=optuna_trials
    )
    results.append(xgb_res); models["xgboost"] = xgb_pipe

    if_model, if_res  = train_isolation_forest(X_train, y_train, X_test, y_test)
    results.append(if_res);  models["isolation_forest"] = if_model

    # ── Save pipelines (each includes SMOTE step — predict_proba skips it) ──
    joblib.dump(xgb_pipe,  os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(lr_pipe,   os.path.join(MODELS_DIR, "lr_model.pkl"))
    joblib.dump(rf_pipe,   os.path.join(MODELS_DIR, "rf_model.pkl"))
    joblib.dump(if_model,  os.path.join(MODELS_DIR, "if_model.pkl"))

    best = xgb_res
    config = {
        "best_threshold":    best["best_threshold"],
        "xgb_params":        xgb_params,
        "smote_strategy":    SMOTE_STRATEGY,
        "feature_names":     list(X_test.columns),
        "metrics": {
            r["name"]: {k: v for k, v in r.items() if k not in ("name", "y_prob")}
            for r in results
        },
    }
    with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save test probabilities for dashboard plotting
    np.save(os.path.join(MODELS_DIR, "y_test.npy"),     y_test.values)
    np.save(os.path.join(MODELS_DIR, "y_prob_xgb.npy"), np.array(xgb_res["y_prob"]))
    np.save(os.path.join(MODELS_DIR, "y_prob_lr.npy"),  np.array(lr_res["y_prob"]))
    np.save(os.path.join(MODELS_DIR, "y_prob_rf.npy"),  np.array(rf_res["y_prob"]))
    np.save(os.path.join(MODELS_DIR, "y_prob_if.npy"),  np.array(if_res["y_prob"]))

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — ALL ARTIFACTS SAVED TO /models/")
    print("=" * 60)
    print(f"\n  Best threshold (XGBoost): {best['best_threshold']}")
    print(f"  Best F1 (tuned thr)     : {best['f1_best']}")
    print(f"  ROC-AUC                 : {best['roc_auc']}")

    return models, results, config


if __name__ == "__main__":
    import sys
    tune_flag = "--no-tune" not in sys.argv
    run_training(tune_xgb=tune_flag, optuna_trials=50)
