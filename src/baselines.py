"""
baselines.py
============
Baseline models for the US Wildfire fire_size_class classification task.

Three baselines, all evaluated with 5-fold stratified cross-validation
on the training split (held-out val/test are NOT touched here):

  1. DummyClassifier(strategy='most_frequent')
       Predicts class B every time. Class B is ~66% of the data, so
       accuracy looks high — that's the point. Demonstrates why
       accuracy alone is misleading on this imbalanced problem.

  2. LogisticRegression
       Linear baseline. Multinomial / lbfgs.

  3. DecisionTreeClassifier
       Non-linear baseline at default depth (will overfit — that is
       intentional, it's what makes it a useful contrast against the
       linear model).

Metrics reported (macro-averaged, mean ± std across folds):
  - precision (macro)
  - recall    (macro)
  - f1        (macro)
  - roc_auc   (one-vs-rest, macro)
  - accuracy  (only here to expose the imbalance trap)

Run from the repo root:
    python src/baselines.py

Author: Rishi (and team)
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# Sibling import: assumes baselines.py lives next to preprocessing.py in src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import build_pipeline, load_and_split, encode_labels  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_FOLDS = 5
RANDOM_STATE = 42

# Suppress the chatty UndefinedMetric warnings that fire when the dummy
# classifier predicts only one class — we already handle this with
# zero_division=0 on the scorers, but sklearn warns anyway during CV.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*y_pred.*")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
# zero_division=0 prevents divide-by-zero noise when a class is never
# predicted (the dummy classifier triggers this on every non-majority class).

SCORING = {
    "accuracy":  "accuracy",
    "precision": make_scorer(precision_score, average="macro", zero_division=0),
    "recall":    make_scorer(recall_score,    average="macro", zero_division=0),
    "f1":        make_scorer(f1_score,        average="macro", zero_division=0),
    # roc_auc_ovr defaults to macro-averaging across classes.
    # Requires predict_proba — all three of our classifiers expose it.
    "roc_auc":   "roc_auc_ovr",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_model(name: str, classifier, X_train, y_train,
                   n_folds: int = N_FOLDS,
                   random_state: int = RANDOM_STATE) -> dict:
    """
    Run stratified k-fold CV on (X_train, y_train) using the full
    leakage-free preprocessing pipeline + the supplied classifier.

    Returns a dict {metric: (mean, std)}.
    """
    pipeline = build_pipeline(classifier=classifier)
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state,
    )

    cv = cross_validate(
        pipeline, X_train, y_train,
        cv=skf,
        scoring=SCORING,
        n_jobs=-1,
        return_train_score=False,
    )

    results = {
        metric: (cv[f"test_{metric}"].mean(), cv[f"test_{metric}"].std())
        for metric in SCORING
    }

    # Pretty print this model's block
    print(f"\n{'=' * 64}")
    print(f"  {name}")
    print(f"{'=' * 64}")
    for metric, (m, s) in results.items():
        print(f"  {metric:10s}:  {m:.4f}  ±  {s:.4f}")

    return results


def print_summary(all_results: dict) -> None:
    """Print a compact comparison table across all baselines."""
    print(f"\n{'=' * 78}")
    print("  SUMMARY  —  mean ± std over 5-fold stratified CV (training split only)")
    print(f"{'=' * 78}")
    header = f"{'Model':<42}{'Acc':<14}{'F1':<14}{'AUC':<14}"
    print(header)
    print("-" * len(header))
    for name, results in all_results.items():
        acc_m, acc_s = results["accuracy"]
        f1_m,  f1_s  = results["f1"]
        auc_m, auc_s = results["roc_auc"]
        print(f"{name:<42}"
              f"{acc_m:.3f}±{acc_s:.3f}  "
              f"{f1_m:.3f}±{f1_s:.3f}  "
              f"{auc_m:.3f}±{auc_s:.3f}")


def save_results(all_results: dict, out_dir: str) -> str:
    """Flatten results into a long-form CSV for the report."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for model_name, results in all_results.items():
        for metric, (mean, std) in results.items():
            rows.append({
                "model":  model_name,
                "metric": metric,
                "mean":   mean,
                "std":    std,
            })
    out_path = os.path.join(out_dir, "baseline_cv_results.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "..", "data", "FW_Veg_Rem_Combined.csv")
    out_dir   = os.path.join(here, "..", "outputs")

    if not os.path.exists(data_path):
        print(
            f"Dataset not found at {data_path}\n"
            "Download it first — see README.md > Dataset Setup."
        )
        sys.exit(1)

    print("=" * 64)
    print("  Wildfire baselines — 5-fold stratified CV")
    print("=" * 64)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(data_path)

    # Encode B–G → 0–5 deterministically. ROC-AUC scorer needs integer labels.
    enc = encode_labels(y_train)        # returns [y_train_enc, le]
    y_train_enc = enc[0]

    models = {
        "Dummy (most_frequent → class B)": DummyClassifier(
            strategy="most_frequent", random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
    }

    all_results: dict = {}
    for name, clf in models.items():
        all_results[name] = evaluate_model(name, clf, X_train, y_train_enc)

    print_summary(all_results)

    out_path = save_results(all_results, out_dir)
    print(f"\nResults saved → {os.path.relpath(out_path, here)}")

    # ------------------------------------------------------------------
    # Reading guide for the report
    # ------------------------------------------------------------------
    # If the dummy's accuracy ≈ 0.66 but its macro F1 ≈ 0.13 and AUC ≈ 0.50,
    # that's the imbalance story in one line: a useless classifier looks
    # respectable on accuracy. Macro-averaged F1 and AUC penalise it because
    # they weight every class equally — so failure on rare classes (F, G)
    # drags the score down, which is exactly the behaviour we want.
    #
    # Logistic Regression should beat the dummy on F1 and AUC even if its
    # accuracy is similar. The Decision Tree at default depth will probably
    # have higher train-fold scores than test-fold scores, which previews
    # the overfitting story for the modelling section.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
