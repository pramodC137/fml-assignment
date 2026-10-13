"""Shared evaluation code, so every comparison in the project is scored identically.

``cross_validate`` scores a preprocessing choice with one or more classifiers
on the expanding-window folds from ``src.preprocessing.expanding_year_folds``.
The score is PR-AUC (average precision), the project's primary metric for the
rare outbreak class.

``classification_metrics`` returns every metric the report uses for one set of
predictions. Baselines (step 9), both models (step 10) and the final comparison
(step 11) all call it, so they are scored identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src import config

# Column order for metric tables: ranking metrics first, then metrics at a
# decision threshold, then the confusion matrix counts.
METRIC_NAMES = ["PR-AUC", "ROC-AUC", "precision", "recall", "F1", "MCC", "accuracy",
                "TP", "FP", "FN", "TN"]


def classification_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    """Every metric the project reports, for one set of predictions.

    ``scores`` are outbreak probabilities, or any score where higher means an
    outbreak is more likely (persistence, for example, scores 0 or 1).
    PR-AUC and ROC-AUC use the scores directly and do not depend on a threshold.
    The other metrics use the predictions ``scores >= threshold``.

    A rule that never predicts an outbreak gets precision, recall, F1 and MCC
    of 0 rather than a warning. If ``y_true`` holds only one class, the two
    ranking metrics are undefined and returned as NaN.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    both_classes = np.unique(y_true).size == 2
    return {
        "PR-AUC": average_precision_score(y_true, scores) if both_classes else np.nan,
        "ROC-AUC": roc_auc_score(y_true, scores) if both_classes else np.nan,
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted) if both_classes else np.nan,
        "accuracy": accuracy_score(y_true, predicted),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def reference_models() -> dict:
    """The two project algorithms at fixed default settings, for comparing inputs rather than tuning.

    Class weighting is on for both, because outbreaks are a small minority of rows.
    """
    return {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=5000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_SEED,
        ),
    }


def cross_validate(preprocessor_factory, X: pd.DataFrame, y: pd.Series, folds, models=None,
                   label: str = "") -> pd.DataFrame:
    """PR-AUC for each model on each fold.

    ``preprocessor_factory`` is a zero-argument function returning a fresh,
    unfitted preprocessing step, so nothing is shared between folds. The
    preprocessing is fitted on each fold's training rows only.

    Returns one row per (model, fold) with columns: setting, model, fold, pr_auc.
    """
    models = models or reference_models()
    rows = []
    for model_name, model in models.items():
        for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
            pipeline = Pipeline([("prep", preprocessor_factory()), ("clf", clone(model))])
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
            rows.append({
                "setting": label, "model": model_name, "fold": fold_number,
                "pr_auc": average_precision_score(y.iloc[val_idx], scores),
            })
    return pd.DataFrame(rows)


def persistence_scores(train_frame: pd.DataFrame, folds, label: str = "persistence baseline") -> pd.DataFrame:
    """PR-AUC of the no-model rule "outbreak in two weeks if there is one now", per fold."""
    rows = []
    for fold_number, (_, val_idx) in enumerate(folds, start=1):
        part = train_frame.iloc[val_idx]
        rows.append({
            "setting": label, "model": "Persistence", "fold": fold_number,
            "pr_auc": average_precision_score(part["target"], part["outbreak_now"]),
        })
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of PR-AUC across folds, one row per setting and model."""
    summary = results.groupby(["setting", "model"], sort=False)["pr_auc"].agg(["mean", "std"])
    summary.columns = ["mean PR-AUC", "std across folds"]
    return summary


def compare_to_reference(results: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Paired comparison of every setting against ``reference``, fold by fold.

    Scores vary far more between folds (years) than between settings, so each
    setting is compared with the reference on the same fold. Returns the mean
    change in PR-AUC and how many folds improved or got worse, per model.
    """
    ref = results[results["setting"] == reference].set_index(["model", "fold"])["pr_auc"]
    rows = []
    for (setting, model), part in results[results["setting"] != reference].groupby(["setting", "model"], sort=False):
        if model not in ref.index.get_level_values("model"):
            continue
        change = part.set_index("fold")["pr_auc"] - ref.loc[model]
        rows.append({
            "setting": setting, "model": model,
            "mean PR-AUC": part["pr_auc"].mean(),
            "change vs reference": change.mean(),
            "folds better": int((change > 0).sum()),
            "folds worse": int((change < 0).sum()),
        })
    return pd.DataFrame(rows).set_index(["setting", "model"])


def fold_table(results: pd.DataFrame, years=None) -> pd.DataFrame:
    """PR-AUC per fold as columns, optionally labelled with the validation year."""
    table = results.pivot_table(index=["setting", "model"], columns="fold", values="pr_auc", sort=False)
    if years is not None:
        table.columns = [f"{y}" for y in years]
    return table

