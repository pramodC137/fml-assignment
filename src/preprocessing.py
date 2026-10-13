"""Feature engineering and the shared preprocessing pipeline.

Both models import from this module, so they are trained on exactly the same
inputs. Typical use in a model notebook::

    from sklearn.pipeline import Pipeline
    from src.preprocessing import build_model_frame, get_xy, make_selected_preprocessor

    frame = build_model_frame()
    X_train, y_train = get_xy(frame, "train")
    model = Pipeline([("prep", make_selected_preprocessor()), ("clf", SomeClassifier())])
    model.fit(X_train, y_train)

``make_selected_preprocessor`` is the preprocessing agreed in step 8: the
shared pipeline from ``make_preprocessor`` followed by L1 feature selection.
``make_preprocessor`` (all 42 inputs) stays available for comparisons.

Timing convention
-----------------
A prediction is made at the end of the **origin week t** and concerns the
**target week t + FORECAST_HORIZON**. Every feature for week t uses only data
from week t and earlier.

Leakage safeguards
------------------
1. Lags and rolling windows look backwards only (``shift`` and trailing windows),
   computed within each district.
2. Statistics that are learned from data (outbreak thresholds, the temperature
   baseline, the district density constant) use training years only.
3. Imputation and scaling live inside a scikit-learn pipeline, so they are
   refitted on the training portion of every cross-validation fold.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config
from src.data_loader import fit_outbreak_thresholds, load_dataset, load_raw

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# Skewed counts and rainfall: imputed, then log(1 + x), then standardised.
LOG_FEATURES = [
    "cases_now", "cases_lag1", "cases_lag2", "cases_lag3", "cases_lag4",
    "cases_roll4", "cases_roll8", "cases_rel_threshold",
    "rain_recent4", "rain_w5_8",
]
# Roughly symmetric or already transformed: imputed, then standardised.
NUMERIC_FEATURES = [
    "cases_trend", "humidity_w5_8", "temp_max_anom4",
    "week_sin", "week_cos", "log_density", "outbreak_now",
]
# One-hot encoded.
CATEGORICAL_FEATURES = [config.DISTRICT_COL]

MODEL_FEATURES = LOG_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Thematic groups, used to measure what each kind of information contributes.
FEATURE_GROUPS = {
    "case history": [
        "cases_now", "cases_lag1", "cases_lag2", "cases_lag3", "cases_lag4",
        "cases_roll4", "cases_roll8", "cases_rel_threshold", "cases_trend", "outbreak_now",
    ],
    "climate": ["rain_recent4", "rain_w5_8", "humidity_w5_8", "temp_max_anom4"],
    "calendar": ["week_sin", "week_cos"],
    "district": ["log_density", "district"],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(MODEL_FEATURES)

# Validation years for expanding-window cross-validation inside the training period.
CV_VALIDATION_YEARS = (2016, 2017, 2018, 2019, 2020)

FEATURE_DESCRIPTIONS = {
    "cases_now": "Cases in the origin week t",
    "cases_lag1": "Cases in week t-1",
    "cases_lag2": "Cases in week t-2",
    "cases_lag3": "Cases in week t-3",
    "cases_lag4": "Cases in week t-4",
    "cases_roll4": "Mean weekly cases over weeks t-3 to t",
    "cases_roll8": "Mean weekly cases over weeks t-7 to t",
    "cases_rel_threshold": "Cases in week t divided by the district's outbreak threshold",
    "rain_recent4": "Mean weekly rainfall over weeks t-3 to t",
    "rain_w5_8": "Mean weekly rainfall over weeks t-8 to t-5",
    "cases_trend": "log(1 + cases now) minus log(1 + cases four weeks earlier)",
    "humidity_w5_8": "Mean relative humidity over weeks t-8 to t-5",
    "temp_max_anom4": "Mean of (max temperature minus the district's usual value for that week) over weeks t-3 to t",
    "week_sin": "Sine of the week's position in the year",
    "week_cos": "Cosine of the week's position in the year",
    "log_density": "Log of the district's mean population density over training years (constant per district)",
    "outbreak_now": "1 if week t is already an outbreak week",
    "district": "District, one-hot encoded",
}


# ---------------------------------------------------------------------------
# Reference statistics learned from training years only
# ---------------------------------------------------------------------------
def fit_reference_stats(raw: pd.DataFrame) -> dict:
    """Learn every data-dependent constant from the training years only."""
    train = raw[raw["year"].between(*config.TRAIN_YEARS)]
    week_of_year = train["week"].clip(upper=52)

    temp_baseline = train.groupby([config.DISTRICT_COL, week_of_year])["temp_max_c"].mean()
    temp_baseline.index.names = [config.DISTRICT_COL, "week_of_year"]

    return {
        "thresholds": fit_outbreak_thresholds(raw),
        "temp_baseline": temp_baseline,
        "log_density": np.log(train.groupby(config.DISTRICT_COL)["population_density"].mean()),
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _trailing_mean(grouped, window: int, gap: int = 0) -> pd.Series:
    """Mean over `window` weeks ending `gap` weeks before the current week, per district."""
    return grouped.transform(lambda s: s.shift(gap).rolling(window, min_periods=window).mean())


def engineer_features(raw: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Build the model features for every district-week of the raw panel.

    ``raw`` must be the full, time-ordered panel from ``load_raw``. Rows are
    never dropped here, so lags always refer to the true previous weeks.
    Early weeks without enough history receive missing values, which the
    pipeline imputes.
    """
    raw = raw.sort_values([config.DISTRICT_COL, config.DATE_COL])
    by_district = raw.groupby(config.DISTRICT_COL)
    cases = raw[config.CASES_COL]

    out = raw[[config.DISTRICT_COL, config.DATE_COL]].copy()

    # Case history
    out["cases_now"] = cases
    for lag in range(1, 5):
        out[f"cases_lag{lag}"] = by_district[config.CASES_COL].shift(lag)
    out["cases_roll4"] = _trailing_mean(by_district[config.CASES_COL], 4)
    out["cases_roll8"] = _trailing_mean(by_district[config.CASES_COL], 8)
    out["cases_trend"] = np.log1p(cases) - np.log1p(out["cases_lag4"])

    threshold = raw[config.DISTRICT_COL].map(ref["thresholds"])
    out["cases_rel_threshold"] = cases / threshold
    out["outbreak_now"] = (cases > threshold).astype(int)

    # Climate, at the lags found in 01b_eda_features (about 5 to 8 weeks before week t)
    out["rain_recent4"] = _trailing_mean(by_district["rainfall_mm"], 4)
    out["rain_w5_8"] = _trailing_mean(by_district["rainfall_mm"], 4, gap=5)
    out["humidity_w5_8"] = _trailing_mean(by_district["humidity_pct"], 4, gap=5)

    week_of_year = raw["week"].clip(upper=52)
    baseline_keys = pd.MultiIndex.from_arrays([raw[config.DISTRICT_COL], week_of_year])
    baseline = ref["temp_baseline"].reindex(baseline_keys).to_numpy()
    anomaly = raw["temp_max_c"] - baseline
    out["temp_max_anom4"] = _trailing_mean(anomaly.groupby(raw[config.DISTRICT_COL]), 4)

    # Calendar, encoded on a circle so week 52 sits next to week 1
    angle = 2 * np.pi * raw["week"] / 52.18
    out["week_sin"] = np.sin(angle)
    out["week_cos"] = np.cos(angle)

    # District-level density constant (replaces the year-specific population columns)
    out["log_density"] = raw[config.DISTRICT_COL].map(ref["log_density"])
    return out


def build_model_frame() -> pd.DataFrame:
    """Return one row per modelling district-week: keys, split, target and all features."""
    labelled, _ = load_dataset()
    raw = load_raw()
    features = engineer_features(raw, fit_reference_stats(raw))

    keys = [config.DISTRICT_COL, config.DATE_COL]
    frame = labelled[keys + ["year", "week", "split", "target", "target_year", "outbreak_now"]].merge(
        features, on=keys, how="left", validate="one_to_one", suffixes=("", "_feature"),
    )
    # The loader and the feature code must agree on which weeks are outbreaks.
    if not (frame["outbreak_now"] == frame["outbreak_now_feature"]).all():
        raise ValueError("outbreak_now differs between data_loader and preprocessing")
    return frame.drop(columns="outbreak_now_feature").reset_index(drop=True)


def get_xy(frame: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and target for one split ('train', 'val' or 'test')."""
    part = frame[frame["split"] == split]
    return part[MODEL_FEATURES].reset_index(drop=True), part["target"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def _one_hot() -> OneHotEncoder:
    """OneHotEncoder with dense output on any recent scikit-learn version."""
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _log_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])


def _numeric_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def _branches(features, include_district: bool) -> list:
    unknown = set(features) - set(MODEL_FEATURES)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")
    branches = [
        ("log", _log_pipeline(), [f for f in LOG_FEATURES if f in features]),
        ("num", _numeric_pipeline(), [f for f in NUMERIC_FEATURES if f in features]),
    ]
    if include_district:
        branches.append(("district", _one_hot(), [f for f in CATEGORICAL_FEATURES if f in features]))
    return [b for b in branches if b[2]]  # skip branches with no selected columns


def make_preprocessor(features=MODEL_FEATURES) -> ColumnTransformer:
    """Unfitted preprocessing step. Fit it on training data only (a Pipeline does this).

    ``features`` selects a subset of ``MODEL_FEATURES`` (default: all of them),
    so feature selection can compare smaller sets on equal terms. Branches are
    named "log", "num" and "district".
    """
    return ColumnTransformer(_branches(features, include_district=True), verbose_feature_names_out=False)


def make_district_encoder() -> OneHotEncoder:
    """One-hot encoder for district, for pipelines that assemble their own branches (for example PCA)."""
    return _one_hot()


def make_numeric_transformer(features=MODEL_FEATURES) -> ColumnTransformer:
    """The "log" and "num" branches only, for the numeric features in ``features``.

    Used where the numeric block is processed further (for example by PCA)
    while district is encoded separately.
    """
    return ColumnTransformer(_branches(features, include_district=False), verbose_feature_names_out=False)


# ---------------------------------------------------------------------------
# Feature selection agreed in step 8
# ---------------------------------------------------------------------------
# Joint decision in notebooks/02c_feature_selection.ipynb: L1 selection at C = 0.01
# was the simplest feature set within 0.01 mean PR-AUC of the best, for both
# algorithms. Fitted on 2007 to 2020 it keeps 9 of the 42 encoded inputs.
SELECTION_C = 0.01

# Coefficients at or below this size count as zero, so the selector keeps exactly
# the inputs the L1 penalty leaves in the model. Set explicitly because the
# automatic default depends on the scikit-learn version.
SELECTION_THRESHOLD = 1e-5


def l1_logistic(C: float = SELECTION_C) -> LogisticRegression:
    """L1-penalised logistic regression, written for old and new scikit-learn versions.

    scikit-learn 1.8 deprecated ``penalty`` in favour of ``l1_ratio``. On older
    versions ``l1_ratio`` alone would silently fit an L2 model, so the argument
    is chosen from the installed version.
    """
    settings = dict(C=C, solver="liblinear", class_weight="balanced", max_iter=5000)
    penalty = inspect.signature(LogisticRegression).parameters.get("penalty")
    if penalty is None or penalty.default == "deprecated":
        return LogisticRegression(l1_ratio=1.0, **settings)
    return LogisticRegression(penalty="l1", **settings)


def make_selected_preprocessor(C: float = SELECTION_C) -> Pipeline:
    """Unfitted preprocessing agreed for both models: ``make_preprocessor`` then L1 selection.

    The selector keeps the encoded inputs whose L1 coefficient is non-zero. As a
    step inside the model pipeline it is refitted on the training rows of every
    fit, including each cross-validation fold, so it never sees the rows it is
    scored on.
    """
    return Pipeline([
        ("prep", make_preprocessor()),
        ("select", SelectFromModel(l1_logistic(C), threshold=SELECTION_THRESHOLD)),
    ])


def selected_inputs(fitted) -> list:
    """Names of the encoded inputs kept by a fitted ``make_selected_preprocessor``.

    Accepts the fitted preprocessing itself or a fitted model pipeline whose first
    step is that preprocessing.
    """
    step = fitted
    if isinstance(fitted, Pipeline) and "select" not in fitted.named_steps:
        step = fitted.steps[0][1]
    return list(step.get_feature_names_out())


# ---------------------------------------------------------------------------
# Time-series cross-validation
# ---------------------------------------------------------------------------
def expanding_year_folds(train_frame: pd.DataFrame, val_years=CV_VALIDATION_YEARS) -> list:
    """Expanding-window folds over the training years, usable as ``cv=`` in scikit-learn.

    For each validation year Y the model trains on every earlier year and is
    scored on year Y. Training rows whose target week falls in year Y or later
    are left out (the same two-week embargo used between the main splits).
    Indices are positions in ``train_frame``.
    """
    year = train_frame["year"].to_numpy()
    target_year = train_frame["target_year"].to_numpy()
    folds = []
    for val_year in val_years:
        train_idx = np.flatnonzero((year < val_year) & (target_year < val_year))
        val_idx = np.flatnonzero(year == val_year)
        folds.append((train_idx, val_idx))
    return folds
