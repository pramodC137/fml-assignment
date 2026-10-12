"""Load the raw dengue data, build the outbreak target and assign the temporal split.

Typical use in a notebook::

    from src.data_loader import load_dataset
    df, thresholds = load_dataset()

Every row of the returned frame is one district-week with three added columns:

* ``outbreak_now``  1 if this week's cases exceed the district threshold
* ``target``        1 if an outbreak occurs ``FORECAST_HORIZON`` weeks later
* ``split``         "train", "val" or "test"

Leakage safeguards built in here:

1. Outbreak thresholds are fitted on training years only.
2. Rows whose target week falls in a different split from the row itself are
   removed (an embargo at each boundary), so no training row is labelled with
   a validation or test outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def load_raw(path=config.RAW_DATA_FILE) -> pd.DataFrame:
    """Read the raw CSV, parse dates and sort each district's series in time order."""
    df = pd.read_csv(path, parse_dates=["start.date", "end.date"])
    df = df.sort_values([config.DISTRICT_COL, config.DATE_COL]).reset_index(drop=True)

    # A complete panel is assumed by the lag and target logic below, so fail loudly if not.
    duplicated = df.duplicated([config.DISTRICT_COL, config.DATE_COL]).sum()
    if duplicated:
        raise ValueError(f"{duplicated} duplicate district-weeks found in {path}")
    return df


def year_to_split(year: pd.Series) -> pd.Series:
    """Map calendar years to 'train', 'val', 'test' or 'excluded'."""
    split = pd.Series("excluded", index=year.index, dtype=object)
    for name, (start, end) in {
        "train": config.TRAIN_YEARS,
        "val": config.VAL_YEARS,
        "test": config.TEST_YEARS,
    }.items():
        split[(year >= start) & (year <= end)] = name
    return split


def fit_outbreak_thresholds(df: pd.DataFrame, k: float = config.OUTBREAK_K) -> pd.Series:
    """Return the outbreak threshold (mean + k * SD of weekly cases) for each district.

    Only training years are used, so validation and test outcomes never
    influence how an outbreak is defined.
    """
    train_mask = df["year"].between(*config.TRAIN_YEARS)
    stats = df.loc[train_mask].groupby(config.DISTRICT_COL)[config.CASES_COL].agg(["mean", "std"])
    thresholds = stats["mean"] + k * stats["std"]
    thresholds.name = "outbreak_threshold"
    return thresholds


def add_outbreak_target(
    df: pd.DataFrame,
    thresholds: pd.Series,
    horizon: int = config.FORECAST_HORIZON,
) -> pd.DataFrame:
    """Add ``outbreak_now``, ``target``, ``target_date`` and ``target_year`` columns.

    The target is shifted within each district, so it never borrows a value
    from a neighbouring district's series. The last ``horizon`` weeks of each
    district have no future outcome and receive a missing target.
    """
    out = df.copy()
    district_threshold = out[config.DISTRICT_COL].map(thresholds)
    out["outbreak_now"] = (out[config.CASES_COL] > district_threshold).astype(int)

    grouped = out.groupby(config.DISTRICT_COL)
    out["target"] = grouped["outbreak_now"].shift(-horizon)
    out["target_date"] = grouped[config.DATE_COL].shift(-horizon)
    # Epidemiological year of the target week. This can differ from the calendar
    # year for late-December weeks, so the split must use this, not target_date.
    out["target_year"] = grouped["year"].shift(-horizon)
    return out


def load_dataset(
    k: float = config.OUTBREAK_K,
    horizon: int = config.FORECAST_HORIZON,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the modelling dataset with the outbreak target and temporal split.

    Returns
    -------
    df : DataFrame
        One row per district-week with ``outbreak_now``, ``target`` and ``split``.
        Rows without a target, rows from 2006, and rows whose target week falls
        in a different split are removed.
    thresholds : Series
        Outbreak threshold per district, fitted on training years.
    """
    raw = load_raw()
    thresholds = fit_outbreak_thresholds(raw, k=k)
    df = add_outbreak_target(raw, thresholds, horizon=horizon)

    df["split"] = year_to_split(df["year"])
    target_split = year_to_split(df["target_year"])

    keep = (
        df["target"].notna()
        & (df["split"] != "excluded")
        & (df["split"] == target_split)  # embargo at each split boundary
    )
    df = df.loc[keep].copy()
    df["target"] = df["target"].astype(int)
    return df.reset_index(drop=True), thresholds


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows, date range and outbreak prevalence for each split."""
    order = ["train", "val", "test"]
    summary = df.groupby("split").agg(
        rows=("target", "size"),
        first_week=(config.DATE_COL, "min"),
        last_week=(config.DATE_COL, "max"),
        outbreaks=("target", "sum"),
    ).reindex(order)
    summary["outbreak_rate_%"] = (100 * summary["outbreaks"] / summary["rows"]).round(1)
    return summary


if __name__ == "__main__":
    data, thr = load_dataset()
    print(split_summary(data))
    print(f"\nThresholds fitted for {thr.size} districts; median {np.median(thr):.1f} cases/week")
