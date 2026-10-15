"""Project-wide settings shared by every notebook and module.

Keeping paths, the random seed, the temporal split and the outbreak label
definition in one place guarantees that both models are trained and evaluated
under identical conditions, which is what makes the final comparison valid.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "dengue_srilanka_full_2006_2025.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Temporal split (inclusive year ranges)
# ---------------------------------------------------------------------------
# A random split would let the model see later weeks while predicting earlier
# ones, so the data is split by time. 2006 contains a single week and is
# excluded from every split.
TRAIN_YEARS = (2007, 2020)
VAL_YEARS = (2021, 2022)
TEST_YEARS = (2023, 2025)

# ---------------------------------------------------------------------------
# Outbreak label
# ---------------------------------------------------------------------------
# A district-week is an outbreak when its cases exceed the district's mean plus
# OUTBREAK_K standard deviations. The mean and SD are fitted on TRAIN_YEARS only.
OUTBREAK_K = 1.0

# The model predicts whether an outbreak occurs FORECAST_HORIZON weeks ahead.
FORECAST_HORIZON = 2

# ---------------------------------------------------------------------------
# Column groups (see data/README.md for the full data dictionary)
# ---------------------------------------------------------------------------
DATE_COL = "start.date"
DISTRICT_COL = "district"
CASES_COL = "cases"

# Derived from the same week's cases or from hindsight. Must never be used as features.
LEAKY_COLUMNS = ["incidence_rate", "cases_zscore", "is_epidemic_year"]

# Fully determined by another column, or metadata rather than a district attribute.
# 02a_preprocessing.ipynb checks that none of these, or the leaky columns, reach the model.
REDUNDANT_COLUMNS = ["end.date", "province", "population_source"]
