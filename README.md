# Fundamentals of Machine Learning: Programming Assignment

## Problem

Dengue outbreaks in Sri Lanka strain hospitals and vector-control teams, and a response is most effective when it starts before cases peak. This project predicts whether a district will be in a dengue **outbreak two weeks ahead**, using recent case counts, rainfall, temperature, humidity and district characteristics. An outbreak is defined as weekly cases above the district’s mean plus one standard deviation, with the threshold fitted on training years only.

## Dataset

Sri Lanka Dengue Surveillance, 2006 to 2025: weekly dengue cases for all 25 districts with climate, geographic and demographic features. 24,775 rows x 28 columns.

- Source: https://www.kaggle.com/datasets/tharinduheshan/dengue-research-2006-2026
- Full description, data dictionary, leakage audit and a cross-check against official case totals: `data/README.md`

## Approach

- Supervised binary classification, two weeks ahead, for every district and week
- Temporal split: train 2007 to 2020, validation 2021 to 2022, test 2023 to 2025, with a two-week gap at each boundary
- Shared preprocessing for both models (`src/preprocessing.py`): 17 engineered features plus district, then L1 feature selection, which keeps 9 inputs when fitted on 2007 to 2020
- Tuning: grid search scored on expanding-window cross-validation over the years 2016 to 2020
- Baselines: always no outbreak, random guessing, persistence, and the persistence score (this week's cases divided by the district threshold)
- Primary metric: PR-AUC, with ROC-AUC, precision, recall, F1, MCC and accuracy also reported

## Algorithms

- Member A: Logistic Regression
- Member B: Random Forest

## Results on the test years, 2023 to 2025

Both models were refitted on 2007 to 2022 with the settings and decision thresholds chosen in cross-validation, then scored once on 3,825 district-weeks, 367 of them outbreak weeks.

| Model | PR-AUC | Recall | F1 | MCC |
|---|---|---|---|---|
| Random Forest | 0.603 | 0.638 | 0.580 | 0.534 |
| Logistic Regression | 0.595 | 0.662 | 0.585 | 0.540 |
| Persistence score (baseline) | 0.576 | 0.569 | 0.566 | 0.520 |

The two models cannot be separated statistically. Both catch more outbreak weeks than persistence, and every extra catch is a new outbreak, which persistence cannot foresee. Logistic Regression ranks new outbreaks better, while Random Forest judges better whether an outbreak will continue. The full analysis is in `notebooks/04b_comparison_analysis.ipynb`.

## Notebooks

| Notebook | Step | Owner | Content |
|---|---|---|---|
| `01a_eda_target.ipynb` | 6 | Member A | Case distribution, seasonality, district differences, outbreak label |
| `01b_eda_features.ipynb` | 6 | Member B | Climate lags, feature relevance, collinearity, leaky columns |
| `02a_preprocessing.ipynb` | 7 | Member B | Feature engineering, leakage tests, pipeline, cross-validation folds |
| `02b_dimensionality_reduction.ipynb` | 8 | Member B | Feature-group ablation and PCA |
| `02c_feature_selection.ipynb` | 8 | Member A | Filter and L1 selection, joint feature-set decision |
| `03_baselines.ipynb` | 9 | Member A | Four no-learning baselines |
| `03_model_b_random_forest.ipynb` | 10 | Member B | Random Forest tuning, threshold, validation check |
| `03_model_a_logistic_regression.ipynb` | 10 | Member A | Logistic Regression tuning, threshold, validation check |
| `04a_test_results.ipynb` | 11 | Member B | Final models scored once on the test years |
| `04b_comparison_analysis.ipynb` | 11 | Member A | Significance, onsets, calibration, why the algorithms differ, conclusion |

## How to run

1. Use Python 3.14 (the version used for the committed results; 3.13 also works) and create a virtual environment.
2. `pip install -r requirements.txt`
3. Open each notebook in the order of the table above and use Restart Kernel and Run All. The Random Forest grid search takes about 8 minutes on a laptop; every other notebook finishes in under 2 minutes.

Every notebook saves its figures to `results/figures/` and its tables to `results/metrics/`. The later notebooks read files saved by earlier ones, so a fresh clone reproduces every result.

## Repository structure

- `data/` raw dataset, data dictionary and dataset documentation
- `notebooks/` EDA, preprocessing, feature selection, baselines, models, test results and comparison
- `src/` shared modules used by every notebook (configuration, data loading, preprocessing, evaluation, plotting)
- `results/` exported figures and metric tables
- `report/` report drafts and final PDF

## Team

See `members.txt`.
