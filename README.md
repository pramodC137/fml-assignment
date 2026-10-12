# Fundamentals of Machine Learning: Programming Assignment

## Problem

Dengue outbreaks in Sri Lanka strain hospitals and vector-control teams, and a response is most effective when it starts before cases peak. This project predicts whether a district will be in a dengue **outbreak two weeks ahead**, using recent case counts, rainfall, temperature, humidity and district characteristics. An outbreak is defined as weekly cases above the district’s mean plus one standard deviation, with the threshold fitted on training years only.

## Dataset

Sri Lanka Dengue Surveillance, 2006 to 2025: weekly dengue cases for all 25 districts with climate, geographic and demographic features. 24,775 rows x 28 columns.

- Source: https://www.kaggle.com/datasets/tharinduheshan/dengue-research-2006-2026
- Full description, data dictionary and leakage audit: `data/README.md`

## Approach

- Supervised binary classification
- Temporal split: train 2007 to 2020, validation 2021 to 2022, test 2023 to 2025
- Baselines: persistence model and dummy classifier
- Primary metric: PR-AUC, with recall, precision, F1, MCC and ROC-AUC

## Algorithms

- Member A: Logistic Regression
- Member B: Random Forest

## Team

See `members.txt`.

## How to run

1. `pip install -r requirements.txt`
2. Run notebooks in numerical order from `notebooks/`

## Repository structure

- `data/` raw and processed datasets, data dictionary
- `notebooks/` EDA, preprocessing, models, comparison
- `src/` shared modules used across notebooks
- `results/` exported figures and metric tables
- `report/` report drafts and final PDF
