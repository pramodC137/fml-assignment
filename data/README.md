# Dataset: Sri Lanka Dengue Surveillance (2006 to 2025)

## Source

| Item | Detail |
| --- | --- |
| Title | Dengue_Research_2006-2026 (Weekly Dengue Surveillance Dataset for Sri Lanka, 2006 to 2025) |
| Publisher | Tharindu Heshan, Kaggle |
| Link | https://www.kaggle.com/datasets/tharinduheshan/dengue-research-2006-2026 |
| Licence | Apache 2.0 |
| Version | 1 (initial release, 26 June 2026) |
| File | `raw/dengue_srilanka_full_2006_2025.csv` (6,063,376 bytes) |
| Downloaded on | 10 October 2026 |

**Provenance (open item):** the Kaggle page does not state where the case counts, climate and population figures come from. These sources are to be confirmed with the uploader. Until confirmed, units marked *assumed* below are inferred from the data, not documented by the publisher. The case counts are also below the official national totals in most years (see *Cross-check against official case totals* below).

## Context

Dengue is a mosquito-borne viral disease and a major public health burden in Sri Lanka, with national epidemics in 2009, 2017 and 2019. This dataset combines weekly district-level dengue case counts with climate, geographic and demographic attributes, for outbreak forecasting and early-warning research.

## Structure

- **24,775 rows x 28 columns**, one row per district per epidemiological week
- **Complete panel:** all 25 districts x 991 weeks, no duplicate district-weeks
- **Time span:** week starting 23 December 2006 (week 52 of 2006) to the week of 13 to 19 December 2025
- 2006 contains a single week (25 rows); three years contain a week 53 (75 rows)
- Weeks normally span 6 days from start to end date; one week spans 5 days and one spans 7 days (25 rows each)
- **Missing values** occur only in the three lag columns, and only at the start of each district’s series where no earlier week exists: `cases_lag1` 25 rows, `rainfall_lag2` 50 rows, `rainfall_lag4` 100 rows
- **No invalid values:** no negative cases or rainfall, no minimum temperature above the maximum, no humidity above 100%

## Cross-check against official case totals

The yearly sums of `cases` (all 25 districts, by the dataset's `year` column) were compared with the national totals published by the Epidemiology Unit of the Ministry of Health.

| Year | Dataset | Official total | Dataset as share of official |
| --- | --- | --- | --- |
| 2012 | 23,527 | 44,461 | 53% |
| 2013 | 22,387 | 32,063 | 70% |
| 2014 | 35,314 | 47,502 | 74% |
| 2015 | 22,313 | 29,777 | 75% |
| 2016 | 35,192 | 55,150 | 64% |
| 2017 | 174,687 | 186,101 | 94% |
| 2018 | 51,321 | 51,659 | 99% |
| 2019 | 94,780 | 105,049 | 90% |
| 2020 | 32,222 | 31,162 | 103% |
| 2022, 1 January to 27 October | 26,297 | over 62,000 | about 42% |

Sources of the official figures: Epidemiology Unit annual totals for 2012 to 2020 as compiled at https://en.wikipedia.org/wiki/Dengue_pandemic_in_Sri_Lanka (to be confirmed against the annual reports on epid.gov.lk); the 2022 figure as reported by News 1st on 27 October 2022 (https://newsfirst.lk/2022/10/27/dengue-cases-spike-in-sri-lanka-over-60-000-reported-so-far).

**Finding.** The dataset covers every week for all 25 districts, so the shortfall is not caused by missing rows. It is close to the official totals in 2017 to 2020 but holds only about half of them in some years (2012 and 2022). The reason is not documented. Possible explanations, none yet confirmed, are provisional weekly figures rather than final totals, a different set of reporting areas (the Epidemiology Unit reports 26 health areas, including Kalmunai, against 25 districts here), or partly reconstructed values.

**Consequence for the project.** The outbreak thresholds are set per district from this same data, and both models and all baselines are scored on it, so the comparison between them stays valid. The results describe this dataset, however, and may not carry over to the complete official counts. This is stated as a limitation in the report.

## Data dictionary

| # | Column | Type | Unit | Range / values | Missing % | Unique | Meaning | Planned role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `year` | Integer | year | 2006 to 2025 | 0.0 | 20 | Calendar year of the epidemiological week | Split key / feature |
| 2 | `week` | Integer | week number | 1 to 53 | 0.0 | 53 | Epidemiological week of the year (1 to 53) | Feature |
| 3 | `start.date` | Date (YYYY-MM-DD) | date | 2006-12-23 to 2025-12-13 | 0.0 | 991 | First day of the epidemiological week | Split key and sort order, not a feature |
| 4 | `end.date` | Date (YYYY-MM-DD) | date | 2006-12-29 to 2025-12-19 | 0.0 | 991 | Last day of the epidemiological week | Drop (redundant with start.date) |
| 5 | `month` | Integer | month | 1 to 12 | 0.0 | 12 | Calendar month of the week | Feature |
| 6 | `season` | Categorical (2) | - | Maha, Yala | 0.0 | 2 | Agricultural season. Maha = October to March, Yala = April to September | Candidate feature (derived from month) |
| 7 | `monsoon_season` | Categorical (3) | - | Inter-monsoon, NE Monsoon, SW Monsoon | 0.0 | 3 | Monsoon period. NE Monsoon = November to February, SW Monsoon = May to September, Inter-monsoon = March, April, October | Candidate feature (derived from month) |
| 8 | `is_epidemic_year` | Binary (0/1) | - | 0 / 1 (share of 1s: 15.8%) | 0.0 | 2 | Flag for the national epidemic years 2009, 2017 and 2019, applied with hindsight (2023 is not flagged) | DROP: leakage (hindsight label) |
| 9 | `district` | Categorical (25) | - | 25 values, e.g. Ampara, Anuradhapura, Badulla | 0.0 | 25 | Administrative district (all 25 districts of Sri Lanka) | Feature (encoded) |
| 10 | `province` | Categorical (9) | - | 9 values, e.g. Central, Eastern, North Central | 0.0 | 9 | Province containing the district | Drop (fully determined by district) |
| 11 | `latitude` | Float | decimal degrees | 5.95 to 9.66 | 0.0 | 25 | District latitude (one fixed value per district) | Candidate feature (district proxy) |
| 12 | `longitude` | Float | decimal degrees | 79.83 to 81.7 | 0.0 | 25 | District longitude (one fixed value per district) | Candidate feature (district proxy) |
| 13 | `coastal_district` | Binary (0/1) | - | 0 / 1 (share of 1s: 56.0%) | 0.0 | 2 | 1 if the district has a coastline (14 of 25 districts) | Feature |
| 14 | `elevation_category` | Categorical (3) | - | Highland, Low, Mid | 0.0 | 3 | Elevation band. Low (17 districts), Mid (7), Highland (Nuwara Eliya only) | Feature (ordinal) |
| 15 | `area_km2` | Integer | km2 | 699 to 7,179 | 0.0 | 25 | Land area of the district (one fixed value per district) | Candidate feature (district proxy) |
| 16 | `population` | Integer | persons | 89,200 to 2,498,473 | 0.0 | 468 | District population, one value per district per year | Feature |
| 17 | `population_source` | Categorical (2) | - | actual, predicted | 0.0 | 2 | Whether population is a census or official figure (‘actual’, 2014 to 2024) or interpolated or projected (‘predicted’, 2006 to 2025) | Metadata, not a feature (report as limitation) |
| 18 | `population_density` | Float | persons per km2 | 35.24 to 3,574.35 | 0.0 | 482 | population / area_km2 | Feature |
| 19 | `cases` | Integer | cases per week | 0 to 2,631 | 0.0 | 532 | Reported dengue cases in the district for the week | TARGET SOURCE (outbreak label built from this) |
| 20 | `incidence_rate` | Float | cases per 100,000 per week | 0 to 150.89 | 0.0 | 10480 | Exactly cases / population x 100,000 for the same week (verified) | DROP: leakage (rescaled target) |
| 21 | `cases_lag1` | Float | cases per week | 0 to 2,631 | 0.1 | 531 | Cases in the same district the previous week (verified 100% match) | Feature (legitimate lag) |
| 22 | `cases_zscore` | Float | standard deviations | -0.87 to 13.95 | 0.0 | 3102 | (cases - district mean) / district SD, computed over the full 2006 to 2025 series (verified) | DROP: leakage (target plus future statistics) |
| 23 | `rainfall_mm` | Float | mm (weekly total, assumed) | 0 to 647.7 | 0.0 | 2395 | Rainfall in the district during the week | Feature |
| 24 | `rainfall_lag2` | Float | mm | 0 to 647.7 | 0.2 | 2394 | Rainfall in the same district two weeks earlier (verified 100% match) | Feature (legitimate lag) |
| 25 | `rainfall_lag4` | Float | mm | 0 to 605.1 | 0.4 | 2370 | Rainfall in the same district four weeks earlier (verified 100% match) | Feature (legitimate lag) |
| 26 | `temp_max_c` | Float | deg C (weekly mean, assumed) | 16.67 to 38.8 | 0.0 | 1619 | Mean daily maximum temperature for the week | Feature |
| 27 | `temp_min_c` | Float | deg C (weekly mean, assumed) | 7.49 to 29.13 | 0.0 | 1377 | Mean daily minimum temperature for the week | Feature |
| 28 | `humidity_pct` | Float | % (weekly mean, assumed) | 50.57 to 97.57 | 0.0 | 346 | Mean relative humidity for the week | Feature |

A machine-readable copy is in `data_dictionary.csv`.

The *Planned role* column records the plan made before preprocessing. The final decision for every column, with its reason, is in `results/metrics/preprocessing_column_decisions.csv` (`notebooks/02a_preprocessing.ipynb`, section 1).

## Leakage audit

Each derived column was recomputed from the raw columns to establish exactly how it was built.

| Column | Finding | Decision |
| --- | --- | --- |
| `incidence_rate` | Exactly `cases / population x 100,000` for the same week | Drop: it is the target rescaled |
| `cases_zscore` | Standardised with each district’s mean and SD over the full 2006 to 2025 series (population SD) | Drop: contains the target and statistics from future years |
| `is_epidemic_year` | 1 only for 2009, 2017 and 2019. 2023 (about 71,000 cases) is 0, so this is a hand-applied historical label | Drop: hindsight not available at prediction time |
| `cases_lag1` | Previous week’s cases, same district (100% match) | Keep |
| `rainfall_lag2`, `rainfall_lag4` | Rainfall 2 and 4 weeks earlier, same district (100% match) | Keep |

## Redundant columns

- `latitude`, `longitude` and `area_km2` take exactly 25 values, one per district, so they act as district identifiers
- `province` is fully determined by `district`
- `season` and `monsoon_season` are deterministic functions of `month`
- `end.date` is always `start.date` plus 5 to 7 days
- `population_source` describes data quality rather than the district: population is interpolated or projected (‘predicted’) for 10,450 rows

## Rules for this folder

Raw files in `raw/` are never modified. All transformations are done in code by the shared modules in `src/` and are rebuilt in memory each time a notebook runs, so no processed copy of the data is stored.
