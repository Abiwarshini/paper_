# Base Paper Replication — Child Malnutrition Prediction (DHS India)

Exact-methodology replication of:
Mgomezulu et al. (2025), "Advancing predictive analytics in child malnutrition:
Machine, ensemble and deep learning models with balanced class distribution
for early detection of stunting and wasting," Human Nutrition & Metabolism 42, 200340.

## Setup

1. Put IAKR7EFL.DTA in data/raw/
2. python -m venv venv  (use Python 3.12, NOT 3.14 -- TensorFlow needs <=3.13)
3. venv\Scripts\activate
4. pip install -r requirements.txt
5. cd src
6. python 01_prepare_data.py
7. python 02_run_models.py    (this is the long-running full pipeline)

## Variable mapping: paper (LSMS) -> this replication (DHS)

| Paper variable       | DHS variable used          | Notes                                                          |
|-----------------------|------------------------------|------------------------------------------------------------------|
| Stunting outcome      | hw70 (HAZ, WHO) <= -200      | WHO cutoff z <= -2.00 SD                                        |
| Wasting outcome       | hw72 (WHZ, WHO) <= -200      | WHO cutoff z <= -2.00 SD                                        |
| Education (hh head)  | v106 (mother's educ level)   | DHS KR only has mother's education                              |
| Land size             | NOT AVAILABLE                | DHS does not collect landholding data                           |
| Gender (hh head)      | v151                         | Direct match                                                     |
| Age (hh head)         | v012 (mother's age)          | Proxy                                                            |
| Household size        | v136                         | Direct match                                                     |
| Total household income| v190/v191 (wealth index)     | DHS proxy for income                                              |
| Residence urban/rural | v025                         | Direct match                                                     |
| Distance to market     | v467d (categorical proxy)    | Not a continuous km distance like LSMS                          |
| HDDS/FCS/MAHFP         | NOT AVAILABLE                | DHS has no equivalent module                                    |

## Known limitation

Land, market-distance, and dietary-diversity variables (top feature importances
in the original paper) are unavailable in DHS. Flag this explicitly to reviewers.
