"""
Step 1: Extract, clean and label the DHS Kids Recode (KR) file
so it mirrors the LSMS-based dataset used in the base paper:
Mgomezulu et al. (2025) "Advancing predictive analytics in child
malnutrition..." Human Nutrition & Metabolism 42, 200340.

Outcome variables (WHO standard, matches paper's HAZ/WHZ Table 1 defn):
    stunting = 1 if HAZ <= -2.00  (hw70 <= -200)
    wasting  = 1 if WHZ <= -2.00  (hw72 <= -200)

Predictor mapping paper -> DHS (documented in chat):
    education         -> v106 (mother's educ level, 0-3) [head-level unavailable in KR]
    land               -> NOT AVAILABLE in DHS (dropped)
    gender (hh head)   -> v151
    age (hh head)      -> v012 (mother's age; head age unavailable in KR)
    household size     -> v136
    total income       -> v190 / v191 (wealth index quintile / factor score, DHS's
                           standard income proxy since DHS never collects cash income)
    residence           -> v025 (urban=1/rural=2 in DHS coding)
    distance to market  -> v467d (categorical "distance to facility is a problem":
                           0=not asked,1=big problem,2=not big problem) -- DHS has
                           no continuous km distance-to-market variable, so this is
                           a categorical proxy, not equivalent to the paper's km measure
    HDDS/FCS/MAHFP       -> NOT AVAILABLE in DHS (dropped; flagged as a limitation)

Extra child-level covariates the paper's household model implicitly needed but
DHS requires at child level (kept since HAZ/WHZ are inherently child-specific):
    child age in months -> hw1
    child sex            -> b4
    birth order           -> bord
"""
import pandas as pd
import numpy as np

SRC = "../data/raw/IAKR7EFL.DTA"

extra_cols = ['m18', 'm19', 'm19a', 'm4', 'h9', 'h11', 'h22', 'h31', 
              'v437', 'v438', 'v445', 'v501', 'm14', 'v113', 'v116', 'v161']

cols = ['caseid','v001','v002','v003',
        'hw70','hw72',
        'hw1','b4','bord',
        'v106','v012','v136','v190','v191','v025','v151','v467d'] + extra_cols

print("Reading DHS KR file (selected columns only)...")
df = pd.read_stata(SRC, columns=cols, convert_categoricals=False)
print("Raw shape:", df.shape)

# ---- Drop rows with missing / flagged (9998) anthropometric z-scores ----
for c in ['hw70', 'hw72']:
    df = df[df[c].notna()]
    df = df[df[c] != 9998]
    df = df[(df[c] >= -600) & (df[c] <= 600)]

print("Shape after dropping missing/flagged HAZ & WHZ:", df.shape)

# ---- Outcome labels (WHO cutoff: z <= -2.00 SD) ----
df['stunting'] = (df['hw70'] <= -200).astype(int)
df['wasting']  = (df['hw72'] <= -200).astype(int)

print("\nStunting prevalence:", df['stunting'].mean().round(4))
print("Wasting prevalence:", df['wasting'].mean().round(4))

# ---- Clean and preprocess extra features ----
# m19 (birth weight): continuous. Map 9996 (not weighed) and 9998 (don't know) to NaN
df['m19'] = df['m19'].astype(float)
df.loc[df['m19'] >= 9996, 'm19'] = np.nan

# m4 (duration of breastfeeding): continuous. 94 -> 0. 95 -> child_age_months.
df['m4'] = df['m4'].astype(float)
df.loc[df['m4'] == 94, 'm4'] = 0.0
df.loc[df['m4'] == 95, 'm4'] = df.loc[df['m4'] == 95, 'hw1'].astype(float)

# v437 (mother weight): continuous. Map >= 9990 (not present/refused) to NaN. Convert decigrams to kg.
df['v437'] = df['v437'].astype(float)
df.loc[df['v437'] >= 9990, 'v437'] = np.nan
df['mother_weight'] = df['v437'] / 10.0

# v438 (mother height): continuous. Map >= 9990 to NaN. Convert mm to cm.
df['v438'] = df['v438'].astype(float)
df.loc[df['v438'] >= 9990, 'v438'] = np.nan
df['mother_height'] = df['v438'] / 10.0

# v445 (mother BMI): continuous. Map >= 9990 to NaN. Convert 2-decimal int to float.
df['v445'] = df['v445'].astype(float)
df.loc[df['v445'] >= 9990, 'v445'] = np.nan
df['mother_bmi'] = df['v445'] / 100.0

# m14 (number of antenatal visits): continuous. Map >= 98 (don't know) to NaN.
df['m14'] = df['m14'].astype(float)
df.loc[df['m14'] >= 98, 'm14'] = np.nan

# ---- Rename to paper-style variable names for clarity ----
df = df.rename(columns={
    'v106': 'education',
    'v012': 'age_hh_head_proxy',
    'v136': 'hhsize',
    'v190': 'wealth_quintile',
    'v191': 'wealth_score',
    'v025': 'residence',
    'v151': 'gender_hh_head',
    'v467d': 'dist_market_proxy',
    'hw1': 'child_age_months',
    'b4': 'child_sex',
    'bord': 'birth_order',
    
    # Extra columns
    'm18': 'birth_size',
    'm19': 'birth_weight',
    'm19a': 'birth_weight_source',
    'm4': 'breastfeeding_duration',
    'h9': 'measles_vaccine',
    'h11': 'diarrhea_recent',
    'h22': 'fever_recent',
    'h31': 'cough_recent',
    'v501': 'mother_marital_status',
    'm14': 'anc_visits',
    'v113': 'water_source',
    'v116': 'toilet_type',
    'v161': 'cooking_fuel',
})

keep = ['caseid','v001','v002','v003',
        'stunting','wasting',
        'education','age_hh_head_proxy','hhsize',
        'wealth_quintile','wealth_score','residence',
        'gender_hh_head','dist_market_proxy',
        'child_age_months','child_sex','birth_order',
        # Extra columns
        'birth_size', 'birth_weight', 'birth_weight_source', 'breastfeeding_duration',
        'measles_vaccine', 'diarrhea_recent', 'fever_recent', 'cough_recent',
        'mother_weight', 'mother_height', 'mother_bmi', 'mother_marital_status',
        'anc_visits', 'water_source', 'toilet_type', 'cooking_fuel']

df = df[keep]

df.to_parquet("../data/processed/dhs_clean.parquet", index=False)
print("\nSaved dhs_clean.parquet, final shape:", df.shape)
print(df.head())

