import pandas as pd
<<<<<<< HEAD:ICU_stay.py

=======
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
>>>>>>> 276fecd (Update chart values and cpt code):src/ICU_stay.py
# --- 1) Load and preprocess ICUSTAYS ---
icustays = pd.read_csv(
    '../data/ICUSTAYS.csv',
    usecols=[
        'HADM_ID',
        'ICUSTAY_ID',
        'FIRST_CAREUNIT',
        'LAST_CAREUNIT',
        'INTIME',
        'OUTTIME'
    ]
)

# parse timestamps
icustays['INTIME']  = pd.to_datetime(icustays['INTIME'])
icustays['OUTTIME'] = pd.to_datetime(icustays['OUTTIME'])

# compute LOS in hours
icustays['icu_los_hours'] = (
    icustays['OUTTIME'] - icustays['INTIME']
).dt.total_seconds() / 3600
<<<<<<< HEAD:ICU_stay.py
icustays['switch'] = (
    icustays['FIRST_CAREUNIT'] != icustays['LAST_CAREUNIT']
).astype(int)

# --- 2) Aggregate core ICU metrics ---

# a) number of distinct ICU stays
=======


# number of distinct ICU stays
>>>>>>> 276fecd (Update chart values and cpt code):src/ICU_stay.py
num_stays = (
    icustays
    .groupby('HADM_ID')['ICUSTAY_ID']
    .nunique()
    .rename('num_icu_stays')
)

# --- 3) First & last ICU stay LOS ---

# sort so that .first()/.last() map to chronological order
icustays_sorted = icustays.sort_values(['HADM_ID','INTIME'])
mean_los = (
    icustays_sorted
    .groupby('HADM_ID')['icu_los_hours']
    .mean()
    .rename('mean_icu_los_hours')
)
last_careunit = (
    icustays_sorted
    .groupby('HADM_ID')['LAST_CAREUNIT']
    .last()
    .rename('last_careunit')
)
# --- 4) Combine into one ICU‐features table ---
icu_agg = pd.concat(
    [
        num_stays,                           # num_icu_stays
        mean_los,     # mean_icu_los_hours only
        last_careunit                        # new categorical column
    ],
    axis=1
).reset_index()

# --- 5) Inspect the results ---
pd.set_option('display.max_columns', None)
print(icu_agg.sample(5).to_string())
print(icu_agg.shape)

icu_agg.to_csv('icu_features.csv', index=False)

# # correlation heat map
# numeric_cols = icu_agg.drop(columns=['HADM_ID']).columns
# corr_mat = icu_agg[numeric_cols].corr(method='pearson')
#
# plt.figure(figsize=(10, 8))
# sns.heatmap(
#     corr_mat,
#     cmap='coolwarm',
#     square=True,
#     cbar_kws={'label': 'Pearson r'},
#     annot=False,          # use True for numbers in each cell
#     fmt='.2f',
#     linewidths=0.5
# )
# plt.title('Correlation matrix of aggregated ICU-stay features')
# plt.tight_layout()
# plt.show()
#
# # report highly correlated pairs
# high_corr_pairs = (
#     corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
#             .stack()
#             .rename('r')
#             .reset_index()
#             .query('abs(r) >= 0.8')
#             .sort_values('r', ascending=False)
# )
# print("\nHighly correlated feature pairs (|r| ≥ 0.8):")
# print(high_corr_pairs.to_string(index=False))