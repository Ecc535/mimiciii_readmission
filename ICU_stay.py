import pandas as pd

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
icustays['switch'] = (
    icustays['FIRST_CAREUNIT'] != icustays['LAST_CAREUNIT']
).astype(int)

# --- 2) Aggregate core ICU metrics ---

# a) number of distinct ICU stays
num_stays = (
    icustays
    .groupby('HADM_ID')['ICUSTAY_ID']
    .nunique()
    .rename('num_icu_stays')
)

# b) total ICU hours across all stays
total_los = (
    icustays
    .groupby('HADM_ID')['icu_los_hours']
    .sum()
    .rename('total_icu_hours')
)
icu_switch = (
    icustays
    .groupby('HADM_ID')['switch']
    .sum()
    .rename('icu_switch')
)
# c) distinct care‐unit visits (union of first+last care units)
visits_long = pd.concat([
    icustays[['HADM_ID','ICUSTAY_ID','FIRST_CAREUNIT']]
      .rename(columns={'FIRST_CAREUNIT':'care_unit'}),
    icustays[['HADM_ID','ICUSTAY_ID','LAST_CAREUNIT']]
      .rename(columns={'LAST_CAREUNIT':'care_unit'})
], ignore_index=True)
visits_per_stay = visits_long.drop_duplicates(
    subset=['ICUSTAY_ID','care_unit']
)
# Now group and pivot to get counts per care_unit
unit_counts = (
    visits_per_stay
    .groupby(['HADM_ID','care_unit'])
    .size()
    .unstack(fill_value=0)
)

# --- 3) First & last ICU stay LOS ---

# sort so that .first()/.last() map to chronological order
icustays_sorted = icustays.sort_values(['HADM_ID','INTIME'])

los_agg = (
    icustays_sorted
    .groupby('HADM_ID')['icu_los_hours']
    .agg(
        first_icu_los_hours='first',
        last_icu_los_hours='last',
        mean_icu_los_hours  = 'mean',
        # std_icu_los_hours = 'std',
        diff_icu_los_hours  = lambda x: x.max() - x.min()
    )
)


# --- 4) Combine into one ICU‐features table ---
icu_agg = pd.concat(
    [num_stays, total_los, unit_counts, los_agg, icu_switch],
    axis=1
).reset_index()

# --- 5) Inspect the results ---
pd.set_option('display.max_columns', None)
print(icu_agg.sample(5).to_string())
print(icu_agg.shape)

icu_agg.to_csv('icu_features.csv', index=False)