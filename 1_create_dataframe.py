import pandas as pd
import numpy as np
from pathlib import Path


# load admission, patients, labevents, chartevents
admissions = pd.read_csv("../data/ADMISSIONS.csv")
patients = pd.read_csv("../data/PATIENTS.csv")

# demographics: age, gender
admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
patients['DOB']         = pd.to_datetime(patients['DOB'])

df = (
    admissions[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]
    .merge(patients[['SUBJECT_ID', 'DOB', 'GENDER']], on='SUBJECT_ID', how='left')
)

# calculate age
df['AGE'] = df['ADMITTIME'].dt.year - df['DOB'].dt.year

mask = (
    (df['ADMITTIME'].dt.month  < df['DOB'].dt.month) |
    ((df['ADMITTIME'].dt.month == df['DOB'].dt.month) &
     (df['ADMITTIME'].dt.day   < df['DOB'].dt.day))
)
df.loc[mask, 'AGE'] -= 1

# cap at 90 per MIMIC policy
df.loc[df['AGE'] > 89, 'AGE'] = 90

age = df.set_index('SUBJECT_ID')['AGE']
gender = df.set_index('SUBJECT_ID')['GENDER']
# print(gender.head())

# load chart values
chart_values = pd.read_csv("../data/CHARTEVENTS.csv",
                           usecols=['HADM_ID', 'ITEMID', 'VALUENUM'],
                           low_memory=True)
# define needed features
chart_itemids = {
    211:     "HeartRate",
    618:     "RespRate", # respiratory rate
    220050:  "Art_SBP", # arterial blood pressure systolic
    220179:  "NIBP_SBP", # non invasive blood pressure systolic
    220051:  "Art_DBP", # diastolic
    220180:  "NIBP_DBP",
    220052:  "Art_MBP", # mean
    220181:  "NIBP_MBP",
    223761:  "Temp_F",
    228232:  "SpO2" # oxygen saturation
}
# filter
mask = chart_values["ITEMID"].isin(chart_itemids.keys())
chart_filtered = chart_values.loc[mask].copy()
chart_filtered["measurement"] = chart_filtered["ITEMID"].map(chart_itemids)
chart_agg = (
    chart_filtered
    .groupby(["HADM_ID", "measurement"])["VALUENUM"]
    .mean()
    .unstack(fill_value=np.nan)
    .reset_index()
)

# print(chart_agg.head())

# load lab values
lab_values = pd.read_csv("../data/LABEVENTS.csv",
                         usecols=['HADM_ID', 'ITEMID', 'VALUENUM'],
                         low_memory=True)
lab_itemids = {
    # in blood fluid
    50809: "Glucose",
    50912: "Creatinine",
    50983: "Sodium",
    50971: "Potassium",
    50882: "Bicarbonate",
    50902: "Chloride",
    51006: "Urea_Nitrogen",
    51301: "WBC", # white blood cells
    51221: "Hematocrit",
    51222: "Hemoglobin",
    51265: "Platelet_Count",
    51049: "Total_Bilirubin",
    50862: "Albumin",
    50813: "Lactate",
    50820: "pH",
    50818: "pCO2",
    50821: "pO2"
}
relevant_itemids = list(lab_itemids.keys())
lab_filtered = lab_values[lab_values["ITEMID"].isin(relevant_itemids)].copy()
lab_filtered["measurement"] = lab_filtered["ITEMID"].map(lab_itemids)
lab_agg = (
    lab_filtered
    .groupby(["HADM_ID", "measurement"])["VALUENUM"]
    .mean()
    .unstack(fill_value=np.nan)
    .reset_index()
)
lab_agg.columns.name = None
# print(lab_agg.head())

# CPT codes
# chartdate for timing of procedure
cpt_events = pd.read_csv('../data/CPTEVENTS.csv',
                         usecols=['SUBJECT_ID','HADM_ID','CHARTDATE','CPT_NUMBER'])
d_cpt      = pd.read_csv('../data/D_CPT.csv',
                         usecols=['SECTIONHEADER','SUBSECTIONHEADER',
                                  'MINCODEINSUBSECTION','MAXCODEINSUBSECTION'])

# map code to section
intervals = pd.IntervalIndex.from_arrays(
    left=d_cpt['MINCODEINSUBSECTION'],
    right=d_cpt['MAXCODEINSUBSECTION'],
    closed='both'
)
d_cpt = d_cpt.assign(interval=intervals)

# cpt timing
cpt_events['CHARTDATE'] = pd.to_datetime(cpt_events['CHARTDATE'])
cpt_events = cpt_events.merge(admissions, on='HADM_ID', how='left')
cpt_events['days_since_admission'] = (
    cpt_events['CHARTDATE'] - cpt_events['ADMITTIME']
).dt.days

# look up
def lookup_cpt_sections(cpt_num):
    mask = intervals.contains(cpt_num)
    matches = d_cpt[mask]
    if matches.empty:
        return pd.Series({'sections': None, 'subsections': None})
    else:
        return pd.Series({
            'sections':    ';'.join(matches['SECTIONHEADER'].astype(str)),
            'subsections': ';'.join(matches['SUBSECTIONHEADER'].astype(str)),
        })

cpt_events[['section','subsection']] = (
    cpt_events['CPT_NUMBER']
              .apply(lookup_cpt_sections)
)
agg = cpt_events.groupby('HADM_ID').agg(
    total_cpt_count    = ('CPT_NUMBER','size'),
    unique_cpt_codes   = ('CPT_NUMBER','nunique'),
    distinct_sections  = ('section','nunique'),
    distinct_subsects  = ('subsection','nunique'),
)

timing_agg = cpt_events.groupby('HADM_ID').agg(
first_cpt_day   = ('days_since_admission','min'),   # e.g. 0 = same day
    last_cpt_day    = ('days_since_admission','max'),
    cpt_span_days   = ('days_since_admission', lambda x: x.max() - x.min()),
)

code_agg = cpt_events.groupby('HADM_ID').agg(
    min_cpt_number     = ('CPT_NUMBER','min'),
    max_cpt_number     = ('CPT_NUMBER','max')
    # cpt_number_range   = ('cpt_number', lambda x: x.max() - x.min()),
)

# high risk indicator
# high_risk_secs = {'Surgery','Cardiovascular','Neurosurgery'}  # customize as needed
# cpt_events['is_high_risk_section'] = cpt_events['section'].isin(high_risk_secs).astype(int)
#
# high_risk_agg = cpt_events.groupby('hadm_id').agg(
#     had_any_high_risk = ('is_high_risk_section','max'),
#     high_risk_count   = ('is_high_risk_section','sum')
# )
cpt_features = (
    agg
    # .join(high_risk_agg,  how='left')
    .join(timing_agg,     how='left')
    .join(code_agg,       how='left')
    .fillna(0)
    .reset_index()
)
print(cpt_features.head())

# ICD 9 code
diag_agg = pd.read_csv("diag_features.csv")
proc_agg = pd.read_csv("proc_features.csv")

# ICU stay
icu_agg = pd.read_csv("icu_features.csv")
# prepare the feature data frame
demo_feats = (
    df[['HADM_ID','SUBJECT_ID', 'AGE', 'GENDER']]
    .drop_duplicates(subset='HADM_ID')
    .set_index('HADM_ID')
)
lab_agg = lab_agg.dropna(subset=['HADM_ID']).set_index('HADM_ID')
chart_agg = chart_agg.dropna(subset=['HADM_ID']).set_index('HADM_ID')
cpt_features = cpt_features.dropna(subset=['HADM_ID']).set_index('HADM_ID')
diag_agg = diag_agg.dropna(subset=['HADM_ID']).set_index('HADM_ID')
proc_agg = proc_agg.dropna(subset=['HADM_ID']).set_index('HADM_ID')
icu_agg = icu_agg.dropna(subset=['HADM_ID']).set_index('HADM_ID')
feature_df = (
    demo_feats
      .join(lab_agg, how='left')
      .join(chart_agg, how='left')
      .join(cpt_features, how='left')
      .join(diag_agg, how='left')
      .join(proc_agg, how='left')
      .join(icu_agg, how='left')
      .reset_index()
)

print(feature_df.shape)
print(feature_df.columns)
print(feature_df.head())

# target dataframe
adm_counts = (
    admissions
      .groupby("SUBJECT_ID")["HADM_ID"]
      .nunique()
      .reset_index(name="n_admissions")
)

# 2. Filter to patients with >1 admissions & drop outlier
p99 = adm_counts["n_admissions"].quantile(0.99)
multi_adm = adm_counts[
    (adm_counts["n_admissions"] > 1) &
    (adm_counts["n_admissions"] <= p99)
].copy()

print(f"99th percentile cutoff = {p99:.0f} admissions")
print(multi_adm)

admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
# sort with order
admissions = admissions.sort_values(['SUBJECT_ID', 'ADMITTIME'])
admissions['next_admittime'] = (
    admissions
      .groupby('SUBJECT_ID')['ADMITTIME']
      .shift(-1)
)
admissions['readmit_gap'] = (
    admissions['next_admittime'] - admissions['DISCHTIME']
).dt.days
admissions['readmission_30'] = (
    admissions['readmit_gap']
      .between(1, 30)
      .fillna(0)
      .astype(int)
)
admissions['readmission_60'] = (
    admissions['readmit_gap']
      .between(1, 60)        # 1–60 days
      .astype(int)
)
admissions['n_admissions'] = (
    admissions.groupby('SUBJECT_ID')['HADM_ID']
              .transform('nunique')
)
mask = (
    (admissions['n_admissions'] == 1) |
    (admissions['next_admittime'].notna())
)
target_df = admissions.loc[mask,
    ['SUBJECT_ID','HADM_ID','readmission_30','readmission_60']
].copy()
# print(target_df.sample(10))

# prepare the final data frame
final_df = feature_df.merge(
    target_df[['HADM_ID','readmission_30','readmission_60']],
    on='HADM_ID',
    how='inner'
)
print(final_df.shape)
print(final_df[['HADM_ID','readmission_30','readmission_60']].head())
final_df.to_csv("../src/dataframe_clean.csv", index=False)