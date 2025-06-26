import pandas as pd
import numpy as np

proc_events = pd.read_csv("../data/PROCEDURES_ICD.csv",
                           usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])
proc_events['ICD9_CODE'] = proc_events['ICD9_CODE'].astype(str).str.zfill(4)

# extract prefix
proc_events['icd9_prefix'] = (
    proc_events['ICD9_CODE']
               .str.extract(r'^(\d{2})')[0]
)
ranges = [
    (1,   5,   1),
    (6,   7,   2),
    (8,   16,  3),
    (17,  25,  4),
    (30,  34,  5),
    (35,  39,  6),
    (40,  41,  7),
    (42,  54,  8),
    (55,  59,  9),
    (60,  64, 10),
    (65,  67, 11),
    (68,  68, 12),
    (69,  71, 13),
    (72,  72, 14),
    (73,  73, 15),
    (74,  74, 16),
    (75,  79, 17),
]
prefix_to_procchap_id = {}
for lo, hi, cid in ranges:
    for i in range(lo, hi+1):
        prefix_to_procchap_id[f"{i:02d}"] = cid

proc_events['proc_chapter_id'] = proc_events['icd9_prefix']\
    .map(prefix_to_procchap_id)\
    .fillna(0).astype(int)

proc_agg = (
  proc_events
  .groupby('HADM_ID')
  .agg(
    total_proc_count        = ('ICD9_CODE',       'size'),
    unique_proc_codes       = ('ICD9_CODE',       'nunique'),
    distinct_proc_chapters  = ('proc_chapter_id', 'nunique'),
    proc_chapter_list       = ('proc_chapter_id', lambda x: list(x.unique()))
  )
  .reset_index()
)
chap_counts = (
    proc_events
    .groupby(['HADM_ID','proc_chapter_id'])
    .size()
    .unstack(fill_value=0)
)
chap_counts.columns = [
    f"proc_chap_count_{int(chap)}"
    for chap in chap_counts.columns
]

proc_features = (
    proc_agg
    .drop(columns=['proc_chapter_list'])
    .merge(chap_counts, on='HADM_ID', how='left')
    .fillna(0)
)

# ensure integer dtype for count columns
count_cols = [c for c in proc_features.columns if c.startswith('proc_chap_count_')]
proc_features[count_cols] = proc_features[count_cols].astype(int)

pd.set_option('display.max_columns', None)
print(proc_features.head().to_string())
print("Shape:", proc_features.shape)
proc_features.to_csv("proc_features.csv", index=False)