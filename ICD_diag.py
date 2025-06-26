import pandas as pd
import numpy as np

diag_events = pd.read_csv("../data/DIAGNOSES_ICD.csv",
                          usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])
def normalize_icd9(code: str) -> str:
    code = code.strip().upper()
    if not code:
        return "00000"
    # Handle E/V codes (“E000–E999” or “V000–V999”)
    if code[0] in ("E", "V"):
        letter = code[0]
        digits = code[1:].zfill(4)
        return letter + digits
    return code.zfill(5)
# fill NaNs
diag_events['ICD9_CODE'] = (
    diag_events['ICD9_CODE']
      .fillna('')
      .astype(str)
)
diag_events["icd9_norm"] = diag_events["ICD9_CODE"].apply(normalize_icd9)

# extract 3-digit prefix
diag_events['icd_prefix'] = (
    diag_events['icd9_norm']
               .astype(str)
               .str.extract(r'^(\d{3})')[0]
)
# define ID range
ranges = [
    (1,   139,  1),   # Infectious & parasitic
    (140, 239,  2),   # Neoplasms
    (240, 279,  3),   # Endocrine/metabolic
    (280, 289,  4),   # Blood disorders
    (290, 319,  5),   # Mental disorders
    (320, 389,  6),   # Nervous system
    (390, 459,  7),   # Circulatory
    (460, 519,  8),   # Respiratory
    (520, 579,  9),   # Digestive
    (580, 629, 10),   # Genitourinary
    (630, 679, 11),   # Pregnancy
    (680, 709, 12),   # Skin
    (710, 739, 13),   # musculoskeletal system & skeletal
    (740, 759, 14),   # Congenital anomalies
    (760, 779, 15),   # Certain perinatal conditions
    (780, 799, 16),   # Symptoms, signs & ill-defined conditions
    (800, 999, 17),   # Injury & poisoning
]
prefix_to_id = {}
for start, end, gid in ranges:
    for i in range(start, end+1):
        prefix_to_id[f"{i:03d}"] = gid

prefix_to_id['E'] = 18
prefix_to_id['V'] = 19
# extract the chapter key
def get_icd9_chapter_id(code: str) -> int:
    code = code.strip()
    if not code:
        return 0
    first = code[0]
    if first in ('E', 'V'):
        return prefix_to_id[first]
    pref = code[:3]
    return prefix_to_id.get(pref, 0)
diag_events['icd_chapter_id'] = diag_events['ICD9_CODE'].apply(get_icd9_chapter_id)

diag_agg = (
    diag_events
    .groupby('HADM_ID')
    .agg(
        total_diag_count       = ('ICD9_CODE',      'size'),
        unique_diag_codes      = ('ICD9_CODE',      'nunique'),
        distinct_diag_chapters = ('icd_chapter_id', 'nunique'),
        chapter_diag_list      = ('icd_chapter_id', lambda x: sorted(x.dropna().unique()))
    )
    .reset_index()
)

# chapter-count matrix
chap_count_matrix = (
    diag_events
    .groupby(['HADM_ID','icd_chapter_id'])
    .size()                              # count occurrences
    .unstack(fill_value=0)              # pivot to wide form
)

# 2) rename columns for clarity
chap_count_matrix.columns = [
    f"diag_chap_count_{int(chap)}"
    for chap in chap_count_matrix.columns
]

# 3) merge back onto your diag_agg (dropping the old list column)
diag_features_counts = (
    diag_agg
    .drop(columns=['chapter_diag_list'])
    .merge(chap_count_matrix, on='HADM_ID', how='left')
    .fillna(0)                          # admissions with no diagnoses get zeros
)

# 4) optionally convert to int
diag_features_counts[chap_count_matrix.columns] = (
    diag_features_counts[chap_count_matrix.columns].astype(int)
)

pd.set_option('display.max_columns', None)
print(diag_features_counts.head().to_string())
print("Shape:", diag_features_counts.shape)

diag_features_counts.to_csv("diag_features.csv", index=False)