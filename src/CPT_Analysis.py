from pathlib import Path
import pandas as pd

"""
CPTcodes.py
-----------
Clean MIMIC‑III CPTEVENTS and produce a **bag‑of‑codes** (one‑hot / binary)
feature matrix per hospital admission.

Pipeline
========
1.  Load CPTEVENTS and ADMISSIONS.
2.  *Basic cleaning*  
    • strip modifiers → 5‑digit base code  
    • drop daily E/M codes (99200‑99499)  
    • deduplicate ⟨HADM_ID, code⟩  
    • prune codes that occur in < MIN_FREQ admissions
3.  Pivot to one‑hot rows (HADM_ID × CPT_CODE) and save as `cpt_bag.csv`.

Tune `DATA_DIR` and `MIN_FREQ` as needed.
"""
# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MIN_FREQ  = 25                            # keep codes seen ≥ this many hadm
OUT_FILE  = Path("cpt_values.csv")           # output feature matrix

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
admissions_cols = ["HADM_ID", "ADMITTIME"]
cptevents_cols  = ["HADM_ID", "CPT_NUMBER", "CPT_CD", "CPT_SUFFIX", "CHARTDATE"]

adm = pd.read_csv( "../data/ADMISSIONS.csv",
                  usecols=admissions_cols,
                  parse_dates=["ADMITTIME"])

cpt = pd.read_csv( "../data/CPTEVENTS.csv",
                  usecols=cptevents_cols,
                  parse_dates=["CHARTDATE"])

# ------------------------------------------------------------------
# 2. Basic cleaning steps
# ------------------------------------------------------------------
# 2‑a) core 5‑digit code, zero‑padded
cpt['BASE_CODE'] = pd.to_numeric(cpt['CPT_NUMBER'], errors='coerce')
cpt = cpt.dropna(subset=['BASE_CODE'])
cpt['BASE_CODE'] = (cpt['BASE_CODE'].astype(int).astype(str).str.zfill(5))

# 2‑b) drop daily Evaluation & Management (E/M) codes 99200‑99499
mask_em = cpt["BASE_CODE"].between("99200", "99499")
cpt = cpt.loc[~mask_em]

# 2‑c) deduplicate within an admission
cpt = cpt.drop_duplicates(subset=["HADM_ID", "BASE_CODE"])

# 2‑d) prune rare codes (sparsity control)
freq = cpt.groupby("BASE_CODE")["HADM_ID"].nunique()
keep_codes = freq[freq >= MIN_FREQ].index
cpt = cpt[cpt["BASE_CODE"].isin(keep_codes)]

print(f"Retained {len(keep_codes)} CPT codes that appear in ≥ {MIN_FREQ} admissions.")

# ------------------------------------------------------------------
# 3. Bag‑of‑codes pivot (binary)
# ------------------------------------------------------------------
cpt["flag"] = 1
bag = (
    cpt.pivot_table(index="HADM_ID",
                    columns="BASE_CODE",
                    values="flag",
                    aggfunc="max",
                    fill_value=0)
    .astype("int8")
    .reset_index()
)

print("Bag‑of‑codes matrix shape:", bag.shape)

# Optional extra: total code count per admission (before dedup)
totals = (
    cpt.groupby("HADM_ID")["CPT_NUMBER"].size()
        .rename("total_cpt_count")
        .astype("int16")
)

bag = bag.merge(totals, on="HADM_ID", how="left")

# ------------------------------------------------------------------
# 4. Save
# ------------------------------------------------------------------
bag.to_csv(OUT_FILE, index=False)
print("Saved →", OUT_FILE.resolve())

