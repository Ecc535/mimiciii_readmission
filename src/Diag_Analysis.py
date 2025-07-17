from pathlib import Path
import pandas as pd

DATA_DIR   = Path("../data")
OUT_FILE   = Path("icd_features.csv")

DX_FILE    = DATA_DIR / "DIAGNOSES_ICD.csv"
CCS_MAP    = DATA_DIR / "ccs_single_dx_icd9.csv"   # download from HCUP
CCI_MAP    = DATA_DIR / "cci_icd9_map.csv"         # e.g. from quan_cci_crosswalk

MIN_CCS_FREQ = 25

# ------------------------------------------------------------
# 1. Load & basic cleaning
# ------------------------------------------------------------
dx = (pd.read_csv(DX_FILE,
                  usecols=["HADM_ID", "SEQ_NUM", "ICD9_CODE"],
                  dtype={"ICD9_CODE": str})
        .assign(ICD9_CODE=lambda d: d["ICD9_CODE"].str.zfill(5))
        .loc[lambda d: ~d["ICD9_CODE"].str.startswith(("E", "V"),na=False)]      # strip E/V
        .drop_duplicates(subset=["HADM_ID", "ICD9_CODE"])               # de-dupe
)

print("Clean diagnoses shape:", dx.shape)

# ------------------------------------------------------------
# 2. Charlson Comorbidity Index (CCI)
# ------------------------------------------------------------
cci_map = (pd.read_csv(CCI_MAP, dtype={"ICD9": str})
             .assign(ICD9=lambda d: d["ICD9"].str.zfill(5)))

dx_cci  = (dx.merge(cci_map, left_on="ICD9_CODE", right_on="ICD9", how="inner")
             .drop_duplicates(subset=["HADM_ID", "CCI_CONDITION"]))

cci_flags = (dx_cci.assign(flag=1)
                     .pivot_table(index="HADM_ID",
                                  columns="CCI_CONDITION",
                                  values="flag",
                                  aggfunc="max",
                                  fill_value=0)
                     .astype("int8"))

# Charlson weighted score
weights = (cci_map[["CCI_CONDITION", "WEIGHT"]]
             .drop_duplicates()
             .set_index("CCI_CONDITION")["WEIGHT"])
cci_score = (cci_flags * weights).sum(axis=1).rename("CCI_SCORE").astype("int8")

print("CCI flags shape:", cci_flags.shape)

# ------------------------------------------------------------
# 3. CCS single-level bag-of-codes
# ------------------------------------------------------------
ccs_map = (pd.read_csv(CCS_MAP, dtype={"ICD9": str})
             .assign(ICD9=lambda d: d["ICD9"].str.zfill(5)))

dx_ccs = (dx.merge(ccs_map, left_on="ICD9_CODE", right_on="ICD9", how="left")
            .dropna(subset=["CCS_CATEGORY"]))

# optional: prune very rare CCS groups to control sparsity
freq = dx_ccs.groupby("CCS_CATEGORY")["HADM_ID"].nunique()
keep  = freq[freq >= MIN_CCS_FREQ].index
dx_ccs = dx_ccs[dx_ccs["CCS_CATEGORY"].isin(keep)]

ccs_bag = (dx_ccs.assign(flag=1)
                    .pivot_table(index="HADM_ID",
                                 columns="CCS_CATEGORY",
                                 values="flag",
                                 aggfunc="max",
                                 fill_value=0)
                    .astype("int8"))

print("CCS bag shape:", ccs_bag.shape)

# ------------------------------------------------------------
# 4. Combine & save
# ------------------------------------------------------------
features = (pd.concat([ccs_bag, cci_flags, cci_score], axis=1)
              .reset_index()
              .fillna(0)
              .astype({col: "int8" for col in ccs_bag.columns})   # tighten dtypes
)

features.to_csv(OUT_FILE, index=False)
print("Saved →", OUT_FILE.resolve())