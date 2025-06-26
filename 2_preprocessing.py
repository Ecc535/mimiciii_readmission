import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder
import numpy as np
visualize = False
df = pd.read_csv("../src/dataframe_clean.csv")

# split X and Y
Y_df = df.set_index('HADM_ID')[['readmission_30']]
Y_df.to_csv("../src/Y_df.csv")
X = df.drop(
    columns=[
        "SUBJECT_ID",
        "readmission_30",
        "readmission_60",
        "readmit_gap",
    ],
    errors="ignore"
)
# identify numeric and categorical
X_cat = X[['GENDER']].copy()
X_num = X.drop(columns=['GENDER'])

# categorical encode 0 -> female; 1 -> male
ohe = OneHotEncoder(sparse_output=False, drop='first')
X_cat_encoded = ohe.fit_transform(X_cat)
cat_feature_names = ohe.get_feature_names_out(X_cat.columns)
X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=cat_feature_names, index=X_cat.index)

# outliers filter
# p_low, p_high = 0.01, 0.99
# lower = X_num.quantile(p_low)
# upper = X_num.quantile(p_high)
#
# # 2) Build mask: keep rows where *all* features are within those percentiles
# mask = ((X_num >= lower) & (X_num <= upper)).all(axis=1)
#
# # 3) Filter
# X_num_trim = X_num.loc[mask]
# X_cat_trim = X_cat.loc[mask]
# Y_trim     = Y.loc[mask]
#
# print(f"Kept {len(X_num_trim)} of {len(X_num)} rows after 1–99% trimming")


X_processed = pd.concat([X_cat_encoded_df, X_num], axis=1)
print(X_processed.head())
X_processed.to_csv("../src/X_df.csv")
# visualization
if visualize:
    const_cols = [
        col
        for col in X_num.columns
        if X_num[col].nunique(dropna=True) <= 1
    ]
    print("Dropping constant columns:", const_cols)
    X_num_plot = X_num.drop(columns=const_cols)
    # Scatter matrix of raw numeric features (no outlier filter)
    fig = scatter_matrix(
        X_num_plot,
        alpha=0.3,
        figsize=(30, 30),
        diagonal='kde'
    )
    plt.suptitle("Raw Numeric Features (No Outlier Filter)", y=0.92)
    plt.savefig('../figures/scatter_pre_no_outlier_filter.png')
    plt.clf()

