import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score

# feature and target loading
Y_df = pd.read_csv("Y_df.csv")
X_df = pd.read_csv("X_df.csv")

# merge on hadm_id
df = pd.merge(X_df, Y_df, on="HADM_ID", how="inner")

# separate
X = df.drop(columns=["HADM_ID", "readmission_30"])
y = df["readmission_30"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(y_train.value_counts(normalize=True))

# initialize and fit random: forest parameter refine
clf = RandomForestClassifier(
    n_estimators=30,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt",
    class_weight="balanced"
)
clf.fit(X_train, y_train)

# predict on test
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

# F-1
f1 = f1_score(y_test, y_pred)
print(f"F₁ Score: {f1:.4f}")
# evaluate
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))