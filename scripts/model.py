import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── 1. Load data ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed"
df = pd.read_csv(DATA_DIR / "fhb_final_dataset.csv")
print(f"Total rows: {len(df)}")
print(f"Known labels: {df['severity'].notna().sum()}")
print(f"Missing labels: {df['severity'].isna().sum()}")

# ── 2. Impute missing severity using weather thresholds ───────────
# Thresholds derived from EDA boxplots and correlation analysis
def impute_severity(row):
    if row["avg_humidity"] > 68 and row["avg_tmax_c"] < 24:
        return 3
    elif row["avg_humidity"] > 63 and row["avg_tmax_c"] < 26:
        return 2
    elif row["avg_humidity"] < 63 and row["avg_tmax_c"] > 26:
        return 0
    else:
        return 1

df["severity"] = df.apply(
    lambda row: impute_severity(row) if pd.isna(row["severity"]) else row["severity"],
    axis=1
)

print(f"\nAfter imputation — severity distribution:")
print(df["severity"].value_counts().sort_index())

# ── 3. Features and target ────────────────────────────────────────
FEATURES = ["rainy_days", "avg_humidity", "high_humidity_days",
            "avg_tmax_c", "avg_tmin_c"]

X = df[FEATURES].values
y = df["severity"].astype(int).values

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {np.unique(y)}")

# ── 4. Define models ──────────────────────────────────────────────
# The scaler must be fitted inside each CV training fold, not on all rows.
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=3, random_state=42
    ),
}

# ── 6. LOOCV evaluation ───────────────────────────────────────────
OUT_DIR = DATA_DIR / "models"
os.makedirs(OUT_DIR, exist_ok=True)

loo = LeaveOneOut()
results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")

    y_pred = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)

    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    results[name] = {"accuracy": acc, "y_pred": y_pred}

    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[0, 1, 2, 3],
                yticklabels=[0, 1, 2, 3])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual Severity")
    plt.xlabel("Predicted Severity")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_{name.replace(' ', '_')}.png")
    plt.close()
    print("Confusion matrix saved")

# ── 7. Model comparison ───────────────────────────────────────────
print(f"\n{'='*50}")
print("MODEL COMPARISON")
print(f"{'='*50}")
for name, result in results.items():
    print(f"{name:25s} Accuracy: {result['accuracy']:.2%}")

# ── 8. Feature importance ─────────────────────────────────────────
print(f"\n{'='*50}")
print("FEATURE IMPORTANCE — Random Forest")
print(f"{'='*50}")

rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X, y)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print(importance_df.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="importance", y="feature",
            hue=importance_df["feature"], palette="viridis", legend=False)
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png")
plt.close()
print("Feature importance plot saved")

# ── 9. Save imputed dataset ───────────────────────────────────────
imputed_path = DATA_DIR / "fhb_final_imputed.csv"
df.to_csv(imputed_path, index=False)
print(f"\nImputed dataset saved to: {imputed_path}")
print(f"\nAll outputs saved to: {OUT_DIR}")