import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── 1. Load imputed dataset ───────────────────────────────────────
df = pd.read_csv("/Users/priaa/Desktop/wheat-fhb-forecasting/processed/fhb_final_imputed.csv")
print(f"Total rows: {len(df)}")

# ── 2. Convert to binary labels ───────────────────────────────────
# 0, 1 → 0 (no outbreak)
# 2, 3 → 1 (outbreak)
df["outbreak"] = (df["severity"] >= 2).astype(int)
print(f"\nBinary label distribution:")
print(df["outbreak"].value_counts())
print(f"\n0 = No outbreak, 1 = Outbreak")

# ── 3. Features and target ────────────────────────────────────────
FEATURES = ["rainy_days", "avg_humidity", "high_humidity_days",
            "avg_tmax_c", "avg_tmin_c"]

X = df[FEATURES].values
y = df["outbreak"].values

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Class balance: {y.mean():.1%} outbreak rate")

# ── 4. Scale features ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 5. Define models ──────────────────────────────────────────────
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        X_scaled
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        X
    ),
}

# ── 6. LOOCV evaluation ───────────────────────────────────────────
OUT_DIR = "/Users/priaa/Desktop/wheat-fhb-forecasting/processed/models_binary"
os.makedirs(OUT_DIR, exist_ok=True)

loo = LeaveOneOut()
results = {}

for name, (model, X_input) in models.items():
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")

    y_pred = np.zeros(len(y), dtype=int)
    y_prob = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X_input):
        X_train, X_test = X_input[train_idx], X_input[test_idx]
        y_train = y[train_idx]
        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)
        y_prob[test_idx] = model.predict_proba(X_test)[0][1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    print(f"Accuracy: {acc:.2%}")
    print(f"ROC-AUC:  {auc:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred,
          target_names=["No Outbreak", "Outbreak"],
          zero_division=0))

    results[name] = {
        "accuracy": acc,
        "auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Outbreak", "Outbreak"],
                yticklabels=["No Outbreak", "Outbreak"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_{name.replace(' ', '_')}.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="blue")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_{name.replace(' ', '_')}.png")
    plt.close()

    print("Plots saved")

# ── 7. Model comparison ───────────────────────────────────────────
print(f"\n{'='*50}")
print("MODEL COMPARISON")
print(f"{'='*50}")
print(f"{'Model':25s} {'Accuracy':10s} {'ROC-AUC':10s}")
print(f"{'-'*45}")
for name, result in results.items():
    print(f"{name:25s} {result['accuracy']:.2%}     {result['auc']:.3f}")

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
plt.title("Feature Importance — Binary Classification")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png")
plt.close()
print("Feature importance saved")

print(f"\nAll outputs saved to: {OUT_DIR}")