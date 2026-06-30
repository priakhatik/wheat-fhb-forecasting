import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Load final dataset ────────────────────────────────────────────
df = pd.read_csv("/Users/priaa/Desktop/wheat-fhb-forecasting/processed/fhb_final_dataset.csv")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

OUT_DIR = "/Users/priaa/Desktop/wheat-fhb-forecasting/processed/eda"
os.makedirs(OUT_DIR, exist_ok=True)

# use only rows where severity is known
known = df[df["severity"].notna()].copy()
print(f"\nRows with known severity: {len(known)} out of {len(df)}")

# ── Plot 1: Severity distribution ─────────────────────────────────
plt.figure(figsize=(8, 5))
sns.countplot(data=known, x="severity", palette="Reds")
plt.title("FHB Severity Distribution")
plt.xlabel("Severity (0=none, 1=low, 2=moderate, 3=severe)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/1_severity_distribution.png")
plt.close()
print("Plot 1 saved")

# ── Plot 2: Total rainfall by severity ────────────────────────────
plt.figure(figsize=(8, 5))
sns.boxplot(data=known, x="severity", y="total_precip_mm", palette="Blues")
plt.title("Total Rainfall in 15-Day Pre-Heading Window by Severity")
plt.xlabel("Severity")
plt.ylabel("Total Precipitation (mm)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_rain_by_severity.png")
plt.close()
print("Plot 2 saved")

# ── Plot 3: Humidity by severity ──────────────────────────────────
plt.figure(figsize=(8, 5))
sns.boxplot(data=known, x="severity", y="avg_humidity", palette="Greens")
plt.title("Average Humidity in 15-Day Pre-Heading Window by Severity")
plt.xlabel("Severity")
plt.ylabel("Average Relative Humidity (%)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_humidity_by_severity.png")
plt.close()
print("Plot 3 saved")

# ── Plot 4: Temperature by severity ───────────────────────────────
plt.figure(figsize=(8, 5))
sns.boxplot(data=known, x="severity", y="avg_tmax_c", palette="Oranges")
plt.title("Average Max Temperature in 15-Day Pre-Heading Window by Severity")
plt.xlabel("Severity")
plt.ylabel("Avg Max Temperature (°C)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/4_temperature_by_severity.png")
plt.close()
print("Plot 4 saved")

# ── Plot 5: Correlation heatmap ───────────────────────────────────
plt.figure(figsize=(10, 8))
features = ["total_precip_mm", "rainy_days", "avg_humidity",
            "high_humidity_days", "avg_tmax_c", "avg_tmin_c",
            "avg_vpd", "severity"]
corr = known[features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Between Weather Features and FHB Severity")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/5_correlation_heatmap.png")
plt.close()
print("Plot 5 saved")

# ── Plot 6: Severity over time ────────────────────────────────────
plt.figure(figsize=(12, 5))
for state in ["ND", "MN", "SD"]:
    state_df = known[known["state"] == state]
    plt.plot(state_df["year"], state_df["severity"],
             marker="o", label=state)
plt.title("FHB Severity Over Time by State")
plt.xlabel("Year")
plt.ylabel("Severity")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/6_severity_over_time.png")
plt.close()
print("Plot 6 saved")

print(f"\nAll plots saved to: {OUT_DIR}")

# ── Summary statistics ────────────────────────────────────────────
print("\nWeather summary by severity:")
print(known.groupby("severity")[["total_precip_mm", "rainy_days",
                                 "avg_humidity", "avg_tmax_c"]].mean().round(2))