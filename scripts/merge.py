import pandas as pd
import os

# ── 1. Load weather ──────────────────────────────────────────────
weather = pd.read_parquet("/Users/priaa/Desktop/wheat-fhb-forecasting/processed/weather_combined.parquet")
weather["time"] = pd.to_datetime(weather["time"])
weather["tmmx_c"] = weather["tmmx (K)"] - 273.15
weather["tmmn_c"] = weather["tmmn (K)"] - 273.15
weather["avg_rh"] = (weather["rmin (%)"] + weather["rmax (%)"]) / 2
print("Weather loaded:", weather.shape)

# ── 2. Load crop progress ─────────────────────────────────────────
progress_files = [
    "/Users/priaa/Desktop/wheat-fhb-forecasting/raw/nass/progress_ND.csv",
    "/Users/priaa/Desktop/wheat-fhb-forecasting/raw/nass/progress_MN.csv",
    "/Users/priaa/Desktop/wheat-fhb-forecasting/raw/nass/progress_SD.csv",
]
progress = pd.concat([pd.read_csv(f) for f in progress_files], ignore_index=True)
print("Progress loaded:", progress.shape)

# Filter to PCT HEADED (closest to flowering for spring wheat)
headed = progress[
    progress["unit_desc"] == "PCT HEADED"
].copy()
print("Headed rows found:", len(headed))

# ── 3. Find heading week per state per year ───────────────────────
headed["week_ending"] = pd.to_datetime(headed["week_ending"])
headed["Value"] = pd.to_numeric(headed["Value"], errors="coerce")

flowering_dates = []
for (state, year), group in headed.groupby(["state_alpha", "year"]):
    group = group.sort_values("week_ending")
    crossed = group[group["Value"] >= 50]
    if not crossed.empty:
        flower_date = crossed.iloc[0]["week_ending"]
        flowering_dates.append({
            "state": state,
            "year": int(year),
            "flowering_date": flower_date
        })

flowering_dates = pd.DataFrame(flowering_dates)
print("Heading dates found:", len(flowering_dates))
print(flowering_dates.head())

# ── 4. Compute weather features for 15-day pre-heading window ─────
rows = []
for _, row in flowering_dates.iterrows():
    state = row["state"]
    year = row["year"]
    flower_date = row["flowering_date"]
    window_start = flower_date - pd.Timedelta(days=15)

    subset = weather[
        (weather["state"] == state) &
        (weather["time"] >= window_start) &
        (weather["time"] <= flower_date)
    ]

    if subset.empty:
        continue

    rows.append({
        "state": state,
        "year": year,
        "flowering_date": flower_date,
        "total_precip_mm": subset["pr (mm)"].sum(),
        "rainy_days": (subset["pr (mm)"] > 1.0).sum(),
        "avg_humidity": subset["avg_rh"].mean(),
        "high_humidity_days": (subset["rmax (%)"] > 80).sum(),
        "avg_tmax_c": subset["tmmx_c"].mean(),
        "avg_tmin_c": subset["tmmn_c"].mean(),
        "avg_vpd": subset["vpd (kPa)"].mean(),
    })

features = pd.DataFrame(rows)
print("\nFeature rows built:", len(features))
print(features.head())

# ── 5. Load disease labels ────────────────────────────────────────
labels = pd.read_csv("/Users/priaa/Desktop/wheat-fhb-forecasting/raw/labels/fhb_labels.csv")
labels["year"] = labels["year"].astype(int)
print("\nLabels loaded:", labels.shape)

# ── 6. Join everything ────────────────────────────────────────────
final = features.merge(labels, on=["state", "year"], how="left")
print("\nFinal dataset shape:", final.shape)
print(final.head(10))

# ── 7.
out_path = "/Users/priaa/Desktop/wheat-fhb-forecasting/processed/fhb_final_dataset.csv"
final.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
print("Done.")