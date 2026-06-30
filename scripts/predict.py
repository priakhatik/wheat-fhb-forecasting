import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── 1. Train the final model on all data ─────────────────────────
df = pd.read_csv("/Users/priaa/Desktop/wheat-fhb-forecasting/processed/fhb_final_imputed.csv")
df["outbreak"] = (df["severity"] >= 2).astype(int)

FEATURES = ["rainy_days", "avg_humidity", "high_humidity_days",
            "avg_tmax_c", "avg_tmin_c"]

X = df[FEATURES].values
y = df["outbreak"].values

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X, y)
print("Model trained on", len(df), "observations")

# ── 2. Prediction function ────────────────────────────────────────
def predict_fhb_risk(rainy_days, avg_humidity, high_humidity_days,
                     avg_tmax_c, avg_tmin_c):
    """
    Predict FHB outbreak risk given weather conditions
    during the 15-day pre-heading window.

    Parameters:
        rainy_days        -- number of days with rain > 1mm
        avg_humidity      -- average relative humidity (%)
        high_humidity_days -- days with max humidity > 80%
        avg_tmax_c        -- average max temperature (Celsius)
        avg_tmin_c        -- average min temperature (Celsius)

    Returns:
        Dictionary with probability, risk level, and drivers
    """
    features = np.array([[rainy_days, avg_humidity, high_humidity_days,
                          avg_tmax_c, avg_tmin_c]])

    prob = model.predict_proba(features)[0][1]

    if prob >= 0.70:
        risk_level = "HIGH"
        recommendation = "Apply fungicide at heading. High FHB risk."
    elif prob >= 0.40:
        risk_level = "MODERATE"
        recommendation = "Monitor closely. Consider fungicide if rain continues."
    else:
        risk_level = "LOW"
        recommendation = "Low risk. Standard monitoring sufficient."

    # Identify key drivers
    drivers = []
    if avg_humidity > 68:
        drivers.append(f"High humidity ({avg_humidity:.1f}%)")
    if avg_tmax_c < 24:
        drivers.append(f"Cool temperatures ({avg_tmax_c:.1f}°C max)")
    if rainy_days > 8:
        drivers.append(f"Frequent rain ({rainy_days} days)")
    if not drivers:
        drivers.append("No major risk factors detected")

    return {
        "outbreak_probability": f"{prob:.1%}",
        "risk_level": risk_level,
        "recommendation": recommendation,
        "key_drivers": drivers
    }

# ── 3. Print results nicely ───────────────────────────────────────
def print_forecast(result, location=""):
    print(f"\n{'='*50}")
    if location:
        print(f"Location: {location}")
    print(f"{'='*50}")
    print(f"Outbreak Probability : {result['outbreak_probability']}")
    print(f"Risk Level           : {result['risk_level']}")
    print(f"Recommendation       : {result['recommendation']}")
    print(f"Key Drivers          :")
    for d in result['key_drivers']:
        print(f"  → {d}")
    print(f"{'='*50}")

# ── 4. Example predictions ────────────────────────────────────────
print("\nRunning example predictions...\n")

# Example 1 — High risk conditions (warm humid rainy)
result1 = predict_fhb_risk(
    rainy_days=12,
    avg_humidity=72,
    high_humidity_days=10,
    avg_tmax_c=22,
    avg_tmin_c=14
)
print_forecast(result1, "ND Cass County — High Risk Scenario")

# Example 2 — Low risk conditions (dry hot)
result2 = predict_fhb_risk(
    rainy_days=3,
    avg_humidity=48,
    high_humidity_days=2,
    avg_tmax_c=31,
    avg_tmin_c=18
)
print_forecast(result2, "SD Brown County — Low Risk Scenario")

# Example 3 — Moderate risk
result3 = predict_fhb_risk(
    rainy_days=7,
    avg_humidity=63,
    high_humidity_days=6,
    avg_tmax_c=26,
    avg_tmin_c=13
)
print_forecast(result3, "MN Polk County — Moderate Risk Scenario")

# ── 5. Interactive mode ───────────────────────────────────────────
print("\n" + "="*50)
print("INTERACTIVE MODE — Enter your own weather data")
print("="*50)

try:
    rainy_days = int(input("Rainy days in last 15 days: "))
    avg_humidity = float(input("Average humidity (%): "))
    high_humidity_days = int(input("Days with humidity > 80%: "))
    avg_tmax_c = float(input("Average max temperature (°C): "))
    avg_tmin_c = float(input("Average min temperature (°C): "))

    result = predict_fhb_risk(rainy_days, avg_humidity, high_humidity_days,
                               avg_tmax_c, avg_tmin_c)
    print_forecast(result, "Your Location")

except KeyboardInterrupt:
    print("\nExited interactive mode.")