"""Reproducible FHB risk prediction from the project's imputed training data.

The trained model is loaded lazily: importing this module neither reads a
developer-specific file path nor starts an interactive prompt.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "processed" / "fhb_final_imputed.csv"
FEATURES = [
    "rainy_days", "avg_humidity", "high_humidity_days",
    "avg_tmax_c", "avg_tmin_c",
]


@lru_cache(maxsize=4)
def load_model(dataset_path=DATA_PATH):
    """Train once per dataset path; fail clearly on invalid training inputs."""
    df = pd.read_csv(dataset_path)
    required = FEATURES + ["severity"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing required training columns: {', '.join(missing)}")
    if df[required].isna().any().any():
        raise ValueError("training data contains missing features or severity labels")

    features = df[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("training data contains non-finite weather features")
    # Non-finite or out-of-domain severity labels must not silently become
    # outbreak/non-outbreak classes (e.g. +inf previously became outbreak).
    try:
        severity = df["severity"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("training severity labels must be numeric") from exc
    if not np.isfinite(severity).all() or not np.isin(severity, [0, 1, 2, 3]).all():
        raise ValueError("training severity labels must be finite classes 0 through 3")
    labels = (severity >= 2).astype(int)
    if set(labels) != {0, 1}:
        raise ValueError("training data must contain both outbreak classes")

    model = RandomForestClassifier(
        n_estimators=100, max_depth=3, random_state=42
    )
    model.fit(features, labels)
    return model


def predict_fhb_risk(rainy_days, avg_humidity, high_humidity_days,
                     avg_tmax_c, avg_tmin_c):
    """Predict outbreak risk from pre-heading weather features.

    These probabilities come from a small dataset with weather-imputed labels;
    they are exploratory, not independently validated field-level forecasts.
    """
    features = np.array([[
        rainy_days, avg_humidity, high_humidity_days,
        avg_tmax_c, avg_tmin_c,
    ]], dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("weather inputs must be finite numbers")
    # The source dataset can cover a longer window than the example's
    # 15-day description; enforce valid counts without inventing an upper
    # bound that would reject legitimate training observations.
    if any(isinstance(value, (bool, np.bool_)) for value in (rainy_days, high_humidity_days)) or any(
        value < 0 or not float(value).is_integer() for value in (features[0, 0], features[0, 2])
    ):
        raise ValueError("rainy_days and high_humidity_days must be nonnegative whole-day counts")
    if not 0 <= features[0, 1] <= 100:
        raise ValueError("avg_humidity must be a percentage between 0 and 100")
    prob = float(load_model().predict_proba(features)[0, 1])

    if prob >= 0.70:
        risk_level = "HIGH"
        recommendation = "High modeled risk; consult local extension guidance."
    elif prob >= 0.40:
        risk_level = "MODERATE"
        recommendation = "Moderate modeled risk; monitor local conditions."
    else:
        risk_level = "LOW"
        recommendation = "Low modeled risk; continue routine monitoring."

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
        "key_drivers": drivers,
    }


def print_forecast(result, location=""):
    print("\n" + "=" * 50)
    if location:
        print(f"Location: {location}")
    print("=" * 50)
    print(f"Outbreak Probability : {result['outbreak_probability']}")
    print(f"Risk Level           : {result['risk_level']}")
    print(f"Recommendation       : {result['recommendation']}")
    print("Key Drivers          :")
    for driver in result["key_drivers"]:
        print(f"  → {driver}")
    print("=" * 50)


def main():
    print("Running example predictions...\n")
    scenarios = [
        ("ND Cass County — High Risk Scenario", (12, 72, 10, 22, 14)),
        ("SD Brown County — Low Risk Scenario", (3, 48, 2, 31, 18)),
        ("MN Polk County — Moderate Risk Scenario", (7, 63, 6, 26, 13)),
    ]
    for location, inputs in scenarios:
        print_forecast(predict_fhb_risk(*inputs), location)

    print("\nINTERACTIVE MODE — Enter your own weather data")
    try:
        rainy_days = int(input("Rainy days in last 15 days: "))
        avg_humidity = float(input("Average humidity (%): "))
        high_humidity_days = int(input("Days with humidity > 80%: "))
        avg_tmax_c = float(input("Average max temperature (°C): "))
        avg_tmin_c = float(input("Average min temperature (°C): "))
    except (KeyboardInterrupt, EOFError):
        print("\nExited interactive mode.")
        return

    print_forecast(
        predict_fhb_risk(
            rainy_days, avg_humidity, high_humidity_days,
            avg_tmax_c, avg_tmin_c,
        ),
        "Your Location",
    )


if __name__ == "__main__":
    main()
