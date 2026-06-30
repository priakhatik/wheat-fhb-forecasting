import pandas as pd
import os

OUT_DIR = "/Users/priaa/Desktop/wheat-fhb-forecasting/raw/labels"
os.makedirs(OUT_DIR, exist_ok=True)

# Known epidemic years from research
KNOWN_LABELS = [
    {"state": "ND", "year": 2009, "severity": 3, "notes": "Major outbreak", "source": "NDSU Extension"},
    {"state": "MN", "year": 2009, "severity": 3, "notes": "Major outbreak", "source": "UMN Extension"},
    {"state": "ND", "year": 2010, "severity": 3, "notes": "Major epidemic", "source": "NDSU Extension"},
    {"state": "MN", "year": 2010, "severity": 3, "notes": "Major epidemic", "source": "UMN Extension"},
    {"state": "SD", "year": 2010, "severity": 3, "notes": "Major epidemic", "source": "SDSU Extension"},
    {"state": "ND", "year": 2013, "severity": 2, "notes": "Moderate outbreak", "source": "NDSU Extension"},
    {"state": "MN", "year": 2013, "severity": 2, "notes": "Moderate outbreak", "source": "UMN Extension"},
    {"state": "ND", "year": 2015, "severity": 2, "notes": "Moderate localized", "source": "NDSU Extension"},
    {"state": "SD", "year": 2015, "severity": 2, "notes": "Moderate localized", "source": "SDSU Extension"},
    {"state": "ND", "year": 2016, "severity": 3, "notes": "Severe epidemic", "source": "NDSU Extension"},
    {"state": "MN", "year": 2016, "severity": 3, "notes": "Severe epidemic", "source": "UMN Extension"},
    {"state": "SD", "year": 2016, "severity": 3, "notes": "Severe epidemic", "source": "SDSU Extension"},
    {"state": "ND", "year": 2019, "severity": 2, "notes": "Moderate", "source": "NDSU Extension"},
    {"state": "ND", "year": 2023, "severity": 1, "notes": "Low, fungicide mitigated", "source": "World Grain 2024"},
    {"state": "MN", "year": 2023, "severity": 0, "notes": "Minimal presence", "source": "World Grain 2024"},
    {"state": "SD", "year": 2023, "severity": 0, "notes": "Minimal presence", "source": "World Grain 2024"},
]

# Build full grid - every state x every year
states = ["ND", "MN", "SD"]
years = range(2006, 2026)

rows = []
known_lookup = {(r["state"], r["year"]): r for r in KNOWN_LABELS}

for state in states:
    for year in years:
        key = (state, year)
        if key in known_lookup:
            rows.append(known_lookup[key])
        else:
            rows.append({
                "state": state,
                "year": year,
                "severity": None,
                "notes": "NEEDS RESEARCH",
                "source": ""
            })

df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/fhb_labels.csv", index=False)

print(f"Total rows: {len(df)}")
print(f"Filled in: {df['severity'].notna().sum()}")
print(f"Needs research: {df['severity'].isna().sum()}")
print(df.head(10))