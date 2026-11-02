"""Generate the state/year FHB label grid at a repository-relative path.

Missing labels remain explicitly missing until supported by source evidence.
Importing this module does not write files or create local directories.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "raw" / "labels"

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


def build_label_grid():
    """Return one row per state/year with explicit unknown-label provenance."""
    states = ("ND", "MN", "SD")
    years = range(2006, 2026)
    known_lookup = {(row["state"], row["year"]): row for row in KNOWN_LABELS}
    rows = []
    for state in states:
        for year in years:
            key = (state, year)
            if key in known_lookup:
                rows.append(dict(known_lookup[key]))
            else:
                rows.append({
                    "state": state,
                    "year": year,
                    "severity": None,
                    "notes": "NEEDS RESEARCH",
                    "source": "",
                })
    return pd.DataFrame(rows)


def main():
    df = build_label_grid()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUT_DIR / "fhb_labels.csv"
    df.to_csv(destination, index=False)
    print(f"Saved labels to: {destination}")
    print(f"Total rows: {len(df)}")
    print(f"Filled in: {df['severity'].notna().sum()}")
    print(f"Needs research: {df['severity'].isna().sum()}")
    print(df.head(10))


if __name__ == "__main__":
    main()
