import os
import time
import requests
import pandas as pd

API_KEY = "498944D6-2CEA-3B62-8DB1-BDFEF4E4418B"

BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

OUT_DIR = "/Users/priaa/Desktop/wheat-fhb-forecasting/raw/nass"
os.makedirs(OUT_DIR, exist_ok=True)

STATES = ["ND", "MN", "SD"]

def query_nass(params, label):
    params = {**params, "key": API_KEY, "format": "CSV"}
    resp = requests.get(BASE_URL, params=params, timeout=60)
    
    if resp.status_code != 200:
        print(f"[FAIL] {label}: {resp.text[:200]}")
        return None
    
    fpath = os.path.join(OUT_DIR, f"{label}.csv")
    with open(fpath, "wb") as f:
        f.write(resp.content)
    
    df = pd.read_csv(fpath)
    print(f"[ok] {label}: {len(df)} rows")
    return df

for state in STATES:
    # 1. Crop progress - flowering dates
    query_nass({
        "commodity_desc": "WHEAT",
        "statisticcat_desc": "PROGRESS",
        "agg_level_desc": "STATE",
        "state_alpha": state,
        "freq_desc": "WEEKLY",
        "year__GE": 2006,
        "year__LE": 2025,
    }, f"progress_{state}")
    time.sleep(1)

    # 2. Crop condition ratings
    query_nass({
        "commodity_desc": "WHEAT",
        "statisticcat_desc": "CONDITION",
        "agg_level_desc": "STATE",
        "state_alpha": state,
        "freq_desc": "WEEKLY",
        "year__GE": 2006,
        "year__LE": 2025,
    }, f"condition_{state}")
    time.sleep(1)

    # 3. County yield
    query_nass({
        "commodity_desc": "WHEAT",
        "statisticcat_desc": "YIELD",
        "agg_level_desc": "COUNTY",
        "state_alpha": state,
        "year__GE": 2006,
        "year__LE": 2025,
    }, f"yield_{state}")
    time.sleep(1)

print("Done. All NASS data saved.")