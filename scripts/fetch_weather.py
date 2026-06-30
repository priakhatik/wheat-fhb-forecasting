import os
import time
import pandas as pd
import pygridmet as gridmet
from counties import COUNTIES, DATE_START, DATE_END

OUT_DIR = "/Users/priaa/Desktop/wheat-fhb-forecasting/raw/weather"
os.makedirs(OUT_DIR, exist_ok=True)

VARIABLES = ["pr", "tmmn", "tmmx", "rmin", "rmax", "vpd", "vs"]

def fetch_county(county):
    state = county["state"]
    name = county["county"]
    lat = county["lat"]
    lon = county["lon"]

    fname = f"{state}_{name.replace(' ', '')}.csv"
    fpath = os.path.join(OUT_DIR, fname)

    if os.path.exists(fpath):
        print(f"  [skip] {fname} already exists")
        return pd.read_csv(fpath)

    df = gridmet.get_bycoords(
        (lon, lat),
        dates=(DATE_START, DATE_END),
        variables=VARIABLES,
    )
    df = df.reset_index()
    df["state"] = state
    df["county"] = name
    df["fips"] = county["fips"]

    df.to_csv(fpath, index=False)
    print(f"  [ok] {fname} — {len(df)} rows")
    return df

all_dfs = []
for i, county in enumerate(COUNTIES):
    print(f"[{i+1}/{len(COUNTIES)}] {county['state']} - {county['county']}")
    df = fetch_county(county)
    if df is not None:
        all_dfs.append(df)
    time.sleep(0.5)

combined = pd.concat(all_dfs, ignore_index=True)
combined.to_parquet("/Users/priaa/Desktop/wheat-fhb-forecasting/processed/weather_combined.parquet", index=False)
print(f"Done. {len(combined)} total rows saved.")