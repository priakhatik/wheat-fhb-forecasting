import pandas as pd

df = pd.read_parquet("/Users/priaa/Desktop/wheat-fhb-forecasting/processed/weather_combined.parquet")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nDate range:")
print(df["time"].min(), "to", df["time"].max())
print("\nCounties:")
print(df["county"].unique())