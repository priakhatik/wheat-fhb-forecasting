"""
Reference list of top wheat-producing counties in ND, MN, SD.
Centroid lat/lon pulled from US Census Gazetteer county centroids.
This defines the spatial scope for all weather + crop data pulls.
"""

COUNTIES = [
    # North Dakota - top spring/durum wheat counties
    {"state": "ND", "county": "Cass",       "fips": "38017", "lat": 46.85, "lon": -97.30},
    {"state": "ND", "county": "Barnes",     "fips": "38003", "lat": 46.90, "lon": -98.10},
    {"state": "ND", "county": "Walsh",      "fips": "38099", "lat": 48.40, "lon": -97.75},
    {"state": "ND", "county": "Pembina",    "fips": "38067", "lat": 48.75, "lon": -97.55},
    {"state": "ND", "county": "Traill",     "fips": "38095", "lat": 47.30, "lon": -97.30},
    {"state": "ND", "county": "Grand Forks","fips": "38035", "lat": 47.90, "lon": -97.30},
    {"state": "ND", "county": "Stutsman",   "fips": "38093", "lat": 46.95, "lon": -98.85},
    {"state": "ND", "county": "Bottineau",  "fips": "38009", "lat": 48.85, "lon": -100.45},
    {"state": "ND", "county": "McHenry",    "fips": "38049", "lat": 48.05, "lon": -100.45},
    {"state": "ND", "county": "Williams",   "fips": "38105", "lat": 48.25, "lon": -103.55},

    # South Dakota - top wheat counties
    {"state": "SD", "county": "Brown",      "fips": "46013", "lat": 45.45, "lon": -98.25},
    {"state": "SD", "county": "Day",        "fips": "46037", "lat": 45.30, "lon": -97.65},
    {"state": "SD", "county": "Marshall",   "fips": "46091", "lat": 45.75, "lon": -97.60},
    {"state": "SD", "county": "Roberts",    "fips": "46109", "lat": 45.55, "lon": -97.00},
    {"state": "SD", "county": "Spink",      "fips": "46115", "lat": 44.95, "lon": -98.10},

    # Minnesota - top spring/durum wheat counties
    {"state": "MN", "county": "Polk",       "fips": "27119", "lat": 47.65, "lon": -96.40},
    {"state": "MN", "county": "Marshall",   "fips": "27089", "lat": 48.40, "lon": -96.30},
    {"state": "MN", "county": "Norman",     "fips": "27107", "lat": 47.50, "lon": -96.45},
    {"state": "MN", "county": "Kittson",    "fips": "27069", "lat": 48.75, "lon": -96.65},
    {"state": "MN", "county": "Pennington", "fips": "27113", "lat": 47.95, "lon": -96.10},
]

DATE_START = "2006-01-01"
DATE_END = "2025-12-31"

# Wheat growing season window (spring wheat: planted Apr/May, flowering ~late June-July, harvest Aug)
# Used to subset weather features for disease-risk-relevant months
SEASON_START_MONTH = 4
SEASON_END_MONTH = 9
