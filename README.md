---

## 🔬 Data Sources

| Source | What it provides |
|--------|-----------------|
| [gridMET](https://www.climatologylab.org/gridmet.html) | Daily weather (temp, humidity, precip, wind) 1979–present |
| [USDA NASS QuickStats](https://quickstats.nass.usda.gov) | Weekly crop progress, condition ratings, county yield |
| USWBSI / State Extension | Historical FHB outbreak severity labels |

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/priakhatik/wheat-fhb-forecasting.git
cd wheat-fhb-forecasting

# Install dependencies
pip install pygridmet pandas requests pyarrow matplotlib seaborn scikit-learn xgboost

# Set NASS API key (free from https://quickstats.nass.usda.gov/api)
export NASS_API_KEY="your-key-here"
```

---

## 🚀 Run the Pipeline

```bash
# Step 1 — Collect weather data
python scripts/fetch_weather.py

# Step 2 — Collect crop data
python scripts/fetch_nass.py

# Step 3 — Build disease labels
python scripts/build_labels.py

# Step 4 — Merge into final dataset
python scripts/merge.py

# Step 5 — Exploratory data analysis
python scripts/eda.py

# Step 6 — Train and evaluate models
python scripts/model_binary.py

# Step 7 — Live prediction
python scripts/predict.py
```

---

## 🔮 Live Prediction

```python
from scripts.predict import predict_fhb_risk

result = predict_fhb_risk(
    rainy_days=12,
    avg_humidity=74,
    high_humidity_days=9,
    avg_tmax_c=21,
    avg_tmin_c=13
)

# Output:
# Outbreak Probability : 77.4%
# Risk Level           : HIGH
# Recommendation       : Apply fungicide at heading. High FHB risk.
# Key Drivers          :
#   → High humidity (74.0%)
#   → Cool temperatures (21.0°C max)
#   → Frequent rain (12 days)
```

---

## 📈 Key EDA Findings

- **Humidity** is the strongest single predictor (correlation 0.84 with severity)
- **Temperature** has a negative relationship — cooler pre-heading = more disease
- **Rainfall alone** is a poor predictor — the combination of humidity + temperature matters
- **2009, 2010, 2016** were the worst epidemic years across all three states
- FHB outbreaks are **episodic**, not steadily increasing — driven by year-to-year weather volatility

---

## ⚠️ Limitations

- Disease severity labels for 44 of 60 state-years were imputed from weather 
  thresholds rather than ground-truth field observations
- Dataset covers state-level aggregates — field-level predictions would require 
  finer spatial resolution
- Model trained on 60 observations — performance would improve significantly 
  with more labeled data

---

## 🔭 Next Steps

- Collect ground-truth FHB labels from NDSU/UMN/SDSU Extension archives
- Add DON (vomitoxin) concentration as a continuous target variable
- Incorporate satellite vegetation indices (NDVI) as crop stress proxy
- Build a web dashboard for real-time in-season risk monitoring
- Expand to additional pathosystems (stripe rust, septoria leaf blotch)

---

## 👩‍💻 Author

**Priya Khatik**
MS in AI & Business Analytics, University of South Florida  
[GitHub](https://github.com/priakhatik)
