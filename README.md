# 🌾 Wheat FHB Disease Forecasting Platform

A biology-informed, climate-aware machine learning system for predicting 
Fusarium Head Blight (FHB) outbreaks in spring wheat across North Dakota, 
Minnesota, and South Dakota (2006–2025).

---

## 🎯 Problem

Fusarium Head Blight (FHB) is one of the most destructive wheat diseases 
in the US, causing billions in losses. The fungus infects wheat during a 
narrow 7-10 day flowering window — but only when weather conditions are 
right: warm nights, high humidity, and frequent rain.

Existing tools struggle to provide reliable in-season predictions as 
climate volatility disrupts historical growing patterns.

---

## 💡 Solution

An end-to-end pipeline that:
1. Collects real weather data (gridMET) for 20 top wheat-producing counties
2. Identifies the biological infection window using USDA crop progress data
3. Engineers weather features for the 15-day pre-heading period
4. Predicts outbreak risk using a Random Forest classifier
5. Outputs probability, risk level, and key biological drivers

---

## 📊 Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 80.00% | 0.856 |
| **Random Forest** | **88.33%** | **0.922** |

**Evaluation caveat:** The reported 80.00% / 88.33% accuracy and ROC-AUC
figures were calculated against a dataset containing **44 weather-imputed
severity labels out of 60 state-year rows**. Because the missing labels were
constructed from weather thresholds, these scores are **not independently
validated outbreak prediction performance**; the features partly determine
their own evaluation targets. Do not interpret them as held-out field accuracy.

The original recorded-label dataset has 16 nonmissing severity entries
(13 outbreaks and 3 non-outbreaks). All three recorded non-outbreak labels
occur in 2023, so leaving 2023 out yields a training set with **no negative
class**. A defensible leave-one-year-out binary benchmark is therefore not
currently feasible without more independently verified labels.

Run the offline, CI-tested label/holdout audit:

```bash
python scripts/audit_label_evaluation.py
python -m pytest -q tests/test_label_evaluation_audit.py
```

The audit reports recorded vs. missing labels and whether each year-held-out
training fold contains both classes. It does **not** train a model, verify the
external label sources, or produce a replacement accuracy claim. More
independently verified non-outbreak years are needed before reliable temporal
generalization can be evaluated.

---

## 🗂️ Project Structure

```
wheat-fhb-forecasting/
├── raw/
│   ├── weather/          # Daily gridMET CSVs per county
│   ├── nass/             # USDA crop progress, condition, yield
│   └── labels/           # FHB severity labels
├── processed/
│   ├── weather_combined.parquet
│   ├── fhb_final_dataset.csv
│   ├── fhb_final_imputed.csv
│   ├── eda/              # EDA plots
│   ├── models/           # 4-class model outputs
│   └── models_binary/    # Binary model outputs
└── scripts/
    ├── counties.py        # Study area definition
    ├── fetch_weather.py   # gridMET data collection
    ├── fetch_nass.py      # USDA NASS data collection
    ├── build_labels.py    # Disease label construction
    ├── merge.py           # Feature engineering + dataset merge
    ├── eda.py             # Exploratory data analysis
    ├── model.py           # 4-class classification
    ├── model_binary.py    # Binary classification
    └── predict.py         # Live prediction tool
```

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
