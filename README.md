# 🏠 Zimbabwe Property Valuator

**An AI-powered Automated Valuation Model (AVM) for residential real estate in Zimbabwe.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)  
[![Kaggle Dataset](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/datasets/your-username/your-dataset)

---

## 📌 Overview

This project provides an end‑to‑end machine learning solution for estimating property prices in Zimbabwe (primary data from Harare and Bulawayo). It includes:

- Data cleaning and feature engineering (Phase 2 & 3)  
- XGBoost model training with hyperparameter tuning via Optuna (Phase 4 & 5)  
- SHAP explainability (Phase 6)  
- An **interactive Streamlit web app** (Phase 8) that lets users input property details and get instant price estimates, segment classification, and what‑if analysis.

---

## 🚀 Features

- **Instant Valuation** – Enter property specs and receive a predicted market value with a confidence range.
- **Segment Classification** – Budget, Mid, Premium, or Luxury.
- **What‑If Analysis** – See how changes in floor area, bedrooms, or land size affect the price.
- **Compare Properties** – Side‑by‑side comparison of two properties.
- **SHAP‑Driven Insights** – Understand which features drive the estimate for each property.
- **PDF/JSON Export** – Download a valuation report.

---

## 🧠 Model Performance

| Metric   | Value  |
|----------|--------|
| Model    | XGBoost Regressor |
| Target   | log(1 + price) |
| R² (test)| ~0.81  |
| MAE      | ~USD 91,000 |
| Features | 25 (leakage‑free) |
| Training | ~1,985 listings from property.co.zw |

---

## 📁 Repository Structure

```
property-valuator-zimbabwe/
├── data/
│   └── processed/
│       ├── properties_features.csv   # feature‑engineered dataset
│       └── properties_clean.csv      # (optional) cleaned raw data
├── models/
│   ├── xgb_tuned.pkl                 # final XGBoost model
│   └── model_metadata.json           # metrics and feature list
├── notebooks/
│   ├── phase1_eda.ipynb
│   ├── phase2_clean.ipynb
│   ├── phase3_features.ipynb
│   ├── phase4_train.ipynb
│   ├── phase5_tune.ipynb
│   └── phase6_shap.ipynb
├── streamlit_app.py                  # Streamlit web application
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── LICENSE                           # MIT (or your choice)
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+  
- Git

### 1. Clone the repository
```bash
git clone https://github.com/your-username/property-valuator-zimbabwe.git
cd property-valuator-zimbabwe
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📊 Using the App

### Valuation Tab
- Fill in property details: location, size, bedrooms, bathrooms, amenities.
- Click **Valuate** – the model returns a price estimate, segment, and price per m².
- Explore the **“What’s Driving This Price”** section to see feature contributions.
- View **similar properties** from the training set.

### Compare Tab
- Enter two properties side‑by‑side.
- See which one is estimated higher and by what percentage.

### What‑If Analysis
- Choose a variable (floor area, bedrooms, land size).
- Adjust the slider to see how the predicted price changes.
- View a table of scenario snapshots.

### Engine Diagnostics
- See global feature importance from the XGBoost model.
- Read model validation details and leakage controls.

---

## 🧪 Data & Methodology

- **Data Source**: Scraped from property.co.zw (April 2026).
- **Feature Engineering**: Created 25 features including:
  - Size metrics (floor_area, land_size, room_density)
  - Amenity scores (energy, modern, maintenance burden)
  - Location liquidity (listings per region/suburb)
  - Categorical encodings (house type, furnished)
- **Leakage Prevention**: Price‑derived features (e.g., `price_per_sqm`) were excluded.
- **Hyperparameter Tuning**: Optuna with 100 trials and 5‑fold cross‑validation.
- **Explainability**: SHAP (SHapley Additive exPlanations) used to interpret predictions.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

---


---

## 🙏 Acknowledgements

- Data provided by [property.co.zw](https://www.property.co.zw)
- Built with [Streamlit](https://streamlit.io), [XGBoost](https://xgboost.ai), and [SHAP](https://shap.readthedocs.io)

---

## 📬 Contact

For questions or collaboration, feel free to reach out:  
**MUNASHE MURAMBIWA ** munathe1st@gmail.com 
**GitHub**: https://github.com/Mukundithe1st

---

> **Disclaimer**: This tool provides an **algorithmic estimate** only. It is not a substitute for a professional, on‑site property valuation. Always consult a licensed real estate appraiser before making financial decisions.
