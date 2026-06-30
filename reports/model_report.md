# Zimbabwe Real Estate - Model Evaluation Report

## Dataset Overview
- Records: 1,985 residential property listings
- Features: 25 leakage-free features
- Target: log(1+price), reported in USD
- Train/Test: 80/20 split (random_state=42)
- Price range: $5,000 - $1,800,000

## Leakage Audit
Features EXCLUDED (derived from price or quality flags):
price_per_sqm_floor, price_per_bed, price_per_bath, price_per_hectare,
value_density, bed_quality, segment_encoded, region_price_enc,
suburb_price_enc, data_quality_score, missing_suburb, missing_region,
location_issue, bath_ratio, land_efficiency

## Model Comparison

| Model | R2 (Test) | MAE (USD) | CV R2 |
|-------|-----------|-----------|-------|
| Linear Regression | 0.6019 | $135,306 | -34.8408 +/- 70.9094 |
| Random Forest | 0.7816 | $99,048 | 0.762 +/- 0.0292 |
| Gradient Boosting | 0.8155 | $93,154 | 0.7879 +/- 0.0336 |
| XGBoost | 0.8141 | $91,650 | 0.7845 +/- 0.0313 |

## Best Model: Gradient Boosting

| Metric | Value |
|--------|-------|
| R2 Score | 0.8155 |
| MAE (USD) | $93,154 |
| RMSE (USD) | $172,144 |

## Feature Importance (Top 10)
1. listings_per_region (0.3677)
2. land_size (0.3150)
3. listings_per_suburb (0.0671)
4. total_amenities (0.0486)
5. bedrooms (0.0227)
6. bathrooms (0.0213)
7. floor_area (0.0192)
8. house_category_encoded (0.0168)
9. room_density (0.0166)
10. maintenance_burden (0.0158)

## Limitations
- Dataset is geographically concentrated (Harare-centric)
- No time dimension - market changes not captured
- Luxury segment has fewer listings and higher prediction error
- Suburb-level supply counts may not generalise to unseen suburbs

## Future Improvements
- Add geospatial features (lat/lng, distance to CBD)
- Multi-period data for time-trend features
- Ensemble stacking of XGBoost + RF
- SHAP-based feature selection to reduce noise
- Neighbourhood-level price index from external sources
