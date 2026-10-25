# Time Series Forecasting: ETTh1

## Dataset
- **Name**: Electricity Transformer Temperature (hourly)
- **Source**: https://github.com/zhouhaoyi/ETDataset
- **Domain**: Energy / Industrial
- **Rows**: 17,420 (hourly, Jul 2016 – Jun 2018)
- **Target**: OT (Oil Temperature, °C)
- **Features**: 6 power load columns (HUFL, HULL, MUFL, MULL, LUFL, LULL)

## Task
Predict future oil temperature using historical readings and engineered features.

### Primary horizon: 1 hour ahead
- **Metric**: RMSE
- **Train/Test split**: temporal 70/20 (12,076 / 3,451)

### Extended analysis: 24 hours ahead
- **Train/Test split**: temporal 70/20 (12,059 / 3,446)

## Feature Engineering
54+ features from 6 original: lag features (1h to 168h), rolling statistics (mean, std, min, max at 6h–168h windows), difference features, calendar encoding (hour, dayofweek, month with cyclical sin/cos), cross-feature lags.

## Validation
Expanding-window cross-validation (3 folds) preserving temporal order. No random shuffling.

## Results (1h ahead)
- **LightGBM**: RMSE 0.695
- **CatBoost**: RMSE 0.719
- **Ensemble**: RMSE 0.695 (LGB=1.0, CB=0.0)
- **vs Mean baseline**: -92.3% (9.02 → 0.695)
- **vs Yesterday same hour**: -69.0% (2.24 → 0.695)
- **Overfit gap**: -4.0% (no overfitting)
