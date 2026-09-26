# Drift Detection Analysis

## Configuration
- Reference dataset: random 70% of customer data (training distribution)
- Current dataset: remaining 30% with synthetic drift injected

## Simulated Production Shifts
- `MonthlyCharges`: shifted by ~N(25, 8) per customer record
- `tenure`: increased by random integer offset in [6, 18]
- `Contract`: ~45% of current records forced to `Month-to-month`
- `retention_status`: ~20% of labels flipped to simulate concept drift

## Custom Metrics
- Absolute mean difference in `MonthlyCharges`: **24.408**
  (threshold=5.0)
- Absolute retention-rate shift within `Contract == Month-to-month`: **0.0275**

## Detection Results
- Dataset drift flagged: **True**
- Drifted columns: ['Churn', 'Contract', 'MonthlyCharges', 'tenure']
- Monitored features flagged: ['MonthlyCharges', 'tenure', 'Contract']
- Significant drift (for automation): **True**

## Production Impact
If this pattern occurred in production, feature and target distributions would
no longer match training data. Predictions would be unreliable, especially for
billing-sensitive and month-to-month customer segments. Recommended action:
investigate data pipelines, then retrain and validate before deploying new model.
