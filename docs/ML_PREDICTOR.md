# ML Predictor - Predictive Maintenance Intelligence

## Overview

The ML Predictor module adds machine learning capabilities for predicting repair time and consumable usage based on defect characteristics, demonstrating data-driven decision making.

## Model Architecture

### RandomForestRegressor
- **Trees**: 100 estimators
- **Max Depth**: 10
- **Training Data**: 1000 synthetic samples

### Features
| Feature | Range | Description |
|---------|-------|-------------|
| `defect_area_cm2` | 0.5-15 | Size of the defect in cm² |
| `curvature_complexity` | 1-10 | Surface curvature difficulty |
| `material_hardness` | 1-10 | Material hardness factor |

### Target
- `repair_time_seconds`: Predicted repair duration (5-180s)

### Training Data Formula
```
time = area * 2.5 + curvature * 3.0 + hardness * 1.5 + 10 + interaction + noise
```
Where:
- `interaction = 0.1 * curvature * hardness`
- `noise = ±15% variation`

## Usage

### Agent Tool
Ask the agent: *"How long will this repair take?"*

The agent calls `predict_repair_metrics` and returns:
- Predicted time with 90% confidence interval
- Consumable usage estimate
- Per-defect and total time breakdown

### API Usage
```python
from src.ml import predict_repair_metrics

# Predict from defect dict
result = predict_repair_metrics(defect={
    "type": "crack",
    "severity": "high",
    "position": (0.5, 0.1, 0.3)
})

# Direct feature input
result = predict_repair_metrics(
    defect_area_cm2=5.0,
    curvature_complexity=7,
    material_hardness=6
)

print(result)
# {
#     "repair_time_seconds": 42.5,
#     "confidence_interval": {"lower": 35.2, "upper": 49.8},
#     "consumable_estimate": "Medium: Standard tool wear + filler material",
#     "feature_contributions": {"area": 12.5, "curvature": 21.0, "hardness": 9.0}
# }
```

### UI Visualization
In the Plan workflow step, a bar chart displays:
- **ML Predicted** (orange): Time predicted by the RandomForest model
- **Rule Estimated** (green): Time from rule-based fallback strategies

## Performance Metrics
- **MAE**: ~4-6 seconds
- **R² Score**: ~0.92

## File Structure
```
src/ml/
├── __init__.py          # Package exports
└── ml_predictor.py      # Core predictor implementation
```
