"""
Test script for the ML Predictor module.
"""

import pytest
import numpy as np


def test_generate_training_data():
    """Test synthetic data generation."""
    from src.ml.ml_predictor import generate_training_data
    
    X, y = generate_training_data(n_samples=100, seed=42)
    
    # Check shapes
    assert X.shape == (100, 3), f"Expected (100, 3), got {X.shape}"
    assert y.shape == (100,), f"Expected (100,), got {y.shape}"
    
    # Check feature ranges
    assert X[:, 0].min() >= 0.5, "Area should be >= 0.5"
    assert X[:, 0].max() <= 15, "Area should be <= 15"
    assert X[:, 1].min() >= 1, "Curvature should be >= 1"
    assert X[:, 1].max() <= 10, "Curvature should be <= 10"
    assert X[:, 2].min() >= 1, "Hardness should be >= 1"
    assert X[:, 2].max() <= 10, "Hardness should be <= 10"
    
    # Check target range
    assert y.min() >= 5, "Min repair time should be >= 5s"
    assert y.max() <= 180, "Max repair time should be <= 180s"
    
    print("✓ Synthetic data generation test passed")


def test_predictor_training():
    """Test that the predictor trains successfully."""
    from src.ml.ml_predictor import RepairTimePredictor
    
    predictor = RepairTimePredictor(n_estimators=10, random_state=42)
    
    assert predictor.is_trained, "Predictor should be trained"
    assert predictor.model is not None, "Model should not be None"
    assert "mae_seconds" in predictor.train_metrics, "Should have MAE metric"
    assert "r2_score" in predictor.train_metrics, "Should have R² metric"
    assert predictor.train_metrics["r2_score"] > 0.5, "R² should be > 0.5"
    
    print(f"✓ Predictor training test passed (R²={predictor.train_metrics['r2_score']:.3f})")


def test_prediction_output():
    """Test prediction output structure."""
    from src.ml.ml_predictor import get_predictor
    
    predictor = get_predictor()
    result = predictor.predict(
        defect_area_cm2=5.0,
        curvature_complexity=5.0,
        material_hardness=5.0
    )
    
    # Check output structure
    assert hasattr(result, 'repair_time_seconds'), "Should have repair_time_seconds"
    assert hasattr(result, 'confidence_lower'), "Should have confidence_lower"
    assert hasattr(result, 'confidence_upper'), "Should have confidence_upper"
    assert hasattr(result, 'consumable_estimate'), "Should have consumable_estimate"
    
    # Check value ranges
    assert 5 <= result.repair_time_seconds <= 180, "Time should be within range"
    assert result.confidence_lower < result.repair_time_seconds, "Lower bound should be less than mean"
    assert result.confidence_upper > result.repair_time_seconds, "Upper bound should be greater than mean"
    
    print(f"✓ Prediction output test passed (predicted={result.repair_time_seconds:.1f}s)")


def test_predict_for_defect():
    """Test prediction from defect dictionary."""
    from src.ml.ml_predictor import get_predictor
    
    predictor = get_predictor()
    
    defect = {
        "type": "crack",
        "severity": "high",
        "position": (0.5, 0.1, 0.3),
    }
    
    result = predictor.predict_for_defect(defect)
    
    assert result.repair_time_seconds > 0, "Time should be positive"
    assert isinstance(result.consumable_estimate, str), "Consumable should be string"
    
    print(f"✓ Defect prediction test passed (type=crack, predicted={result.repair_time_seconds:.1f}s)")


def test_predict_repair_metrics_api():
    """Test the main API function."""
    from src.ml import predict_repair_metrics
    
    # Test with defect dict
    result1 = predict_repair_metrics(defect={"type": "rust", "severity": "medium"})
    assert "repair_time_seconds" in result1, "Should have repair_time_seconds"
    assert "confidence_interval" in result1, "Should have confidence_interval"
    
    # Test with direct features
    result2 = predict_repair_metrics(defect_area_cm2=3.0)
    assert result2["repair_time_seconds"] > 0, "Time should be positive"
    
    print(f"✓ API function test passed")


def test_chart_data_generation():
    """Test chart data generation for UI."""
    from src.ml.ml_predictor import get_predictor
    
    predictor = get_predictor()
    
    defects = [
        {"type": "crack", "severity": "high", "position": (0.5, 0.1, 0.3)},
        {"type": "rust", "severity": "medium", "position": (0.3, 0.2, 0.2)},
    ]
    
    plans = [
        {"estimated_time_seconds": 45},
        {"estimated_time_seconds": 30},
    ]
    
    chart_data = predictor.get_actual_vs_predicted_data(defects, plans)
    
    assert "labels" in chart_data, "Should have labels"
    assert "predicted_times" in chart_data, "Should have predicted_times"
    assert "estimated_times" in chart_data, "Should have estimated_times"
    assert len(chart_data["labels"]) == 2, "Should have 2 labels"
    
    print(f"✓ Chart data generation test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("ML Predictor Module Tests")
    print("=" * 60)
    
    test_generate_training_data()
    test_predictor_training()
    test_prediction_output()
    test_predict_for_defect()
    test_predict_repair_metrics_api()
    test_chart_data_generation()
    
    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
