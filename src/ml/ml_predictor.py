"""
ML Predictor - Predictive Maintenance Intelligence

Predicts repair time and consumable usage based on defect characteristics
using a trained RandomForest model on synthetic historical data.

Features:
    - defect_area_cm2: Size of the defect area in cm²
    - curvature_complexity: Surface curvature difficulty (1-10)
    - material_hardness: Material hardness factor

Target:
    - repair_time_seconds: Predicted repair duration

Relation (training data):
    time = area * 0.5 + curvature * 2 + hardness * 0.3 + noise
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. ML predictions unavailable.")


# ============ DATA CLASSES ============
@dataclass
class PredictionResult:
    """Result of a repair time prediction."""
    repair_time_seconds: float
    confidence_lower: float
    confidence_upper: float
    consumable_estimate: str
    feature_contributions: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "repair_time_seconds": round(self.repair_time_seconds, 1),
            "confidence_interval": {
                "lower": round(self.confidence_lower, 1),
                "upper": round(self.confidence_upper, 1),
            },
            "consumable_estimate": self.consumable_estimate,
            "feature_contributions": self.feature_contributions,
        }


# ============ SYNTHETIC DATA GENERATION ============
def generate_training_data(n_samples: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for the repair time predictor.
    
    The relation simulates real-world repair dynamics:
    - Larger defects take longer to repair
    - Complex curved surfaces require more careful work
    - Harder materials are slower to process
    
    Args:
        n_samples: Number of training samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (features, targets) as numpy arrays
        Features shape: (n_samples, 3) - [area, curvature, hardness]
        Targets shape: (n_samples,) - repair time in seconds
    """
    rng = np.random.default_rng(seed)
    
    # Generate features with realistic distributions
    # Defect area: 0.5 to 15 cm² (skewed towards smaller)
    defect_area_cm2 = rng.exponential(scale=3, size=n_samples).clip(0.5, 15)
    
    # Curvature complexity: 1 to 10 (uniform with some peaks at common values)
    curvature_complexity = rng.integers(1, 11, size=n_samples).astype(float)
    curvature_complexity += rng.normal(0, 0.5, size=n_samples)  # Add noise
    curvature_complexity = curvature_complexity.clip(1, 10)
    
    # Material hardness: 1 to 10 (some materials more common)
    # Simulate: soft aluminum(3), steel(6), titanium(8), composite(5)
    material_base = rng.choice([3, 5, 6, 8], size=n_samples, p=[0.25, 0.25, 0.35, 0.15])
    material_hardness = material_base + rng.normal(0, 0.8, size=n_samples)
    material_hardness = material_hardness.clip(1, 10)
    
    # Generate target: repair_time_seconds
    # Base formula with realistic coefficients
    base_time = (
        defect_area_cm2 * 2.5 +           # 2.5 seconds per cm²
        curvature_complexity * 3.0 +       # 3 seconds per complexity level
        material_hardness * 1.5 +          # 1.5 seconds per hardness level
        10  # Base setup time
    )
    
    # Add interaction effects (curved + hard = even slower)
    interaction = 0.1 * curvature_complexity * material_hardness
    
    # Add random noise (±15% variation)
    noise = rng.normal(0, base_time * 0.15, size=n_samples)
    
    repair_time_seconds = base_time + interaction + noise
    repair_time_seconds = repair_time_seconds.clip(5, 180)  # 5 sec to 3 min
    
    # Stack features
    features = np.column_stack([defect_area_cm2, curvature_complexity, material_hardness])
    
    return features, repair_time_seconds


# ============ MODEL CLASS ============
class RepairTimePredictor:
    """
    Machine Learning predictor for repair time estimation.
    
    Uses RandomForestRegressor trained on synthetic historical data.
    Provides predictions with confidence intervals.
    """
    
    FEATURE_NAMES = ["defect_area_cm2", "curvature_complexity", "material_hardness"]
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the predictor.
        
        Args:
            n_estimators: Number of trees in the random forest
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None
        self.is_trained = False
        self.train_metrics: Dict[str, float] = {}
        
        if HAS_SKLEARN:
            self._train()
    
    def _train(self):
        """Train the model on synthetic data."""
        print("🤖 Training ML predictor on synthetic repair data...")
        
        # Generate training data
        X, y = generate_training_data(n_samples=1000, seed=self.random_state)
        
        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.train_metrics = {
            "mae_seconds": round(mae, 2),
            "r2_score": round(r2, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        
        self.is_trained = True
        print(f"   ✓ Model trained: MAE={mae:.1f}s, R²={r2:.3f}")
    
    def predict(
        self,
        defect_area_cm2: float,
        curvature_complexity: float = 5.0,
        material_hardness: float = 5.0
    ) -> PredictionResult:
        """
        Predict repair time for a defect.
        
        Args:
            defect_area_cm2: Size of the defect in cm²
            curvature_complexity: Surface curvature difficulty (1-10)
            material_hardness: Material hardness factor (1-10)
            
        Returns:
            PredictionResult with time estimate and confidence interval
        """
        if not HAS_SKLEARN or not self.is_trained:
            # Fallback to simple formula
            time = defect_area_cm2 * 2.5 + curvature_complexity * 3.0 + material_hardness * 1.5 + 10
            return PredictionResult(
                repair_time_seconds=time,
                confidence_lower=time * 0.8,
                confidence_upper=time * 1.2,
                consumable_estimate=self._estimate_consumables(time),
                feature_contributions={
                    "area": defect_area_cm2 * 2.5,
                    "curvature": curvature_complexity * 3.0,
                    "hardness": material_hardness * 1.5,
                }
            )
        
        # Prepare features
        X = np.array([[defect_area_cm2, curvature_complexity, material_hardness]])
        
        # Get predictions from all trees for confidence interval
        tree_predictions = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
        
        mean_pred = np.mean(tree_predictions)
        std_pred = np.std(tree_predictions)
        
        # 90% confidence interval
        confidence_lower = mean_pred - 1.645 * std_pred
        confidence_upper = mean_pred + 1.645 * std_pred
        
        # Feature importance contributions (approximate)
        importances = self.model.feature_importances_
        contributions = {
            "area": float(importances[0] * defect_area_cm2 * 2.5),
            "curvature": float(importances[1] * curvature_complexity * 3.0),
            "hardness": float(importances[2] * material_hardness * 1.5),
        }
        
        return PredictionResult(
            repair_time_seconds=float(mean_pred),
            confidence_lower=float(max(5, confidence_lower)),
            confidence_upper=float(min(180, confidence_upper)),
            consumable_estimate=self._estimate_consumables(mean_pred),
            feature_contributions=contributions
        )
    
    def _estimate_consumables(self, repair_time: float) -> str:
        """Estimate consumable usage based on repair time."""
        if repair_time < 20:
            return "Low: Minor abrasive disc usage"
        elif repair_time < 45:
            return "Medium: Standard tool wear + filler material"
        elif repair_time < 90:
            return "High: Multiple disc changes + significant filler"
        else:
            return "Very High: Full tool replacement + extensive materials"
    
    def predict_for_defect(self, defect: Dict[str, Any]) -> PredictionResult:
        """
        Predict repair time from a defect dictionary.
        
        Extracts/estimates features from defect metadata.
        
        Args:
            defect: Defect dictionary with 'type', 'severity', 'position', etc.
            
        Returns:
            PredictionResult with predictions
        """
        # Extract or estimate features from defect data
        defect_type = defect.get("type", "unknown").lower()
        severity = defect.get("severity", "medium").lower()
        size = defect.get("size", 4.0)  # Default 4 cm²
        
        # Estimate area from type and severity
        base_areas = {
            "crack": 2.0,
            "corrosion": 5.0,
            "rust": 5.0,
            "wear": 3.0,
            "pitting": 1.5,
            "dent": 4.0,
            "erosion": 6.0,
        }
        base_area = base_areas.get(defect_type, 4.0)
        
        severity_multipliers = {"low": 0.6, "medium": 1.0, "high": 1.5}
        severity_mult = severity_multipliers.get(severity, 1.0)
        
        defect_area_cm2 = base_area * severity_mult
        
        # Estimate curvature from position (higher z = more curved surface typically)
        position = defect.get("position", (0, 0, 0))
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            curvature_complexity = min(10, max(1, abs(position[2]) * 10 + 3))
        else:
            curvature_complexity = 5.0
        
        # Estimate hardness from defect type
        type_hardness = {
            "crack": 7.0,      # Usually in harder materials
            "corrosion": 4.0,  # Softer corroded surface
            "rust": 3.5,       # Soft oxidized layer
            "wear": 6.0,       # Standard steel
            "pitting": 5.0,    # Average
            "dent": 5.5,       # Deformable material
            "erosion": 4.5,    # Partially eroded
        }
        material_hardness = type_hardness.get(defect_type, 5.0)
        
        return self.predict(defect_area_cm2, curvature_complexity, material_hardness)
    
    def get_actual_vs_predicted_data(self, defects: List[Dict], plans: List[Dict]) -> Dict[str, List]:
        """
        Generate data for Predicted vs Actual chart.
        
        Args:
            defects: List of defect dictionaries
            plans: List of repair plan dictionaries
            
        Returns:
            Dict with labels, predicted_times, and estimated_times for charting
        """
        labels = []
        predicted_times = []
        estimated_times = []
        
        for i, (defect, plan) in enumerate(zip(defects, plans)):
            # Get ML prediction
            prediction = self.predict_for_defect(defect)
            
            # Get rule-based estimate from plan
            estimated = plan.get("estimated_time_seconds", plan.get("estimated_time", 30))
            
            labels.append(f"Defect {i+1}: {defect.get('type', 'Unknown')}")
            predicted_times.append(prediction.repair_time_seconds)
            estimated_times.append(estimated)
        
        return {
            "labels": labels,
            "predicted_times": predicted_times,
            "estimated_times": estimated_times,
        }


# ============ GLOBAL PREDICTOR ============
# Initialize predictor on module load (trains the model)
_predictor: Optional[RepairTimePredictor] = None


def get_predictor() -> RepairTimePredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = RepairTimePredictor()
    return _predictor


def predict_repair_metrics(
    defect: Dict[str, Any] = None,
    defect_area_cm2: float = None,
    curvature_complexity: float = 5.0,
    material_hardness: float = 5.0
) -> Dict[str, Any]:
    """
    Predict repair metrics for a defect.
    
    This is the main API function for external use.
    
    Args:
        defect: Optional defect dictionary (if provided, extracts features automatically)
        defect_area_cm2: Manual area specification
        curvature_complexity: Manual curvature specification
        material_hardness: Manual hardness specification
        
    Returns:
        Dictionary with prediction results
    """
    predictor = get_predictor()
    
    if defect is not None:
        result = predictor.predict_for_defect(defect)
    elif defect_area_cm2 is not None:
        result = predictor.predict(defect_area_cm2, curvature_complexity, material_hardness)
    else:
        # Default prediction for demo
        result = predictor.predict(4.0, 5.0, 5.0)
    
    return result.to_dict()


# Initialize on import (lazy training)
if HAS_SKLEARN:
    # Pre-warm the predictor (trains the model)
    get_predictor()
