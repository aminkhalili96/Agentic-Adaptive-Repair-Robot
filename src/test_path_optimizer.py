"""
Unit tests for the Path Optimizer (TSP) module.

Tests the efficiency calculation and optimization logic.
"""

import pytest
from dataclasses import dataclass
from typing import Tuple

# Import the module under test
from src.planning.tsp import (
    distance,
    nearest_neighbor_tsp,
    two_opt_improve,
    optimize_defect_order,
    calculate_total_distance,
    optimize_with_metrics,
    PathOptimizationResult,
)


# ============ MOCK DEFECT CLASS ============
@dataclass
class MockDefect:
    """Simple defect for testing with position attribute."""
    id: int
    position: Tuple[float, float, float]


# ============ DISTANCE TESTS ============
class TestDistance:
    """Tests for the Euclidean distance function."""
    
    def test_same_point(self):
        """Distance to same point should be 0."""
        assert distance((0, 0, 0), (0, 0, 0)) == 0.0
    
    def test_unit_distance_x(self):
        """Distance of 1 along X axis."""
        assert distance((0, 0, 0), (1, 0, 0)) == 1.0
    
    def test_3d_diagonal(self):
        """Distance along 3D diagonal (1,1,1)."""
        result = distance((0, 0, 0), (1, 1, 1))
        assert abs(result - 1.732) < 0.01  # sqrt(3) ≈ 1.732


# ============ CALCULATE TOTAL DISTANCE TESTS ============
class TestCalculateTotalDistance:
    """Tests for calculating total path distance."""
    
    def test_empty_list(self):
        """Empty list should return 0."""
        result = calculate_total_distance([], lambda x: x, (0, 0, 0))
        assert result == 0.0
    
    def test_single_item_with_start(self):
        """Single item with start position."""
        defects = [MockDefect(1, (1, 0, 0))]
        result = calculate_total_distance(
            defects,
            lambda d: d.position,
            start_pos=(0, 0, 0)
        )
        assert result == 1.0  # Distance from origin to (1,0,0)
    
    def test_two_items_in_line(self):
        """Two items along a line from origin."""
        defects = [
            MockDefect(1, (1, 0, 0)),
            MockDefect(2, (3, 0, 0)),
        ]
        result = calculate_total_distance(
            defects,
            lambda d: d.position,
            start_pos=(0, 0, 0)
        )
        # 0 -> (1,0,0) = 1, (1,0,0) -> (3,0,0) = 2
        assert result == 3.0


# ============ NEAREST NEIGHBOR TSP TESTS ============
class TestNearestNeighborTSP:
    """Tests for the nearest-neighbor heuristic."""
    
    def test_single_item(self):
        """Single item should return as is."""
        items = [MockDefect(1, (5, 5, 5))]
        result = nearest_neighbor_tsp(items, lambda d: d.position)
        assert len(result) == 1
        assert result[0].id == 1
    
    def test_orders_by_distance(self):
        """Should order items by nearest distance."""
        items = [
            MockDefect(1, (10, 0, 0)),  # Far
            MockDefect(2, (1, 0, 0)),   # Near
            MockDefect(3, (5, 0, 0)),   # Medium
        ]
        result = nearest_neighbor_tsp(
            items,
            lambda d: d.position,
            start_pos=(0, 0, 0)
        )
        # From origin: nearest is (1,0,0), then (5,0,0), then (10,0,0)
        assert result[0].id == 2
        assert result[1].id == 3
        assert result[2].id == 1


# ============ OPTIMIZE WITH METRICS TESTS ============
class TestOptimizeWithMetrics:
    """Tests for the optimization with efficiency metrics."""
    
    def test_single_defect(self):
        """Single defect returns 0% efficiency gain."""
        defects = [MockDefect(1, (1, 0, 0))]
        result = optimize_with_metrics(defects)
        
        assert isinstance(result, PathOptimizationResult)
        assert result.efficiency_gain_percent == 0.0
        assert "single or no defects" in result.algorithm_used
    
    def test_already_optimal_path(self):
        """Path already in optimal order."""
        defects = [
            MockDefect(1, (1, 0, 0)),
            MockDefect(2, (2, 0, 0)),
            MockDefect(3, (3, 0, 0)),
        ]
        result = optimize_with_metrics(defects, robot_pos=(0, 0, 0))
        
        # Already optimal linear path
        assert result.efficiency_gain_percent >= 0.0
        assert result.optimized_distance <= result.original_distance
    
    def test_suboptimal_path_improves(self):
        """Suboptimal path should show efficiency gain."""
        # Defects in worst order: farthest first
        defects = [
            MockDefect(1, (10, 0, 0)),  # Far - listed first
            MockDefect(2, (1, 0, 0)),   # Near - listed second
            MockDefect(3, (5, 0, 0)),   # Medium - listed third
        ]
        result = optimize_with_metrics(defects, robot_pos=(0, 0, 0))
        
        # Should reorder to: 2 (near) -> 3 (medium) -> 1 (far)
        assert result.optimized_distance < result.original_distance
        assert result.efficiency_gain_percent > 0
    
    def test_summary_message_format(self):
        """Summary message should be properly formatted."""
        defects = [
            MockDefect(1, (10, 0, 0)),
            MockDefect(2, (1, 0, 0)),
        ]
        result = optimize_with_metrics(defects, robot_pos=(0, 0, 0))
        message = result.get_summary_message()
        
        assert "path optimization" in message.lower() or "optimal" in message.lower()
        assert "m" in message  # Distance in meters


# ============ EFFICIENCY CALCULATION TESTS ============
class TestEfficiencyCalculation:
    """Tests specifically for efficiency gain calculation."""
    
    def test_50_percent_improvement(self):
        """Test roughly 50% path improvement scenario."""
        # Reverse order path (worst case for linear layout)
        defects = [
            MockDefect(1, (2, 0, 0)),  # Listed first but should be second
            MockDefect(2, (1, 0, 0)),  # Listed second but should be first
        ]
        result = optimize_with_metrics(defects, robot_pos=(0, 0, 0), use_2opt=False)
        
        # Original: 0->(2,0,0)=2 + (2,0,0)->(1,0,0)=1 = 3
        # Optimal:  0->(1,0,0)=1 + (1,0,0)->(2,0,0)=1 = 2
        # Gain: (3-2)/3 = 33%
        assert result.original_distance == 3.0
        assert result.optimized_distance == 2.0
        assert abs(result.efficiency_gain_percent - 33.33) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
