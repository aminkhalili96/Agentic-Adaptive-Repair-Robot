"""
Simple test runner for Path Optimizer (TSP) module.
No pytest dependency - just run with: python src/test_path_optimizer_simple.py
"""

from dataclasses import dataclass
from typing import Tuple
import sys

# Add project to path
sys.path.insert(0, '/Users/amin/dev/Robotic AI')

from src.planning.tsp import (
    distance,
    nearest_neighbor_tsp,
    calculate_total_distance,
    optimize_with_metrics,
    PathOptimizationResult,
)


@dataclass
class MockDefect:
    """Simple defect for testing."""
    id: int
    position: Tuple[float, float, float]


def test_distance():
    """Test Euclidean distance calculation."""
    print("Testing distance()...")
    
    assert distance((0, 0, 0), (0, 0, 0)) == 0.0, "Same point should be 0"
    assert distance((0, 0, 0), (1, 0, 0)) == 1.0, "Unit X distance should be 1"
    assert abs(distance((0, 0, 0), (1, 1, 1)) - 1.732) < 0.01, "3D diagonal should be sqrt(3)"
    
    print("  ✓ distance() tests passed")


def test_calculate_total_distance():
    """Test total path distance calculation."""
    print("Testing calculate_total_distance()...")
    
    # Empty list
    result = calculate_total_distance([], lambda x: x, (0, 0, 0))
    assert result == 0.0, "Empty list should be 0"
    
    # Two items in line
    defects = [MockDefect(1, (1, 0, 0)), MockDefect(2, (3, 0, 0))]
    result = calculate_total_distance(defects, lambda d: d.position, start_pos=(0, 0, 0))
    assert result == 3.0, f"Expected 3.0, got {result}"
    
    print("  ✓ calculate_total_distance() tests passed")


def test_nearest_neighbor():
    """Test nearest-neighbor TSP."""
    print("Testing nearest_neighbor_tsp()...")
    
    items = [
        MockDefect(1, (10, 0, 0)),  # Far
        MockDefect(2, (1, 0, 0)),   # Near
        MockDefect(3, (5, 0, 0)),   # Medium
    ]
    result = nearest_neighbor_tsp(items, lambda d: d.position, start_pos=(0, 0, 0))
    
    assert result[0].id == 2, "First should be nearest (id=2)"
    assert result[1].id == 3, "Second should be medium (id=3)"
    assert result[2].id == 1, "Third should be farthest (id=1)"
    
    print("  ✓ nearest_neighbor_tsp() tests passed")


def test_optimize_with_metrics():
    """Test optimization with efficiency metrics."""
    print("Testing optimize_with_metrics()...")
    
    # Single defect - no optimization
    defects = [MockDefect(1, (1, 0, 0))]
    result = optimize_with_metrics(defects)
    assert isinstance(result, PathOptimizationResult), "Should return PathOptimizationResult"
    assert result.efficiency_gain_percent == 0.0, "Single defect should have 0% gain"
    
    # Suboptimal order should improve
    defects = [
        MockDefect(1, (10, 0, 0)),  # Far - listed first
        MockDefect(2, (1, 0, 0)),   # Near - listed second
    ]
    result = optimize_with_metrics(defects, robot_pos=(0, 0, 0))
    assert result.optimized_distance <= result.original_distance, "Optimized should be <= original"
    
    print("  ✓ optimize_with_metrics() tests passed")


def test_efficiency_calculation():
    """Test efficiency gain calculation."""
    print("Testing efficiency calculation...")
    
    # Reverse order path
    defects = [
        MockDefect(1, (2, 0, 0)),  # Should be second
        MockDefect(2, (1, 0, 0)),  # Should be first
    ]
    result = optimize_with_metrics(defects, robot_pos=(0, 0, 0), use_2opt=False)
    
    # Original: 0->(2,0,0)=2 + (2,0,0)->(1,0,0)=1 = 3
    # Optimal:  0->(1,0,0)=1 + (1,0,0)->(2,0,0)=1 = 2
    # Gain: (3-2)/3 = 33%
    assert result.original_distance == 3.0, f"Original should be 3.0, got {result.original_distance}"
    assert result.optimized_distance == 2.0, f"Optimized should be 2.0, got {result.optimized_distance}"
    assert abs(result.efficiency_gain_percent - 33.33) < 1.0, f"Gain should be ~33%, got {result.efficiency_gain_percent}"
    
    print("  ✓ Efficiency calculation tests passed")


def test_summary_message():
    """Test summary message generation."""
    print("Testing get_summary_message()...")
    
    defects = [MockDefect(1, (10, 0, 0)), MockDefect(2, (1, 0, 0))]
    result = optimize_with_metrics(defects, robot_pos=(0, 0, 0))
    message = result.get_summary_message()
    
    assert "path optimization" in message.lower() or "optimal" in message.lower(), "Message should mention optimization"
    assert "m" in message, "Message should include distance unit"
    
    print("  ✓ get_summary_message() tests passed")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Path Optimizer Unit Tests")
    print("=" * 60 + "\n")
    
    try:
        test_distance()
        test_calculate_total_distance()
        test_nearest_neighbor()
        test_optimize_with_metrics()
        test_efficiency_calculation()
        test_summary_message()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)
