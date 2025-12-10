"""
TSP (Traveling Salesman Problem) solver for multi-defect ordering.

Uses nearest-neighbor heuristic with optional 2-opt improvement
per Codex feedback.
"""

import numpy as np
from typing import List, TypeVar
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class PathOptimizationResult:
    """
    Result of path optimization with efficiency metrics.
    
    Attributes:
        original_order: List of items in original order
        optimized_order: List of items in optimized order
        original_distance: Total travel distance before optimization (meters)
        optimized_distance: Total travel distance after optimization (meters)
        efficiency_gain_percent: Percentage improvement ((orig - opt) / orig * 100)
        algorithm_used: Name of the algorithm used
    """
    original_order: list
    optimized_order: list
    original_distance: float
    optimized_distance: float
    efficiency_gain_percent: float
    algorithm_used: str
    
    def get_summary_message(self) -> str:
        """Generate a human-readable summary of the optimization."""
        if self.efficiency_gain_percent > 0:
            return (
                f"I have re-sequenced the repair order using a path optimization algorithm. "
                f"Original Path: {self.original_distance:.2f}m. "
                f"Optimized Path: {self.optimized_distance:.2f}m. "
                f"AI Efficiency Gain: +{self.efficiency_gain_percent:.0f}%."
            )
        else:
            return (
                f"Path is already optimal. "
                f"Total travel distance: {self.optimized_distance:.2f}m."
            )


def distance(pos1: tuple, pos2: tuple) -> float:
    """
    Calculate Euclidean distance between two 3D positions.
    
    Args:
        pos1: (x, y, z) first position
        pos2: (x, y, z) second position
        
    Returns:
        Distance between positions
    """
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


def nearest_neighbor_tsp(
    items: List[T],
    get_position: callable,
    start_pos: tuple = None
) -> List[T]:
    """
    Order items using nearest-neighbor heuristic.
    
    Args:
        items: List of items to order
        get_position: Function that extracts (x, y, z) position from an item
        start_pos: Optional starting position (default: first item)
        
    Returns:
        Reordered list minimizing total travel distance
    """
    if len(items) <= 1:
        return items
    
    remaining = list(items)
    ordered = []
    
    # Start from first item or given position
    if start_pos is None:
        current = remaining.pop(0)
        ordered.append(current)
        current_pos = get_position(current)
    else:
        current_pos = start_pos
    
    while remaining:
        # Find nearest
        nearest_idx = min(
            range(len(remaining)),
            key=lambda i: distance(current_pos, get_position(remaining[i]))
        )
        
        current = remaining.pop(nearest_idx)
        ordered.append(current)
        current_pos = get_position(current)
    
    return ordered


def two_opt_improve(
    items: List[T],
    get_position: callable,
    max_iterations: int = 100
) -> List[T]:
    """
    Improve TSP solution using 2-opt swaps (per Codex feedback).
    
    Args:
        items: Initial ordering
        get_position: Function to extract position
        max_iterations: Maximum improvement iterations
        
    Returns:
        Improved ordering
    """
    if len(items) <= 2:
        return items
    
    def total_distance(order):
        return sum(
            distance(get_position(order[i]), get_position(order[i + 1]))
            for i in range(len(order) - 1)
        )
    
    best = list(items)
    best_dist = total_distance(best)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(best) - 2):
            for j in range(i + 2, len(best)):
                # Try reversing segment between i+1 and j
                new_order = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                new_dist = total_distance(new_order)
                
                if new_dist < best_dist:
                    best = new_order
                    best_dist = new_dist
                    improved = True
                    break
            
            if improved:
                break
    
    return best


def optimize_defect_order(
    defects: List,
    robot_pos: tuple = (0, 0, 0.5),
    use_2opt: bool = True
) -> List:
    """
    Optimize the order of defects to minimize robot travel.
    
    Args:
        defects: List of defects (with .position attribute or Pose3D objects)
        robot_pos: Starting robot position
        use_2opt: Whether to apply 2-opt improvement
        
    Returns:
        Reordered defect list
    """
    if len(defects) <= 1:
        return defects
    
    # Extract position from defect
    def get_pos(defect):
        if hasattr(defect, 'position'):
            return defect.position
        elif hasattr(defect, 'centroid_px'):
            # For DetectedDefect, use centroid (will be 2D but still useful for ordering)
            return (*defect.centroid_px, 0)
        else:
            return (0, 0, 0)
    
    # Initial ordering with nearest neighbor
    ordered = nearest_neighbor_tsp(defects, get_pos, robot_pos)
    
    # Improve with 2-opt if requested
    if use_2opt:
        ordered = two_opt_improve(ordered, get_pos)
    
    return ordered


def calculate_total_distance(
    items: List,
    get_position: callable,
    start_pos: tuple = None
) -> float:
    """
    Calculate the total travel distance for a given order of items.
    
    Args:
        items: Ordered list of items
        get_position: Function to extract (x, y, z) position from item
        start_pos: Optional starting position
        
    Returns:
        Total Euclidean distance in meters
    """
    if len(items) == 0:
        return 0.0
    
    total = 0.0
    
    # Distance from start to first item
    if start_pos is not None:
        total += distance(start_pos, get_position(items[0]))
    
    # Distance between consecutive items
    for i in range(len(items) - 1):
        total += distance(get_position(items[i]), get_position(items[i + 1]))
    
    return total


def optimize_with_metrics(
    defects: List,
    robot_pos: tuple = (0, 0, 0.5),
    use_2opt: bool = True
) -> PathOptimizationResult:
    """
    Optimize defect order and calculate efficiency metrics.
    
    This wraps the TSP solver to provide before/after distance comparison
    and efficiency gain calculation.
    
    Args:
        defects: List of defects to optimize
        robot_pos: Starting robot position (x, y, z)
        use_2opt: Whether to apply 2-opt improvement
        
    Returns:
        PathOptimizationResult with optimization metrics
    """
    if len(defects) <= 1:
        return PathOptimizationResult(
            original_order=list(defects),
            optimized_order=list(defects),
            original_distance=0.0,
            optimized_distance=0.0,
            efficiency_gain_percent=0.0,
            algorithm_used="none (single or no defects)"
        )
    
    # Position extractor
    def get_pos(defect):
        if hasattr(defect, 'position'):
            return defect.position
        elif hasattr(defect, 'centroid_px'):
            return (*defect.centroid_px, 0)
        else:
            return (0, 0, 0)
    
    # Keep original order
    original_order = list(defects)
    original_distance = calculate_total_distance(original_order, get_pos, robot_pos)
    
    # Optimize
    optimized_order = optimize_defect_order(defects, robot_pos, use_2opt)
    optimized_distance = calculate_total_distance(optimized_order, get_pos, robot_pos)
    
    # Calculate efficiency gain
    if original_distance > 0:
        efficiency_gain = ((original_distance - optimized_distance) / original_distance) * 100
    else:
        efficiency_gain = 0.0
    
    algorithm = "nearest-neighbor + 2-opt" if use_2opt else "nearest-neighbor"
    
    return PathOptimizationResult(
        original_order=original_order,
        optimized_order=optimized_order,
        original_distance=original_distance,
        optimized_distance=optimized_distance,
        efficiency_gain_percent=max(0.0, efficiency_gain),  # Ensure non-negative
        algorithm_used=algorithm
    )
