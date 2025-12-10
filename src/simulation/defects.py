"""
Defect spawning utilities for the simulation.

Creates visual markers for different defect types:
- Rust: Red spheres
- Crack: Black lines
- Dent: Blue concave markers
"""

try:
    import pybullet as p
except ImportError:
    class MockPyBullet:
        GEOM_BOX = 1
        GEOM_SPHERE = 2
        def createVisualShape(self, *args, **kwargs): return 0
        def createMultiBody(self, *args, **kwargs): return 0
        def changeVisualShape(self, *args, **kwargs): pass
        def removeBody(self, *args): pass
    p = MockPyBullet()
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from src.config import config


class DefectType(Enum):
    """Types of surface defects."""
    RUST = "rust"
    CRACK = "crack"
    DENT = "dent"


@dataclass
class Defect:
    """Represents a surface defect."""
    id: int  # PyBullet body ID
    type: DefectType
    position: Tuple[float, float, float]
    size: float
    repaired: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "position": list(self.position),
            "size": self.size,
            "repaired": self.repaired,
        }


# Defect colors (RGBA)
DEFECT_COLORS = {
    DefectType.RUST: [0.8, 0.2, 0.1, 1.0],    # Red-brown
    DefectType.CRACK: [0.1, 0.1, 0.1, 1.0],   # Black
    DefectType.DENT: [0.2, 0.4, 0.8, 1.0],    # Blue
}

# Repaired color (grey)
REPAIRED_COLOR = [0.5, 0.5, 0.5, 1.0]


def spawn_defect(
    defect_type: DefectType,
    position: Tuple[float, float, float],
    size: float = 0.03
) -> Defect:
    """
    Spawn a single defect marker in the simulation.
    
    Args:
        defect_type: Type of defect (rust, crack, dent)
        position: [x, y, z] position in world frame
        size: Radius/size of the defect marker
        
    Returns:
        Defect object with PyBullet ID
    """
    color = DEFECT_COLORS.get(defect_type, [1, 0, 0, 1])
    
    if defect_type == DefectType.RUST:
        # Rust: Sphere
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=size,
            rgbaColor=color
        )
    elif defect_type == DefectType.CRACK:
        # Crack: Thin box (line-like)
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[size * 2, size * 0.2, size * 0.1],
            rgbaColor=color
        )
    elif defect_type == DefectType.DENT:
        # Dent: Slightly larger blue sphere
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=size * 1.2,
            rgbaColor=color
        )
    else:
        # Default: sphere
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=size,
            rgbaColor=color
        )
    
    # Create the body (visual only, no collision for markers)
    defect_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )
    
    return Defect(
        id=defect_id,
        type=defect_type,
        position=position,
        size=size
    )


def spawn_random_defects(
    workpiece_position: Tuple[float, float, float],
    workpiece_size: float,
    count: int = 3,
    types: Optional[List[DefectType]] = None
) -> List[Defect]:
    """
    Spawn random defects on a workpiece surface.
    
    Args:
        workpiece_position: Center position of the workpiece
        workpiece_size: Size of the workpiece (assumes cube)
        count: Number of defects to spawn
        types: List of allowed defect types (default: all)
        
    Returns:
        List of Defect objects
    """
    if types is None:
        types = list(DefectType)
    
    defects = []
    
    # Get random seed from config for reproducibility
    rng = np.random.default_rng(config.get("simulation", {}).get("seed", 42))
    
    for i in range(count):
        # Random position on top surface of workpiece
        half_size = workpiece_size / 2 * 0.8  # Stay within bounds
        x = workpiece_position[0] + rng.uniform(-half_size, half_size)
        y = workpiece_position[1] + rng.uniform(-half_size, half_size)
        z = workpiece_position[2] + workpiece_size / 2 + 0.01  # Slightly above surface
        
        # Random type
        defect_type = rng.choice(types)
        
        # Random size
        size = rng.uniform(0.02, 0.04)
        
        defect = spawn_defect(defect_type, (x, y, z), size)
        defects.append(defect)
    
    return defects


def spawn_demo_defects(
    workpiece_position: Tuple[float, float, float],
    workpiece_size: float
) -> List[Defect]:
    """
    Spawn a predefined set of demo defects (deterministic).
    
    Args:
        workpiece_position: Center position of the workpiece
        workpiece_size: Size of the workpiece
        
    Returns:
        List of 3 defects (rust, crack, dent)
    """
    z_surface = workpiece_position[2] + workpiece_size / 2 + 0.01
    
    defects = [
        spawn_defect(
            DefectType.RUST,
            (workpiece_position[0] - 0.05, workpiece_position[1] + 0.05, z_surface),
            size=0.03
        ),
        spawn_defect(
            DefectType.CRACK,
            (workpiece_position[0] + 0.05, workpiece_position[1] - 0.03, z_surface),
            size=0.025
        ),
        spawn_defect(
            DefectType.DENT,
            (workpiece_position[0], workpiece_position[1], z_surface),
            size=0.035
        ),
    ]
    
    return defects


def mark_defect_repaired(defect: Defect) -> None:
    """
    Mark a defect as repaired by changing its color to grey.
    
    Args:
        defect: The defect to mark as repaired
    """
    p.changeVisualShape(defect.id, -1, rgbaColor=REPAIRED_COLOR)
    defect.repaired = True


def remove_defect(defect: Defect) -> None:
    """
    Remove a defect from the simulation.
    
    Args:
        defect: The defect to remove
    """
    p.removeBody(defect.id)
