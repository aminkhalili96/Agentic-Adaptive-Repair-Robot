"""
Localization module for converting pixel coordinates to 3D world coordinates.

Includes:
- Pixel-to-ray conversion
- Camera-to-world transformation
- Surface normal estimation via ray casting
- Tool orientation calculation

Per Codex feedback: includes normal smoothing for jitter reduction.
"""

try:
    import pybullet as p
except ImportError:
    class MockPyBullet:
        def rayTest(self, rayFromPosition, rayToPosition): return []
    p = MockPyBullet()
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from src.config import config


@dataclass  
class Pose3D:
    """
    A 3D pose with position and orientation.
    
    Attributes:
        position: [x, y, z] in world frame
        orientation: [qx, qy, qz, qw] quaternion
        normal: [nx, ny, nz] surface normal vector
        confidence: Detection confidence (0-1)
    """
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    normal: Tuple[float, float, float]
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "position": list(self.position),
            "orientation": list(self.orientation),
            "normal": list(self.normal),
            "confidence": self.confidence,
        }


class Localizer:
    """
    Converts 2D pixel detections to 3D world poses with surface normals.
    """
    
    def __init__(self, environment):
        """
        Initialize localizer with simulation environment.
        
        Args:
            environment: SimulationEnvironment instance
        """
        self.env = environment
        self.K = environment.get_camera_intrinsics()
        self.K_inv = np.linalg.inv(self.K)
        
        # Camera position and target from config
        cam_config = config.get("camera", {})
        self.camera_pos = np.array(cam_config.get("position", [0.5, 0.0, 1.2]))
        self.camera_target = np.array(cam_config.get("target", [0.5, 0.0, 0.2]))
        
    def pixel_to_ray(self, u: int, v: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel coordinates to a ray in world space.
        
        Args:
            u: Horizontal pixel coordinate
            v: Vertical pixel coordinate
            
        Returns:
            Tuple of (ray_origin, ray_direction) in world frame
        """
        # Normalized camera coordinates
        p_norm = self.K_inv @ np.array([u, v, 1.0])
        p_norm = p_norm / np.linalg.norm(p_norm)
        
        # Camera coordinate system
        forward = self.camera_target - self.camera_pos
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Rotation matrix from camera to world
        R_cam_to_world = np.column_stack([right, -up, forward])
        
        # Ray direction in world frame
        ray_dir = R_cam_to_world @ p_norm
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        return self.camera_pos.copy(), ray_dir
    
    def pixel_to_world(
        self,
        u: int,
        v: int,
        depth: float
    ) -> np.ndarray:
        """
        Convert pixel coordinates to 3D world position using depth.
        
        Args:
            u: Horizontal pixel coordinate
            v: Vertical pixel coordinate
            depth: Depth value at (u, v)
            
        Returns:
            [x, y, z] position in world frame
        """
        ray_origin, ray_dir = self.pixel_to_ray(u, v)
        
        # Scale by depth
        # Note: PyBullet depth buffer is linearized distance
        position = ray_origin + ray_dir * depth
        
        return position
    
    def get_surface_normal(
        self,
        pixel_pos: Tuple[int, int],
        max_distance: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get surface normal at a pixel location via ray casting.
        
        Args:
            pixel_pos: (u, v) pixel coordinates
            max_distance: Maximum ray distance
            
        Returns:
            Tuple of (hit_position, surface_normal) in world frame
        """
        u, v = pixel_pos
        ray_origin, ray_dir = self.pixel_to_ray(u, v)
        
        # Cast ray
        ray_end = ray_origin + ray_dir * max_distance
        
        result = p.rayTest(
            rayFromPosition=ray_origin.tolist(),
            rayToPosition=ray_end.tolist()
        )
        
        if result and result[0][0] != -1:
            # Hit something
            hit_position = np.array(result[0][3])
            hit_normal = np.array(result[0][4])
            return hit_position, hit_normal
        else:
            # No hit - return estimated position and default normal
            return ray_origin + ray_dir * 0.5, np.array([0, 0, 1])
    
    def normal_to_orientation(
        self,
        normal: np.ndarray
    ) -> np.ndarray:
        """
        Calculate tool orientation quaternion from surface normal.
        
        The tool Z-axis should point INTO the surface (negative normal).
        
        Args:
            normal: [nx, ny, nz] surface normal vector
            
        Returns:
            [qx, qy, qz, qw] quaternion for tool orientation
        """
        # Target direction: point INTO the surface
        target = -np.array(normal)
        target = target / np.linalg.norm(target)
        
        # Source direction: default tool Z-axis
        source = np.array([0, 0, 1])
        
        # Handle edge cases
        dot = np.dot(source, target)
        
        if dot > 0.9999:
            # Already aligned
            return np.array([0, 0, 0, 1])
        
        if dot < -0.9999:
            # Opposite direction - 180° rotation around X
            return np.array([1, 0, 0, 0])
        
        # Rotation axis and angle
        axis = np.cross(source, target)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1, 1))
        
        # Convert to quaternion using scipy
        rot = Rotation.from_rotvec(axis * angle)
        return rot.as_quat()  # [x, y, z, w]
    
    def localize_defect(
        self,
        centroid_px: Tuple[int, int],
        confidence: float = 1.0
    ) -> Pose3D:
        """
        Localize a detected defect to a 3D pose with orientation.
        
        Args:
            centroid_px: (u, v) pixel coordinates of defect centroid
            confidence: Detection confidence
            
        Returns:
            Pose3D with position, orientation, and normal
        """
        # Get surface position and normal via ray casting
        position, normal = self.get_surface_normal(centroid_px)
        
        # Calculate tool orientation
        orientation = self.normal_to_orientation(normal)
        
        return Pose3D(
            position=tuple(position),
            orientation=tuple(orientation),
            normal=tuple(normal),
            confidence=confidence,
        )
    
    def smooth_normals(
        self,
        poses: List[Pose3D],
        window: int = 3
    ) -> List[Pose3D]:
        """
        Apply moving average smoothing to normals (per Codex feedback).
        
        Reduces jitter in tool orientation along a path.
        
        Args:
            poses: List of Pose3D objects
            window: Smoothing window size
            
        Returns:
            List of Pose3D with smoothed normals and orientations
        """
        if len(poses) <= 1:
            return poses
        
        normals = np.array([p.normal for p in poses])
        smoothed_normals = np.zeros_like(normals)
        
        for i in range(len(normals)):
            start = max(0, i - window // 2)
            end = min(len(normals), i + window // 2 + 1)
            avg = np.mean(normals[start:end], axis=0)
            smoothed_normals[i] = avg / np.linalg.norm(avg)
        
        # Recalculate orientations with smoothed normals
        smoothed_poses = []
        for i, (pose, normal) in enumerate(zip(poses, smoothed_normals)):
            new_orientation = self.normal_to_orientation(normal)
            smoothed_poses.append(Pose3D(
                position=pose.position,
                orientation=tuple(new_orientation),
                normal=tuple(normal),
                confidence=pose.confidence,
            ))
        
        return smoothed_poses
