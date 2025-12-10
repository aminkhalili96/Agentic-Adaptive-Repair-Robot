"""
Robot controller for KUKA iiwa arm.

Handles:
- Inverse Kinematics (IK) with orientation
- Motor control
- Path execution
- Collision checking

Per Codex feedback: includes workspace bounds checking.
"""

try:
    import pybullet as p
except ImportError:
    class MockPyBullet:
        POSITION_CONTROL = 1
        def calculateInverseKinematics(self, *args, **kwargs): return [0]*7
        def setJointMotorControl2(self, *args, **kwargs): pass
        def getClosestPoints(self, *args, **kwargs): return []
    p = MockPyBullet()
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from src.config import config
from src.planning.paths import Waypoint


@dataclass
class ControllerState:
    """Current state of the controller."""
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    joint_positions: List[float]
    is_moving: bool = False


class RobotController:
    """
    Controller for KUKA iiwa robot arm.
    """
    
    def __init__(self, environment):
        """
        Initialize controller with simulation environment.
        
        Args:
            environment: SimulationEnvironment instance
        """
        self.env = environment
        self.robot_id = environment.robot_id
        self.end_effector_link = environment.end_effector_link
        self.num_joints = environment.num_joints
        
        # Get safety bounds
        safety_config = config.get("safety", {})
        bounds = safety_config.get("workspace_bounds", {})
        self.workspace_bounds = {
            "x": bounds.get("x", [0.2, 0.8]),
            "y": bounds.get("y", [-0.4, 0.4]),
            "z": bounds.get("z", [0.05, 0.6]),
        }
        
        self.collision_distance = safety_config.get("collision_distance", 0.01)
        
        # Path config
        path_config = config.get("path", {})
        self.collision_check_step = path_config.get("collision_check_step", 5)
        
        # Joint limits for KUKA iiwa
        self.joint_lower_limits = [-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05]
        self.joint_upper_limits = [2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05]
        self.joint_ranges = [ul - ll for ll, ul in zip(self.joint_lower_limits, self.joint_upper_limits)]
        
    def get_state(self) -> ControllerState:
        """Get current robot state."""
        state = self.env.get_robot_state()
        return ControllerState(
            position=state["ee_position"],
            orientation=state["ee_orientation"],
            joint_positions=state["joint_positions"],
        )
    
    def check_workspace_bounds(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if position is within workspace bounds.
        
        Args:
            position: [x, y, z] target position
            
        Returns:
            True if within bounds, False otherwise
        """
        x, y, z = position
        return (
            self.workspace_bounds["x"][0] <= x <= self.workspace_bounds["x"][1] and
            self.workspace_bounds["y"][0] <= y <= self.workspace_bounds["y"][1] and
            self.workspace_bounds["z"][0] <= z <= self.workspace_bounds["z"][1]
        )
    
    def check_collision(self, obstacles: List[int] = None) -> bool:
        """
        Check for collisions with obstacles.
        
        Args:
            obstacles: List of obstacle body IDs (default: workpiece)
            
        Returns:
            True if collision detected, False otherwise
        """
        if obstacles is None:
            obstacles = [self.env.workpiece_id] if self.env.workpiece_id else []
        
        for obs_id in obstacles:
            contacts = p.getClosestPoints(
                self.robot_id, obs_id,
                distance=self.collision_distance
            )
            if contacts:
                return True
        
        return False
    
    def solve_ik(
        self,
        target_position: Tuple[float, float, float],
        target_orientation: Tuple[float, float, float, float] = None
    ) -> Optional[List[float]]:
        """
        Solve inverse kinematics for target pose.
        
        Args:
            target_position: [x, y, z] target position
            target_orientation: [qx, qy, qz, qw] target orientation (optional)
            
        Returns:
            Joint positions if solution found, None otherwise
        """
        # Check workspace bounds
        if not self.check_workspace_bounds(target_position):
            return None
        
        # Solve IK
        if target_orientation is not None:
            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                targetPosition=target_position,
                targetOrientation=target_orientation,
                lowerLimits=self.joint_lower_limits,
                upperLimits=self.joint_upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=[0, 0.5, 0, -1.0, 0, 0.5, 0],
                maxNumIterations=100,
            )
        else:
            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                targetPosition=target_position,
                lowerLimits=self.joint_lower_limits,
                upperLimits=self.joint_upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=[0, 0.5, 0, -1.0, 0, 0.5, 0],
                maxNumIterations=100,
            )
        
        return list(joint_positions[:self.num_joints])
    
    def move_to(
        self,
        target_position: Tuple[float, float, float],
        target_orientation: Tuple[float, float, float, float] = None,
        speed: float = 1.0
    ) -> bool:
        """
        Move robot to target pose.
        
        Args:
            target_position: [x, y, z] target position
            target_orientation: [qx, qy, qz, qw] target orientation
            speed: Speed multiplier (0-1)
            
        Returns:
            True if move successful, False otherwise
        """
        # Solve IK
        joint_positions = self.solve_ik(target_position, target_orientation)
        if joint_positions is None:
            return False
        
        # Apply motor control
        max_force = 500.0
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=max_force,
                maxVelocity=speed * 2.0,
            )
        
        return True
    
    def execute_path(
        self,
        waypoints: List[Waypoint],
        steps_per_waypoint: int = 20,
        check_collisions: bool = True
    ) -> Dict:
        """
        Execute a path of waypoints.
        
        Args:
            waypoints: List of Waypoint objects
            steps_per_waypoint: Simulation steps per waypoint
            check_collisions: Whether to check for collisions
            
        Returns:
            Dict with execution results
        """
        results = {
            "success": True,
            "waypoints_completed": 0,
            "collision_at": None,
            "out_of_bounds_at": None,
        }
        
        for i, waypoint in enumerate(waypoints):
            # Check bounds
            if not self.check_workspace_bounds(waypoint.position):
                results["success"] = False
                results["out_of_bounds_at"] = i
                break
            
            # Move to waypoint
            speed = max(0.1, waypoint.velocity / 0.1) if waypoint.velocity > 0 else 0.5
            success = self.move_to(waypoint.position, waypoint.orientation, speed)
            
            if not success:
                results["success"] = False
                break
            
            # Simulate and check collisions
            for _ in range(steps_per_waypoint):
                self.env.step()
                
                if check_collisions and i % self.collision_check_step == 0:
                    if self.check_collision():
                        results["success"] = False
                        results["collision_at"] = i
                        break
            
            if not results["success"]:
                break
            
            results["waypoints_completed"] = i + 1
        
        return results
    
    def go_home(self) -> bool:
        """
        Move robot to home/neutral position.
        
        Returns:
            True if successful
        """
        home_joints = [0, 0.5, 0, -1.0, 0, 0.5, 0]
        
        for i, pos in enumerate(home_joints):
            p.setJointMotorControl2(
                self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pos,
                force=500.0,
            )
        
        # Wait for robot to reach home
        for _ in range(100):
            self.env.step()
        
        return True
