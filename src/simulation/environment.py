"""
PyBullet simulation environment with KUKA iiwa robot.

This module sets up the 3D simulation world including:
- Ground plane
- KUKA iiwa 7-DOF robot arm
- Workpiece (cube/cylinder)
- Overhead camera
"""

try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False
    class MockPyBullet:
        GUI = 1
        DIRECT = 2
        COV_ENABLE_GUI = 0
        COV_ENABLE_SHADOWS = 0
        GEOM_BOX = 1
        GEOM_CYLINDER = 2
        ER_BULLET_HARDWARE_OPENGL = 1
        ER_TINY_RENDERER = 2
        
        def connect(self, mode): return 0
        def disconnect(self): pass
        def configureDebugVisualizer(self, flag, enable): pass
        def setAdditionalSearchPath(self, path): pass
        def setGravity(self, x, y, z): pass
        def setTimeStep(self, timestep): pass
        def loadURDF(self, fileName, basePosition=[0,0,0], baseOrientation=[0,0,0,1], useFixedBase=False): return 0
        def getNumJoints(self, bodyUniqueId): return 7
        def resetJointState(self, bodyUniqueId, jointIndex, targetValue): pass
        def createVisualShape(self, shapeType, halfExtents=None, radius=None, length=None, rgbaColor=None): return 0
        def createCollisionShape(self, shapeType, halfExtents=None, radius=None, height=None): return 0
        def createMultiBody(self, baseMass=0, baseCollisionShapeIndex=0, baseVisualShapeIndex=0, basePosition=[0,0,0]): return 0
        def computeViewMatrix(self, cameraEyePosition, cameraTargetPosition, cameraUpVector): return []
        def computeProjectionMatrixFOV(self, fov, aspect, nearVal, farVal): return []
        def getCameraImage(self, width, height, viewMatrix, projectionMatrix, renderer):
            # Return dummy data: width, height, rgb, depth, seg
            rgb = np.zeros((height, width, 4), dtype=np.uint8)
            depth = np.zeros((height, width), dtype=np.float32)
            seg = np.zeros((height, width), dtype=np.int32)
            return width, height, rgb.flatten(), depth.flatten(), seg.flatten()
        def getJointState(self, bodyUniqueId, jointIndex): return [0, 0, 0, 0]
        def getLinkState(self, bodyUniqueId, linkIndex): return [[0.5, 0, 0.5], [0, 0, 0, 1]]
        def stepSimulation(self): pass
        def removeBody(self, bodyUniqueId): pass
        def getQuaternionFromEuler(self, eulerAngles): return [0, 0, 0, 1]

    p = MockPyBullet()
    
    class MockData:
        def getDataPath(self): return ""
    pybullet_data = MockData()

import numpy as np
from typing import Tuple, Optional, Dict, Any

from src.config import config


class SimulationEnvironment:
    """
    PyBullet simulation environment for the repair robot.
    
    Attributes:
        physics_client: PyBullet physics client ID
        robot_id: ID of the loaded robot
        workpiece_id: ID of the workpiece object
        defect_ids: List of defect marker IDs
    """
    
    def __init__(self, gui: bool = True):
        """
        Initialize the simulation environment.
        
        Args:
            gui: If True, open GUI window. If False, run headless.
        """
        self.gui = gui
        self.physics_client = None
        self.robot_id = None
        self.workpiece_id = None
        self.defect_ids = []
        self.ground_id = None
        
        # Camera parameters from config
        cam_config = config.get("camera", {})
        self.camera_width = cam_config.get("width", 640)
        self.camera_height = cam_config.get("height", 480)
        self.camera_fov = cam_config.get("fov", 60)
        self.camera_near = cam_config.get("near", 0.1)
        self.camera_far = cam_config.get("far", 10.0)
        self.camera_position = cam_config.get("position", [0.5, 0.0, 1.2])
        self.camera_target = cam_config.get("target", [0.5, 0.0, 0.2])
        
    def setup(self) -> None:
        """Set up the complete simulation environment."""
        self._connect()
        self._load_ground()
        self._load_robot()
        self._load_workpiece()
        self._setup_camera()
        
    def _connect(self) -> None:
        """Connect to PyBullet physics server."""
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            # Configure GUI
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Set simulation time step
        time_step = config.get("simulation", {}).get("time_step", 1.0/240.0)
        p.setTimeStep(time_step)
        
    def _load_ground(self) -> None:
        """Load the ground plane."""
        self.ground_id = p.loadURDF("plane.urdf")
        
    def _load_robot(self) -> None:
        """Load the KUKA iiwa robot arm."""
        robot_start_pos = [0, 0, 0]
        robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            robot_start_pos,
            robot_start_orn,
            useFixedBase=True
        )
        
        # Get robot info
        self.num_joints = p.getNumJoints(self.robot_id)
        self.end_effector_link = config.get("robot", {}).get("end_effector_link", 6)
        
        # Set initial joint positions (neutral pose)
        initial_positions = [0, 0.5, 0, -1.0, 0, 0.5, 0]
        for i, pos in enumerate(initial_positions[:self.num_joints]):
            p.resetJointState(self.robot_id, i, pos)
            
    def _load_workpiece(
        self,
        shape: str = "cube",
        position: Tuple[float, float, float] = (0.6, 0.0, 0.1),
        size: float = 0.3
    ) -> None:
        """
        Load a workpiece into the simulation.
        
        Args:
            shape: "cube" or "cylinder"
            position: [x, y, z] position
            size: Scale factor
        """
        if shape == "cube":
            # Create a visual and collision shape
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[size/2, size/2, size/2],
                rgbaColor=[0.7, 0.7, 0.7, 1.0]  # Grey metal color
            )
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[size/2, size/2, size/2]
            )
        elif shape == "cylinder":
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=size/2,
                length=size,
                rgbaColor=[0.7, 0.7, 0.7, 1.0]
            )
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=size/2,
                height=size
            )
        else:
            # Default to cube
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[size/2, size/2, size/2],
                rgbaColor=[0.7, 0.7, 0.7, 1.0]
            )
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[size/2, size/2, size/2]
            )
            
        self.workpiece_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.workpiece_position = position
        self.workpiece_size = size
        
    def _setup_camera(self) -> None:
        """Configure the camera view matrices."""
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=[0, 1, 0]
        )
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_width / self.camera_height,
            nearVal=self.camera_near,
            farVal=self.camera_far
        )
        
    def capture_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture RGB, depth, and segmentation images from the camera.
        
        Returns:
            Tuple of (rgb_image, depth_image, segmentation_image)
        """
        _, _, rgb, depth, seg = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        )
        
        # Convert to numpy arrays
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(
            self.camera_height, self.camera_width, 4
        )[:, :, :3]  # Remove alpha channel
        
        depth_array = np.array(depth, dtype=np.float32).reshape(
            self.camera_height, self.camera_width
        )
        
        seg_array = np.array(seg, dtype=np.int32).reshape(
            self.camera_height, self.camera_width
        )
        
        return rgb_array, depth_array, seg_array
    
    def get_camera_intrinsics(self) -> np.ndarray:
        """
        Get the camera intrinsic matrix K.
        
        Returns:
            3x3 intrinsic matrix
        """
        fx = self.camera_width / (2 * np.tan(np.radians(self.camera_fov / 2)))
        fy = fx  # Assuming square pixels
        cx = self.camera_width / 2
        cy = self.camera_height / 2
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return K
    
    def step(self) -> None:
        """Step the simulation forward."""
        p.stepSimulation()
        
    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot joint positions and velocities."""
        joint_positions = []
        joint_velocities = []
        
        for i in range(self.num_joints):
            state = p.getJointState(self.robot_id, i)
            joint_positions.append(state[0])
            joint_velocities.append(state[1])
            
        # Get end effector pose
        ee_state = p.getLinkState(self.robot_id, self.end_effector_link)
        ee_position = ee_state[0]
        ee_orientation = ee_state[1]
        
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_position": ee_position,
            "ee_orientation": ee_orientation,
        }
    
    def reset(self) -> None:
        """Reset the simulation to initial state."""
        # Reset robot to initial pose
        initial_positions = [0, 0.5, 0, -1.0, 0, 0.5, 0]
        for i, pos in enumerate(initial_positions[:self.num_joints]):
            p.resetJointState(self.robot_id, i, pos)
            
        # Remove all defects
        for defect_id in self.defect_ids:
            p.removeBody(defect_id)
        self.defect_ids = []
        
    def close(self) -> None:
        """Disconnect from the physics server."""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None


def create_environment(gui: bool = True) -> SimulationEnvironment:
    """
    Factory function to create and set up a simulation environment.
    
    Args:
        gui: Whether to show the GUI window
        
    Returns:
        Configured SimulationEnvironment instance
    """
    env = SimulationEnvironment(gui=gui)
    env.setup()
    return env
