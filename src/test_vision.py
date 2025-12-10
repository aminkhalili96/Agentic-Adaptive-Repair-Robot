"""
Test script for the vision pipeline.

Runs the simulation, captures an image, detects defects, and localizes them.
"""

import time
import sys

from src.simulation.environment import create_environment
from src.simulation.defects import spawn_demo_defects
from src.vision.camera import Camera
from src.vision.detector import DefectDetector
from src.vision.localization import Localizer


try:
    import pybullet as p
except ImportError:
    class MockP:
        error = Exception
    p = MockP()

def test_vision():
    """Test the complete vision pipeline."""
    print("=" * 50)
    print("Vision Pipeline Test")
    print("=" * 50)
    
    env = None
    try:
        # 1. Setup simulation
        print("\n[1/5] Setting up simulation...")
        env = create_environment(gui=True)
        defects = spawn_demo_defects(env.workpiece_position, env.workpiece_size)
        env.defect_ids = [d.id for d in defects]
        print(f"  ✓ Created {len(defects)} defects")
        
        # Let simulation settle
        for _ in range(50):
            env.step()
        
        # 2. Initialize vision components
        print("\n[2/5] Initializing vision pipeline...")
        camera = Camera(env)
        detector = DefectDetector()
        localizer = Localizer(env)
        print("  ✓ Camera, Detector, Localizer ready")
        
        # 3. Capture image
        print("\n[3/5] Capturing image...")
        frame = camera.capture()
        rgb = frame['rgb']
        depth = frame['depth']
        print(f"  ✓ Captured RGB: {rgb.shape}, Depth: {depth.shape}")
        
        # 4. Detect defects
        print("\n[4/5] Detecting defects...")
        detected = detector.detect(rgb)
        print(f"  ✓ Detected {len(detected)} defects:")
        for d in detected:
            print(f"    - {d.type.value}: px=({d.centroid_px[0]}, {d.centroid_px[1]}), conf={d.confidence:.0%}")
        
        # 5. Localize to 3D
        print("\n[5/5] Localizing to 3D poses...")
        poses = []
        for d in detected:
            pose = localizer.localize_defect(d.centroid_px, d.confidence)
            poses.append(pose)
            print(f"    - {d.type.value}: pos=({pose.position[0]:.3f}, {pose.position[1]:.3f}, {pose.position[2]:.3f})")
            print(f"      normal=({pose.normal[0]:.2f}, {pose.normal[1]:.2f}, {pose.normal[2]:.2f})")
        
        print("\n" + "=" * 50)
        print("Vision pipeline test PASSED!")
        print("=" * 50)
        
        # Keep window open briefly
        print("\nSimulation window open for 5 seconds...")
        for _ in range(5 * 240):
            env.step()
            time.sleep(1.0 / 240.0)

    except p.error as e:
        print(f"\nPyBullet GUI closed or error occurred: {e}")
    except KeyboardInterrupt:
        print("\n\nCtrl+C detected. Shutting down...")
    finally:
        if env:
            print("Closing simulation environment.")
            env.close()
        print("Done!")


if __name__ == "__main__":
    test_vision()
