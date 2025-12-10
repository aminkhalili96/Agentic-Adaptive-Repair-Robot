"""
Test script for the complete Phase A pipeline.

Tests: Simulation → Vision → Path Planning → Robot Control
"""

import time
import sys

from src.simulation.environment import create_environment
from src.simulation.defects import spawn_demo_defects, mark_defect_repaired
from src.vision.camera import Camera
from src.vision.detector import DefectDetector
from src.vision.localization import Localizer
from src.planning.paths import PathGenerator
from src.planning.tsp import optimize_defect_order
from src.control.controller import RobotController


try:
    import pybullet as p
except ImportError:
    class MockP:
        error = Exception
    p = MockP()

def test_full_pipeline():
    """Test the complete scan-to-path pipeline."""
    print("=" * 60)
    print("Phase A Pipeline Test: Scan → Detect → Plan → Execute")
    print("=" * 60)
    
    env = None
    try:
        # 1. Setup simulation
        print("\n[1/6] Setting up simulation...")
        env = create_environment(gui=True)
        defects = spawn_demo_defects(env.workpiece_position, env.workpiece_size)
        env.defect_ids = [d.id for d in defects]
        print(f"  ✓ Created {len(defects)} defects")
        
        # Let simulation settle
        for _ in range(100):
            env.step()
        
        # 2. Initialize components
        print("\n[2/6] Initializing components...")
        camera = Camera(env)
        detector = DefectDetector()
        localizer = Localizer(env)
        path_gen = PathGenerator()
        controller = RobotController(env)
        print("  ✓ All components ready")
        
        # 3. Capture and detect
        print("\n[3/6] Detecting defects...")
        frame = camera.capture()
        detected = detector.detect(frame['rgb'])
        print(f"  ✓ Detected {len(detected)} defects")
        
        # 4. Localize to 3D
        print("\n[4/6] Localizing to 3D...")
        poses = []
        for d in detected:
            pose = localizer.localize_defect(d.centroid_px, d.confidence)
            poses.append((d, pose))
            print(f"  - {d.type.value}: ({pose.position[0]:.2f}, {pose.position[1]:.2f}, {pose.position[2]:.2f})")
        
        # 5. Generate and execute path for first defect
        if poses:
            print("\n[5/6] Generating path for first defect...")
            defect, pose = poses[0]
            
            # Generate spiral path
            waypoints = path_gen.generate_spiral(
                pose,
                radius=0.04,
                num_loops=2,
                hover_height=0.03
            )
            print(f"  ✓ Generated {len(waypoints)} waypoints")
            
            # Get approach point
            approach = path_gen.get_approach_point(pose, approach_distance=0.1)
            
            print("\n[6/6] Executing path...")
            print("  Moving to approach point...")
            controller.move_to(approach.position, approach.orientation)
            
            for _ in range(100):
                env.step()
                time.sleep(1.0 / 240.0)
            
            print("  Executing spiral path...")
            results = controller.execute_path(waypoints[:20], steps_per_waypoint=10)  # First 20 waypoints
            
            if results["success"]:
                print(f"  ✓ Completed {results['waypoints_completed']} waypoints")
                # Mark defect as repaired
                mark_defect_repaired(defects[0])
                print("  ✓ Defect marked as repaired (turned grey)")
            else:
                if results["collision_at"]:
                    print(f"  ⚠ Collision at waypoint {results['collision_at']}")
                elif results["out_of_bounds_at"]:
                    print(f"  ⚠ Out of bounds at waypoint {results['out_of_bounds_at']}")
        
        print("\n" + "=" * 60)
        print("Phase A Pipeline Test COMPLETE!")
        print("=" * 60)
        
        # Keep window open
        print("\nSimulation window open for 10 seconds...")
        for _ in range(10 * 240):
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
    test_full_pipeline()
