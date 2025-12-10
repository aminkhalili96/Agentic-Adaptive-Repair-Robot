# Mathematical Foundations

This document explains the key mathematical concepts used in the AARR system.

---

## 1. Coordinate Frames

The system uses four coordinate frames:

| Frame | Origin | Axes | Convention |
|-------|--------|------|------------|
| **World (W)** | Robot base | X-fwd, Y-left, Z-up | Right-handed |
| **Camera (C)** | Lens center | X-right, Y-down, Z-fwd | OpenCV standard |
| **Tool (T)** | Tool Center Point | Z points outward | DH convention |
| **Object (O)** | Workpiece center | Aligned with W | -- |

```
        Z (up)
        │
        │
        │_____ Y (left)
       /
      /
     X (forward)

     World Frame (Right-handed)
```

---

## 2. Camera Intrinsic Matrix

The intrinsic matrix `K` maps 3D camera coordinates to 2D pixel coordinates:

```
        ┌         ┐
        │ fx  0  cx│
    K = │ 0  fy  cy│
        │ 0   0   1│
        └         ┘
```

**Parameters:**
- `fx, fy`: Focal lengths in pixels
- `cx, cy`: Principal point (image center)

**Calculation from FOV:**
```python
fx = width / (2 * tan(fov / 2))
fy = fx  # Assuming square pixels
cx = width / 2
cy = height / 2
```

---

## 3. Pixel-to-Ray Conversion

Given a pixel `(u, v)`, we convert to a 3D ray in camera frame:

### Step 1: Normalize pixel coordinates
```python
p_norm = K⁻¹ @ [u, v, 1]ᵀ
p_norm = p_norm / ||p_norm||
```

### Step 2: Transform to world frame
```python
# Camera coordinate system
forward = camera_target - camera_pos
right = cross(forward, [0, 1, 0])
up = cross(right, forward)

# Rotation matrix
R = [right | -up | forward]

# Ray direction in world
ray_dir = R @ p_norm
```

---

## 4. Depth to 3D Position

Given depth `d` at pixel `(u, v)`:

```python
position = ray_origin + ray_direction * d
```

Where:
- `ray_origin` = camera position in world frame
- `ray_direction` = normalized ray from pixel-to-ray conversion

---

## 5. Surface Normal Estimation

We use PyBullet ray casting to get the surface normal at a defect location:

```python
result = pybullet.rayTest(ray_from, ray_to)

if result[0][0] != -1:  # Hit something
    hit_position = result[0][3]
    surface_normal = result[0][4]  # Unit vector
```

**Normal Convention:**
- Points **outward** from the surface
- Unit length (||n|| = 1)

---

## 6. Tool Orientation from Surface Normal

To orient the tool perpendicular to the surface, we compute a quaternion rotation.

### Goal
Align tool Z-axis with the **negative** surface normal (pointing INTO the surface).

### Algorithm

```python
target = -surface_normal  # Point into surface
source = [0, 0, 1]        # Default tool Z-axis

# Rotation axis (perpendicular to both)
axis = cross(source, target)
axis = axis / ||axis||

# Rotation angle
angle = arccos(dot(source, target))

# Convert to quaternion [x, y, z, w]
quaternion = axis * sin(angle/2) + [0, 0, 0, cos(angle/2)]
```

### Edge Cases
1. **Aligned** (`dot > 0.9999`): Return identity quaternion `[0, 0, 0, 1]`
2. **Opposite** (`dot < -0.9999`): 180° rotation around X-axis `[1, 0, 0, 0]`

---

## 7. Normal Smoothing

Per Codex feedback, we smooth normals along a path to reduce jitter:

```python
def smooth_normals(normals, window=3):
    smoothed = []
    for i in range(len(normals)):
        start = max(0, i - window // 2)
        end = min(len(normals), i + window // 2 + 1)
        avg = mean(normals[start:end], axis=0)
        smoothed.append(avg / ||avg||)
    return smoothed
```

---

## 8. Spiral Path Generation

An Archimedean spiral on a surface with normal `n`:

### Local Coordinate System
```python
z_axis = surface_normal
x_axis = cross(z_axis, [0, 0, 1])  # or [1, 0, 0] if z_axis ≈ [0, 0, 1]
y_axis = cross(z_axis, x_axis)
```

### Spiral Parametric Equation
```python
for θ in linspace(0, 2π * num_loops, num_points):
    r = spacing * θ / (2π)  # Radius grows with angle
    r = min(r, max_radius)
    
    local = [r * cos(θ), r * sin(θ), hover_height]
    world = center + local[0]*x + local[1]*y + local[2]*z
```

---

## 9. Quaternion Convention

We use the **[x, y, z, w]** format (SciPy/PyBullet convention):

```
q = [x, y, z, w]

where:
- (x, y, z) = rotation axis * sin(θ/2)
- w = cos(θ/2)
- Identity: [0, 0, 0, 1]
```

---

## 10. Inverse Kinematics

PyBullet's `calculateInverseKinematics` solves for joint angles:

```python
joint_positions = p.calculateInverseKinematics(
    robot_id,
    end_effector_link,
    targetPosition=[x, y, z],
    targetOrientation=[qx, qy, qz, qw],  # Tool orientation
    lowerLimits=joint_lower,
    upperLimits=joint_upper,
)
```

**Key:** Orientation is critical for surface-perpendicular tool alignment.

---

## References

1. Hartley & Zisserman, "Multiple View Geometry" (camera models)
2. Craig, "Introduction to Robotics" (DH convention)
3. PyBullet documentation (ray casting, IK)
4. SciPy documentation (Rotation class)

---

## 11. TSP Path Optimization

The system uses TSP (Traveling Salesman Problem) algorithms to optimize the order of defect repairs, minimizing robot travel time.

### Algorithm: Nearest-Neighbor Heuristic

A greedy algorithm that always visits the nearest unvisited defect:

```python
def nearest_neighbor(defects, start_pos):
    current = start_pos
    ordered = []
    remaining = list(defects)
    
    while remaining:
        nearest = min(remaining, key=lambda d: distance(current, d.position))
        ordered.append(nearest)
        remaining.remove(nearest)
        current = nearest.position
    
    return ordered
```

**Complexity:** O(n²) where n = number of defects

### 2-Opt Improvement

Local search that improves the initial solution by reversing path segments:

```python
for i in range(len(path) - 2):
    for j in range(i + 2, len(path)):
        new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
        if total_distance(new_path) < total_distance(path):
            path = new_path
```

**Complexity:** O(n²) per iteration, typically converges in O(n) iterations

### Efficiency Gain Calculation

```
Efficiency Gain (%) = ((D_original - D_optimized) / D_original) × 100
```

Where:
- `D_original` = total path distance in original order
- `D_optimized` = total path distance after optimization

### Example

```
Defects: A(10,0,0), B(1,0,0), C(5,0,0)
Robot starts at: (0,0,0)

Original Order: A → B → C
Distance: 10 + 9 + 4 = 23m

Optimized Order: B → C → A  
Distance: 1 + 4 + 5 = 10m

Efficiency Gain: (23 - 10) / 23 × 100 = 56%
```

---

## 12. Interactive Segmentation with SAM

The system uses Segment Anything Model (SAM) for zero-shot interactive segmentation, enabling instant defect detection without custom model training.

### Overview

SAM is a foundation model trained on 11 million images and 1.1 billion masks. It can segment any object when given a point prompt, making it ideal for interactive defect detection.

```
Architecture:
┌─────────────────────────────────────────────────────────────┐
│                         SAM Pipeline                         │
├──────────────────┬───────────────────┬──────────────────────┤
│   Image Encoder  │  Prompt Encoder   │    Mask Decoder      │
│   (ViT-based)    │  (Point/Box/Text) │   (Lightweight)      │
│        ↓         │        ↓          │         ↓            │
│ Image Embeddings │ Point Embeddings  │   Binary Mask        │
└──────────────────┴───────────────────┴──────────────────────┘
```

### Point-Prompt Pipeline

1. **User clicks** on image → capture (x, y) pixel coordinates
2. **Image encoder** extracts dense image embeddings (one-time per image)
3. **Prompt encoder** processes the point as a foreground prompt
4. **Mask decoder** generates binary mask from combined embeddings

```python
# Pseudocode
predictor.set_image(rgb_image)           # Step 2
input_point = np.array([[x, y]])         # Step 1
input_label = np.array([1])              # 1 = foreground
masks, scores, _ = predictor.predict(    # Steps 3-4
    point_coords=input_point,
    point_labels=input_label
)
```

### MobileSAM vs Full SAM

| Model | Size | Inference | Use Case |
|-------|------|-----------|----------|
| SAM (ViT-H) | 2.4 GB | ~1s | Server deployment |
| MobileSAM | 38 MB | ~50ms | Edge/laptop |

### Coverage Calculation

```
Coverage (%) = (mask_pixels / total_image_pixels) × 100
```

Where:
- `mask_pixels` = count of pixels where mask = 1
- `total_image_pixels` = image width × height

### Mask Overlay Visualization

```python
# Alpha blending for translucent overlay
overlay[mask] = (1 - α) × original[mask] + α × color[mask]
```

Where:
- `α` = transparency (default 0.5)
- `color` = (255, 0, 0) for red defect highlight

### Fallback: OpenCV Flood Fill

When SAM is unavailable, the system uses color-based flood fill:

```python
cv2.floodFill(gray, mask, (x, y), 255,
              loDiff=20, upDiff=20,
              flags=cv2.FLOODFILL_MASK_ONLY)
```

This provides basic region segmentation based on color similarity around the clicked point.

### References

1. Kirillov et al., "Segment Anything" (Meta AI, 2023)
2. Zhang et al., "Faster Segment Anything" (MobileSAM, 2023)
