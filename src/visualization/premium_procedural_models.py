"""
Premium Procedural Models - High-Fidelity Industrial Parts

Computational geometry for realistic 3D models with:
- Proper thickness (not paper-thin sheets)
- High vertex density (50x50+ for smooth vertex coloring)
- Built-in defects with vertex coloring
- Industrial Metal rendering compatible
"""

import numpy as np
import plotly.graph_objects as go
from typing import Tuple, List


# ============ LIGHTING CONFIG ============
METAL_LIGHTING = dict(
    ambient=0.4,
    diffuse=0.6,
    roughness=0.2,
    specular=0.8,
    fresnel=0.1
)

LIGHT_POSITION = dict(x=1, y=1, z=5)


# ============ COLOR UTILITIES ============
def interpolate_color(base: Tuple[int, int, int], target: Tuple[int, int, int], t: float) -> str:
    """Interpolate between two RGB colors."""
    r = int(base[0] * (1 - t) + target[0] * t)
    g = int(base[1] * (1 - t) + target[1] * t)
    b = int(base[2] * (1 - t) + target[2] * t)
    return f'rgb({r}, {g}, {b})'


STEEL_GREY = (169, 169, 169)
RUST_ORANGE = (255, 69, 0)
HEAT_STRESS_RED = (220, 20, 60)
HEAT_STRESS_PURPLE = (138, 43, 226)
CORROSION_BROWN = (139, 69, 19)


# ============ 1. FLANGED PIPE ============
def generate_premium_pipe(
    length: float = 0.4,
    outer_radius: float = 0.08,
    wall_thickness: float = 0.01,
    flange_radius: float = 0.12,
    flange_thickness: float = 0.02,
    n_circumference: int = 60,
    n_length: int = 80
) -> go.Mesh3d:
    """
    Generate a flanged pipe with corrosion ring defect.
    
    Geometry:
    - Main cylinder body with wall thickness
    - Two flanges (widened rings) at top and bottom
    - Corrosion ring defect around the center
    
    Args:
        length: Pipe length
        outer_radius: Main pipe outer radius
        wall_thickness: Pipe wall thickness
        flange_radius: Flange outer radius
        flange_thickness: Flange height
        n_circumference: Points around circumference
        n_length: Points along length
        
    Returns:
        Plotly Mesh3d trace
    """
    vertices = []
    faces = []
    colors = []
    
    theta = np.linspace(0, 2 * np.pi, n_circumference, endpoint=False)
    
    # Generate profile segments
    # Bottom flange outer
    z_levels = [0, flange_thickness]
    r_levels = [flange_radius, flange_radius]
    
    # Transition to pipe
    z_levels.extend([flange_thickness, flange_thickness + 0.01])
    r_levels.extend([flange_radius, outer_radius])
    
    # Main pipe body (many segments for vertex coloring)
    pipe_z = np.linspace(flange_thickness + 0.01, length - flange_thickness - 0.01, n_length)
    z_levels.extend(pipe_z.tolist())
    r_levels.extend([outer_radius] * len(pipe_z))
    
    # Transition to top flange
    z_levels.extend([length - flange_thickness - 0.01, length - flange_thickness])
    r_levels.extend([outer_radius, flange_radius])
    
    # Top flange
    z_levels.extend([length - flange_thickness, length])
    r_levels.extend([flange_radius, flange_radius])
    
    # Generate outer surface vertices
    vertex_index = 0
    ring_indices = []
    
    for z, r in zip(z_levels, r_levels):
        ring_start = vertex_index
        for t in theta:
            x = r * np.cos(t)
            y = r * np.sin(t)
            vertices.append([x, y, z])
            
            # Corrosion ring defect at center (z ~ length/2)
            dist_from_center = abs(z - length / 2)
            if dist_from_center < 0.03:
                t_defect = 1.0 - (dist_from_center / 0.03)
                colors.append(interpolate_color(STEEL_GREY, CORROSION_BROWN, t_defect ** 2))
            else:
                colors.append(f'rgb{STEEL_GREY}')
            
            vertex_index += 1
        ring_indices.append((ring_start, vertex_index))
    
    # Generate faces between rings
    for i in range(len(ring_indices) - 1):
        start1, end1 = ring_indices[i]
        start2, end2 = ring_indices[i + 1]
        n = end1 - start1
        
        for j in range(n):
            j_next = (j + 1) % n
            # Two triangles per quad
            faces.append([start1 + j, start1 + j_next, start2 + j])
            faces.append([start1 + j_next, start2 + j_next, start2 + j])
    
    # Convert to arrays
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=colors,
        flatshading=False,
        lighting=METAL_LIGHTING,
        lightposition=LIGHT_POSITION,
        hoverinfo='skip',
        name='Flanged Pipe'
    )


# ============ 2. NACA AIRFOIL TURBINE BLADE ============
def naca_airfoil_4digit(x: np.ndarray, thickness: float = 0.12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate NACA 4-digit airfoil Y coordinates.
    
    Uses the standard NACA formula:
    y_t = 5t * (0.2969√x - 0.126x - 0.3516x² + 0.2843x³ - 0.1015x⁴)
    
    Args:
        x: Chordwise positions (0 to 1)
        thickness: Maximum thickness as fraction of chord
        
    Returns:
        (y_upper, y_lower) coordinates
    """
    t = thickness
    y_t = 5 * t * (
        0.2969 * np.sqrt(x) - 
        0.1260 * x - 
        0.3516 * x**2 + 
        0.2843 * x**3 - 
        0.1015 * x**4
    )
    return y_t, -y_t


def generate_turbine_blade_v2(
    chord: float = 0.15,
    span: float = 0.3,
    twist_angle: float = 30.0,
    thickness_ratio: float = 0.12,
    n_chord: int = 50,
    n_span: int = 60
) -> go.Mesh3d:
    """
    Generate a twisted turbine blade with NACA airfoil cross-section.
    
    Geometry:
    - NACA airfoil profile (not flat!)
    - Twisted 30° from root to tip
    - Heat stress defect on leading edge
    
    Args:
        chord: Blade chord length
        span: Blade span (height)
        twist_angle: Total twist in degrees
        thickness_ratio: Airfoil thickness ratio
        n_chord: Points along chord
        n_span: Points along span
        
    Returns:
        Plotly Mesh3d trace
    """
    vertices = []
    faces = []
    colors = []
    
    # Chordwise coordinates (0 to 1)
    x_chord = np.linspace(0, 1, n_chord)
    
    # Get NACA airfoil profile
    y_upper, y_lower = naca_airfoil_4digit(x_chord, thickness_ratio)
    
    # Span positions
    z_span = np.linspace(0, span, n_span)
    
    vertex_index = 0
    section_indices = []
    
    for z_idx, z in enumerate(z_span):
        section_start = vertex_index
        
        # Calculate twist at this span location
        twist_frac = z / span
        twist_rad = np.radians(twist_angle * twist_frac)
        cos_t, sin_t = np.cos(twist_rad), np.sin(twist_rad)
        
        # Taper factor (blade gets narrower toward tip)
        taper = 1.0 - 0.3 * twist_frac
        
        # Generate upper surface vertices
        for i, x in enumerate(x_chord):
            # Airfoil coordinates (scaled)
            x_local = (x - 0.25) * chord * taper  # Center at quarter chord
            y_local = y_upper[i] * chord * taper
            
            # Apply twist rotation
            x_rot = x_local * cos_t - y_local * sin_t
            y_rot = x_local * sin_t + y_local * cos_t
            
            vertices.append([x_rot, y_rot, z])
            
            # Heat stress on leading edge (x near 0)
            if x < 0.2:
                stress_factor = (0.2 - x) / 0.2
                # Blend to purple/red for heat stress
                color = interpolate_color(STEEL_GREY, HEAT_STRESS_RED, stress_factor ** 1.5)
            else:
                color = f'rgb{STEEL_GREY}'
            colors.append(color)
            vertex_index += 1
        
        # Generate lower surface vertices (reverse order for proper winding)
        for i in range(n_chord - 2, 0, -1):  # Skip first and last (shared with upper)
            x = x_chord[i]
            x_local = (x - 0.25) * chord * taper
            y_local = y_lower[i] * chord * taper
            
            x_rot = x_local * cos_t - y_local * sin_t
            y_rot = x_local * sin_t + y_local * cos_t
            
            vertices.append([x_rot, y_rot, z])
            
            # Heat stress on leading edge
            if x < 0.2:
                stress_factor = (0.2 - x) / 0.2
                color = interpolate_color(STEEL_GREY, HEAT_STRESS_PURPLE, stress_factor ** 1.5)
            else:
                color = f'rgb{STEEL_GREY}'
            colors.append(color)
            vertex_index += 1
        
        section_indices.append((section_start, vertex_index))
    
    # Generate faces between sections
    for s in range(len(section_indices) - 1):
        start1, end1 = section_indices[s]
        start2, end2 = section_indices[s + 1]
        n = end1 - start1
        
        for j in range(n):
            j_next = (j + 1) % n
            faces.append([start1 + j, start1 + j_next, start2 + j])
            faces.append([start1 + j_next, start2 + j_next, start2 + j])
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=colors,
        flatshading=False,
        lighting=METAL_LIGHTING,
        lightposition=LIGHT_POSITION,
        hoverinfo='skip',
        name='Turbine Blade'
    )


# ============ 3. CAR HOOD WITH THICKNESS ============
def generate_car_hood_v2(
    width: float = 0.5,
    length: float = 0.6,
    crown_height: float = 0.08,
    thickness: float = 0.008,
    n_x: int = 60,
    n_y: int = 60
) -> go.Mesh3d:
    """
    Generate a car hood panel with proper thickness (solid volume).
    
    Geometry:
    - Top surface: Parabolic crown (automotive styling)
    - Bottom surface: Parallel offset for thickness
    - Edge stitching to create closed volume
    - Surface rust defect on top
    
    Args:
        width: Hood width (Y direction)
        length: Hood length (X direction)
        crown_height: Maximum crown height at center
        thickness: Sheet metal thickness
        n_x: Grid points in X
        n_y: Grid points in Y
        
    Returns:
        Plotly Mesh3d trace
    """
    vertices = []
    faces = []
    colors = []
    
    # Create grid
    x = np.linspace(-length/2, length/2, n_x)
    y = np.linspace(-width/2, width/2, n_y)
    X, Y = np.meshgrid(x, y)
    
    # Parabolic crown surface
    # Z = h * (1 - (x/L)^2) * (1 - (y/W)^2)
    Z_top = crown_height * (1 - (X / (length/2))**2) * (1 - (Y / (width/2))**2)
    Z_bottom = Z_top - thickness
    
    # Defect: Rust patch centered at (0.1, 0.05)
    defect_x, defect_y = 0.1, 0.05
    defect_radius = 0.08
    
    # Generate top surface vertices
    top_start = 0
    for i in range(n_y):
        for j in range(n_x):
            vertices.append([X[i, j], Y[i, j], Z_top[i, j]])
            
            # Calculate distance to defect
            dist = np.sqrt((X[i, j] - defect_x)**2 + (Y[i, j] - defect_y)**2)
            if dist < defect_radius:
                t = 1.0 - (dist / defect_radius)
                colors.append(interpolate_color(STEEL_GREY, RUST_ORANGE, t ** 2))
            else:
                colors.append(f'rgb{STEEL_GREY}')
    
    top_end = len(vertices)
    
    # Generate bottom surface vertices (no defects visible)
    bottom_start = len(vertices)
    for i in range(n_y):
        for j in range(n_x):
            vertices.append([X[i, j], Y[i, j], Z_bottom[i, j]])
            colors.append(f'rgb{STEEL_GREY}')  # Clean underside
    bottom_end = len(vertices)
    
    # Generate faces for top surface
    for i in range(n_y - 1):
        for j in range(n_x - 1):
            idx = i * n_x + j
            faces.append([top_start + idx, top_start + idx + 1, top_start + idx + n_x])
            faces.append([top_start + idx + 1, top_start + idx + n_x + 1, top_start + idx + n_x])
    
    # Generate faces for bottom surface (reverse winding for correct normals)
    for i in range(n_y - 1):
        for j in range(n_x - 1):
            idx = i * n_x + j
            faces.append([bottom_start + idx, bottom_start + idx + n_x, bottom_start + idx + 1])
            faces.append([bottom_start + idx + 1, bottom_start + idx + n_x, bottom_start + idx + n_x + 1])
    
    # Stitch edges to create closed volume
    # Front edge (x = -length/2)
    for i in range(n_y - 1):
        top_idx = top_start + i * n_x
        bottom_idx = bottom_start + i * n_x
        faces.append([top_idx, bottom_idx, top_idx + n_x])
        faces.append([bottom_idx, bottom_idx + n_x, top_idx + n_x])
    
    # Back edge (x = length/2)
    for i in range(n_y - 1):
        top_idx = top_start + i * n_x + (n_x - 1)
        bottom_idx = bottom_start + i * n_x + (n_x - 1)
        faces.append([top_idx, top_idx + n_x, bottom_idx])
        faces.append([bottom_idx, top_idx + n_x, bottom_idx + n_x])
    
    # Left edge (y = -width/2)
    for j in range(n_x - 1):
        top_idx = top_start + j
        bottom_idx = bottom_start + j
        faces.append([top_idx, top_idx + 1, bottom_idx])
        faces.append([bottom_idx, top_idx + 1, bottom_idx + 1])
    
    # Right edge (y = width/2)
    for j in range(n_x - 1):
        top_idx = top_start + (n_y - 1) * n_x + j
        bottom_idx = bottom_start + (n_y - 1) * n_x + j
        faces.append([top_idx, bottom_idx, top_idx + 1])
        faces.append([bottom_idx, bottom_idx + 1, top_idx + 1])
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=colors,
        flatshading=False,
        lighting=METAL_LIGHTING,
        lightposition=LIGHT_POSITION,
        hoverinfo='skip',
        name='Car Hood'
    )


# ============ REGISTRY & CONVENIENCE ============
def get_premium_procedural_models() -> dict:
    """Get all available premium procedural models."""
    return {
        "flanged_pipe": {
            "generator": generate_premium_pipe,
            "name": "Flanged Pipe",
            "desc": "Industrial pipe with flanges and corrosion ring"
        },
        "turbine_blade_v2": {
            "generator": generate_turbine_blade_v2,
            "name": "NACA Turbine Blade",
            "desc": "Twisted airfoil with heat stress"
        },
        "car_hood_v2": {
            "generator": generate_car_hood_v2,
            "name": "Car Hood Panel",
            "desc": "Automotive panel with thickness and rust"
        },
    }


def generate_all_premium_models() -> List[go.Mesh3d]:
    """Generate all premium procedural models."""
    return [
        generate_premium_pipe(),
        generate_turbine_blade_v2(),
        generate_car_hood_v2(),
    ]
