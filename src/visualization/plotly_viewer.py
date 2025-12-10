"""
Industrial 3D Visualization Pipeline

Professional PBR-style rendering with:
- "Industrial Metal" material (proper lighting for steel/aluminum)
- Vertex-colored defects ("painted on" the mesh, not floating)
- Transparent background for seamless UI integration
- Proper camera positioning
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from src.visualization.mesh_loader import MeshData


# ============ INDUSTRIAL METAL MATERIAL ============
METAL_LIGHTING = dict(
    ambient=0.4,       # Bright enough to see in shadows
    diffuse=0.6,       # Shows surface curves well
    roughness=0.2,     # Slightly polished metal
    specular=0.8,      # High shine for metallic look
    fresnel=0.1        # Subtle edge rim lighting
)

# Factory-style top-down lighting (nothing in total darkness)
LIGHT_POSITION = dict(x=1, y=1, z=5)

# Color palette
COLORS = {
    'steel_grey': 'rgb(169, 169, 169)',       # Standard steel color
    'defect_rust': 'rgb(255, 69, 0)',         # Safety Orange/Rust
    'defect_high': 'rgb(244, 67, 54)',        # Red for severe
    'defect_medium': 'rgb(255, 152, 0)',      # Orange for medium
    'defect_low': 'rgb(76, 175, 80)',         # Green for low
    'toolpath': 'rgb(33, 150, 243)',          # Blue toolpath
    'highlight': 'rgb(255, 215, 0)',          # Gold highlight
}


def generate_vertex_colors(
    vertices: np.ndarray,
    defects: List[Dict],
    base_color: str = 'rgb(169, 169, 169)',
    defect_color: str = 'rgb(255, 69, 0)',
    defect_radius: float = 0.03
) -> List[str]:
    """
    Generate vertex colors with defects "painted" onto the mesh.
    
    This creates a smooth defect heatmap by interpolating colors
    based on distance from defect centers.
    
    Args:
        vertices: Nx3 array of vertex positions
        defects: List of defect dicts with 'position' key
        base_color: Default steel grey color
        defect_color: Color for defect regions
        defect_radius: Radius of defect coloring
        
    Returns:
        List of color strings for each vertex
    """
    n_vertices = len(vertices)
    
    # Initialize all vertices to base steel color
    colors = [base_color] * n_vertices
    
    if not defects:
        return colors
    
    # Parse base RGB values
    base_rgb = _parse_rgb(base_color)
    defect_rgb = _parse_rgb(defect_color)
    
    # Calculate defect influence on each vertex
    influence = np.zeros(n_vertices)
    
    for defect in defects:
        if 'position' not in defect:
            continue
            
        defect_pos = np.array(defect['position'])
        
        # Get severity-based color if available
        severity = defect.get('severity', 'medium')
        if severity == 'high':
            defect_rgb = _parse_rgb(COLORS['defect_high'])
        elif severity == 'low':
            defect_rgb = _parse_rgb(COLORS['defect_low'])
        else:
            defect_rgb = _parse_rgb(COLORS['defect_rust'])
        
        # Calculate distance from each vertex to this defect
        distances = np.linalg.norm(vertices - defect_pos, axis=1)
        
        # Smooth falloff within radius (1 at center, 0 at edge)
        local_influence = np.clip(1.0 - (distances / defect_radius), 0, 1)
        
        # Quadratic falloff for smoother blend
        local_influence = local_influence ** 2
        
        # Update vertex colors based on influence
        for i in range(n_vertices):
            if local_influence[i] > 0.01:  # Skip negligible influence
                t = local_influence[i]
                # Interpolate between base and defect color
                r = int(base_rgb[0] * (1 - t) + defect_rgb[0] * t)
                g = int(base_rgb[1] * (1 - t) + defect_rgb[1] * t)
                b = int(base_rgb[2] * (1 - t) + defect_rgb[2] * t)
                colors[i] = f'rgb({r}, {g}, {b})'
    
    return colors


def _parse_rgb(color_str: str) -> Tuple[int, int, int]:
    """Parse 'rgb(r, g, b)' string to tuple."""
    try:
        parts = color_str.replace('rgb(', '').replace(')', '').split(',')
        return (int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip()))
    except:
        return (169, 169, 169)  # Default grey


def create_industrial_mesh_trace(
    mesh_data: MeshData,
    defects: Optional[List[Dict]] = None,
    defect_radius: float = 0.03,
    show_edges: bool = False
) -> go.Mesh3d:
    """
    Create a Mesh3d trace with Industrial Metal rendering.
    
    Features:
    - Proper PBR-style lighting
    - Vertex-colored defects painted on surface
    - No floating markers or cones
    
    Args:
        mesh_data: MeshData object with vertices/faces
        defects: Optional list of defect dicts with 'position'
        defect_radius: Radius for defect color bleeding
        show_edges: Whether to show wireframe edges
        
    Returns:
        Plotly Mesh3d trace
    """
    vertices = mesh_data.vertices
    faces = mesh_data.faces
    
    # Generate vertex colors with defects painted on
    vertex_colors = generate_vertex_colors(
        vertices, 
        defects or [],
        defect_radius=defect_radius
    )
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=vertex_colors,
        opacity=1.0,
        flatshading=False,  # Smooth shading for metal look
        lighting=METAL_LIGHTING,
        lightposition=LIGHT_POSITION,
        hoverinfo='skip',
        name='Part',
        showlegend=False
    )


def create_industrial_layout(
    mesh_data: MeshData,
    transparent_bg: bool = True
) -> Dict[str, Any]:
    """
    Create Plotly layout for industrial visualization.
    
    Features:
    - Transparent background (blends with dark UI)
    - aspectmode='data' (no stretching)
    - Proper camera positioning
    
    Args:
        mesh_data: MeshData for bounds calculation
        transparent_bg: Use transparent background
        
    Returns:
        Layout dict for go.Figure
    """
    # Calculate camera position to fill the screen
    bounds = mesh_data.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = np.max(bounds[1] - bounds[0])
    
    # Camera distance to fill view
    distance = size * 2.0
    
    bg_color = 'rgba(0,0,0,0)' if transparent_bg else '#0D0D0D'
    
    return dict(
        scene=dict(
            xaxis=dict(
                visible=False,
                showbackground=False,
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
            ),
            aspectmode='data',
            bgcolor=bg_color,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision='industrial',  # Preserve camera on rerender
    )


class Mesh3DViewer:
    """
    Industrial 3D Mesh Viewer.
    
    Renders meshes with PBR-style "Industrial Metal" material
    and vertex-colored defects (painted on, not floating).
    """
    
    def __init__(self, mesh_data: MeshData):
        """Initialize with mesh data."""
        self.mesh_data = mesh_data
        self.fig = None
        self._defects: List[Dict] = []
        self._defect_radius = 0.03
        self._toolpath: List[Tuple[float, float, float]] = []
        self._highlight_position: Optional[Tuple[float, float, float]] = None
        self._highlight_radius: float = 0.02
        
    def add_defect_markers(
        self, 
        positions: List[Tuple[float, float, float]],
        labels: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
        normals: Optional[List[Tuple[float, float, float]]] = None,
        confidences: Optional[List[float]] = None
    ):
        """
        Add defects to be painted onto the mesh.
        
        Note: These are NOT floating markers - they become vertex colors.
        
        Args:
            positions: Defect center positions
            labels: Optional labels (for data)
            severities: 'high', 'medium', or 'low'
            normals: Surface normals (for toolpath)
            confidences: Detection confidence values
        """
        for i, pos in enumerate(positions):
            self._defects.append({
                'position': pos,
                'label': labels[i] if labels and i < len(labels) else f'Defect {i+1}',
                'severity': severities[i] if severities and i < len(severities) else 'medium',
                'normal': normals[i] if normals and i < len(normals) else None,
                'confidence': confidences[i] if confidences and i < len(confidences) else 0.9,
            })
    
    def add_toolpath(self, waypoints: List[Tuple[float, float, float]]):
        """Add repair toolpath visualization."""
        self._toolpath = waypoints
    
    def highlight_region(self, center: Tuple[float, float, float], radius: float = 0.02):
        """Highlight a region (adds extra defect coloring)."""
        self._highlight_position = center
        self._highlight_radius = radius
    
    def create_figure(self) -> go.Figure:
        """
        Create the industrial-styled Plotly figure.
        
        Returns:
            Plotly Figure with mesh and defects
        """
        # Create main mesh trace with vertex-colored defects
        mesh_trace = create_industrial_mesh_trace(
            self.mesh_data,
            defects=self._defects,
            defect_radius=self._defect_radius
        )
        
        traces = [mesh_trace]
        
        # Add toolpath if present (this IS a line trace)
        if self._toolpath:
            toolpath_trace = go.Scatter3d(
                x=[w[0] for w in self._toolpath],
                y=[w[1] for w in self._toolpath],
                z=[w[2] for w in self._toolpath],
                mode='lines',
                line=dict(color=COLORS['toolpath'], width=4),
                name='Toolpath',
                showlegend=False,
                hoverinfo='skip'
            )
            traces.append(toolpath_trace)
        
        # Create figure
        self.fig = go.Figure(data=traces)
        
        # Apply industrial layout
        layout = create_industrial_layout(self.mesh_data, transparent_bg=True)
        self.fig.update_layout(**layout)
        
        return self.fig
    
    def set_camera_view(self, target: Tuple[float, float, float]) -> Dict:
        """
        Calculate camera to focus on a target position.
        
        Args:
            target: Position to focus on
            
        Returns:
            Camera dict for Plotly layout
        """
        target = np.array(target)
        bounds = self.mesh_data.bounds
        size = np.max(bounds[1] - bounds[0])
        distance = size * 1.2
        
        # Camera positioned looking at target from 45 degrees
        eye = target + np.array([distance * 0.6, distance * 0.6, distance * 0.4])
        
        return dict(
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
            center=dict(x=target[0], y=target[1], z=target[2]),
            up=dict(x=0, y=0, z=1)
        )


# ============ CONVENIENCE FUNCTIONS ============

def render_industrial_mesh(
    mesh_data: MeshData,
    defects: Optional[List[Dict]] = None,
    height: int = 600
) -> go.Figure:
    """
    One-liner to render a mesh with industrial styling.
    
    Args:
        mesh_data: MeshData object
        defects: Optional defect list
        height: Figure height in pixels
        
    Returns:
        Ready-to-display Plotly Figure
    """
    viewer = Mesh3DViewer(mesh_data)
    
    if defects:
        positions = [d['position'] for d in defects]
        severities = [d.get('severity', 'medium') for d in defects]
        viewer.add_defect_markers(positions, severities=severities)
    
    fig = viewer.create_figure()
    fig.update_layout(height=height)
    
    return fig
