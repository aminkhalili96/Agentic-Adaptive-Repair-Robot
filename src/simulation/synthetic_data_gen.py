"""
Synthetic Data Generation Pipeline for ML Training.

This module generates synthetic training datasets for defect detection models
by procedurally creating diverse training images with ground truth masks.

Solves the "Data Scarcity" problem in industrial AI by simulating:
- Random camera angles (azimuth/elevation)
- Random lighting conditions
- Random defect positions and sizes

Output for each sample:
- image_{i}.png: Rendered RGB image
- mask_{i}.png: Ground truth binary mask (white=defect)
- metadata_{i}.json: Camera pose, lighting params, defect info
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import plotly.graph_objects as go

# Import existing mesh utilities
from src.visualization.mesh_loader import MeshData, sample_surface_points


@dataclass
class LightingParams:
    """Lighting configuration for a scene."""
    azimuth: float  # 0-360 degrees
    elevation: float  # 10-80 degrees
    intensity: float  # 0.5-1.5 multiplier
    ambient: float  # 0.2-0.6 ambient light level
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CameraParams:
    """Camera configuration for a scene."""
    azimuth: float  # 0-360 degrees
    elevation: float  # 10-60 degrees
    distance: float  # Distance from mesh center
    
    def to_eye_position(self, center: np.ndarray = np.array([0, 0, 0])) -> Dict:
        """Convert spherical coords to Plotly eye position."""
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)
        
        x = center[0] + self.distance * np.cos(el_rad) * np.cos(az_rad)
        y = center[1] + self.distance * np.cos(el_rad) * np.sin(az_rad)
        z = center[2] + self.distance * np.sin(el_rad)
        
        return dict(x=float(x), y=float(y), z=float(z))
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SyntheticDefect:
    """A synthetic defect for training data."""
    position: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    size: float  # Radius in mesh units
    defect_type: str  # rust, crack, dent
    
    def to_dict(self) -> Dict:
        return {
            "position": list(self.position),
            "normal": list(self.normal),
            "size": self.size,
            "defect_type": self.defect_type
        }


@dataclass
class SampleMetadata:
    """Metadata for a single training sample."""
    sample_id: int
    camera: CameraParams
    lighting: LightingParams
    defects: List[SyntheticDefect]
    mesh_name: str
    image_width: int
    image_height: int
    
    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "camera": self.camera.to_dict(),
            "lighting": self.lighting.to_dict(),
            "defects": [d.to_dict() for d in self.defects],
            "mesh_name": self.mesh_name,
            "image_size": [self.image_width, self.image_height]
        }


class SyntheticDataGenerator:
    """
    Generates synthetic training data for defect detection models.
    
    Uses Plotly + Kaleido for rendering (consistent with AARR visualization).
    Produces image/mask/metadata triplets for ML training.
    """
    
    def __init__(
        self,
        mesh_data: MeshData,
        seed: int = 42,
        image_width: int = 800,
        image_height: int = 600
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            mesh_data: MeshData object containing the mesh to render
            seed: Random seed for reproducibility
            image_width: Output image width in pixels
            image_height: Output image height in pixels
        """
        self.mesh_data = mesh_data
        self.rng = np.random.default_rng(seed)
        self.image_width = image_width
        self.image_height = image_height
        
        # Calculate mesh scale for appropriate defect sizes
        self.mesh_scale = float(np.max(mesh_data.bounds[1] - mesh_data.bounds[0]))
        
        # Defect types with visual properties
        self.defect_types = ["rust", "crack", "dent"]
        self.defect_colors = {
            "rust": "rgb(255, 69, 0)",      # Red-orange
            "crack": "rgb(30, 30, 30)",      # Dark
            "dent": "rgb(100, 149, 237)",    # Blue-ish
        }
    
    def randomize_lighting(self) -> LightingParams:
        """Generate random lighting parameters."""
        return LightingParams(
            azimuth=float(self.rng.uniform(0, 360)),
            elevation=float(self.rng.uniform(10, 80)),
            intensity=float(self.rng.uniform(0.7, 1.3)),
            ambient=float(self.rng.uniform(0.2, 0.5))
        )
    
    def randomize_camera(self) -> CameraParams:
        """Generate random camera parameters."""
        # Distance based on mesh scale
        base_distance = self.mesh_scale * 2.5
        
        return CameraParams(
            azimuth=float(self.rng.uniform(0, 360)),
            elevation=float(self.rng.uniform(15, 60)),
            distance=float(base_distance * self.rng.uniform(0.8, 1.2))
        )
    
    def randomize_defects(self, num_defects: int = 3) -> List[SyntheticDefect]:
        """
        Generate random defects on the mesh surface.
        
        Args:
            num_defects: Number of defects to generate
            
        Returns:
            List of SyntheticDefect objects
        """
        # Sample points on mesh surface
        positions, normals = sample_surface_points(self.mesh_data, n_points=num_defects)
        
        defects = []
        for i in range(num_defects):
            # Random size based on mesh scale
            size = float(self.mesh_scale * self.rng.uniform(0.02, 0.08))
            
            defects.append(SyntheticDefect(
                position=tuple(positions[i]),
                normal=tuple(normals[i]),
                size=size,
                defect_type=self.rng.choice(self.defect_types)
            ))
        
        return defects
    
    def _create_mesh_trace(
        self,
        defects: List[SyntheticDefect],
        lighting: LightingParams
    ) -> go.Mesh3d:
        """Create a Plotly Mesh3d trace with defect coloring."""
        vertices = self.mesh_data.vertices
        faces = self.mesh_data.faces
        
        # Generate vertex colors with defects "painted" on
        vertex_colors = self._generate_vertex_colors(vertices, defects)
        
        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            vertexcolor=vertex_colors,
            flatshading=False,
            lighting=dict(
                ambient=lighting.ambient,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(
                x=lighting.intensity * np.cos(np.radians(lighting.elevation)) * np.cos(np.radians(lighting.azimuth)) * 100,
                y=lighting.intensity * np.cos(np.radians(lighting.elevation)) * np.sin(np.radians(lighting.azimuth)) * 100,
                z=lighting.intensity * np.sin(np.radians(lighting.elevation)) * 100
            ),
            hoverinfo='skip'
        )
    
    def _generate_vertex_colors(
        self,
        vertices: np.ndarray,
        defects: List[SyntheticDefect],
        base_color: Tuple[int, int, int] = (169, 169, 169)  # Grey metal
    ) -> List[str]:
        """
        Generate vertex colors with defects painted onto the mesh.
        
        Uses distance-based interpolation for smooth defect appearance.
        """
        colors = []
        
        for vertex in vertices:
            # Start with base color
            r, g, b = base_color
            
            # Check distance to each defect
            for defect in defects:
                defect_pos = np.array(defect.position)
                dist = np.linalg.norm(vertex - defect_pos)
                
                if dist < defect.size:
                    # Inside defect radius - interpolate color
                    blend = 1.0 - (dist / defect.size)
                    blend = blend ** 0.5  # Softer falloff
                    
                    # Parse defect color
                    defect_rgb = self._parse_rgb(self.defect_colors[defect.defect_type])
                    
                    # Blend colors
                    r = int(r * (1 - blend) + defect_rgb[0] * blend)
                    g = int(g * (1 - blend) + defect_rgb[1] * blend)
                    b = int(b * (1 - blend) + defect_rgb[2] * blend)
            
            colors.append(f"rgb({r},{g},{b})")
        
        return colors
    
    def _parse_rgb(self, color_str: str) -> Tuple[int, int, int]:
        """Parse 'rgb(r, g, b)' string to tuple."""
        inner = color_str.replace("rgb(", "").replace(")", "")
        parts = [int(x.strip()) for x in inner.split(",")]
        return tuple(parts)
    
    def render_scene(
        self,
        defects: List[SyntheticDefect],
        camera: CameraParams,
        lighting: LightingParams
    ) -> bytes:
        """
        Render a scene with the given parameters.
        
        Returns:
            PNG image as bytes
        """
        # Create mesh trace
        mesh_trace = self._create_mesh_trace(defects, lighting)
        
        fig = go.Figure(data=[mesh_trace])
        
        # Apply camera
        eye = camera.to_eye_position(self.mesh_data.center)
        
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                bgcolor='#0D0D0D',
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                camera=dict(
                    eye=eye,
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            paper_bgcolor='#0D0D0D',
            plot_bgcolor='#0D0D0D',
            width=self.image_width,
            height=self.image_height,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        
        return fig.to_image(format="png")
    
    def generate_mask(
        self,
        defects: List[SyntheticDefect],
        camera: CameraParams
    ) -> bytes:
        """
        Generate a binary mask image showing defect locations.
        
        Renders defects as white circles on black background.
        
        Returns:
            PNG image as bytes
        """
        vertices = self.mesh_data.vertices
        faces = self.mesh_data.faces
        
        # Create mask colors (white for defects, black otherwise)
        mask_colors = []
        for vertex in vertices:
            is_defect = False
            for defect in defects:
                defect_pos = np.array(defect.position)
                dist = np.linalg.norm(vertex - defect_pos)
                if dist < defect.size:
                    is_defect = True
                    break
            
            if is_defect:
                mask_colors.append("rgb(255,255,255)")
            else:
                mask_colors.append("rgb(0,0,0)")
        
        # Create mask mesh trace
        mask_trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            vertexcolor=mask_colors,
            flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0, specular=0),
            hoverinfo='skip'
        )
        
        fig = go.Figure(data=[mask_trace])
        
        # Apply same camera as scene
        eye = camera.to_eye_position(self.mesh_data.center)
        
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                bgcolor='#000000',
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                camera=dict(
                    eye=eye,
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            paper_bgcolor='#000000',
            plot_bgcolor='#000000',
            width=self.image_width,
            height=self.image_height,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        
        return fig.to_image(format="png")
    
    def generate_sample(
        self,
        sample_id: int,
        num_defects: int = 3
    ) -> Tuple[bytes, bytes, SampleMetadata]:
        """
        Generate a single training sample.
        
        Args:
            sample_id: ID for this sample
            num_defects: Number of defects to generate
            
        Returns:
            Tuple of (image_bytes, mask_bytes, metadata)
        """
        # Randomize all parameters
        camera = self.randomize_camera()
        lighting = self.randomize_lighting()
        defects = self.randomize_defects(num_defects)
        
        # Render scene and mask
        image_bytes = self.render_scene(defects, camera, lighting)
        mask_bytes = self.generate_mask(defects, camera)
        
        # Create metadata
        metadata = SampleMetadata(
            sample_id=sample_id,
            camera=camera,
            lighting=lighting,
            defects=defects,
            mesh_name=self.mesh_data.name,
            image_width=self.image_width,
            image_height=self.image_height
        )
        
        return image_bytes, mask_bytes, metadata
    
    def save_sample(
        self,
        output_dir: Path,
        sample_id: int,
        image_bytes: bytes,
        mask_bytes: bytes,
        metadata: SampleMetadata
    ) -> Dict[str, str]:
        """
        Save a training sample to disk.
        
        Returns:
            Dict with paths to saved files
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        image_path = output_dir / f"image_{sample_id:04d}.png"
        mask_path = output_dir / f"mask_{sample_id:04d}.png"
        metadata_path = output_dir / f"metadata_{sample_id:04d}.json"
        
        # Save files
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        with open(mask_path, 'wb') as f:
            f.write(mask_bytes)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        return {
            "image": str(image_path),
            "mask": str(mask_path),
            "metadata": str(metadata_path)
        }


def generate_dataset(
    mesh_data: MeshData,
    num_samples: int = 50,
    output_dir: str = "synthetic_data",
    seed: int = 42,
    num_defects_per_sample: int = 3,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, str]]:
    """
    Generate a complete synthetic training dataset.
    
    Args:
        mesh_data: MeshData object for the mesh to render
        num_samples: Number of training samples to generate
        output_dir: Directory to save output files
        seed: Random seed for reproducibility
        num_defects_per_sample: Number of defects per sample
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        List of dicts containing paths to generated files
    """
    # Create generator
    generator = SyntheticDataGenerator(mesh_data, seed=seed)
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    results = []
    for i in range(num_samples):
        # Generate sample
        image_bytes, mask_bytes, metadata = generator.generate_sample(
            sample_id=i,
            num_defects=num_defects_per_sample
        )
        
        # Save sample
        paths = generator.save_sample(
            output_path, i, image_bytes, mask_bytes, metadata
        )
        results.append(paths)
        
        # Report progress
        if progress_callback:
            progress_callback(i + 1, num_samples)
    
    # Save dataset summary
    summary = {
        "num_samples": num_samples,
        "mesh_name": mesh_data.name,
        "seed": seed,
        "num_defects_per_sample": num_defects_per_sample,
        "image_size": [generator.image_width, generator.image_height],
        "samples": results
    }
    
    with open(output_path / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results
