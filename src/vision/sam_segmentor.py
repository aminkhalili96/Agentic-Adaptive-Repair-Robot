"""
SAM Segmentor - Interactive Zero-Shot Segmentation using Segment Anything Model.

This module provides interactive defect segmentation using SAM/MobileSAM.
Users can click on any point in an image to automatically segment the region
containing that point, enabling zero-shot defect detection without custom training.

Architecture:
    Image → SAM Encoder → Image Embeddings
    Click (x,y) → Prompt Encoder → Point Embeddings  
    Combined → Mask Decoder → Binary Mask

Features:
    - Zero-shot segmentation (no training required)
    - MobileSAM support for lightweight inference (~50ms on CPU)
    - Fallback to OpenCV contour detection if SAM unavailable
    - Mask overlay with configurable color and transparency
    - Coverage percentage calculation
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """
    Result of SAM segmentation at a point.
    
    Attributes:
        mask: Binary mask (H, W) where 1 = segmented region
        overlay: RGB image with mask overlay applied
        coverage_percent: Percentage of image covered by mask
        confidence: Model confidence score (0-1)
        click_point: (x, y) coordinates that triggered segmentation
    """
    mask: np.ndarray
    overlay: np.ndarray
    coverage_percent: float
    confidence: float
    click_point: Tuple[int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "coverage_percent": self.coverage_percent,
            "confidence": self.confidence,
            "click_point": self.click_point,
            "mask_shape": self.mask.shape,
        }


class SAMSegmentor:
    """
    Zero-shot image segmentor using Segment Anything Model (SAM).
    
    This class wraps MobileSAM for efficient interactive segmentation.
    When SAM is not available, it falls back to OpenCV-based methods.
    
    Example:
        >>> segmentor = SAMSegmentor()
        >>> result = segmentor.segment_at_point(image, x=100, y=150)
        >>> print(f"Defect covers {result.coverage_percent:.1f}% of the part")
    """
    
    # Default model paths
    MOBILE_SAM_CHECKPOINT = "mobile_sam.pt"
    SAM_MODEL_URL = "https://github.com/ChaoningZhang/MobileSAM/releases/download/v1.0/mobile_sam.pt"
    
    def __init__(self, model_type: str = "mobile_sam", checkpoint_path: Optional[str] = None):
        """
        Initialize SAM segmentor with specified model.
        
        Args:
            model_type: Type of SAM model ("mobile_sam", "vit_b", "vit_l", "vit_h")
            checkpoint_path: Optional path to model checkpoint. If None, uses default.
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self.model_loaded = False
        
        # Try to load SAM model
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the SAM model and predictor.
        
        Falls back gracefully if model is unavailable.
        """
        try:
            if self.model_type == "mobile_sam":
                # Try MobileSAM first (lightweight)
                from mobile_sam import sam_model_registry, SamPredictor
                
                # Look for checkpoint in common locations
                checkpoint = self._find_checkpoint()
                
                if checkpoint:
                    logger.info(f"Loading MobileSAM from {checkpoint}")
                    sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
                    sam.eval()
                    self.predictor = SamPredictor(sam)
                    self.model_loaded = True
                    logger.info("MobileSAM loaded successfully")
                else:
                    logger.warning(f"MobileSAM checkpoint not found. Download from: {self.SAM_MODEL_URL}")
                    
            else:
                # Standard SAM models
                from segment_anything import sam_model_registry, SamPredictor
                
                checkpoint = self._find_checkpoint()
                if checkpoint:
                    sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                    sam.eval()
                    self.predictor = SamPredictor(sam)
                    self.model_loaded = True
                    
        except ImportError as e:
            logger.warning(f"SAM not available: {e}. Using fallback segmentation.")
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
    
    def _find_checkpoint(self) -> Optional[str]:
        """
        Search for model checkpoint in common locations.
        
        Returns:
            Path to checkpoint if found, None otherwise.
        """
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            return self.checkpoint_path
        
        # Search locations
        search_paths = [
            Path.cwd() / self.MOBILE_SAM_CHECKPOINT,
            Path.cwd() / "models" / self.MOBILE_SAM_CHECKPOINT,
            Path.cwd() / "assets" / self.MOBILE_SAM_CHECKPOINT,
            Path.home() / ".cache" / "sam" / self.MOBILE_SAM_CHECKPOINT,
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def segment_at_point(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        multimask_output: bool = False
    ) -> SegmentationResult:
        """
        Segment the region containing the clicked point.
        
        This is the main entry point for interactive segmentation. Given an image
        and (x, y) click coordinates, it returns the segmented mask for the
        region containing that point.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            x: X coordinate of click point (column)
            y: Y coordinate of click point (row)
            multimask_output: If True, return multiple masks with different granularities
            
        Returns:
            SegmentationResult with mask, overlay, and metrics
        """
        h, w = image.shape[:2]
        
        # Validate click coordinates
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        if self.model_loaded and self.predictor is not None:
            mask, confidence = self._sam_segment(image, x, y, multimask_output)
        else:
            # Fallback to color-based segmentation
            mask, confidence = self._fallback_segment(image, x, y)
        
        # Create overlay
        overlay = self.create_overlay(image, mask)
        
        # Calculate coverage
        coverage = self.calculate_coverage(mask, h * w)
        
        return SegmentationResult(
            mask=mask,
            overlay=overlay,
            coverage_percent=coverage,
            confidence=confidence,
            click_point=(x, y)
        )
    
    def _sam_segment(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        multimask: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Run SAM segmentation using point prompt.
        
        Args:
            image: RGB image
            x, y: Click coordinates
            multimask: Return multiple masks if True
            
        Returns:
            Tuple of (binary mask, confidence score)
        """
        # Set image for predictor
        self.predictor.set_image(image)
        
        # Create point prompt (1 = foreground, 0 = background)
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # Foreground
        
        # Predict mask
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=multimask,
        )
        
        # Select best mask (highest score)
        if multimask:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            confidence = float(scores[best_idx])
        else:
            mask = masks[0]
            confidence = float(scores[0])
        
        return mask.astype(np.uint8), confidence
    
    def _fallback_segment(
        self,
        image: np.ndarray,
        x: int,
        y: int
    ) -> Tuple[np.ndarray, float]:
        """
        Fallback segmentation using OpenCV flood fill when SAM unavailable.
        
        This provides basic region segmentation based on color similarity
        around the clicked point.
        
        Args:
            image: RGB image
            x, y: Click coordinates
            
        Returns:
            Tuple of (binary mask, confidence score)
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale for flood fill
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create mask slightly larger for flood fill algorithm
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Flood fill from clicked point
        # Lower/upper difference determines color tolerance
        lo_diff = 20
        up_diff = 20
        
        cv2.floodFill(
            gray.copy(),
            flood_mask,
            (x, y),
            255,
            loDiff=lo_diff,
            upDiff=up_diff,
            flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        )
        
        # Extract mask (remove border padding)
        mask = flood_mask[1:-1, 1:-1]
        
        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Confidence is lower for fallback method
        confidence = 0.5 if np.sum(mask) > 0 else 0.0
        
        return (mask > 0).astype(np.uint8), confidence
    
    def create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create an overlay visualization of the mask on the image.
        
        Args:
            image: Original RGB image (H, W, 3)
            mask: Binary mask (H, W) where 1 = segmented region
            color: RGB color for the overlay (default: red)
            alpha: Transparency of overlay (0 = transparent, 1 = opaque)
            
        Returns:
            RGB image with colored mask overlay
        """
        overlay = image.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Blend with alpha
        mask_bool = mask > 0
        overlay[mask_bool] = (
            (1 - alpha) * image[mask_bool] + alpha * colored_mask[mask_bool]
        ).astype(np.uint8)
        
        # Draw contour for clarity
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
        
        return overlay
    
    def calculate_coverage(self, mask: np.ndarray, total_pixels: int) -> float:
        """
        Calculate the percentage of image covered by the mask.
        
        Args:
            mask: Binary mask (H, W)
            total_pixels: Total number of pixels in the image
            
        Returns:
            Coverage percentage (0-100)
        """
        mask_pixels = np.sum(mask > 0)
        coverage = (mask_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        return round(coverage, 2)
    
    @property
    def is_available(self) -> bool:
        """Check if SAM model is loaded and ready."""
        return self.model_loaded
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current segmentor status for debugging.
        
        Returns:
            Dictionary with model status information
        """
        return {
            "model_type": self.model_type,
            "model_loaded": self.model_loaded,
            "checkpoint_path": self.checkpoint_path,
            "fallback_available": True,
        }


# Singleton instance for app-wide use
_segmentor_instance: Optional[SAMSegmentor] = None


def get_segmentor(model_type: str = "mobile_sam") -> SAMSegmentor:
    """
    Get or create the global SAMSegmentor instance.
    
    This provides a singleton pattern for efficient model reuse.
    
    Args:
        model_type: Type of SAM model to use
        
    Returns:
        SAMSegmentor instance
    """
    global _segmentor_instance
    
    if _segmentor_instance is None:
        _segmentor_instance = SAMSegmentor(model_type=model_type)
    
    return _segmentor_instance
