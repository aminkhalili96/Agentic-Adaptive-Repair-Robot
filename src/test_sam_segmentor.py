"""
Test suite for SAM Segmentor.

Tests the interactive segmentation module including:
- Model initialization and fallback behavior
- Point-based segmentation
- Mask overlay generation
- Coverage calculation
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.sam_segmentor import SAMSegmentor, SegmentationResult, get_segmentor


class TestSAMSegmentor:
    """Test cases for SAMSegmentor class."""
    
    @pytest.fixture
    def segmentor(self):
        """Create a SAMSegmentor instance for testing."""
        return SAMSegmentor(model_type="mobile_sam")
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image for testing."""
        # 100x100 image with a red square in the center
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Gray background
        image[40:60, 40:60] = [255, 0, 0]  # Red square
        return image
    
    def test_initialization(self, segmentor):
        """Test that SAMSegmentor initializes correctly."""
        assert segmentor is not None
        assert segmentor.model_type == "mobile_sam"
        # Status should be available regardless of model loading
        status = segmentor.get_status()
        assert "model_type" in status
        assert "model_loaded" in status
        assert "fallback_available" in status
        assert status["fallback_available"] is True
    
    def test_segment_at_point_returns_result(self, segmentor, sample_image):
        """Test that segment_at_point returns a SegmentationResult."""
        result = segmentor.segment_at_point(sample_image, x=50, y=50)
        
        assert isinstance(result, SegmentationResult)
        assert result.mask is not None
        assert result.overlay is not None
        assert result.click_point == (50, 50)
        assert 0 <= result.confidence <= 1.0
        assert 0 <= result.coverage_percent <= 100
    
    def test_segment_mask_shape(self, segmentor, sample_image):
        """Test that the mask has correct shape."""
        result = segmentor.segment_at_point(sample_image, x=50, y=50)
        
        h, w = sample_image.shape[:2]
        assert result.mask.shape == (h, w)
        assert result.overlay.shape == sample_image.shape
    
    def test_segment_at_edge_coordinates(self, segmentor, sample_image):
        """Test segmentation at edge coordinates."""
        # Should handle edge cases without error
        result_corner = segmentor.segment_at_point(sample_image, x=0, y=0)
        assert result_corner.click_point == (0, 0)
        
        result_edge = segmentor.segment_at_point(sample_image, x=99, y=99)
        assert result_edge.click_point == (99, 99)
    
    def test_segment_coordinates_clamped(self, segmentor, sample_image):
        """Test that out-of-bounds coordinates are clamped."""
        # Coordinates beyond image bounds should be clamped
        result = segmentor.segment_at_point(sample_image, x=1000, y=1000)
        assert result.click_point[0] <= sample_image.shape[1]
        assert result.click_point[1] <= sample_image.shape[0]
    
    def test_overlay_generation(self, segmentor, sample_image):
        """Test mask overlay generation."""
        # Create a simple mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 1  # Square region
        
        # Use green overlay instead of red (red matches the sample image's red square)
        overlay = segmentor.create_overlay(
            sample_image, mask,
            color=(0, 255, 0),  # Green overlay
            alpha=0.5
        )
        
        # Overlay should have same shape as input
        assert overlay.shape == sample_image.shape
        
        # Overlay should be different from original in masked region
        # Green overlay on red region should produce different color
        assert not np.array_equal(overlay[50, 50], sample_image[50, 50])
    
    def test_coverage_calculation(self, segmentor):
        """Test coverage percentage calculation."""
        # 10% coverage
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:100] = 1  # 10 rows = 1000 pixels = 10%
        
        coverage = segmentor.calculate_coverage(mask, 10000)
        assert coverage == 10.0
        
        # 25% coverage
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:50, 0:50] = 1  # 2500 pixels = 25%
        
        coverage = segmentor.calculate_coverage(mask, 10000)
        assert coverage == 25.0
        
        # 0% coverage
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        coverage = segmentor.calculate_coverage(empty_mask, 10000)
        assert coverage == 0.0
    
    def test_result_to_dict(self, segmentor, sample_image):
        """Test SegmentationResult serialization."""
        result = segmentor.segment_at_point(sample_image, x=50, y=50)
        result_dict = result.to_dict()
        
        assert "coverage_percent" in result_dict
        assert "confidence" in result_dict
        assert "click_point" in result_dict
        assert "mask_shape" in result_dict
        assert result_dict["click_point"] == (50, 50)
    
    def test_is_available_property(self, segmentor):
        """Test is_available property."""
        # Should be boolean
        assert isinstance(segmentor.is_available, bool)


class TestSingleton:
    """Test the singleton pattern for get_segmentor."""
    
    def test_get_segmentor_returns_same_instance(self):
        """Test that get_segmentor returns singleton instance."""
        seg1 = get_segmentor()
        seg2 = get_segmentor()
        
        # Should be the same object (singleton)
        assert seg1 is seg2
    
    def test_get_segmentor_returns_valid_instance(self):
        """Test that get_segmentor returns a valid SAMSegmentor."""
        segmentor = get_segmentor()
        
        assert isinstance(segmentor, SAMSegmentor)
        assert segmentor.get_status() is not None


class TestFallbackBehavior:
    """Test fallback behavior when SAM model is not available."""
    
    def test_fallback_produces_mask(self):
        """Test that fallback method produces valid output."""
        segmentor = SAMSegmentor(model_type="mobile_sam")
        
        # Create test image with distinct region
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        image[40:60, 40:60] = [100, 100, 100]  # Darker square
        
        # Force fallback by checking if model loaded
        if not segmentor.model_loaded:
            result = segmentor.segment_at_point(image, x=50, y=50)
            
            # Should still produce valid result via fallback
            assert result.mask is not None
            assert result.overlay is not None
            # Fallback confidence should be lower
            assert result.confidence <= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
