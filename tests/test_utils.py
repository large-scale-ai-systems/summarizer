"""
Tests for utility functions.
"""

import os
import pytest
from PIL import Image

from image_summarizer.utils import (
    find_images, validate_images, get_image_info, batch_images,
    estimate_processing_time, format_duration
)


def test_find_images_in_directory(temp_dir, sample_images):
    """Test finding images in a directory."""
    found_images = find_images(temp_dir, recursive=False)
    
    # Should find all sample images
    assert len(found_images) == len(sample_images)
    
    # All found files should be in our sample images
    for img in found_images:
        assert img in sample_images


def test_find_images_nonexistent_directory():
    """Test error handling for nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        find_images("/nonexistent/directory")


def test_validate_images(sample_images, invalid_image_path):
    """Test image validation function."""
    all_paths = sample_images + [invalid_image_path]
    valid, invalid = validate_images(all_paths)
    
    assert len(valid) == len(sample_images)
    assert len(invalid) == 1
    assert invalid[0] == invalid_image_path


def test_get_image_info(sample_images):
    """Test getting image information."""
    if sample_images:
        info = get_image_info(sample_images[0])
        
        assert "path" in info
        assert "filename" in info
        assert "format" in info
        assert "size" in info
        assert info["format"] == "JPEG"
        assert info["size"] == (100, 100)


def test_get_image_info_invalid_file(invalid_image_path):
    """Test getting info for invalid image."""
    info = get_image_info(invalid_image_path)
    
    assert "error" in info
    assert "path" in info


def test_batch_images():
    """Test batching of image paths."""
    image_paths = [f"image_{i}.jpg" for i in range(10)]
    
    batches = batch_images(image_paths, batch_size=3)
    
    assert len(batches) == 4  # 10 images, batch size 3: [3, 3, 3, 1]
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3  
    assert len(batches[2]) == 3
    assert len(batches[3]) == 1


def test_batch_images_invalid_size():
    """Test error handling for invalid batch size."""
    with pytest.raises(ValueError):
        batch_images(["image.jpg"], batch_size=0)


def test_estimate_processing_time():
    """Test processing time estimation."""
    time_estimate = estimate_processing_time(
        num_images=6,
        batch_size=2, 
        seconds_per_image=1.0
    )
    
    # 6 images, batch size 2 = 3 batches
    # 3 batches * 2 images * 1 second = 6 seconds
    assert time_estimate == 6.0


def test_format_duration():
    """Test duration formatting."""
    assert format_duration(30) == "30.0 seconds"
    assert format_duration(90) == "1.5 minutes"  
    assert format_duration(3660) == "1.0 hours"