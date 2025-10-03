"""
Test configuration and fixtures.
"""

import pytest
from pathlib import Path
from PIL import Image
import tempfile
import os

from image_summarizer.config import Config, ModelConfig, WorkflowConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return Config(
        description_model=ModelConfig(
            provider="test",
            model_name="test-vision-model",
            api_key="test-key",
            temperature=0.5,
        ),
        summarization_model=ModelConfig(
            provider="test",
            model_name="test-text-model", 
            api_key="test-key",
            temperature=0.3,
        ),
        workflow=WorkflowConfig(
            batch_size=2,
            max_retries=1,
            timeout=60,
        )
    )


@pytest.fixture
def sample_images(temp_dir):
    """Create sample test images."""
    image_paths = []
    
    # Create some simple test images
    for i in range(3):
        # Create a simple RGB image
        img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
        image_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
        img.save(image_path)
        image_paths.append(image_path)
    
    return image_paths


@pytest.fixture
def invalid_image_path(temp_dir):
    """Create an invalid image file for testing."""
    invalid_path = os.path.join(temp_dir, "invalid.jpg")
    with open(invalid_path, 'w') as f:
        f.write("This is not an image")
    return invalid_path