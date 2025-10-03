"""
Test the simplified workflow and interfaces.
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.workflow import ImageSummarizer
from src.config import load_config
from src.interfaces import ImageDescription, SummaryResult


def test_workflow_creation():
    """Test workflow can be created with valid configuration."""
    config_content = """
default_provider: "bedrock"

workflow:
  batch_size: 2
  max_concurrent_requests: 1
  timeout_seconds: 60
  retry_attempts: 2

bedrock:
  aws_region: "us-east-1"
  aws_access_key_id: "test-key"
  aws_secret_access_key: "test-secret"
  
  image_model:
    model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
    max_tokens: 1000
    temperature: 0.7
    system_prompt: "Describe the image"
  
  text_model:
    model_id: "anthropic.claude-3-haiku-20240307-v1:0"
    max_tokens: 500
    temperature: 0.3
    system_prompt: "Summarize the descriptions"

output:
  save_individual_descriptions: true
  save_final_summary: true
  output_format: "json"

logging:
  level: "INFO"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        
        # This will fail if boto3 is not installed, which is expected
        try:
            summarizer = ImageSummarizer(config)
            info = summarizer.get_info()
            
            assert info['provider'] == 'bedrock'
            assert 'image_model' in info
            assert 'text_model' in info
            
        except ImportError:
            # Expected when boto3 is not installed
            pytest.skip("boto3 not available for testing")
            
    finally:
        os.unlink(config_path)


def test_image_description_dataclass():
    """Test ImageDescription dataclass."""
    desc = ImageDescription(
        image_path="/path/to/image.jpg",
        description="A test image description",
        success=True
    )
    
    assert desc.image_path == "/path/to/image.jpg"
    assert desc.description == "A test image description"
    assert desc.success is True
    assert desc.error_message is None


def test_summary_result_dataclass():
    """Test SummaryResult dataclass."""
    descriptions = [
        ImageDescription("img1.jpg", "Description 1", True),
        ImageDescription("img2.jpg", "Description 2", True)
    ]
    
    result = SummaryResult(
        summary="Overall summary",
        descriptions=descriptions,
        total_images=2,
        successful_descriptions=2,
        failed_images=[]
    )
    
    assert result.summary == "Overall summary"
    assert len(result.descriptions) == 2
    assert result.total_images == 2
    assert result.successful_descriptions == 2
    assert len(result.failed_images) == 0


def test_nonexistent_image_handling():
    """Test handling of nonexistent image files."""
    config_content = """
default_provider: "bedrock"

workflow:
  batch_size: 1

bedrock:
  aws_region: "us-east-1"
  image_model:
    model_id: "test-model"
    max_tokens: 100
    temperature: 0.7
    system_prompt: "Test"
  text_model:
    model_id: "test-model" 
    max_tokens: 100
    temperature: 0.3
    system_prompt: "Test"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        
        try:
            summarizer = ImageSummarizer(config)
            
            # Test with nonexistent files
            async def test_nonexistent():
                result = await summarizer.process_images([
                    "/nonexistent/file1.jpg",
                    "/nonexistent/file2.png"
                ])
                
                assert result.total_images == 2
                assert result.successful_descriptions == 0
                assert len(result.failed_images) == 2
                assert result.summary == "No valid images found to process."
                
            # Run async test
            asyncio.run(test_nonexistent())
            
        except ImportError:
            pytest.skip("Required provider dependencies not available")
            
    finally:
        os.unlink(config_path)


if __name__ == '__main__':
    pytest.main([__file__])