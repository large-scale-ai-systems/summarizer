"""
Tests for configuration management.
"""

import pytest
import tempfile
import os
import yaml

from image_summarizer.config import Config, ModelConfig, WorkflowConfig, load_config, create_example_config


def test_model_config_creation():
    """Test ModelConfig creation and validation."""
    config = ModelConfig(
        provider="openai",
        model_name="gpt-4",
        api_key="test-key",
        temperature=0.7
    )
    
    assert config.provider == "openai"
    assert config.model_name == "gpt-4"
    assert config.api_key == "test-key"
    assert config.temperature == 0.7


def test_model_config_env_var_resolution():
    """Test environment variable resolution in ModelConfig."""
    # Set environment variable
    os.environ["TEST_API_KEY"] = "resolved-key"
    
    config = ModelConfig(
        provider="test",
        model_name="test-model",
        api_key="${TEST_API_KEY}"
    )
    
    assert config.api_key == "resolved-key"
    
    # Clean up
    del os.environ["TEST_API_KEY"]


def test_workflow_config_defaults():
    """Test WorkflowConfig default values."""
    config = WorkflowConfig()
    
    assert config.batch_size == 10
    assert config.max_retries == 3
    assert config.timeout == 300


def test_config_creation(sample_config):
    """Test complete Config creation."""
    assert sample_config.description_model.provider == "test"
    assert sample_config.summarization_model.provider == "test"
    assert sample_config.workflow.batch_size == 2


def test_load_config_from_file(temp_dir):
    """Test loading configuration from YAML file."""
    config_data = {
        'description_model': {
            'provider': 'openai',
            'model_name': 'gpt-4-vision',
            'api_key': 'test-key',
        },
        'summarization_model': {
            'provider': 'openai',
            'model_name': 'gpt-4',
            'api_key': 'test-key',
        }
    }
    
    config_path = os.path.join(temp_dir, "test_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    config = load_config(config_path)
    
    assert config.description_model.provider == "openai"
    assert config.summarization_model.model_name == "gpt-4"


def test_load_config_file_not_found():
    """Test error handling for missing config file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_create_example_config(temp_dir):
    """Test creating example configuration."""
    config_path = os.path.join(temp_dir, "example.yaml")
    create_example_config(config_path)
    
    assert os.path.exists(config_path)
    
    # Verify the created config can be loaded
    config = load_config(config_path)
    assert config.description_model.provider == "openai"