"""
Core configuration management for the Image Summarizer.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_name: str
    max_tokens: int
    temperature: float
    system_prompt: str
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    image_model: ModelConfig
    text_model: ModelConfig
    provider_settings: Dict[str, Any]


@dataclass
class Config:
    """Main configuration class."""
    default_provider: str
    workflow: Dict[str, Any]
    providers: Dict[str, ProviderConfig]
    output: Dict[str, Any]
    logging: Dict[str, Any]
    # Optional cross-provider configuration
    image_provider: Optional[str] = None
    text_provider: Optional[str] = None


def expand_env_variables(data: Any) -> Any:
    """Recursively expand environment variables in configuration."""
    if isinstance(data, dict):
        return {key: expand_env_variables(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [expand_env_variables(item) for item in data]
    elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        env_var = data[2:-1]
        return os.getenv(env_var, data)  # Return original if env var not found
    else:
        return data


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    
    # Expand environment variables
    raw_config = expand_env_variables(raw_config)
    
    # Build provider configurations
    providers = {}
    
    for provider_name in ['bedrock', 'azure_openai', 'openai', 'llava', 'falcon']:
        if provider_name in raw_config:
            provider_data = raw_config[provider_name]
            
            # Extract model configurations
            image_model_data = provider_data.pop('image_model', {})
            text_model_data = provider_data.pop('text_model', {})
            
            # Handle different model naming conventions
            if provider_name == 'bedrock':
                image_model = ModelConfig(
                    model_name=image_model_data.get('model_id', ''),
                    max_tokens=image_model_data.get('max_tokens', 1000),
                    temperature=image_model_data.get('temperature', 0.7),
                    system_prompt=image_model_data.get('system_prompt', '')
                )
                text_model = ModelConfig(
                    model_name=text_model_data.get('model_id', ''),
                    max_tokens=text_model_data.get('max_tokens', 500),
                    temperature=text_model_data.get('temperature', 0.3),
                    system_prompt=text_model_data.get('system_prompt', '')
                )
            elif provider_name == 'azure_openai':
                image_model = ModelConfig(
                    model_name=image_model_data.get('deployment_name', ''),
                    max_tokens=image_model_data.get('max_tokens', 1000),
                    temperature=image_model_data.get('temperature', 0.7),
                    system_prompt=image_model_data.get('system_prompt', '')
                )
                text_model = ModelConfig(
                    model_name=text_model_data.get('deployment_name', ''),
                    max_tokens=text_model_data.get('max_tokens', 500),
                    temperature=text_model_data.get('temperature', 0.3),
                    system_prompt=text_model_data.get('system_prompt', '')
                )
            elif provider_name == 'llava':
                image_model = ModelConfig(
                    model_name=image_model_data.get('model_name', ''),
                    max_tokens=image_model_data.get('max_tokens', 1000),
                    temperature=image_model_data.get('temperature', 0.8),
                    system_prompt=image_model_data.get('system_prompt', '')
                )
                text_model = ModelConfig(
                    model_name=text_model_data.get('model_name', ''),
                    max_tokens=text_model_data.get('max_tokens', 500),
                    temperature=text_model_data.get('temperature', 0.3),
                    system_prompt=text_model_data.get('system_prompt', '')
                )
            elif provider_name == 'falcon':
                image_model = ModelConfig(
                    model_name=image_model_data.get('model_name', ''),
                    max_tokens=image_model_data.get('max_tokens', 1000),
                    temperature=image_model_data.get('temperature', 0.7),
                    system_prompt=image_model_data.get('system_prompt', '')
                )
                text_model = ModelConfig(
                    model_name=text_model_data.get('model_name', ''),
                    max_tokens=text_model_data.get('max_tokens', 500),
                    temperature=text_model_data.get('temperature', 0.3),
                    system_prompt=text_model_data.get('system_prompt', '')
                )
            else:  # openai
                image_model = ModelConfig(
                    model_name=image_model_data.get('model_name', ''),
                    max_tokens=image_model_data.get('max_tokens', 1000),
                    temperature=image_model_data.get('temperature', 0.7),
                    system_prompt=image_model_data.get('system_prompt', '')
                )
                text_model = ModelConfig(
                    model_name=text_model_data.get('model_name', ''),
                    max_tokens=text_model_data.get('max_tokens', 500),
                    temperature=text_model_data.get('temperature', 0.3),
                    system_prompt=text_model_data.get('system_prompt', '')
                )
            
            providers[provider_name] = ProviderConfig(
                image_model=image_model,
                text_model=text_model,
                provider_settings=provider_data  # Remaining settings
            )
    
    # Get cross-provider configuration if available
    providers_config = raw_config.get('providers_config', {})
    image_provider = providers_config.get('image_provider')
    text_provider = providers_config.get('text_provider')
    
    return Config(
        default_provider=raw_config.get('default_provider', 'bedrock'),
        workflow=raw_config.get('workflow', {}),
        providers=providers,
        output=raw_config.get('output', {}),
        logging=raw_config.get('logging', {}),
        image_provider=image_provider,
        text_provider=text_provider
    )