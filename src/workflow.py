"""
Main workflow orchestrator for image summarization.
"""

import asyncio
from typing import List, Dict, Any
from pathlib import Path

from .config import Config, load_config
from .interfaces import ImageDescriber, TextSummarizer, SummaryResult, ImageDescription


class ProviderFactory:
    """Factory to create provider instances."""
    
    @staticmethod
    def create_image_describer(provider_name: str, config: Config) -> ImageDescriber:
        """Create an image describer for the given provider."""
        provider_config = config.providers[provider_name]
        
        if provider_name == 'bedrock':
            from .bedrock_provider import BedrockImageDescriber
            return BedrockImageDescriber(
                provider_config.image_model,
                provider_config.provider_settings
            )
        elif provider_name == 'azure_openai':
            from .azure_provider import AzureOpenAIImageDescriber
            return AzureOpenAIImageDescriber(
                provider_config.image_model,
                provider_config.provider_settings
            )
        elif provider_name == 'openai':
            from .openai_provider import OpenAIImageDescriber
            return OpenAIImageDescriber(
                provider_config.image_model,
                provider_config.provider_settings
            )
        elif provider_name == 'llava':
            from .llava_provider import LLaVAImageDescriber
            return LLaVAImageDescriber(
                provider_config.image_model,
                provider_config.provider_settings
            )
        elif provider_name == 'falcon':
            from .falcon_provider import FalconImageDescriber
            return FalconImageDescriber(
                provider_config.image_model,
                provider_config.provider_settings
            )
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    @staticmethod
    def create_text_summarizer(provider_name: str, config: Config) -> TextSummarizer:
        """Create a text summarizer for the given provider."""
        provider_config = config.providers[provider_name]
        
        if provider_name == 'bedrock':
            from .bedrock_provider import BedrockTextSummarizer
            return BedrockTextSummarizer(
                provider_config.text_model,
                provider_config.provider_settings
            )
        elif provider_name == 'azure_openai':
            from .azure_provider import AzureOpenAITextSummarizer
            return AzureOpenAITextSummarizer(
                provider_config.text_model,
                provider_config.provider_settings
            )
        elif provider_name == 'openai':
            from .openai_provider import OpenAITextSummarizer
            return OpenAITextSummarizer(
                provider_config.text_model,
                provider_config.provider_settings
            )
        elif provider_name == 'llava':
            from .llava_provider import LLaVATextSummarizer
            return LLaVATextSummarizer(
                provider_config.text_model,
                provider_config.provider_settings
            )
        elif provider_name == 'falcon':
            from .falcon_provider import FalconTextSummarizer
            return FalconTextSummarizer(
                provider_config.text_model,
                provider_config.provider_settings
            )
        else:
            raise ValueError(f"Unknown provider: {provider_name}")


class ImageSummarizer:
    """Main class for image summarization workflow."""
    
    def __init__(self, config: Config = None, config_path: str = None):
        """Initialize with configuration."""
        if config is None:
            config = load_config(config_path)
        
        self.config = config
        self.provider_name = config.default_provider
        
        # Set provider attributes for backward compatibility
        self.image_provider = config.default_provider
        self.text_provider = config.default_provider
        
        # Create provider instances
        self.image_describer = ProviderFactory.create_image_describer(
            self.provider_name, config
        )
        self.text_summarizer = ProviderFactory.create_text_summarizer(
            self.provider_name, config
        )
    
    async def process_images(self, image_paths: List[str]) -> SummaryResult:
        """Process a list of images and create a summary."""
        try:
            # Validate image paths
            valid_paths = []
            invalid_paths = []
            
            for path in image_paths:
                if Path(path).exists():
                    valid_paths.append(path)
                else:
                    invalid_paths.append(path)
            
            if not valid_paths:
                return SummaryResult(
                    summary="No valid images found to process.",
                    descriptions=[],
                    total_images=len(image_paths),
                    successful_descriptions=0,
                    failed_images=image_paths,
                    error_message="No valid image files found"
                )
            
            # Process images in batches
            batch_size = self.config.workflow.get('batch_size', 3)
            all_descriptions = []
            failed_images = list(invalid_paths)
            
            for i in range(0, len(valid_paths), batch_size):
                batch = valid_paths[i:i + batch_size]
                batch_results = await self.image_describer.describe_images_batch(batch)
                
                for result in batch_results:
                    if result.success:
                        all_descriptions.append(result)
                    else:
                        failed_images.append(result.image_path)
            
            # Create summary if we have successful descriptions
            if all_descriptions:
                description_texts = [desc.description for desc in all_descriptions]
                summary = await self.text_summarizer.summarize(description_texts)
            else:
                summary = "No images could be processed successfully."
            
            return SummaryResult(
                summary=summary,
                descriptions=all_descriptions,
                total_images=len(image_paths),
                successful_descriptions=len(all_descriptions),
                failed_images=failed_images,
                metadata={
                    'provider': self.provider_name,
                    'batch_size': batch_size
                }
            )
            
        except Exception as e:
            return SummaryResult(
                summary="",
                descriptions=[],
                total_images=len(image_paths) if image_paths else 0,
                successful_descriptions=0,
                failed_images=image_paths if image_paths else [],
                error_message=f"Workflow error: {str(e)}"
            )
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the current configuration."""
        try:
            # Use the provider attributes that were referenced in the error
            image_provider = getattr(self, 'image_provider', self.provider_name)
            text_provider = getattr(self, 'text_provider', self.provider_name)
            
            # Handle case where provider might be None
            if image_provider is None:
                image_provider = self.provider_name
            if text_provider is None:
                text_provider = self.provider_name
                
            image_config = self.config.providers[image_provider]
            text_config = self.config.providers[text_provider]
            
            return {
                'provider': self.provider_name,
                'image_provider': image_provider,
                'text_provider': text_provider,
                'image_model': image_config.image_model.model_name,
                'text_model': text_config.text_model.model_name,
                'workflow_settings': self.config.workflow,
                'optimization': {
                    'single_provider': image_provider == text_provider,
                    'models_shared': image_config.image_model.model_name == text_config.text_model.model_name,
                    'memory_efficient': image_provider == text_provider and image_config.image_model.model_name == text_config.text_model.model_name
                }
            }
        except (KeyError, AttributeError) as e:
            # Fallback for any configuration issues
            return {
                'provider': getattr(self, 'provider_name', 'unknown'),
                'image_provider': getattr(self, 'image_provider', 'unknown'),
                'text_provider': getattr(self, 'text_provider', 'unknown'),
                'image_model': 'Configuration Error',
                'text_model': 'Configuration Error',
                'workflow_settings': getattr(self.config, 'workflow', {}),
                'error': f"Provider configuration error: {str(e)}"
            }