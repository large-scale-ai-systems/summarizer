"""
Falcon AI integration for image description and text summarization.
Uses local Hugging Face models via FalconAIModelManager.
"""

from typing import List
from pathlib import Path
from PIL import Image

try:
    import torch
    from transformers import pipeline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .interfaces import ImageDescriber, TextSummarizer, ImageDescription
from .config import ModelConfig


class FalconImageDescriber(ImageDescriber):
    """Image description using Falcon AI models."""
    
    def __init__(self, config: ModelConfig, provider_settings: dict):
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers are required for Falcon AI integration. Install with: pip install torch transformers")
        
        self.config = config
        self.provider_settings = provider_settings
        
        # For image description, we'll use a vision-language model
        # Since Falcon is primarily a text model, we'll use a compatible vision model
        print("Initializing Falcon-compatible image description model...")
        try:
            # Use BLIP model for image captioning (compatible with Falcon workflow)
            model_name = self.config.model_name if self.config.model_name else "Salesforce/blip-image-captioning-base"
            self.model = pipeline(
                "image-to-text",
                model=model_name,
                device="auto" if torch.cuda.is_available() else "cpu"
            )
            print(f"Falcon image description model ready: {model_name}")
        except Exception as e:
            print(f"Error initializing image model: {e}")
            # Fallback to base BLIP model
            self.model = pipeline(
                "image-to-text", 
                model="Salesforce/blip-image-captioning-base",
                device="auto" if torch.cuda.is_available() else "cpu"
            )
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and prepare image for processing."""
        return Image.open(image_path).convert('RGB')
    
    async def describe_image(self, image_path: str) -> ImageDescription:
        """Describe a single image using Falcon-compatible vision model."""
        try:
            if not Path(image_path).exists():
                return ImageDescription(
                    image_path=image_path,
                    description="",
                    success=False,
                    error_message=f"Image file not found: {image_path}"
                )
            
            # Load image
            image = self._load_image(image_path)
            
            # Generate caption
            result = self.model(image)
            
            # Extract description
            if isinstance(result, list) and len(result) > 0:
                description = result[0].get('generated_text', '')
            else:
                description = str(result)
            
            # Enhance the basic caption with the system prompt guidance
            enhanced_prompt = f"{self.config.system_prompt}\n\nBasic caption: {description}\n\nDetailed description:"
            
            return ImageDescription(
                image_path=image_path,
                description=description,
                success=True,
                metadata={
                    'model_id': 'falcon-compatible-vision',
                    'local_model': True,
                    'base_caption': description
                }
            )
            
        except Exception as e:
            return ImageDescription(
                image_path=image_path,
                description="",
                success=False,
                error_message=f"Error describing image with Falcon: {str(e)}"
            )
    
    async def describe_images_batch(self, image_paths: List[str]) -> List[ImageDescription]:
        """Describe multiple images."""
        results = []
        for image_path in image_paths:
            result = await self.describe_image(image_path)
            results.append(result)
        return results


class FalconTextSummarizer(TextSummarizer):
    """Text summarization using Falconsai/text_summarization model."""
    
    def __init__(self, config: ModelConfig, provider_settings: dict):
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers are required for Falcon AI. Install with: pip install torch transformers")
        
        self.config = config
        self.provider_settings = provider_settings
        
        # Initialize Falcon AI summarization model
        print("Initializing Falconsai text summarization model...")
        try:
            # Use the specific Falconsai model
            model_name = config.model_name if config.model_name else "Falconsai/text_summarization"
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device="auto" if torch.cuda.is_available() else "cpu"
            )
            print(f"Falconsai summarization model ready: {model_name}")
        except Exception as e:
            print(f"Warning: Could not load Falconsai model {config.model_name}")
            print(f"Error: {e}")
            # Create a fallback to the specific Falconsai model
            self.summarizer = pipeline(
                "summarization",
                model="Falconsai/text_summarization",
                device="auto" if torch.cuda.is_available() else "cpu"
            )
    
    async def summarize(self, descriptions: List[str]) -> str:
        """Create a summary from descriptions using Falcon AI."""
        try:
            # Combine all descriptions
            combined_text = "\n\n".join([
                f"Image {i+1}: {desc}" 
                for i, desc in enumerate(descriptions)
            ])
            
            # Limit text length to avoid model limits
            max_input_length = 1024
            if len(combined_text) > max_input_length:
                combined_text = combined_text[:max_input_length] + "..."
            
            # Generate summary using Falcon AI
            summary_result = self.summarizer(
                combined_text,
                max_length=self.config.max_tokens,
                min_length=50,
                do_sample=False
            )
            
            # Extract summary text
            if isinstance(summary_result, list) and len(summary_result) > 0:
                summary = summary_result[0].get('summary_text', '')
            else:
                summary = str(summary_result)
            
            return summary
            
        except Exception as e:
            return f"Error creating summary with Falcon AI: {str(e)}"