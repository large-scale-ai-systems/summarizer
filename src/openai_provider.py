"""
OpenAI integration for image description and text summarization.
"""

import base64
from typing import List
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .interfaces import ImageDescriber, TextSummarizer, ImageDescription
from .config import ModelConfig


class OpenAIImageDescriber(ImageDescriber):
    """Image description using OpenAI."""
    
    def __init__(self, config: ModelConfig, provider_settings: dict):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is required for OpenAI integration. Install with: pip install openai")
        
        self.config = config
        self.provider_settings = provider_settings
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=provider_settings.get('api_key'))
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    async def describe_image(self, image_path: str) -> ImageDescription:
        """Describe a single image using OpenAI."""
        try:
            if not Path(image_path).exists():
                return ImageDescription(
                    image_path=image_path,
                    description="",
                    success=False,
                    error_message=f"Image file not found: {image_path}"
                )
            
            # Encode image
            image_base64 = self._encode_image(image_path)
            image_ext = Path(image_path).suffix.lower()
            
            # Determine media type
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = media_type_map.get(image_ext, 'image/jpeg')
            
            # Make request to OpenAI
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.config.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please describe this image in detail."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            description = response.choices[0].message.content
            
            return ImageDescription(
                image_path=image_path,
                description=description,
                success=True,
                metadata={
                    'model': self.config.model_name,
                    'usage': response.usage.dict() if response.usage else None
                }
            )
            
        except Exception as e:
            return ImageDescription(
                image_path=image_path,
                description="",
                success=False,
                error_message=f"Error describing image: {str(e)}"
            )
    
    async def describe_images_batch(self, image_paths: List[str]) -> List[ImageDescription]:
        """Describe multiple images."""
        results = []
        for image_path in image_paths:
            result = await self.describe_image(image_path)
            results.append(result)
        return results


class OpenAITextSummarizer(TextSummarizer):
    """Text summarization using OpenAI."""
    
    def __init__(self, config: ModelConfig, provider_settings: dict):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is required for OpenAI integration. Install with: pip install openai")
        
        self.config = config
        self.provider_settings = provider_settings
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=provider_settings.get('api_key'))
    
    async def summarize(self, descriptions: List[str]) -> str:
        """Create a summary from descriptions using OpenAI."""
        try:
            # Combine all descriptions
            combined_text = "\n\n".join([
                f"Image {i+1}: {desc}" 
                for i, desc in enumerate(descriptions)
            ])
            
            # Make request to OpenAI
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.config.system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Please create a comprehensive summary of these image descriptions:\n\n{combined_text}"
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error creating summary: {str(e)}"