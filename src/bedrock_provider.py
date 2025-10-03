"""
Amazon Bedrock integration for image description and text summarization.
"""

import json
import base64
from typing import List
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

from .interfaces import ImageDescriber, TextSummarizer, ImageDescription
from .config import ModelConfig


class BedrockImageDescriber(ImageDescriber):
    """Image description using Amazon Bedrock."""
    
    def __init__(self, config: ModelConfig, provider_settings: dict):
        if not BEDROCK_AVAILABLE:
            raise ImportError("boto3 is required for Bedrock integration. Install with: pip install boto3")
        
        self.config = config
        self.provider_settings = provider_settings
        
        # Initialize Bedrock client
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=provider_settings.get('aws_region', 'us-east-1'),
            aws_access_key_id=provider_settings.get('aws_access_key_id'),
            aws_secret_access_key=provider_settings.get('aws_secret_access_key')
        )
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    async def describe_image(self, image_path: str) -> ImageDescription:
        """Describe a single image using Bedrock."""
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
            
            # Prepare request for Claude
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": self.config.system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": "Please describe this image in detail."
                            }
                        ]
                    }
                ]
            }
            
            # Make request to Bedrock
            response = self.client.invoke_model(
                modelId=self.config.model_name,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            description = response_body['content'][0]['text']
            
            return ImageDescription(
                image_path=image_path,
                description=description,
                success=True,
                metadata={'model_id': self.config.model_name}
            )
            
        except ClientError as e:
            return ImageDescription(
                image_path=image_path,
                description="",
                success=False,
                error_message=f"AWS Bedrock error: {str(e)}"
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


class BedrockTextSummarizer(TextSummarizer):
    """Text summarization using Amazon Bedrock."""
    
    def __init__(self, config: ModelConfig, provider_settings: dict):
        if not BEDROCK_AVAILABLE:
            raise ImportError("boto3 is required for Bedrock integration. Install with: pip install boto3")
        
        self.config = config
        self.provider_settings = provider_settings
        
        # Initialize Bedrock client
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=provider_settings.get('aws_region', 'us-east-1'),
            aws_access_key_id=provider_settings.get('aws_access_key_id'),
            aws_secret_access_key=provider_settings.get('aws_secret_access_key')
        )
    
    async def summarize(self, descriptions: List[str]) -> str:
        """Create a summary from descriptions using Bedrock."""
        try:
            # Combine all descriptions
            combined_text = "\n\n".join([
                f"Image {i+1}: {desc}" 
                for i, desc in enumerate(descriptions)
            ])
            
            # Prepare request for Claude
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": self.config.system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Please create a comprehensive summary of these image descriptions:\n\n{combined_text}"
                    }
                ]
            }
            
            # Make request to Bedrock
            response = self.client.invoke_model(
                modelId=self.config.model_name,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            summary = response_body['content'][0]['text']
            
            return summary
            
        except Exception as e:
            return f"Error creating summary: {str(e)}"