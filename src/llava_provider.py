"""
LLaVA (Large Language and Vision Assistant) integration for image description and text summarization.
Uses local Hugging Face models for privacy and cost efficiency.
"""

import base64
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
from .rag_model_manager import LLaVaModelManager


class LLaVAImageDescriber(ImageDescriber):
    """Image description using LLaVA local model."""
    
    def __init__(self, config: ModelConfig, provider_settings: dict):
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers are required for LLaVA integration. Install with: pip install torch transformers")
        
        self.config = config
        self.provider_settings = provider_settings
        
        # Initialize LLaVA model using the manager
        print("Initializing LLaVA model for image description...")
        self.model_manager = LLaVaModelManager()
        self.model = self.model_manager.get_model()
        self.processor = self.model_manager.get_processor()
        print("LLaVA model ready for image description.")
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and prepare image for processing."""
        return Image.open(image_path).convert('RGB')
    
    async def describe_image(self, image_path: str) -> ImageDescription:
        """Describe a single image using LLaVA."""
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
            
            # Prepare the prompt
            prompt = f"[INST] <image>\n{self.config.system_prompt}\n\nPlease describe this image in detail. [/INST]"
            
            # Process the image and text
            inputs = self.processor(prompt, image, return_tensors="pt")
            
            # Move inputs to the same device as model
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate description
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode the response
            generated_text = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract the description (remove the prompt part)
            if "[/INST]" in generated_text:
                description = generated_text.split("[/INST]")[-1].strip()
            else:
                description = generated_text.strip()
            
            return ImageDescription(
                image_path=image_path,
                description=description,
                success=True,
                metadata={
                    'model_id': self.config.model_name,
                    'local_model': True,
                    'device': str(device)
                }
            )
            
        except Exception as e:
            return ImageDescription(
                image_path=image_path,
                description="",
                success=False,
                error_message=f"Error describing image with LLaVA: {str(e)}"
            )
    
    async def describe_images_batch(self, image_paths: List[str]) -> List[ImageDescription]:
        """Describe multiple images."""
        results = []
        for image_path in image_paths:
            result = await self.describe_image(image_path)
            results.append(result)
        return results


class LLaVATextSummarizer(TextSummarizer):
    """Text summarization using LLaVA model itself (efficient - reuses same model)."""
    
    def __init__(self, config: ModelConfig, provider_settings: dict):
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers are required for text summarization. Install with: pip install torch transformers")
        
        self.config = config
        self.provider_settings = provider_settings
        
        # Check if we should use LLaVA itself or a separate model
        if config.model_name == "llava-hf/llava-v1.6-mistral-7b-hf":
            # Use the same LLaVA model for efficiency
            print("Using existing LLaVA model for text summarization (efficient mode)...")
            self.model_manager = LLaVaModelManager()
            self.model = self.model_manager.get_model()
            self.processor = self.model_manager.get_processor()
            self.use_llava = True
            print("LLaVA model ready for text summarization.")
        else:
            # Use a separate summarization model
            print(f"Loading separate text summarization model: {config.model_name}...")
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model=config.model_name,
                    device="auto" if torch.cuda.is_available() else "cpu"
                )
                self.use_llava = False
                print(f"Text summarization model ready: {config.model_name}")
            except Exception as e:
                print(f"Warning: Could not load {config.model_name}, using BART fallback")
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device="auto" if torch.cuda.is_available() else "cpu"
                )
                self.use_llava = False
    
    async def summarize(self, descriptions: List[str]) -> str:
        """Create a summary from descriptions using LLaVA or separate model."""
        try:
            # Combine all descriptions
            combined_text = "\n\n".join([
                f"Image {i+1}: {desc}" 
                for i, desc in enumerate(descriptions)
            ])
            
            if self.use_llava:
                # Use LLaVA model for text summarization
                summary_prompt = f"""Please create a concise summary of these image descriptions:

{combined_text}

Summary:"""
                
                # Tokenize and generate with LLaVA
                inputs = self.processor(text=summary_prompt, return_tensors="pt")
                
                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=min(self.config.max_tokens, 200),
                        temperature=self.config.temperature,
                        do_sample=True,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                # Decode the response
                response = self.processor.batch_decode(
                    generate_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                
                # Extract just the summary part after "Summary:"
                if "Summary:" in response:
                    summary = response.split("Summary:")[-1].strip()
                else:
                    summary = response.strip()
                
                return summary
                
            else:
                # Use separate summarization model (pipeline-based)
                # Limit text length to avoid model limits
                max_input_length = 1024  # Conservative limit for most models
                if len(combined_text) > max_input_length:
                    combined_text = combined_text[:max_input_length] + "..."
                
                # Generate summary
                summary_result = self.summarizer(
                    combined_text,
                    max_length=self.config.max_tokens,
                    min_length=50,
                    do_sample=False,
                    temperature=self.config.temperature
                )
                
                # Extract summary text
                if isinstance(summary_result, list) and len(summary_result) > 0:
                    summary = summary_result[0].get('summary_text', '')
                else:
                    summary = str(summary_result)
                
                return summary
            
        except Exception as e:
            return f"Error creating summary: {str(e)}"