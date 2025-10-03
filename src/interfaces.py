"""
Abstract interfaces for image description and text summarization.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ImageDescription:
    """Represents the description of a single image."""
    image_path: str
    description: str
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class SummaryResult:
    """Represents the result of the summarization process."""
    summary: str
    descriptions: List[ImageDescription]
    total_images: int
    successful_descriptions: int
    failed_images: List[str]
    error_message: Optional[str] = None
    metadata: Optional[dict] = None


class ImageDescriber(ABC):
    """Abstract base class for image description services."""
    
    @abstractmethod
    async def describe_image(self, image_path: str) -> ImageDescription:
        """
        Describe a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ImageDescription with the result
        """
        pass
    
    @abstractmethod
    async def describe_images_batch(self, image_paths: List[str]) -> List[ImageDescription]:
        """
        Describe multiple images in a batch.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of ImageDescription results
        """
        pass


class TextSummarizer(ABC):
    """Abstract base class for text summarization services."""
    
    @abstractmethod
    async def summarize(self, descriptions: List[str]) -> str:
        """
        Create a summary from multiple text descriptions.
        
        Args:
            descriptions: List of text descriptions to summarize
            
        Returns:
            Summary text
        """
        pass