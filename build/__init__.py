"""
Image Summarizer Package

A simple and powerful tool for describing and summarizing images using AI.
Supports Amazon Bedrock, Azure OpenAI, and OpenAI.
"""

from pathlib import Path
import sys

# Add src to path for imports
_src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(_src_path))

from src.workflow import ImageSummarizer
from src.config import load_config, Config
from src.interfaces import ImageDescription, SummaryResult

__version__ = '1.0.0'
__author__ = 'Image Summarizer Team'

__all__ = [
    'ImageSummarizer',
    'load_config',
    'Config',
    'ImageDescription',
    'SummaryResult'
]