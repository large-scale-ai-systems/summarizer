#!/usr/bin/env python3
"""
Image Summarizer - Command Line Interface

Usage:
    python cli.py <image1>             print(f"Error: {result.error_message}")
                return 1
            
            print(f"Successfully processed: {result.successful_descriptions}/{result.total_images} images")ge2> ... [options]
    python cli.py --config config/config.yaml image1.jpg image2.png
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.workflow import ImageSummarizer
from src.config import load_config


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Image Summarizer - Describe and summarize images using AI'
    )
    
    parser.add_argument(
        'images',
        nargs='+',
        help='Paths to image files to process'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--output',
        help='Path to save results (optional)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'text'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--provider',
        choices=['bedrock', 'azure_openai', 'openai', 'llava', 'falcon'],
        help='Override the default provider from config'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override provider if specified
        if args.provider:
            config.default_provider = args.provider
        
        # Initialize workflow
        summarizer = ImageSummarizer(config)
        
        # Display configuration info
        info = summarizer.get_info()
        print(f"Using provider: {info['provider']}")
        print(f"Image model: {info['image_model']}")
        print(f"Text model: {info['text_model']}")
        print(f"Processing {len(args.images)} images...\n")
        
        # Process images
        result = await summarizer.process_images(args.images)
        
        # Display results
        if args.format == 'json':
            output = {
                'summary': result.summary,
                'total_images': result.total_images,
                'successful_descriptions': result.successful_descriptions,
                'failed_images': result.failed_images,
                'descriptions': [
                    {
                        'image_path': desc.image_path,
                        'description': desc.description,
                        'success': desc.success
                    }
                    for desc in result.descriptions
                ],
                'error_message': result.error_message,
                'metadata': result.metadata
            }
            
            print(json.dumps(output, indent=2))
            
        else:  # text format
            print("=" * 60)
            print("IMAGE SUMMARIZATION RESULTS")
            print("=" * 60)
            
            if result.error_message:
                print(f"Error: {result.error_message}")
                return 1
            
            print(f"Successfully processed: {result.successful_descriptions}/{result.total_images} images")
            
            if result.failed_images:
                print(f"\nFailed images ({len(result.failed_images)}):")
                for img in result.failed_images:
                    print(f"  - {Path(img).name}")
            
            if result.descriptions:
                print(f"\nIndividual Descriptions:")
                print("-" * 40)
                for i, desc in enumerate(result.descriptions, 1):
                    print(f"\n{i}. {Path(desc.image_path).name}:")
                    print(f"   {desc.description}")
            
            print(f"\nFINAL SUMMARY:")
            print("-" * 30)
            print(result.summary)
        
        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            
            if args.format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("IMAGE SUMMARIZATION RESULTS\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Successfully processed: {result.successful_descriptions}/{result.total_images} images\n\n")
                    
                    if result.descriptions:
                        f.write("Individual Descriptions:\n")
                        f.write("-" * 40 + "\n")
                        for i, desc in enumerate(result.descriptions, 1):
                            f.write(f"\n{i}. {Path(desc.image_path).name}:\n")
                            f.write(f"{desc.description}\n")
                    
                    f.write(f"\nFINAL SUMMARY:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"{result.summary}\n")
            
            print(f"\nResults saved to: {output_path}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)