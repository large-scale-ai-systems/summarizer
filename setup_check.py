#!/usr/bin/env python3
"""
Setup script to initialize the simplified Image Summarizer structure.
"""

import os
import shutil
from pathlib import Path


def setup_environment():
    """Setup the simplified environment."""
    print("Setting up Image Summarizer - Simplified Structure")
    print("=" * 60)
    
    root_dir = Path(__file__).parent
    
    # Create logs directory if it doesn't exist
    logs_dir = root_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    print(f"Created logs directory: {logs_dir}")
    
    # Create uploads directory for web interface
    uploads_dir = root_dir / 'uploads'
    uploads_dir.mkdir(exist_ok=True)
    print(f"Created uploads directory: {uploads_dir}")
    
    # Check if configuration file exists
    config_file = root_dir / 'config' / 'config.yaml'
    if config_file.exists():
        print(f"Configuration file found: {config_file}")
    else:
        print(f"Configuration file not found: {config_file}")
        print("   Make sure to configure your AI provider settings!")
    
    # Show current structure
    print("\nCurrent Project Structure:")
    print(f"""
{root_dir.name}/
├── src/                    # Source code
│   ├── config.py          # Configuration management
│   ├── interfaces.py      # Abstract interfaces
│   ├── workflow.py        # Main workflow orchestrator
│   ├── bedrock_provider.py    # Amazon Bedrock integration
│   ├── azure_provider.py     # Azure OpenAI integration
│   └── openai_provider.py    # OpenAI integration
├── config/
│   └── config.yaml        # Single configuration file
├── tests/                 # Test files
├── build/                 # Package-related files
├── logs/                  # Log files (created)
├── uploads/               # Upload directory (created)
├── cli.py                 # Command line entry point
├── app.py                 # Web API entry point
└── README.md              # Documentation
""")
    
    print("\nNext Steps:")
    print("1. Configure your AI provider in config/config.yaml")
    print("2. Set environment variables for your chosen provider:")
    print("   - Amazon Bedrock: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
    print("   - Azure OpenAI: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY")
    print("   - OpenAI: OPENAI_API_KEY")
    print("3. Install required dependencies:")
    print("   - pip install boto3              # For Bedrock")
    print("   - pip install openai             # For Azure/OpenAI")  
    print("   - pip install flask flask-cors   # For web interface")
    print("4. Try it out:")
    print("   - python cli.py --help           # Command line help")
    print("   - python app.py                 # Start web interface")
    
    print("\nSetup complete! The simplified structure is ready to use.")


def show_examples():
    """Show usage examples."""
    print("\nUsage Examples:")
    print("-" * 40)
    
    print("Command Line:")
    print("python cli.py image1.jpg image2.png")
    print("python cli.py *.jpg --config config/config.yaml --output results.json")
    
    print("\nWeb Interface:")
    print("python app.py")
    print("# Then open http://localhost:5000")
    
    print("\nPython Code:")
    print("""
import asyncio
from src.workflow import ImageSummarizer
from src.config import load_config

async def main():
    config = load_config('config/config.yaml')
    summarizer = ImageSummarizer(config)
    
    result = await summarizer.process_images([
        'image1.jpg',
        'image2.png'
    ])
    
    print(result.summary)

asyncio.run(main())
""")


def check_dependencies():
    """Check which dependencies are available."""
    print("\nChecking Dependencies:")
    print("-" * 30)
    
    dependencies = {
        'yaml': 'Configuration (Required)',
        'boto3': 'Amazon Bedrock',
        'openai': 'Azure OpenAI / OpenAI',
        'flask': 'Web Interface',
        'flask_cors': 'Web Interface CORS',
        'pytest': 'Testing'
    }
    
    for package, purpose in dependencies.items():
        try:
            __import__(package)
            print(f"[OK] {package:<12} - {purpose}")
        except ImportError:
            print(f"[--] {package:<12} - {purpose} (Not installed)")


if __name__ == '__main__':
    setup_environment()
    check_dependencies()
    show_examples()