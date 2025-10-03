# Image Summarizer

A simple and powerful tool for describing and summarizing images using AI. Supports **Amazon Bedrock**, **Azure OpenAI**, and **OpenAI** with a clean, easy-to-use interface.

## Quick Start

### Installation
```bash
# Basic requirements
pip install pyyaml

# Choose your AI provider
pip install boto3              # For Amazon Bedrock
pip install openai             # For Azure OpenAI or OpenAI
pip install flask flask-cors   # For web interface (optional)
```

### Configuration
1. Edit `config/config.yaml` to choose your AI provider:
```yaml
default_provider: "bedrock"  # or "azure_openai" or "openai"
```

2. Set environment variables:
```bash
# Amazon Bedrock
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"

# Azure OpenAI  
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"

# OpenAI
export OPENAI_API_KEY="your-api-key"
```

### Usage

#### Command Line Interface
```bash
# Basic usage
python cli.py image1.jpg image2.png

# With options
python cli.py *.jpg --config config/config.yaml --output results.json --format json
```

#### Web Interface
```bash
# Start web server
python app.py

# Then open: http://localhost:5000
```

#### Python Package
```python
import asyncio
from src.workflow import ImageSummarizer
from src.config import load_config

async def main():
    config = load_config('config/config.yaml')
    summarizer = ImageSummarizer(config)
    
    result = await summarizer.process_images([
        'image1.jpg', 'image2.png'
    ])
    
    print(result.summary)

asyncio.run(main())
```

## Features

- **Multi-provider support**: Amazon Bedrock, Azure OpenAI, and OpenAI
- **Simple configuration**: Single YAML file for all providers  
- **Multiple interfaces**: CLI, Web API, and Python package
- **Async processing**: Efficient batch processing of images
- **Clean architecture**: Abstract interfaces for easy extension
- **Error handling**: Graceful handling of failed images
- **Flexible output**: JSON, text, or programmatic access

## Project Structure

```
summarizer/
├── src/                    # Source code
│   ├── config.py              # Configuration management
│   ├── interfaces.py          # Abstract interfaces
│   ├── workflow.py            # Main workflow orchestrator
│   ├── bedrock_provider.py    # Amazon Bedrock integration
│   ├── azure_provider.py     # Azure OpenAI integration
│   └── openai_provider.py    # OpenAI integration
├── config/
│   └── config.yaml           # Single configuration file
├── tests/                  # Test files
├── build/                  # Package-related files
├── cli.py                  # Command line entry point
├── app.py                  # Web API entry point
└── README.md              # Documentation
```

## Configuration

The `config/config.yaml` file supports all providers with a unified structure:

### Amazon Bedrock
```yaml
default_provider: "bedrock"

bedrock:
  aws_region: "us-east-1"
  aws_access_key_id: "${AWS_ACCESS_KEY_ID}"
  aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
  
  image_model:
    model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
    max_tokens: 1000
    temperature: 0.7
    system_prompt: "Describe the image in detail..."
    
  text_model:
    model_id: "anthropic.claude-3-haiku-20240307-v1:0"
    max_tokens: 500
    temperature: 0.3
    system_prompt: "Create a concise summary..."
```

### Azure OpenAI
```yaml
default_provider: "azure_openai"

azure_openai:
  endpoint: "${AZURE_OPENAI_ENDPOINT}"
  api_key: "${AZURE_OPENAI_API_KEY}"
  api_version: "2024-02-15-preview"
  
  image_model:
    deployment_name: "gpt-4o"
    max_tokens: 1000
    temperature: 0.7
    
  text_model:
    deployment_name: "gpt-4o-mini"
    max_tokens: 500
    temperature: 0.3
```

### OpenAI
```yaml
default_provider: "openai"

openai:
  api_key: "${OPENAI_API_KEY}"
  
  image_model:
    model_name: "gpt-4o"
    max_tokens: 1000
    temperature: 0.7
    
  text_model:
    model_name: "gpt-4o-mini"
    max_tokens: 500
    temperature: 0.3

## Web Interface

The web interface provides an easy drag-and-drop interface:

1. **Start the server**: `python app.py`
2. **Open your browser**: http://localhost:5000
3. **Upload images**: Drag and drop or click to browse
4. **Get results**: AI-powered descriptions and summary

### API Endpoints

- `GET /` - Web interface
- `POST /summarize` - Upload and process images
- `GET /config` - Current configuration info
- `GET /health` - Server health check

## Development

### Running Tests
```bash
cd tests
python -m pytest test_workflow.py
```

### Setup Development Environment
```bash
# Check setup and dependencies
python setup_check.py

# Install in development mode  
pip install -e build/
```

### Project Architecture

- **Abstract interfaces** (`src/interfaces.py`) for clean architecture
- **Provider implementations** for different AI services
- **Factory pattern** for provider instantiation
- **Async/await** throughout for performance
- **Dataclasses** for type safety and validation

## Requirements

- **Python 3.8+**
- **PyYAML** (required)
- **boto3** (for Amazon Bedrock)
- **openai** (for Azure OpenAI/OpenAI)
- **flask + flask-cors** (for web interface)

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the configuration in `config/config.yaml`
- Verify environment variables are set
- Test with `python setup_check.py`
- Run tests with `pytest tests/`

## Examples

### Process Multiple Images
```bash
python cli.py folder/*.jpg folder/*.png --output summary.json
```

### Custom Configuration
```bash
python cli.py images/ --provider azure_openai --config custom_config.yaml
```

### Web API with Custom Port
```bash
python app.py --port 8080 --host 127.0.0.1
```
    temperature: 0.3
```

```yaml
description_model:
  provider: "openai"
  model_name: "gpt-4o"           # or "gpt-4o-mini" for budget option
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.3
  max_tokens: 800
  
summarization_model:
  provider: "openai" 
  model_name: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.5
  max_tokens: 1200

workflow:
  batch_size: 4
  max_retries: 3
  timeout: 600
```

**Detailed Setup Guide**: See the configuration section above for complete setup instructions, cost optimization, and troubleshooting.

## Architecture

The package uses abstract base classes for easy extensibility:

- `ImageDescriber`: Abstract interface for generating image descriptions
- `TextSummarizer`: Abstract interface for summarizing text descriptions  
- `ImageSummarizerWorkflow`: Intelligent workflow orchestrating the process

## License

MIT License - see LICENSE file for details.