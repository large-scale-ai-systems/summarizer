#!/usr/bin/env python3
"""
Image Summarizer - Flask Web API

Usage:
    python app.py
    python app.py --config config/config.yaml --port 5000
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import uuid

try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    from werkzeug.utils import secure_filename
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.workflow import ImageSummarizer
from src.config import load_config


# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Summarizer</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin-bottom: 20px; }
        .upload-area.dragover { border-color: #007bff; background-color: #f8f9fa; }
        .file-list { margin: 20px 0; }
        .file-item { padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 5px; }
        .button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        .button:hover { background: #0056b3; }
        .button:disabled { background: #ccc; cursor: not-allowed; }
        .results { margin-top: 30px; }
        .description { margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        .summary { margin: 20px 0; padding: 20px; background: #e7f3ff; border-radius: 5px; }
        .error { color: #dc3545; background: #f8d7da; padding: 15px; border-radius: 5px; }
        .success { color: #155724; background: #d4edda; padding: 15px; border-radius: 5px; }
        .loading { display: none; text-align: center; padding: 20px; }
        .config-info { background: #fff3cd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Image Summarizer</h1>
        <p>Upload images to get AI-powered descriptions and summaries</p>
    </div>

    <div class="config-info" id="config-info">
        Loading configuration...
    </div>

    <div class="upload-area" id="upload-area">
        <input type="file" id="file-input" multiple accept="image/*" style="display: none;">
        <p>Drag and drop images here, or <button class="button" onclick="document.getElementById('file-input').click()">Browse Files</button></p>
        <p>Supported formats: JPG, PNG, GIF, BMP, TIFF, WEBP</p>
    </div>

    <div class="file-list" id="file-list"></div>

    <div style="text-align: center;">
        <button class="button" id="process-btn" onclick="processImages()" disabled>Process Images</button>
    </div>

    <div class="loading" id="loading">
        <p>Processing images... This may take a few minutes.</p>
        <p>Please wait while AI analyzes your images.</p>
    </div>

    <div class="results" id="results"></div>

    <script>
        let selectedFiles = [];

        // Load configuration info
        fetch('/config')
            .then(response => response.json())
            .then(data => {
                document.getElementById('config-info').innerHTML = `
                    <strong>Provider:</strong> ${data.provider} | 
                    <strong>Image Model:</strong> ${data.image_model} | 
                    <strong>Text Model:</strong> ${data.text_model}
                `;
            });

        // File upload handling
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const fileList = document.getElementById('file-list');
        const processBtn = document.getElementById('process-btn');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });

        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });

        function handleFiles(files) {
            for (let file of files) {
                if (file.type.startsWith('image/')) {
                    selectedFiles.push(file);
                }
            }
            updateFileList();
            updateProcessButton();
        }

        function updateFileList() {
            fileList.innerHTML = '';
            selectedFiles.forEach((file, index) => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)
                    <button onclick="removeFile(${index})" style="float: right; background: #dc3545; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;">Remove</button>
                `;
                fileList.appendChild(fileItem);
            });
        }

        function removeFile(index) {
            selectedFiles.splice(index, 1);
            updateFileList();
            updateProcessButton();
        }

        function updateProcessButton() {
            processBtn.disabled = selectedFiles.length === 0;
        }

        async function processImages() {
            if (selectedFiles.length === 0) return;

            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('images', file);
            });

            document.getElementById('loading').style.display = 'block';
            processBtn.disabled = true;
            document.getElementById('results').innerHTML = '';

            try {
                const response = await fetch('/summarize', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    displayResults(result);
                } else {
                    displayError(result.error || 'An error occurred');
                }
            } catch (error) {
                displayError('Network error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                processBtn.disabled = false;
            }
        }

        function displayResults(result) {
            let html = '<h2>Results</h2>';
            
            if (result.error_message) {
                html += `<div class="error">Error: ${result.error_message}</div>`;
                document.getElementById('results').innerHTML = html;
                return;
            }

            html += `<div class="success">Successfully processed ${result.successful_descriptions}/${result.total_images} images</div>`;

            if (result.failed_images && result.failed_images.length > 0) {
                html += `<div class="error">Failed to process: ${result.failed_images.join(', ')}</div>`;
            }

            if (result.descriptions && result.descriptions.length > 0) {
                html += '<h3>Individual Descriptions</h3>';
                result.descriptions.forEach((desc, index) => {
                    html += `
                        <div class="description">
                            <strong>${index + 1}. ${desc.image_path.split('/').pop()}</strong><br>
                            ${desc.description}
                        </div>
                    `;
                });
            }

            if (result.summary) {
                html += `
                    <h3>Final Summary</h3>
                    <div class="summary">${result.summary}</div>
                `;
            }

            document.getElementById('results').innerHTML = html;
        }

        function displayError(message) {
            document.getElementById('results').innerHTML = `<div class="error">Error: ${message}</div>`;
        }
    </script>
</body>
</html>
"""


class ImageSummarizerAPI:
    """Flask API for Image Summarizer."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the web API. Install with: pip install flask flask-cors")
        
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Load configuration
        self.config = load_config(config_path)
        self.summarizer = ImageSummarizer(self.config)
        
        # Create upload directory
        self.upload_dir = Path('uploads')
        self.upload_dir.mkdir(exist_ok=True)
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
        
        @self.app.route('/config')
        def get_config():
            info = self.summarizer.get_info()
            return jsonify(info)
        
        @self.app.route('/health')
        def health():
            return jsonify({'status': 'healthy', 'provider': self.config.default_provider})
        
        @self.app.route('/summarize', methods=['POST'])
        def summarize():
            try:
                # Get uploaded files
                if 'images' not in request.files:
                    return jsonify({'error': 'No images uploaded'}), 400
                
                files = request.files.getlist('images')
                if not files:
                    return jsonify({'error': 'No images provided'}), 400
                
                # Save uploaded files
                image_paths = []
                job_id = str(uuid.uuid4())
                job_dir = self.upload_dir / job_id
                job_dir.mkdir(exist_ok=True)
                
                for file in files:
                    if file.filename:
                        filename = secure_filename(file.filename)
                        file_path = job_dir / filename
                        file.save(str(file_path))
                        image_paths.append(str(file_path))
                
                # Process images
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.summarizer.process_images(image_paths))
                finally:
                    loop.close()
                
                # Convert result to JSON-serializable format
                response = {
                    'summary': result.summary,
                    'total_images': result.total_images,
                    'successful_descriptions': result.successful_descriptions,
                    'failed_images': result.failed_images,
                    'descriptions': [
                        {
                            'image_path': desc.image_path,
                            'description': desc.description,
                            'success': desc.success,
                            'error_message': desc.error_message
                        }
                        for desc in result.descriptions
                    ],
                    'error_message': result.error_message,
                    'metadata': result.metadata,
                    'job_id': job_id
                }
                
                return jsonify(response)
                
            except Exception as e:
                return jsonify({'error': f'Server error: {str(e)}'}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application."""
        print(f"Starting Image Summarizer API")
        print(f"Provider: {self.config.default_provider}")
        print(f"Server: http://{host}:{port}")
        print(f"Open your browser to start using the web interface")
        
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function for running the web API."""
    parser = argparse.ArgumentParser(description='Image Summarizer Web API')
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    try:
        api = ImageSummarizerAPI(args.config)
        api.run(host=args.host, port=args.port, debug=args.debug)
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()