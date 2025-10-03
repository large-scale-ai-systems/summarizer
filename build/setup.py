"""
Setup script for the Image Summarizer package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent.parent / 'README.md'
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ''

# Read requirements
requirements = [
    'pyyaml>=6.0',
    'pathlib',
]

# Optional requirements for different providers
extras_require = {
    'bedrock': ['boto3>=1.26.0'],
    'azure': ['openai>=1.0.0'],
    'openai': ['openai>=1.0.0'],
    'web': ['flask>=2.0.0', 'flask-cors>=4.0.0'],
    'all': [
        'boto3>=1.26.0',
        'openai>=1.0.0', 
        'flask>=2.0.0',
        'flask-cors>=4.0.0'
    ]
}

setup(
    name='image-summarizer',
    version='1.0.0',
    description='AI-powered image description and summarization tool',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Image Summarizer Team',
    python_requires='>=3.8',
    packages=find_packages(where='..'),
    package_dir={'': '..'},
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'image-summarizer=cli:main',
            'image-summarizer-web=app:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Graphics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='ai image description summarization computer-vision nlp',
    project_urls={
        'Documentation': 'https://github.com/your-org/image-summarizer',
        'Source': 'https://github.com/your-org/image-summarizer',
        'Tracker': 'https://github.com/your-org/image-summarizer/issues',
    },
)