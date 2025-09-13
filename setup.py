from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="video-event-detection",
    version="1.0.0",
    author="Video Event Detection Team",
    author_email="contact@example.com",
    description="Automatic event detection and extraction from videos using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/video-event-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Display",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "video-event-detection-api=src.api.main:main",
            "video-event-detection-web=src.web.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["**/*.py"],
    },
    keywords=[
        "video analysis",
        "event detection",
        "computer vision",
        "natural language processing",
        "AI",
        "machine learning",
        "OpenCLIP",
        "BLIP",
        "video processing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/example/video-event-detection/issues",
        "Source": "https://github.com/example/video-event-detection",
        "Documentation": "https://github.com/example/video-event-detection/wiki",
    },
)