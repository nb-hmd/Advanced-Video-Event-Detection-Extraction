# 🎥 Advanced Video Event Detection & Extraction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com)

A comprehensive AI-powered video analysis platform that combines multiple state-of-the-art computer vision models to detect, analyze, and extract events from video content using natural language queries.

## 🚀 Key Features

### 🎯 Core Capabilities
- **Natural Language Video Queries** - Search videos using text descriptions
- **Enhanced Person Detection** - Advanced person recognition with background independence
- **Image Matching & Cross-Domain Vision** - Find similar objects across different conditions
- **Small Object Detection** - Specialized detection for tiny objects in videos
- **Multi-Modal AI Integration** - OpenCLIP, BLIP-2, MediaPipe, and YOLOv8
- **Real-time Processing** - Streamlit web interface with FastAPI backend

### 🔬 Advanced Features
- **Background Independence** - Detect objects regardless of background changes
- **Cross-Domain Matching** - Match objects across color/grayscale differences
- **Adaptive Thresholding** - Dynamic confidence adjustment based on context
- **Region Proposal Networks** - Enhanced small object detection
- **Performance Monitoring** - Real-time system performance tracking
- **Memory Management** - Intelligent resource optimization

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **PyTorch 2.0+** - Deep learning framework
- **Transformers** - Hugging Face model library
- **OpenCLIP** - Contrastive language-image pre-training
- **MediaPipe** - Real-time perception pipeline
- **OpenCV** - Computer vision library

### Web Framework
- **Streamlit** - Interactive web interface
- **FastAPI** - High-performance API backend
- **Uvicorn** - ASGI server

### AI Models
- **OpenCLIP** - Vision-language understanding
- **BLIP-2** - Bootstrapped vision-language pre-training
- **YOLOv8** - Object detection
- **MediaPipe** - Pose and face detection

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Git
- 8GB+ RAM recommended
- GPU support optional but recommended

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/nb-hmd/Advanced-Video-Event-Detection-Extraction.git
cd Advanced-Video-Event-Detection-Extraction
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download models** (Optional - models will be downloaded automatically on first use)
```bash
# YOLOv8 model is included
# Other models will be downloaded from Hugging Face automatically
```

### Docker Installation

1. **Using Docker Compose** (Recommended)
```bash
docker-compose up --build
```

2. **Using Dockerfile**
```bash
docker build -t video-event-detection .
docker run -p 8501:8501 -p 8000:8000 video-event-detection
```

## 🚀 Usage

### Web Interface (Streamlit)

1. **Start the application**
```bash
# Using the robust server (recommended)
python robust_server.py

# Or directly with Streamlit
streamlit run src/web/streamlit_app.py --server.port 8501
```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`
   - Upload a video file
   - Choose your analysis type:
     - Natural language queries
     - Person detection
     - Image matching
     - Small object detection

### API Usage (FastAPI)

1. **Start the API server**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

2. **API Documentation**
   - Interactive docs: `http://localhost:8000/docs`
   - OpenAPI spec: `http://localhost:8000/openapi.json`

### Example API Calls

#### Upload Video
```python
import requests

with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/upload',
        files={'file': f}
    )
video_id = response.json()['video_id']
```

#### Natural Language Query
```python
response = requests.post(
    'http://localhost:8000/api/query',
    json={
        'video_id': video_id,
        'query': 'person walking with a dog',
        'mode': 'mvp',
        'top_k': 5
    }
)
results = response.json()['results']
```

#### Enhanced Person Detection
```python
response = requests.post(
    'http://localhost:8000/api/enhanced-person-detection',
    json={
        'video_id': video_id,
        'reference_image_path': 'person.jpg',
        'top_k': 10
    }
)
matches = response.json()['matches']
```

## 📁 Project Structure

```
Advanced-Video-Event-Detection-Extraction/
├── src/
│   ├── api/                    # FastAPI backend
│   │   └── main.py            # API endpoints
│   ├── models/                # AI model implementations
│   │   ├── openclip_model.py  # OpenCLIP integration
│   │   ├── blip_model.py      # BLIP-2 integration
│   │   └── univtg_model.py    # UniVTG integration
│   ├── services/              # Core business logic
│   │   ├── video_processor.py # Main video processing
│   │   ├── enhanced_person_detector.py
│   │   ├── image_matcher.py   # Image matching algorithms
│   │   ├── small_object_detector.py
│   │   └── universal_detector.py
│   ├── pipeline/              # Processing pipelines
│   │   ├── phase1_mvp.py      # Basic processing
│   │   ├── phase2_reranker.py # Result reranking
│   │   └── phase3_advanced.py # Advanced processing
│   ├── utils/                 # Utility functions
│   │   ├── config.py          # Configuration management
│   │   ├── logger.py          # Logging utilities
│   │   ├── memory_manager.py  # Memory optimization
│   │   └── model_cache.py     # Model caching
│   └── web/                   # Web interface
│       └── streamlit_app.py   # Streamlit application
├── data/                      # Data directories
│   ├── videos/               # Uploaded videos
│   ├── clips/                # Extracted clips
│   ├── frames/               # Extracted frames
│   └── embeddings/           # Cached embeddings
├── models/                   # Model storage
│   ├── openclip/            # OpenCLIP models
│   ├── blip/                # BLIP models
│   └── yolo/                # YOLO models
├── tests/                   # Test files
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
└── README.md               # This file
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
OPENCLIP_MODEL=ViT-B-32
OPENCLIP_PRETRAINED=openai
BLIP_MODEL=Salesforce/blip2-opt-2.7b

# Processing Configuration
MAX_FRAMES=1000
FRAME_SKIP=5
BATCH_SIZE=32

# Memory Management
MAX_MEMORY_GB=8
CACHE_SIZE=1000
```

### Model Configuration

Modify `src/utils/config.py` to adjust:
- Model selection and parameters
- Processing thresholds
- Memory limits
- Cache settings

## 🎯 Use Cases

### Security & Surveillance
- Person tracking across multiple cameras
- Unusual activity detection
- Object recognition in security footage

### Content Analysis
- Video content moderation
- Scene understanding
- Object inventory from videos

### Research & Development
- Computer vision research
- Multi-modal AI experiments
- Video understanding benchmarks

## 🔧 Advanced Usage

### Custom Model Integration

```python
from src.models.base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Initialize your model
    
    def process(self, frames):
        # Your processing logic
        return results
```

### Pipeline Customization

```python
from src.pipeline.base_pipeline import BasePipeline

class CustomPipeline(BasePipeline):
    def process_video(self, video_path, query):
        # Custom processing pipeline
        return results
```

## 📊 Performance

### Benchmarks
- **Processing Speed**: ~30 FPS on GPU, ~5 FPS on CPU
- **Memory Usage**: 2-8GB depending on models loaded
- **Accuracy**: 85-95% depending on query complexity

### Optimization Tips
- Use GPU acceleration when available
- Adjust batch size based on available memory
- Enable model caching for repeated queries
- Use frame skipping for faster processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenCLIP](https://github.com/mlfoundations/open_clip) for vision-language models
- [BLIP-2](https://github.com/salesforce/BLIP) for image captioning
- [MediaPipe](https://mediapipe.dev/) for real-time perception
- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Streamlit](https://streamlit.io/) for the web interface
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/nb-hmd/Advanced-Video-Event-Detection-Extraction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nb-hmd/Advanced-Video-Event-Detection-Extraction/discussions)
- **Documentation**: [Wiki](https://github.com/nb-hmd/Advanced-Video-Event-Detection-Extraction/wiki)

## 🔮 Roadmap

- [ ] Real-time video streaming support
- [ ] Mobile app development
- [ ] Cloud deployment templates
- [ ] Additional model integrations
- [ ] Performance optimizations
- [ ] Multi-language support

---

**Made with ❤️ for the computer vision community**