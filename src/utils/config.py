import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    
    # Video processing - High performance settings
    MAX_VIDEO_SIZE: int = 2 * 1024 * 1024 * 1024  # 2GB for complete video processing
    SUPPORTED_FORMATS: list = ["mp4", "avi", "mov", "mkv"]
    FRAME_SAMPLE_RATE: int = 1  # Extract every frame for maximum detail
    WINDOW_SIZE: int = 16  # More frames per window for better context
    WINDOW_STRIDE: int = 8  # Optimal stride for comprehensive coverage
    
    # Frame processing - High quality settings
    MAX_FRAME_WIDTH: int = 512  # Higher resolution for better accuracy
    MAX_FRAME_HEIGHT: int = 512  # Higher resolution for better accuracy
    FRAME_QUALITY: int = 95  # High quality for maximum detail
    MAX_WINDOWS_PER_BATCH: int = 32  # Larger batches for efficient processing
    
    # Model settings
    OPENCLIP_MODEL: str = "ViT-B-32"
    OPENCLIP_PRETRAINED: str = "openai"
    BLIP_MODEL: str = "Salesforce/blip-image-captioning-base"  # Smallest BLIP model for limited memory
    UNIVTG_MODEL: str = "univtg_qvhighlights"
    
    # Advanced matching model settings
    YOLO_MODEL_SIZE: str = "n"  # n, s, m, l, x (n=fastest, x=most accurate)
    SAM_MODEL_TYPE: str = "vit_b"  # vit_b, vit_l, vit_h
    FEATURE_EXTRACTOR_MODEL: str = "efficientnet_b0"
    BACKGROUND_REMOVAL_MODEL: str = "u2net"
    
    # Processing - High performance settings
    BATCH_SIZE: int = 32  # Optimal batch size for maximum throughput
    TOP_K_RESULTS: int = 15  # More results for comprehensive analysis
    CONFIDENCE_THRESHOLD: float = 0.25  # Balanced threshold for quality results
    CLIP_DURATION: int = 30  # Longer clips for complete event coverage
    
    # Advanced matching settings
    MATCHING_MODES: list = ["traditional", "object_focused", "cross_domain", "hybrid", "unlimited"]
    DEFAULT_MATCHING_MODE: str = "traditional"
    
    # Object detection settings
    OBJECT_DETECTION_CONFIDENCE: float = 0.25
    OBJECT_IOU_THRESHOLD: float = 0.45
    MAX_DETECTIONS_PER_FRAME: int = 100
    SUPPORTED_OBJECT_CLASSES: list = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
    
    # Universal/Unlimited Object Detection Settings
    UNLIMITED_DETECTION_ENABLED: bool = True
    UNIVERSAL_DETECTION_MODES: list = ["owlvit", "clip", "hybrid", "yolo_enhanced"]
    DEFAULT_UNIVERSAL_MODE: str = "hybrid"
    
    # Open-Vocabulary Model Settings
    OWLVIT_MODEL: str = "google/owlvit-base-patch32"
    OWLVIT_CONFIDENCE_THRESHOLD: float = 0.1
    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "openai"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    
    # Universal Detection Thresholds
    UNIVERSAL_CONFIDENCE_THRESHOLD: float = 0.1
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.3
    VISUAL_QUALITY_THRESHOLD: float = 0.2
    
    # Open-Vocabulary Matching Settings
    OPEN_VOCAB_MATCHING_MODES: list = ["precise", "balanced", "comprehensive", "semantic", "visual"]
    DEFAULT_OPEN_VOCAB_MODE: str = "balanced"
    
    # Query Processing Settings
    MAX_QUERY_LENGTH: int = 200
    MAX_QUERIES_PER_REQUEST: int = 10
    ENABLE_QUERY_SUGGESTIONS: bool = True
    ENABLE_AUTO_COMPLETE: bool = True
    
    # Small Object Detection Enhancement Settings
    SMALL_OBJECT_DETECTION_ENABLED: bool = True
    
    # Background Independence Settings
    BACKGROUND_INDEPENDENCE_ENABLED: bool = True
    SAM_MODEL_VERSION: str = "sam2_hiera_large"
    CONTRASTIVE_LEARNING_ENABLED: bool = True
    SHAPE_DESCRIPTORS_ENABLED: bool = True
    COLOR_NORMALIZATION_SPACES: list = ["hsv", "lab", "yuv"]
    BACKGROUND_REMOVAL_MODEL: str = "u2net"
    
    # Adaptive Threshold Settings
    ADAPTIVE_THRESHOLDS_ENABLED: bool = True
    SIZE_BASED_THRESHOLD_MAPPING: dict = {
        'tiny': 0.05,
        'small': 0.1,
        'medium': 0.25,
        'large': 0.4
    }
    SIZE_CATEGORIES: dict = {
        'tiny': {'min_area': 0, 'max_area': 32*32},
        'small': {'min_area': 32*32, 'max_area': 96*96},
        'medium': {'min_area': 96*96, 'max_area': 256*256},
        'large': {'min_area': 256*256, 'max_area': None}
    }
    CONFIDENCE_BOOSTERS: dict = {
        'tiny': 2.0,
        'small': 1.5,
        'medium': 1.0,
        'large': 1.0
    }
    TEMPORAL_CONSISTENCY_WINDOW: int = 10
    THRESHOLD_OPTIMIZATION_ENABLED: bool = True
    
    # Small Object Detection Models
    SMALL_OBJECT_MODELS: list = ['fcos_rt', 'retinanet_small', 'yolov8_nano']
    MULTI_SCALE_PROCESSING: list = [256, 512, 1024]
    SCALE_WEIGHTS: dict = {256: 1.2, 512: 1.0, 1024: 0.8}
    
    # Model Paths and Configurations
    FCOS_RT_MODEL_PATH: str = "models/fcos_rt_small.pth"
    RETINANET_SMALL_MODEL_PATH: str = "models/retinanet_small.pth"
    YOLOV8_NANO_MODEL_PATH: str = "models/yolov8n_small.pt"
    
    # Model-specific settings
    FCOS_RT_CONFIG: dict = {
        'input_size': (512, 512),
        'confidence_threshold': 0.05,
        'nms_threshold': 0.3,
        'specialization': 'tiny_objects'
    }
    RETINANET_SMALL_CONFIG: dict = {
        'input_size': (640, 640),
        'confidence_threshold': 0.1,
        'nms_threshold': 0.4,
        'specialization': 'small_objects'
    }
    YOLOV8_NANO_CONFIG: dict = {
        'input_size': (416, 416),
        'confidence_threshold': 0.15,
        'nms_threshold': 0.45,
        'specialization': 'fast_small_objects'
    }
    
    # Region Proposal Network Settings
    RPN_ENABLED: bool = True
    MAX_PROPOSALS_PER_FRAME: int = 100
    PROPOSAL_NMS_THRESHOLD: float = 0.3
    MIN_PROPOSAL_AREA: int = 64
    MAX_PROPOSAL_AREA: int = 10000
    
    # Proposal weights
    SALIENCY_WEIGHT: float = 0.3
    MOTION_WEIGHT: float = 0.4
    RPN_WEIGHT: float = 0.3
    EDGE_WEIGHT: float = 0.2
    TEXTURE_WEIGHT: float = 0.1
    
    # Feature Pyramid Network Settings
    FPN_ENABLED: bool = True
    FPN_CHANNELS: list = [256, 512, 1024, 2048]
    FPN_OUT_CHANNELS: int = 256
    
    # Attention Mechanism Settings
    SPATIAL_ATTENTION_ENABLED: bool = True
    CHANNEL_ATTENTION_ENABLED: bool = True
    ATTENTION_REDUCTION_RATIO: int = 8
    
    # Performance and Caching Settings
    SMALL_OBJECT_CACHE_SIZE: int = 100
    BACKGROUND_INDEPENDENT_CACHE_SIZE: int = 50
    ADAPTIVE_THRESHOLD_CACHE_SIZE: int = 200
    RPN_CACHE_SIZE: int = 50
    
    # Memory Management
    SMALL_OBJECT_MEMORY_THRESHOLD_GB: float = 6.0
    MODEL_QUANTIZATION_ENABLED: bool = True
    PARALLEL_MODEL_PROCESSING: bool = True
    MAX_PARALLEL_MODELS: int = 3
    
    # Ensemble Settings
    ENSEMBLE_DETECTION_ENABLED: bool = True
    ENSEMBLE_VOTING_STRATEGY: str = "weighted_average"  # "majority", "weighted_average", "confidence_based"
    ENSEMBLE_CONFIDENCE_WEIGHTS: dict = {
        'fcos_rt': 0.4,
        'retinanet_small': 0.35,
        'yolov8_nano': 0.25
    }
    
    # Cross-domain matching settings
    CROSS_DOMAIN_COLOR_SPACES: list = ["RGB", "HSV", "LAB", "YUV", "GRAY"]
    CROSS_DOMAIN_FEATURE_METHODS: list = ["lbp", "hog", "orb", "sift", "edges", "texture"]
    
    # Similarity thresholds
    TRADITIONAL_SIMILARITY_THRESHOLD: float = 0.7
    OBJECT_SIMILARITY_THRESHOLD: float = 0.7
    CROSS_DOMAIN_SIMILARITY_THRESHOLD: float = 0.6
    HYBRID_SIMILARITY_THRESHOLD: float = 0.65
    
    # Memory management
    ENABLE_MEMORY_MONITORING: bool = True
    MIN_AVAILABLE_MEMORY_MB: int = 100  # Minimum memory before fallback
    MEMORY_CLEANUP_INTERVAL: int = 5  # Clean memory every N windows
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Redis (for Celery)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    class Config:
        env_file = ".env"

settings = Settings()