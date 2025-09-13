import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import torch
from PIL import Image
import gc
import time
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
import logging
import psutil
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref

# Core imports
try:
    import open_clip
    from transformers import (
        AutoProcessor, AutoModel, AutoTokenizer,
        OwlViTProcessor, OwlViTForObjectDetection,
        BlipProcessor, BlipForConditionalGeneration
    )
    from sentence_transformers import SentenceTransformer
    import timm
    ADVANCED_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced models not available: {e}")
    ADVANCED_MODELS_AVAILABLE = False

# Optional advanced imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.config import settings
from ..utils.memory_manager import memory_manager
from ..utils.model_cache import model_cache
from .background_independent_detector import BackgroundIndependentDetector
from .adaptive_threshold_system import AdaptiveThresholdSystem, DetectionContext
from .small_object_detector import SmallObjectDetector
from .region_proposal_network import RegionProposalNetwork

logger = get_logger(__name__)

class UniversalDetector:
    """
    Universal Object Detection Service - Unlimited Object Classes
    
    This revolutionary service provides unlimited object detection capabilities
    using state-of-the-art open-vocabulary models. It can detect ANY object
    described in natural language, breaking free from traditional class limitations.
    
    Features:
    1. Open-Vocabulary Detection - Detect ANY object using natural language
    2. OWL-ViT Integration - Vision-language model for zero-shot detection
    3. CLIP-based Recognition - Universal image-text understanding
    4. Custom Object Queries - "Find a red bicycle with a basket"
    5. Multi-Scale Detection - From tiny objects to large scenes
    6. Real-time Processing - Optimized for speed and accuracy
    7. Unlimited Classes - No predefined class limitations
    
    Supported Query Types:
    - Simple objects: "person", "car", "dog"
    - Complex descriptions: "person wearing red shirt", "blue car with open door"
    - Scene elements: "traffic light", "street sign", "building window"
    - Custom objects: "wooden chair", "metal fence", "glass bottle"
    - Any imaginable object described in natural language
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Model instances - lazy loaded
        self.owlvit_processor = None
        self.owlvit_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.sentence_transformer = None
        self.yolo_model = None
        
        # Configuration
        self.detection_confidence = 0.1  # Lower threshold for open-vocab
        self.nms_threshold = 0.3
        self.max_detections = 100
        
        # Enhanced small object detection components
        self.background_detector = None  # Lazy loaded
        self.adaptive_threshold_system = None  # Lazy loaded
        self.small_object_detector = None  # Lazy loaded
        self.region_proposal_network = None  # Lazy loaded
        
        # Small object detection settings
        self.small_object_detection_enabled = getattr(settings, 'SMALL_OBJECT_DETECTION_ENABLED', True)
        self.background_independence_enabled = getattr(settings, 'BACKGROUND_INDEPENDENCE_ENABLED', True)
        self.adaptive_thresholds_enabled = getattr(settings, 'ADAPTIVE_THRESHOLDS_ENABLED', True)
        self.rpn_enabled = getattr(settings, 'RPN_ENABLED', True)
        
        # Cache for processed queries and results
        self.query_cache = {}
        self.detection_cache = {}
        self.enable_caching = True
        self.cache_size_limit = getattr(settings, 'DETECTION_CACHE_SIZE', 100)
        
        # Performance optimization features
        self.model_cache = weakref.WeakValueDictionary()  # Weak references for model caching
        self.memory_threshold = getattr(settings, 'MEMORY_THRESHOLD_GB', 8.0)  # GB
        self.gpu_memory_threshold = getattr(settings, 'GPU_MEMORY_THRESHOLD_GB', 6.0)  # GB
        self.batch_size = getattr(settings, 'DETECTION_BATCH_SIZE', 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=2)  # For parallel processing
        self.model_lock = threading.Lock()  # Thread safety for model loading
        
        # Memory monitoring
        self.last_memory_check = 0
        self.memory_check_interval = 30  # seconds
        
        # Supported detection modes
        self.detection_modes = {
            'owlvit': 'OWL-ViT zero-shot detection',
            'clip': 'CLIP-based recognition',
            'hybrid': 'Combined OWL-ViT + CLIP',
            'yolo_enhanced': 'YOLO + open-vocabulary enhancement'
        }
        
        # GPU optimization
        if self.use_gpu:
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        logger.info(f"UniversalDetector initialized on {self.device}")
        logger.info(f"Available detection modes: {list(self.detection_modes.keys())}")
        logger.info(f"Performance optimizations: GPU={self.use_gpu}, Batch={self.batch_size}, Cache={self.cache_size_limit}")
        logger.info(f"Small object enhancements: Detection={self.small_object_detection_enabled}, Background={self.background_independence_enabled}, Adaptive={self.adaptive_thresholds_enabled}, RPN={self.rpn_enabled}")
        self._log_system_resources()
    
    def _log_system_resources(self):
        """Log current system resource usage."""
        try:
            # CPU and RAM info
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_used_gb = memory.used / (1024**3)
            
            logger.info(f"System Resources - CPU: {cpu_percent}%, RAM: {memory_used_gb:.1f}/{memory_gb:.1f}GB ({memory.percent}%)")
            
            # GPU info if available
            if self.use_gpu and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)
                logger.info(f"GPU Resources - Allocated: {gpu_allocated:.1f}GB, Cached: {gpu_cached:.1f}GB, Total: {gpu_memory:.1f}GB")
        except Exception as e:
            logger.warning(f"Could not log system resources: {e}")
    
    def _check_memory_usage(self):
        """Monitor and manage memory usage."""
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return
        
        self.last_memory_check = current_time
        
        try:
            # Check system memory
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            
            if memory_used_gb > self.memory_threshold:
                logger.warning(f"High memory usage detected: {memory_used_gb:.1f}GB > {self.memory_threshold}GB")
                self._cleanup_memory()
            
            # Check GPU memory if available
            if self.use_gpu and torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                if gpu_allocated > self.gpu_memory_threshold:
                    logger.warning(f"High GPU memory usage: {gpu_allocated:.1f}GB > {self.gpu_memory_threshold}GB")
                    self._cleanup_gpu_memory()
                    
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")
    
    def _cleanup_memory(self):
        """Clean up system memory."""
        try:
            # Clear detection cache if it's too large
            if len(self.detection_cache) > self.cache_size_limit // 2:
                cache_keys = list(self.detection_cache.keys())
                for key in cache_keys[:len(cache_keys)//2]:
                    del self.detection_cache[key]
                logger.info(f"Cleared {len(cache_keys)//2} cache entries")
            
            # Force garbage collection
            gc.collect()
            logger.info("Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory."""
        if not self.use_gpu:
            return
            
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleanup completed")
        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {e}")
    
    def _load_small_object_components(self):
        """Lazy load small object detection components"""
        try:
            if self.background_independence_enabled and self.background_detector is None:
                self.background_detector = BackgroundIndependentDetector({
                    'enable_caching': True,
                    'cache_size_limit': getattr(settings, 'BACKGROUND_INDEPENDENT_CACHE_SIZE', 50)
                })
                logger.info("Background independent detector loaded")
            
            if self.adaptive_thresholds_enabled and self.adaptive_threshold_system is None:
                self.adaptive_threshold_system = AdaptiveThresholdSystem({
                    'optimization_enabled': getattr(settings, 'THRESHOLD_OPTIMIZATION_ENABLED', True)
                })
                logger.info("Adaptive threshold system loaded")
            
            if self.small_object_detection_enabled and self.small_object_detector is None:
                self.small_object_detector = SmallObjectDetector({
                    'fcos_rt_path': getattr(settings, 'FCOS_RT_MODEL_PATH', 'models/fcos_rt_small.pth'),
                    'retinanet_path': getattr(settings, 'RETINANET_SMALL_MODEL_PATH', 'models/retinanet_small.pth'),
                    'yolov8_path': getattr(settings, 'YOLOV8_NANO_MODEL_PATH', 'models/yolov8n_small.pt'),
                    'enable_caching': True,
                    'cache_size_limit': getattr(settings, 'SMALL_OBJECT_CACHE_SIZE', 100)
                })
                logger.info("Small object detector loaded")
            
            if self.rpn_enabled and self.region_proposal_network is None:
                self.region_proposal_network = RegionProposalNetwork({
                    'max_proposals_per_frame': getattr(settings, 'MAX_PROPOSALS_PER_FRAME', 100),
                    'nms_threshold': getattr(settings, 'PROPOSAL_NMS_THRESHOLD', 0.3),
                    'min_area': getattr(settings, 'MIN_PROPOSAL_AREA', 64),
                    'max_area': getattr(settings, 'MAX_PROPOSAL_AREA', 10000)
                })
                logger.info("Region proposal network loaded")
                
        except Exception as e:
            logger.error(f"Failed to load small object components: {e}")
    
    def _create_detection_context(self, image: np.ndarray, previous_detections: List = None) -> DetectionContext:
        """Create detection context for adaptive thresholds"""
        try:
            # Analyze image properties
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect motion (simplified)
            motion_detected = False
            if hasattr(self, '_previous_frame') and self._previous_frame is not None:
                diff = cv2.absdiff(gray, self._previous_frame)
                motion_score = np.mean(diff) / 255.0
                motion_detected = motion_score > 0.1
            
            # Detect noise level
            noise_level = np.std(gray) / 255.0
            high_noise = noise_level > 0.15
            
            # Estimate lighting condition
            mean_brightness = np.mean(gray) / 255.0
            if mean_brightness < 0.3:
                lighting_condition = "low"
            elif mean_brightness > 0.7:
                lighting_condition = "high"
            else:
                lighting_condition = "normal"
            
            # Estimate scene complexity
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            if edge_density > 0.1:
                scene_complexity = "high"
            elif edge_density > 0.05:
                scene_complexity = "medium"
            else:
                scene_complexity = "low"
            
            # Calculate frame quality
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            frame_quality = min(laplacian_var / 1000.0, 1.0)  # Normalize
            
            # Calculate temporal consistency
            temporal_consistency = 1.0
            if previous_detections and hasattr(self, '_previous_detections'):
                # Simple consistency check
                if len(previous_detections) > 0 and len(self._previous_detections) > 0:
                    consistency_score = min(len(previous_detections), len(self._previous_detections)) / max(len(previous_detections), len(self._previous_detections))
                    temporal_consistency = consistency_score
            
            # Store for next frame
            self._previous_frame = gray.copy()
            if previous_detections:
                self._previous_detections = previous_detections.copy()
            
            return DetectionContext(
                motion_detected=motion_detected,
                high_noise=high_noise,
                lighting_condition=lighting_condition,
                scene_complexity=scene_complexity,
                frame_quality=frame_quality,
                temporal_consistency=temporal_consistency
            )
            
        except Exception as e:
            logger.error(f"Failed to create detection context: {e}")
            return DetectionContext()
    
    @lru_cache(maxsize=32)
    def _get_cached_model_config(self, model_name: str) -> Dict:
        """Get cached model configuration."""
        return {
            'owlvit': {'model_name': 'google/owlvit-base-patch32', 'precision': torch.float16 if self.use_gpu else torch.float32},
            'clip': {'model_name': 'openai/clip-vit-base-patch32', 'precision': torch.float16 if self.use_gpu else torch.float32}
        }.get(model_name, {})
    
    def _lazy_load_owlvit(self):
        """Lazy load OWL-ViT model for zero-shot object detection with optimizations."""
        if not ADVANCED_MODELS_AVAILABLE:
            logger.warning("Advanced models not available - OWL-ViT disabled")
            return False
            
        with self.model_lock:  # Thread safety
            if self.owlvit_model is None:
                try:
                    # Check memory before loading
                    self._check_memory_usage()
                    
                    logger.info("Loading OWL-ViT model for zero-shot detection...")
                    start_time = time.time()
                    
                    # Check if model is in cache
                    cache_key = "owlvit_model"
                    if cache_key in self.model_cache:
                        self.owlvit_model = self.model_cache[cache_key]
                        logger.info("OWL-ViT model loaded from cache")
                    else:
                        config = self._get_cached_model_config('owlvit')
                        
                        self.owlvit_processor = OwlViTProcessor.from_pretrained(
                            config['model_name'],
                            torch_dtype=config['precision']
                        )
                        self.owlvit_model = OwlViTForObjectDetection.from_pretrained(
                            config['model_name'],
                            torch_dtype=config['precision']
                        )
                        
                        # Cache the model
                        self.model_cache[cache_key] = self.owlvit_model
                    
                    if self.use_gpu:
                        self.owlvit_model = self.owlvit_model.to(self.device)
                        # Use mixed precision for better performance
                        if hasattr(torch.cuda, 'amp'):
                            self.owlvit_model = torch.jit.optimize_for_inference(self.owlvit_model)
                    
                    self.owlvit_model.eval()
                    
                    load_time = time.time() - start_time
                    logger.info(f"OWL-ViT model loaded successfully in {load_time:.2f}s")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error loading OWL-ViT model: {e}")
                    return False
            return True
    
    def _lazy_load_clip(self):
        """Lazy load CLIP model for universal recognition."""
        if self.clip_model is None:
            try:
                logger.info("Loading CLIP model for universal recognition...")
                
                # Load OpenCLIP model
                model_name = "ViT-B-32"
                pretrained = "openai"
                
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device=self.device
                )
                self.clip_tokenizer = open_clip.get_tokenizer(model_name)
                
                self.clip_model.eval()
                
                logger.info("CLIP model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading CLIP model: {e}")
                return False
        return True
    
    def _lazy_load_sentence_transformer(self):
        """Lazy load sentence transformer for text understanding."""
        if self.sentence_transformer is None:
            try:
                logger.info("Loading sentence transformer...")
                
                # Load sentence transformer for text similarity
                model_name = "all-MiniLM-L6-v2"
                self.sentence_transformer = SentenceTransformer(model_name)
                
                if self.use_gpu:
                    self.sentence_transformer = self.sentence_transformer.to(self.device)
                
                logger.info("Sentence transformer loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading sentence transformer: {e}")
                return False
        return True
    
    def _lazy_load_yolo(self):
        """Lazy load YOLO model for enhanced detection."""
        if not YOLO_AVAILABLE:
            return False
            
        if self.yolo_model is None:
            try:
                logger.info("Loading YOLO model for enhanced detection...")
                
                # Use YOLOv8x for maximum accuracy in universal detection
                model_size = getattr(settings, 'YOLO_MODEL_SIZE', 'x')
                self.yolo_model = YOLO(f'yolov8{model_size}.pt')
                
                if self.use_gpu:
                    self.yolo_model.to(self.device)
                
                logger.info(f"YOLO model loaded: yolov8{model_size}")
                return True
            except Exception as e:
                logger.error(f"Error loading YOLO model: {e}")
                return False
        return True
    
    def detect_objects_owlvit(self, image: np.ndarray, text_queries: List[str], 
                             confidence_threshold: float = 0.1) -> List[Dict]:
        """
        Detect objects using OWL-ViT zero-shot detection.
        
        Args:
            image: Input image as numpy array
            text_queries: List of text descriptions of objects to detect
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detection dictionaries with bbox, confidence, and query info
        """
        if not self._lazy_load_owlvit():
            logger.warning("OWL-ViT not available - returning empty results")
            return []
        
        try:
            # Convert numpy to PIL
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Prepare inputs
            inputs = self.owlvit_processor(
                text=text_queries, 
                images=pil_image, 
                return_tensors="pt"
            )
            
            if self.use_gpu:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run detection
            with torch.no_grad():
                outputs = self.owlvit_model(**inputs)
            
            # Process results
            detections = []
            
            # Get image size
            img_height, img_width = image.shape[:2]
            
            # Process each query
            for query_idx, query in enumerate(text_queries):
                # Get predictions for this query
                logits = outputs.logits[0, query_idx]  # [num_boxes]
                boxes = outputs.pred_boxes[0, query_idx]  # [num_boxes, 4]
                
                # Apply confidence threshold
                scores = torch.sigmoid(logits)
                valid_indices = scores > confidence_threshold
                
                if valid_indices.sum() > 0:
                    valid_scores = scores[valid_indices]
                    valid_boxes = boxes[valid_indices]
                    
                    # Convert boxes to pixel coordinates
                    for score, box in zip(valid_scores, valid_boxes):
                        # OWL-ViT returns normalized coordinates
                        x_center, y_center, width, height = box.cpu().numpy()
                        
                        # Convert to pixel coordinates
                        x_center *= img_width
                        y_center *= img_height
                        width *= img_width
                        height *= img_height
                        
                        # Convert to x1, y1, x2, y2 format
                        x1 = max(0, x_center - width / 2)
                        y1 = max(0, y_center - height / 2)
                        x2 = min(img_width, x_center + width / 2)
                        y2 = min(img_height, y_center + height / 2)
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(score.cpu().numpy()),
                            'query': query,
                            'query_index': query_idx,
                            'method': 'owlvit',
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        detections.append(detection)
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"OWL-ViT detected {len(detections)} objects for queries: {text_queries}")
            return detections[:self.max_detections]
            
        except Exception as e:
            logger.error(f"Error in OWL-ViT detection: {e}")
            return []
    
    def detect_objects_clip(self, image: np.ndarray, text_queries: List[str],
                           grid_size: int = 8) -> List[Dict]:
        """
        Detect objects using CLIP-based sliding window approach.
        
        Args:
            image: Input image as numpy array
            text_queries: List of text descriptions
            grid_size: Size of sliding window grid
            
        Returns:
            List of detection dictionaries
        """
        if not self._lazy_load_clip():
            logger.warning("CLIP not available - returning empty results")
            return []
        
        try:
            # Convert to PIL
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            img_width, img_height = pil_image.size
            detections = []
            
            # Create sliding windows
            window_width = img_width // grid_size
            window_height = img_height // grid_size
            
            # Encode text queries
            text_tokens = self.clip_tokenizer(text_queries).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Process each window
            for i in range(grid_size):
                for j in range(grid_size):
                    # Define window coordinates
                    x1 = i * window_width
                    y1 = j * window_height
                    x2 = min((i + 1) * window_width, img_width)
                    y2 = min((j + 1) * window_height, img_height)
                    
                    # Extract window
                    window = pil_image.crop((x1, y1, x2, y2))
                    
                    # Preprocess window
                    window_tensor = self.clip_preprocess(window).unsqueeze(0).to(self.device)
                    
                    # Encode image
                    with torch.no_grad():
                        image_features = self.clip_model.encode_image(window_tensor)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarities
                    similarities = (image_features @ text_features.T).cpu().numpy()[0]
                    
                    # Check each query
                    for query_idx, (query, similarity) in enumerate(zip(text_queries, similarities)):
                        if similarity > 0.2:  # Threshold for CLIP similarity
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(similarity),
                                'query': query,
                                'query_index': query_idx,
                                'method': 'clip',
                                'area': (x2 - x1) * (y2 - y1)
                            }
                            detections.append(detection)
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"CLIP detected {len(detections)} objects for queries: {text_queries}")
            return detections[:self.max_detections]
            
        except Exception as e:
            logger.error(f"Error in CLIP detection: {e}")
            return []
    
    def detect_unlimited_objects(self, image: np.ndarray, 
                               text_queries: Union[str, List[str]],
                               detection_mode: str = 'hybrid',
                               confidence_threshold: float = 0.1,
                               batch_processing: bool = True) -> List[Dict]:
        """
        Universal object detection with unlimited classes and performance optimizations.
        
        Args:
            image: Input image as numpy array
            text_queries: Single query string or list of query strings
            detection_mode: 'owlvit', 'clip', 'hybrid', or 'yolo_enhanced'
            confidence_threshold: Minimum confidence threshold
            batch_processing: Enable batch processing for multiple queries
            
        Returns:
            List of detection dictionaries with unlimited object classes
        """
        try:
            start_time = time.time()
            
            # Monitor memory usage
            self._check_memory_usage()
            
            # Load small object detection components if needed
            self._load_small_object_components()
            
            # Normalize input
            if isinstance(text_queries, str):
                text_queries = [text_queries]
            
            # Check cache first
            if self.enable_caching:
                cache_key = self._create_cache_key(image, text_queries, detection_mode)
                if cache_key in self.detection_cache:
                    logger.debug(f"Cache hit for detection: {cache_key}")
                    return self.detection_cache[cache_key]
            
            # Create detection context for adaptive thresholds
            detection_context = None
            if self.adaptive_thresholds_enabled and self.adaptive_threshold_system:
                detection_context = self._create_detection_context(image)
            
            # Apply region proposal network if enabled
            roi_regions = None
            if self.rpn_enabled and self.region_proposal_network:
                try:
                    roi_regions = self.region_proposal_network.generate_proposals(image)
                    logger.debug(f"Generated {len(roi_regions)} region proposals")
                except Exception as e:
                    logger.warning(f"RPN failed, using full image: {e}")
            
            # Apply background independence if enabled
            processed_image = image
            if self.background_independence_enabled and self.background_detector:
                try:
                    processed_image = self.background_detector.remove_background_bias(image)
                    logger.debug("Applied background independence processing")
                except Exception as e:
                    logger.warning(f"Background independence failed, using original image: {e}")
            
            all_detections = []
            
            # Batch processing for multiple queries
            if batch_processing and len(text_queries) > self.batch_size:
                all_detections = self._process_queries_in_batches(processed_image, text_queries, detection_mode, confidence_threshold)
            else:
                all_detections = self._process_single_batch(processed_image, text_queries, detection_mode, confidence_threshold)
            
            # Apply small object detection if enabled
            if self.small_object_detection_enabled and self.small_object_detector:
                try:
                    small_object_detections = self.small_object_detector.detect_small_objects(
                        processed_image, 
                        roi_regions=roi_regions
                    )
                    
                    # Merge detections
                    all_detections = self._merge_detections(all_detections + small_object_detections)
                    logger.debug(f"Added {len(small_object_detections)} small object detections")
                except Exception as e:
                    logger.warning(f"Small object detection failed: {e}")
            
            # Apply adaptive thresholds if enabled
            if self.adaptive_thresholds_enabled and self.adaptive_threshold_system and detection_context:
                try:
                    all_detections = self.adaptive_threshold_system.apply_adaptive_thresholds(
                        all_detections, 
                        detection_context
                    )
                    logger.debug("Applied adaptive thresholds")
                except Exception as e:
                    logger.warning(f"Adaptive thresholds failed: {e}")
            
            # Filter by confidence threshold
            filtered_detections = [
                det for det in all_detections 
                if det.get('confidence', 0) >= confidence_threshold
            ]
            
            # Cache results
            if self.enable_caching and len(self.detection_cache) < self.cache_size_limit:
                self.detection_cache[cache_key] = filtered_detections
            
            processing_time = time.time() - start_time
            logger.info(f"Enhanced detection completed: {len(filtered_detections)} objects using {detection_mode} mode in {processing_time:.2f}s")
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Error in unlimited object detection: {e}")
            return []
    
    def _process_queries_in_batches(self, image: np.ndarray, text_queries: List[str], 
                                  detection_mode: str, confidence_threshold: float) -> List[Dict]:
        """Process queries in batches for better performance."""
        all_detections = []
        
        # Split queries into batches
        for i in range(0, len(text_queries), self.batch_size):
            batch_queries = text_queries[i:i + self.batch_size]
            batch_detections = self._process_single_batch(image, batch_queries, detection_mode, confidence_threshold)
            all_detections.extend(batch_detections)
            
            # Clean up GPU memory between batches
            if self.use_gpu and i > 0:
                torch.cuda.empty_cache()
        
        return all_detections
    
    def _process_single_batch(self, image: np.ndarray, text_queries: List[str], 
                            detection_mode: str, confidence_threshold: float) -> List[Dict]:
        """Process a single batch of queries."""
        all_detections = []
        
        if detection_mode == 'owlvit':
            if not self._lazy_load_owlvit():
                logger.error("Failed to load OWL-ViT model")
                return []
            
            detections = self.detect_objects_owlvit(image, text_queries, confidence_threshold)
            all_detections.extend(detections)
            
        elif detection_mode == 'clip':
            detections = self.detect_objects_clip(image, text_queries)
            all_detections.extend(detections)
            
        elif detection_mode == 'hybrid':
            # Use parallel processing for hybrid mode
            futures = []
            
            # Submit OWL-ViT detection
            if self._lazy_load_owlvit():
                future_owlvit = self.thread_pool.submit(self.detect_objects_owlvit, image, text_queries, confidence_threshold)
                futures.append(('owlvit', future_owlvit))
            
            # Submit CLIP detection
            future_clip = self.thread_pool.submit(self.detect_objects_clip, image, text_queries)
            futures.append(('clip', future_clip))
            
            # Collect results
            for method, future in futures:
                try:
                    detections = future.result(timeout=30)  # 30 second timeout
                    all_detections.extend(detections)
                except Exception as e:
                    logger.error(f"Error in {method} detection: {e}")
            
            # Merge and deduplicate
            all_detections = self._merge_detections(all_detections)
            
        elif detection_mode == 'yolo_enhanced':
            # Use YOLO for initial detection, then enhance with open-vocab
            yolo_detections = self._detect_with_yolo_enhanced(image, text_queries)
            all_detections.extend(yolo_detections)
        
        else:
            raise ValueError(f"Unknown detection mode: {detection_mode}")
        
        return all_detections
    
    def _detect_with_yolo_enhanced(self, image: np.ndarray, text_queries: List[str]) -> List[Dict]:
        """Enhanced YOLO detection with open-vocabulary matching."""
        if not self._lazy_load_yolo():
            return []
        
        try:
            # Run YOLO detection first
            results = self.yolo_model(image, conf=0.1, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        # Match against text queries using semantic similarity
                        for query in text_queries:
                            semantic_score = self._compute_semantic_similarity(class_name, query)
                            
                            if semantic_score > 0.3:  # Semantic similarity threshold
                                detection = {
                                    'bbox': bbox.tolist(),
                                    'confidence': confidence * semantic_score,  # Combined score
                                    'query': query,
                                    'detected_class': class_name,
                                    'semantic_score': semantic_score,
                                    'method': 'yolo_enhanced',
                                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                }
                                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in YOLO enhanced detection: {e}")
            return []
    
    def _compute_semantic_similarity(self, class_name: str, query: str) -> float:
        """Compute semantic similarity between class name and query."""
        if not self._lazy_load_sentence_transformer():
            return 0.0
        
        try:
            # Encode both texts
            embeddings = self.sentence_transformer.encode([class_name, query])
            
            # Compute cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def _post_process_detections(self, detections: List[Dict], confidence_threshold: float) -> List[Dict]:
        """
        Post-process detections with NMS and filtering.
        
        Args:
            detections: Raw detections
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Filtered and processed detections
        """
        if not detections:
            return []
        
        try:
            # Filter by confidence
            filtered_detections = [
                det for det in detections 
                if det.get('confidence', 0) >= confidence_threshold
            ]
            
            if not filtered_detections:
                return []
            
            # Apply NMS if we have multiple detections
            if len(filtered_detections) > 1:
                filtered_detections = self._apply_nms(filtered_detections)
            
            # Limit number of detections
            if len(filtered_detections) > self.max_detections:
                # Sort by confidence and take top detections
                filtered_detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                filtered_detections = filtered_detections[:self.max_detections]
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return detections
    
    def _merge_detections(self, standard_detections: List[Dict], small_object_detections: List[Dict] = None) -> List[Dict]:
        """
        Merge standard detections with small object detections, removing duplicates.
        
        Args:
            standard_detections: Detections from standard models
            small_object_detections: Detections from small object models
            
        Returns:
            Merged and deduplicated detections
        """
        if small_object_detections is None:
            small_object_detections = []
        
        if not standard_detections and not small_object_detections:
            return []
        
        try:
            all_detections = standard_detections + small_object_detections
            
            if len(all_detections) <= 1:
                return all_detections
            
            # Remove duplicates based on IoU threshold
            merged_detections = []
            iou_threshold = 0.5
            
            # Sort by confidence (highest first)
            all_detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            for detection in all_detections:
                is_duplicate = False
                
                for existing_detection in merged_detections:
                    # Calculate IoU
                    iou = self._calculate_iou(
                        detection.get('bbox', [0, 0, 0, 0]),
                        existing_detection.get('bbox', [0, 0, 0, 0])
                    )
                    
                    # Check if it's a duplicate
                    if iou > iou_threshold:
                        # Keep the one with higher confidence
                        if detection.get('confidence', 0) > existing_detection.get('confidence', 0):
                            # Replace existing with current (higher confidence)
                            merged_detections.remove(existing_detection)
                            merged_detections.append(detection)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    merged_detections.append(detection)
            
            logger.debug(f"Merged {len(all_detections)} detections into {len(merged_detections)} unique detections")
            return merged_detections
            
        except Exception as e:
            logger.error(f"Detection merging failed: {e}")
            return standard_detections + small_object_detections
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        try:
            # Extract coordinates
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection area
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # Calculate union area
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area
            
            if union_area <= 0:
                return 0.0
            
            return intersection_area / union_area
            
        except Exception as e:
            logger.error(f"IoU calculation failed: {e}")
            return 0.0
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute Intersection over Union of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_cache_key(self, image: np.ndarray, queries: List[str], mode: str) -> str:
        """Create cache key for detection results."""
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
        query_hash = hashlib.md5('|'.join(sorted(queries)).encode()).hexdigest()[:8]
        return f"detect_{image_hash}_{query_hash}_{mode}"
    
    def get_supported_queries(self) -> Dict[str, List[str]]:
        """Get examples of supported query types."""
        return {
            'simple_objects': [
                'person', 'car', 'dog', 'cat', 'bicycle', 'motorcycle',
                'bus', 'truck', 'airplane', 'boat', 'chair', 'table'
            ],
            'detailed_descriptions': [
                'person wearing red shirt', 'blue car with open door',
                'dog sitting on grass', 'cat sleeping on sofa',
                'bicycle with basket', 'motorcycle with rider'
            ],
            'scene_elements': [
                'traffic light', 'street sign', 'building window',
                'tree branch', 'road marking', 'fence post'
            ],
            'custom_objects': [
                'wooden chair', 'metal fence', 'glass bottle',
                'plastic container', 'leather bag', 'ceramic mug'
            ],
            'complex_scenes': [
                'person crossing street', 'car parked near building',
                'dog playing in park', 'children on playground'
            ]
        }
    
    def cleanup(self):
        """Clean up model resources and performance optimization components."""
        logger.info("Cleaning up UniversalDetector resources...")
        
        try:
            # Shutdown thread pool
            if hasattr(self, 'thread_pool') and self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                logger.info("Thread pool shutdown completed")
            
            # Clean up models
            models_to_cleanup = [
                'owlvit_model', 'owlvit_processor', 'clip_model', 
                'clip_preprocess', 'clip_tokenizer', 'sentence_transformer', 'yolo_model'
            ]
            
            for model_name in models_to_cleanup:
                if hasattr(self, model_name):
                    model = getattr(self, model_name)
                    if model is not None:
                        try:
                            if hasattr(model, 'cpu'):
                                model.cpu()
                            del model
                            setattr(self, model_name, None)
                            logger.info(f"Cleaned up {model_name}")
                        except Exception as e:
                            logger.warning(f"Error cleaning up {model_name}: {e}")
            
            # Clear all caches
            self.detection_cache.clear()
            self.model_cache.clear()
            
            # Clear LRU cache
            if hasattr(self._get_cached_model_config, 'cache_clear'):
                self._get_cached_model_config.cache_clear()
            
            # Final memory cleanup
            self._cleanup_memory()
            self._cleanup_gpu_memory()
            
            logger.info("UniversalDetector cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction