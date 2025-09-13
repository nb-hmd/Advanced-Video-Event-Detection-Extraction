import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
# from sentence_transformers import SentenceTransformer  # Temporarily commented out due to dependency issues
import numpy as np
from typing import List, Union
import os
import gc
import psutil
from ..utils.logger import get_logger
from ..utils.config import settings
from ..utils.memory_manager import memory_manager
from ..utils.model_cache import model_cache
from ..utils.error_handler import error_handler, handle_model_loading_error, handle_inference_error

logger = get_logger(__name__)

class BLIPModel:
    def __init__(self, lazy_load=True, force_device: str = None):
        # Use memory manager to determine optimal device
        if force_device:
            self.device = torch.device(force_device)
        else:
            optimal_device = memory_manager.get_optimal_device('blip2')
            self.device = torch.device(optimal_device)
        
        self.processor = None
        self.model = None
        self.sentence_model = None
        self.model_loaded = False
        self.memory_optimized = False
        self.lazy_load = lazy_load
        
        if not lazy_load:
            self.load_model()
        else:
            # Only load lightweight sentence transformer for text similarity
            # try:
            #     self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            #     logger.info("Loaded sentence transformer for text similarity (BLIP model will load lazily)")
            # except Exception as e:
            #     logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None  # Temporarily disabled due to dependency issues
    
    def load_model(self):
        """Load BLIP-2 model for captioning with fallback handling."""
        try:
            # Try to load sentence transformer first (lighter dependency)
            # self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            # logger.info("Loaded sentence transformer for text similarity")
            self.sentence_model = None  # Temporarily disabled due to dependency issues
            
            # Try to load BLIP model with error handling
            try:
                # Set environment variable to avoid symlink issues on Windows
                os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
                
                self.processor = Blip2Processor.from_pretrained(
                    settings.BLIP_MODEL,
                    local_files_only=False,
                    force_download=False,
                    resume_download=True
                )
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    settings.BLIP_MODEL,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    local_files_only=False,
                    force_download=False,
                    resume_download=True
                ).to(self.device)
                
                self.model_loaded = True
                logger.info(f"Successfully loaded BLIP model: {settings.BLIP_MODEL}")
                
            except Exception as blip_error:
                logger.warning(f"Failed to load BLIP model due to: {blip_error}")
                logger.info("BLIP model will be loaded lazily when first needed")
                self.model_loaded = False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.warning("Running in fallback mode - some features may be limited")
            self.model_loaded = False
    
    def _check_memory_availability(self) -> bool:
        """Check if sufficient memory is available for BLIP model loading."""
        try:
            # Get available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            # Reduced memory requirement for better compatibility
            required_gb = 0.5  # Much more conservative requirement
            
            logger.info(f"Available memory: {available_gb:.1f}GB, Required: {required_gb}GB")
            
            if available_gb < required_gb:
                logger.warning(f"Insufficient memory for BLIP model: {available_gb:.1f}GB < {required_gb}GB")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not check memory availability: {e}")
            return True  # Assume it's available if we can't check
    
    def _optimize_memory(self):
        """Optimize memory usage before loading BLIP model."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Set memory optimization flags
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            self.memory_optimized = True
            logger.info("Memory optimization completed")
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
    
    @handle_model_loading_error
    def _lazy_load_blip(self):
        """Lazy load BLIP model with caching and enhanced memory management."""
        if self.model_loaded:
            return True
        
        # Generate cache key based on model configuration
        model_config = {
            'model_name': settings.BLIP_MODEL,
            'device': str(self.device),
            'torch_dtype': 'float16' if self.device.type == 'cuda' else 'float32'
        }
        cache_key = f"blip2_{settings.BLIP_MODEL.replace('/', '_')}_{self.device.type}"
        
        try:
            # Try to load from cache first
            cached_components = model_cache.get_model(cache_key, model_config)
            
            if cached_components is not None:
                logger.info(f"Loading BLIP-2 model from cache: {cache_key}")
                self.processor = cached_components['processor']
                self.model = cached_components['model']
                self.model_loaded = True
                logger.info(f"Successfully loaded BLIP-2 model from cache on {self.device}")
                return True
            
            # Cache miss - load model normally
            logger.info(f"Cache miss for BLIP-2, loading fresh model: {cache_key}")
            
            # Check if model can be loaded with memory manager
            can_load, message = memory_manager.can_load_model('blip2', use_gpu=(self.device.type == 'cuda'))
            if not can_load:
                logger.warning(f"Memory check failed for BLIP-2: {message}")
                # Try CPU fallback
                self.device = torch.device('cpu')
                can_load, message = memory_manager.can_load_model('blip2', use_gpu=False)
                if not can_load:
                    logger.error(f"Cannot load BLIP-2 model: {message}")
                    return False
            
            logger.info(f"Loading BLIP-2 model on {self.device}: {message}")
            
            # Aggressive memory optimization for BLIP-2
            memory_manager.aggressive_cleanup()
            memory_manager.optimize_for_model_loading('blip2')
            
            # High performance settings for maximum accuracy
            os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow online fallback
            os.environ['HF_DATASETS_OFFLINE'] = '0'
            
            # Monitor memory during loading
            baseline = memory_manager.monitor_model_loading('blip2')
            
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            
            # Load processor first (lighter)
            logger.info("Loading BLIP processor...")
            
            # Use appropriate processor based on model type
            if 'blip2' in settings.BLIP_MODEL.lower():
                self.processor = Blip2Processor.from_pretrained(
                    settings.BLIP_MODEL,
                    local_files_only=False,
                    force_download=False,
                    resume_download=True
                )
            else:
                self.processor = BlipProcessor.from_pretrained(
                    settings.BLIP_MODEL,
                    local_files_only=False,
                    force_download=False,
                    resume_download=True
                )
            
            # Load model with memory optimization
            logger.info("Loading BLIP model with enhanced memory management...")
            
            # Use high precision for maximum accuracy
            torch_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
            
            model_kwargs = {
                'torch_dtype': torch_dtype,
                'local_files_only': False,
                'force_download': False,
                'resume_download': True,
                'low_cpu_mem_usage': False,  # Disable for high performance
                'load_in_8bit': False,  # Disable quantization for maximum accuracy
            }
            
            # Add device mapping for GPU
            if self.device.type == 'cuda':
                model_kwargs['device_map'] = 'auto'
            else:
                # For CPU, use high precision settings
                model_kwargs['torch_dtype'] = torch.float32  # Use float32 for CPU
                model_kwargs['low_cpu_mem_usage'] = False  # High performance mode
            
            # Use appropriate model class based on model type
            if 'blip2' in settings.BLIP_MODEL.lower():
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    settings.BLIP_MODEL,
                    **model_kwargs
                )
            else:
                self.model = BlipForConditionalGeneration.from_pretrained(
                    settings.BLIP_MODEL,
                    **model_kwargs
                )
            
            # Move to device if not using device_map
            if self.device.type == 'cpu':
                self.model = self.model.to(self.device)
            
            # Enable memory-efficient features
            if hasattr(self.model, 'enable_memory_efficient_attention'):
                self.model.enable_memory_efficient_attention()
            
            # Finalize memory monitoring
            usage = memory_manager.finalize_model_loading('blip2', baseline)
            
            # Cache the loaded components
            components_to_cache = {
                'processor': self.processor,
                'model': self.model
            }
            
            # Cache in memory only (models are too large for disk caching)
            model_cache.cache_model(cache_key, components_to_cache, model_config, persist_to_disk=False)
            
            self.model_loaded = True
            logger.info(f"Successfully loaded and cached BLIP-2 model on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BLIP-2 model: {e}")
            
            # Try emergency fallback
            try:
                logger.info("Attempting emergency CPU fallback for BLIP-2...")
                
                # Try to load fallback from cache first
                fallback_cache_key = f"blip2_{settings.BLIP_MODEL.replace('/', '_')}_cpu_fallback"
                fallback_config = {
                    'model_name': settings.BLIP_MODEL,
                    'device': 'cpu',
                    'torch_dtype': 'float32',
                    'fallback': True
                }
                
                cached_fallback = model_cache.get_model(fallback_cache_key, fallback_config)
                
                if cached_fallback is not None:
                    logger.info("Using cached fallback BLIP-2 model")
                    self.device = torch.device('cpu')
                    self.processor = cached_fallback['processor']
                    self.model = cached_fallback['model']
                    self.model_loaded = True
                    return True
                
                # Load fallback model fresh
                # Aggressive cleanup
                memory_manager.aggressive_cleanup()
                
                # Force CPU device
                self.device = torch.device("cpu")
                
                # Try with minimal settings
                if 'blip2' in settings.BLIP_MODEL.lower():
                    self.processor = Blip2Processor.from_pretrained(
                        settings.BLIP_MODEL,
                        local_files_only=False
                    )
                    
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        settings.BLIP_MODEL,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        local_files_only=False
                    ).to(self.device)
                else:
                    self.processor = BlipProcessor.from_pretrained(
                        settings.BLIP_MODEL,
                        local_files_only=False
                    )
                    
                    self.model = BlipForConditionalGeneration.from_pretrained(
                        settings.BLIP_MODEL,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        local_files_only=False
                    ).to(self.device)
                
                # Cache the fallback model
                fallback_components = {
                    'processor': self.processor,
                    'model': self.model
                }
                model_cache.cache_model(fallback_cache_key, fallback_components, fallback_config, persist_to_disk=False)
                
                self.model_loaded = True
                logger.info("Emergency CPU fallback successful and cached for BLIP-2")
                return True
                
            except Exception as fallback_error:
                logger.error(f"Emergency fallback also failed: {fallback_error}")
                return False
    
    @handle_inference_error
    def generate_caption(self, image: np.ndarray) -> str:
        """Generate caption for a single image."""
        try:
            # Try to lazy load BLIP model if not already loaded
            if not self.model_loaded and not self._lazy_load_blip():
                logger.warning("BLIP model not available, returning generic caption")
                return "image content"
            
            # Convert numpy array to PIL Image
            from PIL import Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
            
            # Process image
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50)
                caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return caption.strip()
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "image content"
    
    @handle_inference_error
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing text similarity: {e}")
            return 0.0