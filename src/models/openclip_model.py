import torch
import open_clip
import numpy as np
from typing import List, Union
from ..utils.logger import get_logger
from ..utils.config import settings
from ..utils.memory_manager import memory_manager
from ..utils.model_cache import model_cache
from ..utils.error_handler import error_handler, handle_model_loading_error

logger = get_logger(__name__)

class OpenCLIPModel:
    def __init__(self, force_device: str = None):
        # Use memory manager to determine optimal device
        if force_device:
            self.device = torch.device(force_device)
        else:
            optimal_device = memory_manager.get_optimal_device('openclip')
            self.device = torch.device(optimal_device)
        
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.model_loaded = False
        self.load_model()
    
    @handle_model_loading_error
    def load_model(self) -> bool:
        """Load OpenCLIP model with caching and enhanced memory management."""
        if self.model_loaded:
            return
        
        # Generate cache key based on model configuration
        model_config = {
            'model_name': settings.OPENCLIP_MODEL,
            'pretrained': settings.OPENCLIP_PRETRAINED,
            'device': str(self.device)
        }
        cache_key = f"openclip_{settings.OPENCLIP_MODEL}_{settings.OPENCLIP_PRETRAINED}_{self.device.type}"
        
        try:
            # Try to load from cache first
            cached_components = model_cache.get_model(cache_key, model_config)
            
            if cached_components is not None:
                logger.info(f"Loading OpenCLIP model from cache: {cache_key}")
                self.model = cached_components['model']
                self.preprocess = cached_components['preprocess']
                self.tokenizer = cached_components['tokenizer']
                self.model_loaded = True
                logger.info(f"Successfully loaded OpenCLIP model from cache on {self.device}")
                return
            
            # Cache miss - load model normally
            logger.info(f"Cache miss for OpenCLIP, loading fresh model: {cache_key}")
            
            # Check if model can be loaded
            can_load, message = memory_manager.can_load_model('openclip', use_gpu=(self.device.type == 'cuda'))
            if not can_load:
                logger.warning(f"Memory check failed for OpenCLIP: {message}")
                # Try CPU fallback
                self.device = torch.device('cpu')
                can_load, message = memory_manager.can_load_model('openclip', use_gpu=False)
                if not can_load:
                    raise RuntimeError(f"Cannot load OpenCLIP model: {message}")
            
            logger.info(f"Loading OpenCLIP model on {self.device}: {message}")
            
            # Optimize system for model loading
            memory_manager.optimize_for_model_loading('openclip')
            
            # Monitor memory during loading
            baseline = memory_manager.monitor_model_loading('openclip')
            
            # Load model with memory monitoring
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                settings.OPENCLIP_MODEL,
                pretrained=settings.OPENCLIP_PRETRAINED,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(settings.OPENCLIP_MODEL)
            self.model.eval()
            
            # Finalize memory monitoring
            usage = memory_manager.finalize_model_loading('openclip', baseline)
            
            # Cache the loaded components
            components_to_cache = {
                'model': self.model,
                'preprocess': self.preprocess,
                'tokenizer': self.tokenizer
            }
            
            # Cache in memory only (models are too large for disk caching)
            model_cache.cache_model(cache_key, components_to_cache, model_config, persist_to_disk=False)
            
            self.model_loaded = True
            logger.info(f"Successfully loaded and cached OpenCLIP model: {settings.OPENCLIP_MODEL} on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading OpenCLIP model: {e}")
            # Try emergency fallback with minimal settings
            try:
                logger.info("Attempting emergency fallback for OpenCLIP...")
                memory_manager.aggressive_cleanup()
                
                # Try to load a smaller model from cache first
                fallback_cache_key = f"openclip_ViT-B-32_openai_cpu"
                fallback_config = {
                    'model_name': 'ViT-B-32',
                    'pretrained': 'openai',
                    'device': 'cpu'
                }
                
                cached_fallback = model_cache.get_model(fallback_cache_key, fallback_config)
                
                if cached_fallback is not None:
                    logger.info("Using cached fallback OpenCLIP model")
                    self.device = torch.device('cpu')
                    self.model = cached_fallback['model']
                    self.preprocess = cached_fallback['preprocess']
                    self.tokenizer = cached_fallback['tokenizer']
                    self.model_loaded = True
                    return
                
                # Load fallback model fresh
                self.device = torch.device('cpu')
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',  # Use smaller model as fallback
                    pretrained='openai',
                    device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
                self.model.eval()
                
                # Cache the fallback model
                fallback_components = {
                    'model': self.model,
                    'preprocess': self.preprocess,
                    'tokenizer': self.tokenizer
                }
                model_cache.cache_model(fallback_cache_key, fallback_components, fallback_config, persist_to_disk=False)
                
                self.model_loaded = True
                logger.info("Emergency fallback successful: loaded and cached smaller OpenCLIP model")
                
            except Exception as fallback_e:
                logger.error(f"Emergency fallback also failed: {fallback_e}")
                raise RuntimeError(f"Failed to load OpenCLIP model: {e}. Fallback also failed: {fallback_e}")
    
    def encode_images(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Encode images to embeddings."""
        from PIL import Image
        
        if isinstance(images, np.ndarray) and len(images.shape) == 4:
            # Batch of images
            batch_size = len(images)
            embeddings = []
            
            for i in range(0, batch_size, settings.BATCH_SIZE):
                batch = images[i:i + settings.BATCH_SIZE]
                batch_tensors = []
                
                for img in batch:
                    # Convert numpy array to PIL Image
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                    # Apply preprocessing
                    processed_tensor = self.preprocess(pil_img)
                    batch_tensors.append(processed_tensor)
                
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    batch_embeddings = self.model.encode_image(batch_tensor)
                    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                    embeddings.append(batch_embeddings.cpu().numpy())
            
            return np.vstack(embeddings)
        else:
            # Single image or list of images
            if isinstance(images, list):
                images = np.array(images)
            
            # Convert numpy array to PIL Image
            if images.dtype != np.uint8:
                images = (images * 255).astype(np.uint8)
            pil_img = Image.fromarray(images)
            
            # Apply preprocessing
            image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                return embedding.cpu().numpy()
    
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        text_tokens = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            return text_embeddings.cpu().numpy()
    
    def compute_similarity(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between image and text embeddings."""
        return np.dot(image_embeddings, text_embeddings.T)