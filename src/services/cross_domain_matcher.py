import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import torch
from PIL import Image
import gc
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import hashlib
import time

# Optional imports for advanced features
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    A = None
    ALBUMENTATIONS_AVAILABLE = False

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    transforms = None
    TORCHVISION_AVAILABLE = False

try:
    import colorspacious
    COLORSPACIOUS_AVAILABLE = True
except ImportError:
    colorspacious = None
    COLORSPACIOUS_AVAILABLE = False

try:
    from skimage import color, feature, filters
    SKIMAGE_AVAILABLE = True
except ImportError:
    color = None
    feature = None
    filters = None
    SKIMAGE_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.config import settings
from ..utils.memory_manager import memory_manager
from ..utils.error_handler import error_handler

logger = get_logger(__name__)

class CrossDomainMatcher:
    """
    Cross-Domain Image Matching Service
    
    This service handles matching between images from different domains:
    - Color vs Grayscale images
    - Different lighting conditions
    - Different color spaces
    - Domain adaptation for robust matching
    
    Features:
    1. Color space normalization and conversion
    2. Domain-invariant feature extraction
    3. Histogram equalization and adaptation
    4. Multi-scale feature matching
    5. Robust similarity computation across domains
    6. Edge and texture-based matching
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Color space conversion methods
        self.color_spaces = ['RGB', 'HSV', 'LAB', 'YUV', 'GRAY']
        
        # Feature extraction methods
        self.feature_methods = {
            'lbp': self._extract_lbp_features,
            'hog': self._extract_hog_features,
            'orb': self._extract_orb_features,
            'sift': self._extract_sift_features,
            'edges': self._extract_edge_features,
            'texture': self._extract_texture_features
        }
        
        # Initialize feature detectors
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.sift = cv2.SIFT_create(nfeatures=1000)
        
        # Augmentation pipeline for domain adaptation
        self.augmentation_pipeline = A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2)
        ])
        
        # Cache for processed features
        self.feature_cache = {}
        self.enable_caching = True
        
        logger.info(f"CrossDomainMatcher initialized on {self.device}")
    
    def normalize_color_space(self, image: np.ndarray, target_space: str = 'LAB') -> np.ndarray:
        """
        Normalize image to a specific color space for domain-invariant processing.
        
        Args:
            image: Input image (RGB or Grayscale)
            target_space: Target color space ('LAB', 'HSV', 'YUV', 'GRAY')
            
        Returns:
            Normalized image in target color space
        """
        try:
            # Handle grayscale input
            if len(image.shape) == 2:
                if target_space == 'GRAY':
                    return image
                # Convert grayscale to RGB first
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Convert to target color space
            if target_space == 'LAB':
                return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            elif target_space == 'HSV':
                return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif target_space == 'YUV':
                return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif target_space == 'GRAY':
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                return image  # Return RGB as default
                
        except Exception as e:
            logger.warning(f"Color space conversion failed: {e}")
            return image
    
    def adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive histogram equalization for better cross-domain matching.
        
        Args:
            image: Input image
            
        Returns:
            Histogram equalized image
        """
        try:
            if len(image.shape) == 2:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
            else:
                # Color image - apply to each channel
                result = image.copy()
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                
                for i in range(image.shape[2]):
                    result[:, :, i] = clahe.apply(image[:, :, i])
                
                return result
                
        except Exception as e:
            logger.warning(f"Histogram equalization failed: {e}")
            return image
    
    def _extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern features."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Compute LBP
            radius = 3
            n_points = 8 * radius
            lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Compute histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-8)  # Normalize
            
            return hist
            
        except Exception as e:
            logger.warning(f"LBP feature extraction failed: {e}")
            return np.zeros(26)  # Default LBP histogram size
    
    def _extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Histogram of Oriented Gradients features."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Resize to standard size
            gray = cv2.resize(gray, (128, 128))
            
            # Compute HOG features
            hog_features = feature.hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=False,
                feature_vector=True
            )
            
            return hog_features
            
        except Exception as e:
            logger.warning(f"HOG feature extraction failed: {e}")
            return np.zeros(3780)  # Default HOG feature size
    
    def _extract_orb_features(self, image: np.ndarray) -> np.ndarray:
        """Extract ORB keypoint features."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is not None and len(descriptors) > 0:
                # Aggregate descriptors using mean
                features = np.mean(descriptors, axis=0)
                return features.astype(float)
            else:
                return np.zeros(32)  # ORB descriptor size
                
        except Exception as e:
            logger.warning(f"ORB feature extraction failed: {e}")
            return np.zeros(32)
    
    def _extract_sift_features(self, image: np.ndarray) -> np.ndarray:
        """Extract SIFT keypoint features."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            if descriptors is not None and len(descriptors) > 0:
                # Aggregate descriptors using mean
                features = np.mean(descriptors, axis=0)
                return features.astype(float)
            else:
                return np.zeros(128)  # SIFT descriptor size
                
        except Exception as e:
            logger.warning(f"SIFT feature extraction failed: {e}")
            return np.zeros(128)
    
    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge-based features."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply different edge detectors
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            
            canny = cv2.Canny(gray, 50, 150)
            
            # Compute edge statistics
            features = [
                np.mean(sobel_mag),
                np.std(sobel_mag),
                np.mean(canny),
                np.std(canny),
                np.sum(canny > 0) / canny.size,  # Edge density
            ]
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"Edge feature extraction failed: {e}")
            return np.zeros(5)
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture-based features."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Compute texture features using filters
            gaussian = filters.gaussian(gray, sigma=1)
            laplacian = filters.laplacian(gray)
            
            # Compute statistics
            features = [
                np.mean(gaussian),
                np.std(gaussian),
                np.mean(laplacian),
                np.std(laplacian),
                np.var(gray),  # Texture variance
            ]
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"Texture feature extraction failed: {e}")
            return np.zeros(5)
    
    def extract_domain_invariant_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract multiple types of domain-invariant features.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of feature vectors
        """
        # Create cache key
        cache_key = hashlib.md5(image.tobytes()).hexdigest()[:16]
        
        if self.enable_caching and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        features = {}
        
        # Preprocess image
        processed_image = self.adaptive_histogram_equalization(image)
        
        # Extract different types of features
        for method_name, method_func in self.feature_methods.items():
            try:
                features[method_name] = method_func(processed_image)
            except Exception as e:
                logger.warning(f"Feature extraction failed for {method_name}: {e}")
                features[method_name] = np.array([])
        
        # Cache results
        if self.enable_caching:
            self.feature_cache[cache_key] = features
        
        return features
    
    def compute_cross_domain_similarity(self, image1: np.ndarray, image2: np.ndarray, 
                                       weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute similarity between images from potentially different domains.
        
        Args:
            image1: First image
            image2: Second image
            weights: Feature weights for similarity computation
            
        Returns:
            Similarity score between 0 and 1
        """
        if weights is None:
            weights = {
                'lbp': 0.25,
                'hog': 0.25,
                'orb': 0.15,
                'sift': 0.15,
                'edges': 0.1,
                'texture': 0.1
            }
        
        try:
            # Extract features from both images
            features1 = self.extract_domain_invariant_features(image1)
            features2 = self.extract_domain_invariant_features(image2)
            
            similarities = []
            total_weight = 0
            
            for feature_name in features1.keys():
                if feature_name in features2 and feature_name in weights:
                    feat1 = features1[feature_name]
                    feat2 = features2[feature_name]
                    
                    if len(feat1) > 0 and len(feat2) > 0 and len(feat1) == len(feat2):
                        # Compute cosine similarity
                        similarity = cosine_similarity(
                            feat1.reshape(1, -1),
                            feat2.reshape(1, -1)
                        )[0, 0]
                        
                        # Handle NaN values
                        if not np.isnan(similarity):
                            similarities.append(similarity * weights[feature_name])
                            total_weight += weights[feature_name]
            
            if total_weight > 0:
                final_similarity = sum(similarities) / total_weight
                return max(0.0, min(1.0, final_similarity))  # Clamp to [0, 1]
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Cross-domain similarity computation failed: {e}")
            return 0.0
    
    def match_cross_domain_images(self, reference_image: np.ndarray, 
                                 target_images: List[np.ndarray],
                                 similarity_threshold: float = 0.6) -> List[Dict]:
        """
        Match reference image against target images across domains.
        
        Args:
            reference_image: Reference image
            target_images: List of target images to match against
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of matches with similarity scores
        """
        matches = []
        
        logger.info(f"Cross-domain matching against {len(target_images)} target images")
        
        for i, target_image in enumerate(target_images):
            similarity = self.compute_cross_domain_similarity(reference_image, target_image)
            
            if similarity >= similarity_threshold:
                match = {
                    'index': i,
                    'similarity': similarity,
                    'method': 'cross_domain_matching'
                }
                matches.append(match)
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Found {len(matches)} cross-domain matches")
        return matches
    
    def adapt_image_domain(self, image: np.ndarray, target_domain: str = 'normalized') -> np.ndarray:
        """
        Adapt image to target domain for better matching.
        
        Args:
            image: Input image
            target_domain: Target domain ('normalized', 'grayscale', 'enhanced')
            
        Returns:
            Domain-adapted image
        """
        try:
            if target_domain == 'normalized':
                # Apply histogram equalization and normalization
                adapted = self.adaptive_histogram_equalization(image)
                adapted = cv2.normalize(adapted, None, 0, 255, cv2.NORM_MINMAX)
                return adapted.astype(np.uint8)
            
            elif target_domain == 'grayscale':
                # Convert to grayscale
                if len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                return image
            
            elif target_domain == 'enhanced':
                # Apply enhancement augmentations
                if len(image.shape) == 2:
                    # Convert grayscale to RGB for augmentation
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                augmented = self.augmentation_pipeline(image=image)
                return augmented['image']
            
            else:
                return image
                
        except Exception as e:
            logger.warning(f"Domain adaptation failed: {e}")
            return image
    
    def cleanup(self):
        """Clean up resources."""
        self.feature_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("CrossDomainMatcher cleanup completed")