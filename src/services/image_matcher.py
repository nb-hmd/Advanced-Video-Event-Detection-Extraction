import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import torch
from PIL import Image
import gc
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import time

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Import new advanced matching services (optional)
try:
    from .object_detector import ObjectDetector
    OBJECT_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ObjectDetector not available: {e}")
    ObjectDetector = None
    OBJECT_DETECTOR_AVAILABLE = False

try:
    from .cross_domain_matcher import CrossDomainMatcher
    CROSS_DOMAIN_MATCHER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CrossDomainMatcher not available: {e}")
    CrossDomainMatcher = None
    CROSS_DOMAIN_MATCHER_AVAILABLE = False
from ..utils.config import settings
from ..utils.memory_manager import memory_manager
from ..models.openclip_model import OpenCLIPModel
from .frame_extractor import FrameExtractor

class ImageMatcher:
    """
    Advanced Image-to-Video Frame Matching Service
    
    This service provides multiple algorithms for matching a reference image
    to video frames with high accuracy and performance:
    
    Traditional Methods:
    1. Deep Learning Features (OpenCLIP embeddings)
    2. Structural Similarity (SSIM)
    3. Histogram Comparison
    4. Feature Point Matching (ORB/SIFT)
    5. Perceptual Hashing
    
    Advanced Methods (NEW):
    6. Object Detection with Background Independence (YOLO + Segmentation)
    7. Cross-Domain Matching (Color/Grayscale adaptation)
    8. Domain-Invariant Feature Extraction
    
    Matching Modes:
    - 'traditional': Original multi-stage approach
    - 'object_focused': Background-independent object matching
    - 'cross_domain': Color/grayscale cross-domain matching
    - 'hybrid': Combination of all methods
    """
    
    def __init__(self, use_gpu: bool = True):
        self.frame_extractor = FrameExtractor()
        self.clip_model = None
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Initialize new advanced matching services
        self.object_detector = None
        self.cross_domain_matcher = None
        
        # Similarity thresholds for different methods
        self.thresholds = {
            'clip_similarity': 0.85,      # High threshold for deep learning
            'ssim_threshold': 0.7,        # Structural similarity
            'histogram_threshold': 0.8,   # Histogram correlation
            'feature_threshold': 10,      # Minimum feature matches
            'hash_threshold': 5,          # Hamming distance for perceptual hash
            'object_similarity': 0.7,     # Object detection similarity
            'cross_domain_similarity': 0.6  # Cross-domain similarity
        }
        
        # Performance settings
        self.max_frames_per_batch = 50
        self.enable_caching = True
        self.cache = {}
        
        # Initialize feature detectors
        self._init_feature_detectors()
        
        logger.info(f"ImageMatcher initialized with device: {self.device}")
    
    def _init_feature_detectors(self):
        """Initialize feature detection algorithms."""
        try:
            # ORB detector for fast feature matching
            self.orb = cv2.ORB_create(nfeatures=1000)
            
            # SIFT detector for high-quality features (if available)
            try:
                self.sift = cv2.SIFT_create()
                self.use_sift = True
                logger.info("SIFT detector initialized")
            except AttributeError:
                self.sift = None
                self.use_sift = False
                logger.info("SIFT not available, using ORB only")
            
            # FLANN matcher for efficient feature matching
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
            
        except Exception as e:
            logger.error(f"Error initializing feature detectors: {e}")
            self.orb = None
            self.sift = None
            self.flann = None
    
    def _lazy_load_clip_model(self):
        """Lazy load OpenCLIP model for deep learning similarity."""
        if self.clip_model is None:
            try:
                logger.info("Loading OpenCLIP model for image matching...")
                self.clip_model = OpenCLIPModel(force_device=str(self.device))
                logger.info("OpenCLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load OpenCLIP model: {e}")
                self.clip_model = None
    
    def _compute_perceptual_hash(self, image: np.ndarray) -> str:
        """Compute perceptual hash for fast similarity filtering."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Resize to 8x8 for hash computation
            resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            
            # Compute average
            avg = resized.mean()
            
            # Create hash
            hash_bits = []
            for pixel in resized.flatten():
                hash_bits.append('1' if pixel > avg else '0')
            
            return ''.join(hash_bits)
        except Exception as e:
            logger.error(f"Error computing perceptual hash: {e}")
            return '0' * 64
    
    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hashes."""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def _compute_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute histogram correlation similarity."""
        try:
            # Convert to HSV for better color representation
            hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
            hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
            
            # Compute histograms
            hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Compute correlation
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return float(correlation)
            
        except Exception as e:
            logger.error(f"Error computing histogram similarity: {e}")
            return 0.0
    
    def _compute_ssim_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Structural Similarity Index."""
        try:
            # Resize images to same size
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1_resized = cv2.resize(img1, (w, h))
            img2_resized = cv2.resize(img2, (w, h))
            
            # Convert to grayscale
            if len(img1_resized.shape) == 3:
                gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)
            else:
                gray1, gray2 = img1_resized, img2_resized
            
            # Compute SSIM
            similarity, _ = ssim(gray1, gray2, full=True)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing SSIM similarity: {e}")
            return 0.0
    
    def _compute_feature_matches(self, img1: np.ndarray, img2: np.ndarray) -> int:
        """Compute number of matching feature points."""
        try:
            if self.orb is None:
                return 0
            
            # Convert to grayscale
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            else:
                gray1, gray2 = img1, img2
            
            # Detect keypoints and descriptors
            detector = self.sift if self.use_sift else self.orb
            kp1, des1 = detector.detectAndCompute(gray1, None)
            kp2, des2 = detector.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                return 0
            
            # Match features
            if self.use_sift and self.flann is not None:
                # Use FLANN for SIFT features
                matches = self.flann.knnMatch(des1, des2, k=2)
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                return len(good_matches)
            else:
                # Use BFMatcher for ORB features
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                # Filter good matches (distance < threshold)
                good_matches = [m for m in matches if m.distance < 50]
                return len(good_matches)
                
        except Exception as e:
            logger.error(f"Error computing feature matches: {e}")
            return 0
    
    def _compute_clip_similarity(self, reference_image: np.ndarray, frame: np.ndarray) -> float:
        """Compute deep learning similarity using OpenCLIP."""
        try:
            if self.clip_model is None:
                self._lazy_load_clip_model()
                if self.clip_model is None:
                    return 0.0
            
            # Encode both images
            ref_embedding = self.clip_model.encode_images(reference_image)
            frame_embedding = self.clip_model.encode_images(frame)
            
            # Compute cosine similarity
            similarity = cosine_similarity(ref_embedding, frame_embedding)[0, 0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing CLIP similarity: {e}")
            return 0.0
    
    def _create_cache_key(self, video_path: str, image_data: np.ndarray) -> str:
        """Create cache key for results."""
        video_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
        image_hash = hashlib.md5(image_data.tobytes()).hexdigest()[:8]
        return f"match_{video_hash}_{image_hash}"
    
    def match_image_to_video(self, 
                           video_path: str, 
                           reference_image: Union[np.ndarray, str, Path],
                           top_k: int = 10,
                           similarity_threshold: float = 0.7,
                           matching_mode: str = 'traditional',
                           target_class: Optional[str] = None,
                           use_multi_stage: bool = True) -> List[Dict]:
        """Match a reference image to video frames and return timestamps of matches.
        
        Args:
            video_path: Path to the video file
            reference_image: Reference image as numpy array or file path
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            matching_mode: Simplified matching mode:
                - 'smart_match': Best overall performance (hybrid approach)
                - 'cross_domain': For color vs grayscale content differences
                - 'object_focused': Find objects regardless of background changes
                - 'fast_match': Quick processing with good accuracy
            target_class: Target object class for object_focused mode (e.g., 'person', 'car')
            use_multi_stage: Whether to use multi-stage matching for better accuracy
            
        Returns:
            List of dictionaries containing timestamp, confidence, and similarity scores
        """
        logger.info(f"Starting image matching for video: {video_path}")
        start_time = time.time()
        
        try:
            # Load reference image
            if isinstance(reference_image, (str, Path)):
                ref_img = cv2.imread(str(reference_image))
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            else:
                ref_img = reference_image.copy()
            
            if ref_img is None:
                raise ValueError("Could not load reference image")
            
            # Check cache
            cache_key = self._create_cache_key(video_path, ref_img)
            if self.enable_caching and cache_key in self.cache:
                logger.info("Returning cached results")
                return self.cache[cache_key]
            
            # Extract frames from video
            logger.info("Extracting frames from video...")
            frames, timestamps = self.frame_extractor.extract_frames(video_path)
            logger.info(f"Extracted {len(frames)} frames")
            
            # Choose matching strategy based on simplified mode
            if matching_mode == 'smart_match':
                # Smart Match: Automatically adapts to content with enhanced capabilities
                results = self._enhanced_smart_matching(ref_img, frames, timestamps, top_k, similarity_threshold, target_class)
            elif matching_mode == 'cross_domain':
                # Cross-Domain Match: For color vs grayscale content
                results = self._cross_domain_matching(ref_img, frames, timestamps, top_k, similarity_threshold)
            elif matching_mode == 'object_focused':
                # Object-Focused Match: Find objects regardless of background
                results = self._object_focused_matching(ref_img, frames, timestamps, top_k, similarity_threshold, target_class)
            elif matching_mode == 'fast_match':
                # Fast Match: Quick processing with good accuracy
                results = self._single_stage_matching(ref_img, frames, timestamps, top_k, similarity_threshold)
            # Legacy mode support for backward compatibility
            elif matching_mode == 'traditional':
                if use_multi_stage:
                    results = self._multi_stage_matching(ref_img, frames, timestamps, top_k, similarity_threshold)
                else:
                    results = self._single_stage_matching(ref_img, frames, timestamps, top_k, similarity_threshold)
            elif matching_mode == 'hybrid':
                results = self._hybrid_matching(ref_img, frames, timestamps, top_k, similarity_threshold, target_class)
            else:
                logger.warning(f"Unknown matching mode: {matching_mode}. Using smart_match as default.")
                results = self._hybrid_matching(ref_img, frames, timestamps, top_k, similarity_threshold, target_class)
            
            # Cache results
            if self.enable_caching:
                self.cache[cache_key] = results
            
            processing_time = time.time() - start_time
            logger.info(f"Image matching completed in {processing_time:.2f}s, found {len(results)} matches")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image matching: {e}")
            return []
    
    def _multi_stage_matching(self, 
                            reference_image: np.ndarray, 
                            frames: np.ndarray, 
                            timestamps: List[float],
                            top_k: int,
                            similarity_threshold: float) -> List[Dict]:
        """
        Multi-stage matching for maximum accuracy.
        
        Stage 1: Fast filtering using perceptual hashing
        Stage 2: Deep learning similarity using OpenCLIP
        Stage 3: Structural similarity refinement
        Stage 4: Feature point verification
        """
        logger.info("Starting multi-stage image matching...")
        
        # Stage 1: Perceptual hash filtering
        logger.info("Stage 1: Perceptual hash filtering...")
        ref_hash = self._compute_perceptual_hash(reference_image)
        
        candidates = []
        for i, frame in enumerate(frames):
            frame_hash = self._compute_perceptual_hash(frame)
            hash_distance = self._hamming_distance(ref_hash, frame_hash)
            
            if hash_distance <= self.thresholds['hash_threshold']:
                candidates.append({
                    'index': i,
                    'timestamp': timestamps[i],
                    'frame': frame,
                    'hash_distance': hash_distance
                })
        
        logger.info(f"Stage 1 filtered to {len(candidates)} candidates")
        
        if not candidates:
            return []
        
        # Stage 2: Deep learning similarity
        logger.info("Stage 2: Deep learning similarity...")
        for candidate in candidates:
            clip_sim = self._compute_clip_similarity(reference_image, candidate['frame'])
            candidate['clip_similarity'] = clip_sim
        
        # Filter by CLIP similarity
        candidates = [c for c in candidates if c['clip_similarity'] >= self.thresholds['clip_similarity']]
        logger.info(f"Stage 2 filtered to {len(candidates)} candidates")
        
        if not candidates:
            return []
        
        # Stage 3: Structural similarity
        logger.info("Stage 3: Structural similarity...")
        for candidate in candidates:
            ssim_score = self._compute_ssim_similarity(reference_image, candidate['frame'])
            candidate['ssim_score'] = ssim_score
        
        # Stage 4: Feature matching for top candidates
        logger.info("Stage 4: Feature point matching...")
        # Sort by CLIP similarity and take top candidates for feature matching
        candidates.sort(key=lambda x: x['clip_similarity'], reverse=True)
        top_candidates = candidates[:min(20, len(candidates))]
        
        for candidate in top_candidates:
            feature_matches = self._compute_feature_matches(reference_image, candidate['frame'])
            candidate['feature_matches'] = feature_matches
            
            # Compute histogram similarity
            hist_sim = self._compute_histogram_similarity(reference_image, candidate['frame'])
            candidate['histogram_similarity'] = hist_sim
        
        # Compute final composite score
        for candidate in top_candidates:
            # Weighted combination of all similarity measures
            composite_score = (
                0.4 * candidate['clip_similarity'] +
                0.25 * candidate['ssim_score'] +
                0.2 * candidate['histogram_similarity'] +
                0.1 * min(candidate['feature_matches'] / 50.0, 1.0) +  # Normalize feature matches
                0.05 * (1.0 - candidate['hash_distance'] / 64.0)  # Normalize hash distance
            )
            candidate['composite_score'] = composite_score
        
        # Sort by composite score and filter by threshold
        top_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
        final_results = []
        
        for candidate in top_candidates[:top_k]:
            if candidate['composite_score'] >= similarity_threshold:
                result = {
                    'timestamp': candidate['timestamp'],
                    'confidence': candidate['composite_score'],
                    'clip_similarity': candidate['clip_similarity'],
                    'ssim_score': candidate['ssim_score'],
                    'histogram_similarity': candidate['histogram_similarity'],
                    'feature_matches': candidate['feature_matches'],
                    'hash_distance': candidate['hash_distance'],
                    'method': 'multi_stage_matching',
                    'frame_index': candidate['index']
                }
                final_results.append(result)
        
        return final_results
    
    def _lazy_load_object_detector(self):
        """Lazy load object detector to save memory."""
        if self.object_detector is None and OBJECT_DETECTOR_AVAILABLE:
            try:
                logger.info("Loading ObjectDetector...")
                self.object_detector = ObjectDetector(use_gpu=self.use_gpu)
                logger.info("ObjectDetector loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ObjectDetector: {e}")
                self.object_detector = None
        elif not OBJECT_DETECTOR_AVAILABLE:
            logger.warning("ObjectDetector not available - skipping object-focused matching")
            self.object_detector = None
    
    def _lazy_load_cross_domain_matcher(self):
        """Lazy load cross-domain matcher to save memory."""
        if self.cross_domain_matcher is None and CROSS_DOMAIN_MATCHER_AVAILABLE:
            try:
                logger.info("Loading CrossDomainMatcher...")
                self.cross_domain_matcher = CrossDomainMatcher(use_gpu=self.use_gpu)
                logger.info("CrossDomainMatcher loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CrossDomainMatcher: {e}")
                self.cross_domain_matcher = None
    
    def _object_focused_matching(self, 
                               reference_image: np.ndarray, 
                               frames: np.ndarray, 
                               timestamps: List[float],
                               top_k: int,
                               similarity_threshold: float,
                               target_class: Optional[str] = None) -> List[Dict]:
        """
        Object-focused matching with background independence.
        """
        logger.info("Starting object-focused matching...")
        
        self._lazy_load_object_detector()
        if self.object_detector is None:
            logger.warning("ObjectDetector not available, falling back to traditional matching")
            return self._multi_stage_matching(reference_image, frames, timestamps, top_k, similarity_threshold)
        
        try:
            # Process reference image to extract objects
            reference_objects = self.object_detector.process_reference_image(reference_image, target_class)
            
            if not reference_objects:
                logger.warning("No objects detected in reference image")
                return []
            
            logger.info(f"Found {len(reference_objects)} objects in reference image")
            
            all_matches = []
            
            # Process frames in batches
            for i in range(0, len(frames), self.max_frames_per_batch):
                batch_end = min(i + self.max_frames_per_batch, len(frames))
                batch_frames = frames[i:batch_end]
                batch_timestamps = timestamps[i:batch_end]
                
                for j, frame in enumerate(batch_frames):
                    frame_matches = self.object_detector.match_objects_in_frame(
                        frame, reference_objects, self.thresholds['object_similarity']
                    )
                    
                    for match in frame_matches:
                        result = {
                            'timestamp': batch_timestamps[j],
                            'confidence': match['similarity'],
                            'object_similarity': match['similarity'],
                            'class_name': match['class_name'],
                            'bbox': match['bbox'].tolist(),
                            'method': 'object_focused_matching',
                            'frame_index': i + j
                        }
                        all_matches.append(result)
            
            # Sort by confidence and return top_k
            all_matches.sort(key=lambda x: x['confidence'], reverse=True)
            final_results = all_matches[:top_k]
            
            logger.info(f"Object-focused matching found {len(final_results)} matches")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in object-focused matching: {e}")
            return []
    
    def _cross_domain_matching(self, 
                             reference_image: np.ndarray, 
                             frames: np.ndarray, 
                             timestamps: List[float],
                             top_k: int,
                             similarity_threshold: float) -> List[Dict]:
        """
        Cross-domain matching for color/grayscale adaptation.
        """
        logger.info("Starting cross-domain matching...")
        
        self._lazy_load_cross_domain_matcher()
        if self.cross_domain_matcher is None:
            logger.warning("CrossDomainMatcher not available, falling back to traditional matching")
            return self._multi_stage_matching(reference_image, frames, timestamps, top_k, similarity_threshold)
        
        try:
            matches = []
            
            # Process frames in batches
            for i in range(0, len(frames), self.max_frames_per_batch):
                batch_end = min(i + self.max_frames_per_batch, len(frames))
                batch_frames = frames[i:batch_end]
                batch_timestamps = timestamps[i:batch_end]
                
                # Use cross-domain matcher
                batch_matches = self.cross_domain_matcher.match_cross_domain_images(
                    reference_image, batch_frames, self.thresholds['cross_domain_similarity']
                )
                
                for match in batch_matches:
                    result = {
                        'timestamp': batch_timestamps[match['index']],
                        'confidence': match['similarity'],
                        'cross_domain_similarity': match['similarity'],
                        'method': 'cross_domain_matching',
                        'frame_index': i + match['index']
                    }
                    matches.append(result)
            
            # Sort by confidence and return top_k
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            final_results = matches[:top_k]
            
            logger.info(f"Cross-domain matching found {len(final_results)} matches")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in cross-domain matching: {e}")
            return []
    
    def _hybrid_matching(self, 
                       reference_image: np.ndarray, 
                       frames: np.ndarray, 
                       timestamps: List[float],
                       top_k: int,
                       similarity_threshold: float,
                       target_class: Optional[str] = None) -> List[Dict]:
        """
        Hybrid matching combining all available methods.
        """
        logger.info("Starting hybrid matching...")
        
        all_results = []
        
        # Run traditional matching
        try:
            traditional_results = self._multi_stage_matching(
                reference_image, frames, timestamps, top_k, similarity_threshold
            )
            for result in traditional_results:
                result['method'] = 'traditional_in_hybrid'
            all_results.extend(traditional_results)
        except Exception as e:
            logger.warning(f"Traditional matching failed in hybrid mode: {e}")
        
        # Run object-focused matching
        try:
            object_results = self._object_focused_matching(
                reference_image, frames, timestamps, top_k, similarity_threshold, target_class
            )
            for result in object_results:
                result['method'] = 'object_focused_in_hybrid'
            all_results.extend(object_results)
        except Exception as e:
            logger.warning(f"Object-focused matching failed in hybrid mode: {e}")
        
        # Run cross-domain matching
        try:
            cross_domain_results = self._cross_domain_matching(
                reference_image, frames, timestamps, top_k, similarity_threshold
            )
            for result in cross_domain_results:
                result['method'] = 'cross_domain_in_hybrid'
            all_results.extend(cross_domain_results)
        except Exception as e:
            logger.warning(f"Cross-domain matching failed in hybrid mode: {e}")
        
        # Combine and deduplicate results
        combined_results = self._combine_hybrid_results(all_results, top_k)
        
        logger.info(f"Hybrid matching found {len(combined_results)} combined matches")
        return combined_results
    
    def _enhanced_smart_matching(self, 
                               reference_image: np.ndarray, 
                               frames: np.ndarray, 
                               timestamps: List[float],
                               top_k: int,
                               similarity_threshold: float,
                               target_class: Optional[str] = None) -> List[Dict]:
        """
        Enhanced Smart Match: Intelligently combines all matching methods with adaptive weighting.
        
        This method automatically detects the best approach based on image characteristics:
        - Background independence for objects with different backgrounds
        - Cross-domain matching for color/grayscale differences
        - Traditional matching for similar conditions
        """
        logger.info("Starting enhanced smart matching...")
        
        # Analyze reference image characteristics
        ref_analysis = self._analyze_image_characteristics(reference_image)
        
        all_results = []
        method_weights = {}
        
        # 1. Always try object-focused matching (background independence)
        try:
            logger.info("Running object-focused matching for background independence...")
            object_results = self._object_focused_matching(
                reference_image, frames, timestamps, top_k, similarity_threshold * 0.8, target_class
            )
            
            # Weight based on object detection confidence
            weight = 1.0 if ref_analysis['has_clear_objects'] else 0.7
            method_weights['object_focused'] = weight
            
            for result in object_results:
                result['method'] = 'object_focused_smart'
                result['method_weight'] = weight
                result['weighted_confidence'] = result['confidence'] * weight
            
            all_results.extend(object_results)
            logger.info(f"Object-focused: {len(object_results)} matches (weight: {weight:.2f})")
            
        except Exception as e:
            logger.warning(f"Object-focused matching failed: {e}")
        
        # 2. Try cross-domain matching (color/grayscale adaptation)
        try:
            logger.info("Running cross-domain matching for color adaptation...")
            cross_domain_results = self._cross_domain_matching(
                reference_image, frames, timestamps, top_k, similarity_threshold * 0.9
            )
            
            # Weight based on color characteristics
            weight = 1.0 if ref_analysis['needs_color_adaptation'] else 0.8
            method_weights['cross_domain'] = weight
            
            for result in cross_domain_results:
                result['method'] = 'cross_domain_smart'
                result['method_weight'] = weight
                result['weighted_confidence'] = result['confidence'] * weight
            
            all_results.extend(cross_domain_results)
            logger.info(f"Cross-domain: {len(cross_domain_results)} matches (weight: {weight:.2f})")
            
        except Exception as e:
            logger.warning(f"Cross-domain matching failed: {e}")
        
        # 3. Traditional multi-stage matching as baseline
        try:
            logger.info("Running traditional matching as baseline...")
            traditional_results = self._multi_stage_matching(
                reference_image, frames, timestamps, top_k, similarity_threshold
            )
            
            # Lower weight for traditional method in smart mode
            weight = 0.6
            method_weights['traditional'] = weight
            
            for result in traditional_results:
                result['method'] = 'traditional_smart'
                result['method_weight'] = weight
                result['weighted_confidence'] = result['confidence'] * weight
            
            all_results.extend(traditional_results)
            logger.info(f"Traditional: {len(traditional_results)} matches (weight: {weight:.2f})")
            
        except Exception as e:
            logger.warning(f"Traditional matching failed: {e}")
        
        # 4. Intelligent result fusion
        if not all_results:
            logger.warning("No results from any matching method")
            return []
        
        # Group results by timestamp with tolerance
        fused_results = self._intelligent_result_fusion(all_results, ref_analysis)
        
        # Sort by weighted confidence and return top_k
        fused_results.sort(key=lambda x: x.get('final_confidence', 0), reverse=True)
        final_results = fused_results[:top_k]
        
        logger.info(f"Enhanced smart matching found {len(final_results)} fused matches")
        logger.info(f"Method weights used: {method_weights}")
        
        return final_results
    
    def _analyze_image_characteristics(self, image: np.ndarray) -> Dict:
        """
        Analyze image characteristics to determine optimal matching strategy.
        """
        analysis = {
            'has_clear_objects': False,
            'needs_color_adaptation': False,
            'is_grayscale': False,
            'background_complexity': 'medium',
            'dominant_colors': [],
            'edge_density': 0.0
        }
        
        try:
            # Check if grayscale
            if len(image.shape) == 2:
                analysis['is_grayscale'] = True
                analysis['needs_color_adaptation'] = True
            elif len(image.shape) == 3:
                # Check if effectively grayscale (all channels similar)
                if image.shape[2] == 3:
                    channel_diff = np.std([image[:,:,0].std(), image[:,:,1].std(), image[:,:,2].std()])
                    if channel_diff < 10:  # Very similar channel variations
                        analysis['is_grayscale'] = True
                        analysis['needs_color_adaptation'] = True
            
            # Analyze edge density (indicates object presence)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            edges = cv2.Canny(gray, 50, 150)
            analysis['edge_density'] = np.sum(edges > 0) / edges.size
            
            # High edge density suggests clear objects
            analysis['has_clear_objects'] = analysis['edge_density'] > 0.05
            
            # Analyze background complexity
            # Use standard deviation of pixel intensities
            intensity_std = gray.std()
            if intensity_std < 30:
                analysis['background_complexity'] = 'simple'
            elif intensity_std > 60:
                analysis['background_complexity'] = 'complex'
            else:
                analysis['background_complexity'] = 'medium'
            
            # Analyze dominant colors (if color image)
            if not analysis['is_grayscale'] and len(image.shape) == 3:
                # Simple dominant color analysis
                pixels = image.reshape(-1, 3)
                unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1])))))  
                total_pixels = pixels.shape[0]
                color_diversity = unique_colors / total_pixels
                
                if color_diversity < 0.1:  # Low color diversity
                    analysis['needs_color_adaptation'] = True
            
            logger.debug(f"Image analysis: {analysis}")
            
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
        
        return analysis
    
    def _intelligent_result_fusion(self, all_results: List[Dict], ref_analysis: Dict) -> List[Dict]:
        """
        Intelligently fuse results from different matching methods.
        """
        if not all_results:
            return []
        
        # Group results by timestamp (with tolerance)
        timestamp_groups = {}
        tolerance = 0.5  # 0.5 second tolerance
        
        for result in all_results:
            timestamp = result['timestamp']
            
            # Find existing group within tolerance
            found_group = None
            for group_timestamp in timestamp_groups.keys():
                if abs(timestamp - group_timestamp) <= tolerance:
                    found_group = group_timestamp
                    break
            
            if found_group is not None:
                timestamp_groups[found_group].append(result)
            else:
                timestamp_groups[timestamp] = [result]
        
        # Fuse results for each timestamp group
        fused_results = []
        
        for group_timestamp, group_results in timestamp_groups.items():
            if len(group_results) == 1:
                # Single result - use as is but add final confidence
                result = group_results[0]
                result['final_confidence'] = result.get('weighted_confidence', result.get('confidence', 0))
                result['fusion_method'] = 'single'
                fused_results.append(result)
            else:
                # Multiple results - intelligent fusion
                fused_result = self._fuse_multiple_results(group_results, ref_analysis)
                if fused_result:
                    fused_results.append(fused_result)
        
        return fused_results
    
    def _fuse_multiple_results(self, results: List[Dict], ref_analysis: Dict) -> Optional[Dict]:
        """
        Fuse multiple results for the same timestamp.
        """
        if not results:
            return None
        
        # Sort by weighted confidence
        results.sort(key=lambda x: x.get('weighted_confidence', 0), reverse=True)
        
        # Use the best result as base
        best_result = results[0]
        
        # Calculate fusion confidence
        confidences = [r.get('weighted_confidence', r.get('confidence', 0)) for r in results]
        methods = [r.get('method', 'unknown') for r in results]
        
        # Weighted average with emphasis on best result
        weights = [0.5] + [0.5 / (len(results) - 1)] * (len(results) - 1) if len(results) > 1 else [1.0]
        fusion_confidence = sum(c * w for c, w in zip(confidences, weights))
        
        # Bonus for method diversity
        unique_methods = len(set(methods))
        if unique_methods > 1:
            diversity_bonus = min(0.1 * (unique_methods - 1), 0.2)
            fusion_confidence += diversity_bonus
        
        # Create fused result
        fused_result = best_result.copy()
        fused_result.update({
            'final_confidence': fusion_confidence,
            'fusion_method': 'intelligent',
            'contributing_methods': methods,
            'method_count': len(results),
            'confidence_range': [min(confidences), max(confidences)]
        })
        
        return fused_result
    
    def _combine_hybrid_results(self, all_results: List[Dict], top_k: int) -> List[Dict]:
        """
        Combine results from different matching methods and remove duplicates.
        """
        if not all_results:
            return []
        
        # Group results by timestamp (with small tolerance)
        timestamp_groups = {}
        tolerance = 0.5  # 0.5 second tolerance
        
        for result in all_results:
            timestamp = result['timestamp']
            
            # Find existing group within tolerance
            found_group = None
            for group_timestamp in timestamp_groups.keys():
                if abs(timestamp - group_timestamp) <= tolerance:
                    found_group = group_timestamp
                    break
            
            if found_group is not None:
                timestamp_groups[found_group].append(result)
            else:
                timestamp_groups[timestamp] = [result]
        
        # Combine results for each timestamp group
        combined_results = []
        for group_timestamp, group_results in timestamp_groups.items():
            if len(group_results) == 1:
                combined_results.append(group_results[0])
            else:
                # Combine multiple results for same timestamp
                combined_result = self._merge_results(group_results)
                combined_results.append(combined_result)
        
        # Sort by confidence and return top_k
        combined_results.sort(key=lambda x: x['confidence'], reverse=True)
        return combined_results[:top_k]
    
    def _merge_results(self, results: List[Dict]) -> Dict:
        """
        Merge multiple results for the same timestamp.
        """
        # Use the result with highest confidence as base
        base_result = max(results, key=lambda x: x['confidence'])
        
        # Combine confidence scores using weighted average
        total_confidence = sum(r['confidence'] for r in results)
        avg_confidence = total_confidence / len(results)
        
        # Create merged result
        merged_result = base_result.copy()
        merged_result['confidence'] = min(avg_confidence * 1.1, 1.0)  # Slight boost for multiple matches
        merged_result['method'] = 'hybrid_combined'
        merged_result['contributing_methods'] = [r['method'] for r in results]
        merged_result['method_count'] = len(results)
        
        return merged_result
    
    def _single_stage_matching(self, 
                             reference_image: np.ndarray, 
                             frames: np.ndarray, 
                             timestamps: List[float],
                             top_k: int,
                             similarity_threshold: float) -> List[Dict]:
        """
        Single-stage matching using only deep learning similarity for speed.
        """
        logger.info("Starting single-stage image matching...")
        
        similarities = []
        
        # Process frames in batches for memory efficiency
        for i in range(0, len(frames), self.max_frames_per_batch):
            batch_end = min(i + self.max_frames_per_batch, len(frames))
            batch_frames = frames[i:batch_end]
            
            for j, frame in enumerate(batch_frames):
                frame_idx = i + j
                clip_sim = self._compute_clip_similarity(reference_image, frame)
                
                similarities.append({
                    'timestamp': timestamps[frame_idx],
                    'confidence': clip_sim,
                    'clip_similarity': clip_sim,
                    'method': 'single_stage_matching',
                    'frame_index': frame_idx
                })
            
            # Memory cleanup
            if i % (self.max_frames_per_batch * 2) == 0:
                memory_manager.aggressive_cleanup()
        
        # Sort by similarity and filter
        similarities.sort(key=lambda x: x['confidence'], reverse=True)
        results = [s for s in similarities[:top_k] if s['confidence'] >= similarity_threshold]
        
        return results
    
    def clear_cache(self):
        """Clear the results cache."""
        self.cache.clear()
        logger.info("Image matching cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'cache_enabled': self.enable_caching
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.clip_model is not None:
            del self.clip_model
            self.clip_model = None
        
        # Clean up new advanced matching services
        if self.object_detector is not None:
            self.object_detector.cleanup()
            del self.object_detector
            self.object_detector = None
        
        if self.cross_domain_matcher is not None:
            self.cross_domain_matcher.cleanup()
            del self.cross_domain_matcher
            self.cross_domain_matcher = None
        
        self.cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("ImageMatcher cleanup completed with advanced services")