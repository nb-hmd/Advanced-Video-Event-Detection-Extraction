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
import json

# Core imports
try:
    import open_clip
    from transformers import (
        AutoProcessor, AutoModel, AutoTokenizer,
        OwlViTProcessor, OwlViTForObjectDetection,
        BlipProcessor, BlipForConditionalGeneration
    )
    from sentence_transformers import SentenceTransformer
    ADVANCED_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced models not available: {e}")
    ADVANCED_MODELS_AVAILABLE = False

from .universal_detector import UniversalDetector
from .frame_extractor import FrameExtractor
from ..utils.logger import get_logger
from ..utils.config import settings
from ..utils.memory_manager import memory_manager

logger = get_logger(__name__)

class OpenVocabMatcher:
    """
    Open Vocabulary Matching Service - Unlimited Object Classes
    
    This revolutionary service enables matching ANY object described in natural language
    across video frames, breaking free from traditional predefined class limitations.
    
    Revolutionary Features:
    1. **Unlimited Object Classes** - Match ANY object imaginable
    2. **Natural Language Queries** - "Find a red bicycle with a basket"
    3. **Semantic Understanding** - Understands object relationships and attributes
    4. **Cross-Modal Matching** - Match text descriptions to visual content
    5. **Temporal Consistency** - Track objects across video frames
    6. **Multi-Scale Detection** - From tiny details to large objects
    7. **Real-time Processing** - Optimized for speed and accuracy
    
    Supported Query Examples:
    - Simple: "person", "car", "dog", "bicycle"
    - Descriptive: "person wearing red shirt", "blue car with open door"
    - Contextual: "person crossing street", "dog playing in park"
    - Attribute-based: "wooden chair", "metal fence", "glass bottle"
    - Complex scenes: "children playing on playground", "cars parked near building"
    - Custom objects: ANY object you can describe in words!
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Initialize core services
        self.universal_detector = UniversalDetector(use_gpu=use_gpu)
        self.frame_extractor = FrameExtractor()
        
        # Matching configuration
        self.matching_modes = {
            'precise': 'High precision matching with strict thresholds',
            'balanced': 'Balanced precision and recall',
            'comprehensive': 'Maximum recall with lower thresholds',
            'semantic': 'Semantic similarity-based matching',
            'visual': 'Visual appearance-based matching'
        }
        
        # Thresholds for different matching modes
        self.mode_thresholds = {
            'precise': {'confidence': 0.7, 'semantic': 0.8, 'visual': 0.75},
            'balanced': {'confidence': 0.5, 'semantic': 0.6, 'visual': 0.6},
            'comprehensive': {'confidence': 0.3, 'semantic': 0.4, 'visual': 0.45},
            'semantic': {'confidence': 0.4, 'semantic': 0.7, 'visual': 0.3},
            'visual': {'confidence': 0.6, 'semantic': 0.3, 'visual': 0.8}
        }
        
        # Cache for results
        self.matching_cache = {}
        self.enable_caching = True
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'successful_matches': 0,
            'processing_time': [],
            'cache_hits': 0
        }
        
        logger.info(f"OpenVocabMatcher initialized on {self.device}")
        logger.info(f"Available matching modes: {list(self.matching_modes.keys())}")
    
    def match_unlimited_objects(self, 
                              video_path: str,
                              object_queries: Union[str, List[str]],
                              matching_mode: str = 'balanced',
                              top_k: int = 10,
                              frame_sampling_rate: int = 1,
                              detection_mode: str = 'hybrid') -> Dict:
        """
        Match unlimited object classes in video using natural language queries.
        
        Args:
            video_path: Path to the video file
            object_queries: Single query or list of object descriptions
            matching_mode: 'precise', 'balanced', 'comprehensive', 'semantic', 'visual'
            top_k: Maximum number of matches to return
            frame_sampling_rate: Sample every Nth frame (1 = every frame)
            detection_mode: Detection method for UniversalDetector
            
        Returns:
            Dictionary containing matches, metadata, and performance info
        """
        start_time = time.time()
        
        # Normalize queries
        if isinstance(object_queries, str):
            object_queries = [object_queries]
        
        # Update performance stats
        self.performance_stats['total_queries'] += len(object_queries)
        
        # Create cache key
        cache_key = self._create_cache_key(video_path, object_queries, matching_mode, detection_mode)
        
        if self.enable_caching and cache_key in self.matching_cache:
            logger.info("Returning cached unlimited matching results")
            self.performance_stats['cache_hits'] += 1
            return self.matching_cache[cache_key]
        
        logger.info(f"Starting unlimited object matching for video: {video_path}")
        logger.info(f"Object queries: {object_queries}")
        logger.info(f"Matching mode: {matching_mode}")
        
        try:
            # Extract frames from video
            logger.info("Extracting frames from video...")
            frames, timestamps = self.frame_extractor.extract_frames(
                video_path, 
                sample_rate=frame_sampling_rate
            )
            
            if not frames:
                return self._create_empty_result("No frames extracted from video")
            
            logger.info(f"Processing {len(frames)} frames with {len(object_queries)} queries")
            
            # Get thresholds for matching mode
            thresholds = self.mode_thresholds.get(matching_mode, self.mode_thresholds['balanced'])
            
            all_matches = []
            frame_results = []
            
            # Process each frame
            for frame_idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                logger.debug(f"Processing frame {frame_idx + 1}/{len(frames)} at {timestamp:.2f}s")
                
                # Detect objects in frame using unlimited detection
                detections = self.universal_detector.detect_unlimited_objects(
                    image=frame,
                    text_queries=object_queries,
                    detection_mode=detection_mode,
                    confidence_threshold=thresholds['confidence']
                )
                
                # Process detections for this frame
                frame_matches = []
                for detection in detections:
                    # Enhance detection with additional matching scores
                    enhanced_detection = self._enhance_detection(
                        detection, frame, object_queries, thresholds
                    )
                    
                    if enhanced_detection:
                        enhanced_detection.update({
                            'frame_index': frame_idx,
                            'timestamp': timestamp,
                            'video_path': video_path
                        })
                        frame_matches.append(enhanced_detection)
                        all_matches.append(enhanced_detection)
                
                frame_results.append({
                    'frame_index': frame_idx,
                    'timestamp': timestamp,
                    'matches': frame_matches,
                    'total_detections': len(detections)
                })
            
            # Post-process and rank matches
            final_matches = self._post_process_matches(
                all_matches, matching_mode, top_k
            )
            
            # Create result dictionary
            processing_time = time.time() - start_time
            self.performance_stats['processing_time'].append(processing_time)
            
            if final_matches:
                self.performance_stats['successful_matches'] += len(final_matches)
            
            result = {
                'status': 'success' if final_matches else 'no_matches',
                'matches': final_matches,
                'total_found': len(final_matches),
                'queries': object_queries,
                'matching_mode': matching_mode,
                'detection_mode': detection_mode,
                'thresholds': thresholds,
                'frame_results': frame_results,
                'metadata': {
                    'video_path': video_path,
                    'total_frames': len(frames),
                    'processing_time': processing_time,
                    'frames_per_second': len(frames) / processing_time if processing_time > 0 else 0,
                    'queries_processed': len(object_queries),
                    'matching_mode': matching_mode,
                    'detection_mode': detection_mode
                },
                'performance': self._get_performance_summary()
            }
            
            # Cache result
            if self.enable_caching:
                self.matching_cache[cache_key] = result
            
            logger.info(f"Unlimited matching completed: {len(final_matches)} matches found in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in unlimited object matching: {e}")
            return self._create_empty_result(f"Processing error: {str(e)}")
    
    def _enhance_detection(self, detection: Dict, frame: np.ndarray, 
                          queries: List[str], thresholds: Dict) -> Optional[Dict]:
        """
        Enhance detection with additional matching scores and validation.
        
        Args:
            detection: Base detection from UniversalDetector
            frame: Original frame image
            queries: List of query strings
            thresholds: Matching thresholds
            
        Returns:
            Enhanced detection dictionary or None if below thresholds
        """
        try:
            # Extract object region from frame
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Ensure valid bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            object_region = frame[y1:y2, x1:x2]
            
            if object_region.size == 0:
                return None
            
            # Compute additional matching scores
            query = detection.get('query', '')
            
            # Visual quality score
            visual_quality = self._compute_visual_quality(object_region)
            
            # Semantic relevance score
            semantic_score = self._compute_semantic_relevance(detection, queries)
            
            # Size and position score
            size_score = self._compute_size_score(bbox, frame.shape)
            
            # Compute composite matching score
            base_confidence = detection.get('confidence', 0.0)
            composite_score = self._compute_composite_score(
                base_confidence, visual_quality, semantic_score, size_score
            )
            
            # Apply thresholds
            if (composite_score >= thresholds['confidence'] and 
                semantic_score >= thresholds['semantic'] and
                visual_quality >= thresholds['visual']):
                
                # Enhance detection with additional information
                enhanced = detection.copy()
                enhanced.update({
                    'composite_score': composite_score,
                    'visual_quality': visual_quality,
                    'semantic_score': semantic_score,
                    'size_score': size_score,
                    'object_region_size': object_region.shape,
                    'enhanced': True
                })
                
                return enhanced
            
            return None
            
        except Exception as e:
            logger.error(f"Error enhancing detection: {e}")
            return None
    
    def _compute_visual_quality(self, object_region: np.ndarray) -> float:
        """
        Compute visual quality score of detected object region.
        
        Args:
            object_region: Cropped object region from frame
            
        Returns:
            Visual quality score (0.0 to 1.0)
        """
        try:
            # Convert to grayscale for analysis
            if len(object_region.shape) == 3:
                gray = cv2.cvtColor(object_region, cv2.COLOR_RGB2GRAY)
            else:
                gray = object_region
            
            # Compute sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
            
            # Compute contrast (standard deviation)
            contrast_score = min(gray.std() / 128.0, 1.0)  # Normalize
            
            # Compute brightness balance
            mean_brightness = gray.mean()
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
            
            # Size factor (larger objects generally have better quality)
            size_factor = min(gray.size / 10000.0, 1.0)  # Normalize
            
            # Combine scores
            quality_score = (
                0.3 * sharpness_score +
                0.3 * contrast_score +
                0.2 * brightness_score +
                0.2 * size_factor
            )
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error computing visual quality: {e}")
            return 0.5  # Default moderate quality
    
    def _compute_semantic_relevance(self, detection: Dict, queries: List[str]) -> float:
        """
        Compute semantic relevance of detection to queries.
        
        Args:
            detection: Detection dictionary
            queries: List of query strings
            
        Returns:
            Semantic relevance score (0.0 to 1.0)
        """
        try:
            query = detection.get('query', '')
            method = detection.get('method', '')
            
            # Base semantic score from detection
            base_score = detection.get('semantic_score', detection.get('confidence', 0.5))
            
            # Method-specific adjustments
            method_multiplier = {
                'owlvit': 1.0,      # OWL-ViT is inherently semantic
                'clip': 0.9,        # CLIP has good semantic understanding
                'yolo_enhanced': 0.8, # YOLO + semantic enhancement
                'hybrid': 0.95      # Hybrid combines multiple methods
            }.get(method, 0.7)
            
            # Query complexity bonus
            query_words = len(query.split()) if query else 1
            complexity_bonus = min(query_words / 10.0, 0.2)  # Up to 20% bonus
            
            # Compute final semantic score
            semantic_score = (base_score * method_multiplier) + complexity_bonus
            
            return float(np.clip(semantic_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error computing semantic relevance: {e}")
            return 0.5
    
    def _compute_size_score(self, bbox: List[float], frame_shape: Tuple[int, int]) -> float:
        """
        Compute size-based score for detection.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Size score (0.0 to 1.0)
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame_shape[:2]
            
            # Compute relative size
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = h * w
            relative_size = bbox_area / frame_area
            
            # Optimal size range (not too small, not too large)
            if 0.01 <= relative_size <= 0.5:  # 1% to 50% of frame
                size_score = 1.0
            elif relative_size < 0.01:  # Too small
                size_score = relative_size / 0.01  # Linear decrease
            else:  # Too large
                size_score = max(0.1, 1.0 - (relative_size - 0.5) / 0.5)
            
            # Aspect ratio consideration
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
            
            # Penalize extreme aspect ratios
            if 0.2 <= aspect_ratio <= 5.0:  # Reasonable aspect ratios
                aspect_score = 1.0
            else:
                aspect_score = 0.5
            
            return float(size_score * aspect_score)
            
        except Exception as e:
            logger.error(f"Error computing size score: {e}")
            return 0.5
    
    def _compute_composite_score(self, confidence: float, visual_quality: float,
                               semantic_score: float, size_score: float) -> float:
        """
        Compute composite matching score from individual components.
        
        Args:
            confidence: Base detection confidence
            visual_quality: Visual quality score
            semantic_score: Semantic relevance score
            size_score: Size-based score
            
        Returns:
            Composite score (0.0 to 1.0)
        """
        # Weighted combination of scores
        composite = (
            0.4 * confidence +      # Base detection confidence (40%)
            0.3 * semantic_score +  # Semantic relevance (30%)
            0.2 * visual_quality +  # Visual quality (20%)
            0.1 * size_score        # Size appropriateness (10%)
        )
        
        return float(np.clip(composite, 0.0, 1.0))
    
    def _post_process_matches(self, matches: List[Dict], 
                            matching_mode: str, top_k: int) -> List[Dict]:
        """
        Post-process and rank matches based on matching mode.
        
        Args:
            matches: List of all matches
            matching_mode: Matching mode for ranking strategy
            top_k: Maximum number of matches to return
            
        Returns:
            Ranked and filtered list of matches
        """
        if not matches:
            return []
        
        # Remove duplicates based on temporal and spatial proximity
        deduplicated = self._remove_duplicate_matches(matches)
        
        # Sort based on matching mode
        if matching_mode == 'precise':
            # Prioritize high confidence and visual quality
            deduplicated.sort(
                key=lambda x: (x.get('composite_score', 0), x.get('visual_quality', 0)),
                reverse=True
            )
        elif matching_mode == 'semantic':
            # Prioritize semantic relevance
            deduplicated.sort(
                key=lambda x: (x.get('semantic_score', 0), x.get('composite_score', 0)),
                reverse=True
            )
        elif matching_mode == 'visual':
            # Prioritize visual quality
            deduplicated.sort(
                key=lambda x: (x.get('visual_quality', 0), x.get('composite_score', 0)),
                reverse=True
            )
        else:  # balanced, comprehensive
            # Use composite score
            deduplicated.sort(
                key=lambda x: x.get('composite_score', 0),
                reverse=True
            )
        
        return deduplicated[:top_k]
    
    def _remove_duplicate_matches(self, matches: List[Dict], 
                                time_threshold: float = 2.0,
                                spatial_threshold: float = 0.5) -> List[Dict]:
        """
        Remove duplicate matches based on temporal and spatial proximity.
        
        Args:
            matches: List of matches
            time_threshold: Time threshold in seconds
            spatial_threshold: IoU threshold for spatial overlap
            
        Returns:
            Deduplicated list of matches
        """
        if not matches:
            return []
        
        # Sort by composite score (best first)
        sorted_matches = sorted(
            matches, 
            key=lambda x: x.get('composite_score', 0), 
            reverse=True
        )
        
        deduplicated = []
        
        for match in sorted_matches:
            is_duplicate = False
            
            for existing in deduplicated:
                # Check temporal proximity
                time_diff = abs(match.get('timestamp', 0) - existing.get('timestamp', 0))
                
                # Check spatial overlap
                iou = self._compute_iou(match.get('bbox', []), existing.get('bbox', []))
                
                # Check query similarity
                same_query = match.get('query', '') == existing.get('query', '')
                
                if (time_diff <= time_threshold and 
                    iou >= spatial_threshold and 
                    same_query):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(match)
        
        return deduplicated
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute Intersection over Union of two bounding boxes."""
        if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
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
    
    def _create_cache_key(self, video_path: str, queries: List[str], 
                         matching_mode: str, detection_mode: str) -> str:
        """Create cache key for matching results."""
        video_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
        query_hash = hashlib.md5('|'.join(sorted(queries)).encode()).hexdigest()[:8]
        return f"match_{video_hash}_{query_hash}_{matching_mode}_{detection_mode}"
    
    def _create_empty_result(self, error_message: str) -> Dict:
        """Create empty result dictionary with error message."""
        return {
            'status': 'error',
            'matches': [],
            'total_found': 0,
            'error': error_message,
            'metadata': {
                'processing_time': 0,
                'frames_processed': 0
            }
        }
    
    def _get_performance_summary(self) -> Dict:
        """Get performance statistics summary."""
        processing_times = self.performance_stats['processing_time']
        
        return {
            'total_queries': self.performance_stats['total_queries'],
            'successful_matches': self.performance_stats['successful_matches'],
            'cache_hits': self.performance_stats['cache_hits'],
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'min_processing_time': np.min(processing_times) if processing_times else 0,
            'max_processing_time': np.max(processing_times) if processing_times else 0,
            'success_rate': (self.performance_stats['successful_matches'] / 
                           max(self.performance_stats['total_queries'], 1)) * 100
        }
    
    def get_query_suggestions(self, partial_query: str = "") -> Dict[str, List[str]]:
        """
        Get smart query suggestions for unlimited object detection.
        
        Args:
            partial_query: Partial query string for context
            
        Returns:
            Dictionary of categorized query suggestions
        """
        base_suggestions = self.universal_detector.get_supported_queries()
        
        # Add advanced query patterns
        advanced_suggestions = {
            'attribute_based': [
                'red car', 'blue shirt', 'wooden table', 'metal fence',
                'glass window', 'plastic bottle', 'leather bag', 'ceramic mug'
            ],
            'action_based': [
                'person walking', 'car driving', 'dog running', 'bird flying',
                'person sitting', 'car parked', 'door opening', 'light blinking'
            ],
            'contextual': [
                'person in park', 'car on street', 'dog in yard', 'cat on roof',
                'bird in tree', 'fish in water', 'person at desk', 'car in garage'
            ],
            'complex_descriptions': [
                'person wearing red shirt and blue jeans',
                'car with open trunk and visible license plate',
                'dog playing with ball in grassy area',
                'bicycle with basket parked near building'
            ]
        }
        
        # Combine suggestions
        all_suggestions = {**base_suggestions, **advanced_suggestions}
        
        # Filter based on partial query if provided
        if partial_query:
            filtered_suggestions = {}
            query_lower = partial_query.lower()
            
            for category, suggestions in all_suggestions.items():
                filtered = [
                    s for s in suggestions 
                    if query_lower in s.lower() or any(word in s.lower() for word in query_lower.split())
                ]
                if filtered:
                    filtered_suggestions[category] = filtered
            
            return filtered_suggestions
        
        return all_suggestions
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up OpenVocabMatcher resources...")
        
        # Cleanup universal detector
        if hasattr(self, 'universal_detector'):
            self.universal_detector.cleanup()
        
        # Clear caches
        self.matching_cache.clear()
        
        # Reset performance stats
        self.performance_stats = {
            'total_queries': 0,
            'successful_matches': 0,
            'processing_time': [],
            'cache_hits': 0
        }
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("OpenVocabMatcher cleanup completed")