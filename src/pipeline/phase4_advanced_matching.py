import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import time
import gc

from ..services.image_matcher import ImageMatcher
from ..services.frame_extractor import FrameExtractor
from ..services.clip_extractor import ClipExtractor
from ..utils.logger import get_logger
from ..utils.config import settings
from ..utils.memory_manager import memory_manager
from ..utils.error_handler import error_handler

logger = get_logger(__name__)

class Phase4AdvancedMatching:
    """
    Phase 4: Advanced Image Matching Pipeline
    
    This phase provides advanced image matching capabilities including:
    1. Object-focused matching with background independence
    2. Cross-domain matching for color/grayscale adaptation
    3. Hybrid matching combining multiple approaches
    4. Enhanced similarity computation with multiple algorithms
    
    Features:
    - YOLO-based object detection and segmentation
    - Background removal and object isolation
    - Domain-invariant feature extraction
    - Multi-modal similarity computation
    - Robust matching across different image conditions
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.image_matcher = ImageMatcher(use_gpu=use_gpu)
        self.frame_extractor = FrameExtractor()
        self.clip_extractor = ClipExtractor()
        
        # Configuration from settings
        self.matching_modes = settings.MATCHING_MODES
        self.default_mode = settings.DEFAULT_MATCHING_MODE
        self.supported_classes = settings.SUPPORTED_OBJECT_CLASSES
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'successful_matches': 0,
            'processing_time': 0.0,
            'mode_usage': {mode: 0 for mode in self.matching_modes}
        }
        
        logger.info(f"Phase4AdvancedMatching initialized with modes: {self.matching_modes}")
    
    def process_image_query(self, 
                          video_path: str, 
                          reference_image: Union[np.ndarray, str, Path],
                          matching_mode: str = None,
                          target_class: Optional[str] = None,
                          top_k: int = 10,
                          similarity_threshold: float = None,
                          extract_clips: bool = True) -> Dict:
        """
        Process image query using advanced matching techniques.
        
        Args:
            video_path: Path to the video file
            reference_image: Reference image or path to image
            matching_mode: Matching mode to use
            target_class: Target object class for object-focused matching
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            extract_clips: Whether to extract video clips for matches
            
        Returns:
            Dictionary containing results and metadata
        """
        start_time = time.time()
        
        # Validate and set defaults
        if matching_mode is None:
            matching_mode = self.default_mode
        
        if matching_mode not in self.matching_modes:
            logger.warning(f"Invalid matching mode: {matching_mode}. Using default: {self.default_mode}")
            matching_mode = self.default_mode
        
        if similarity_threshold is None:
            similarity_threshold = self._get_default_threshold(matching_mode)
        
        if target_class and target_class not in self.supported_classes:
            logger.warning(f"Unsupported target class: {target_class}. Supported classes: {self.supported_classes}")
            target_class = None
        
        logger.info(f"Processing image query with mode: {matching_mode}, target_class: {target_class}")
        
        try:
            # Load reference image if path provided
            if isinstance(reference_image, (str, Path)):
                ref_img = cv2.imread(str(reference_image))
                if ref_img is None:
                    raise ValueError(f"Could not load reference image: {reference_image}")
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            else:
                ref_img = reference_image.copy()
            
            # Perform matching using ImageMatcher
            matches = self.image_matcher.match_image_to_video(
                video_path=video_path,
                reference_image=ref_img,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                matching_mode=matching_mode,
                target_class=target_class,
                use_multi_stage=True
            )
            
            # Extract clips if requested
            clips_info = []
            if extract_clips and matches:
                clips_info = self._extract_clips_for_matches(video_path, matches)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats['total_processed'] += 1
            self.processing_stats['successful_matches'] += len(matches)
            self.processing_stats['processing_time'] += processing_time
            self.processing_stats['mode_usage'][matching_mode] += 1
            
            # Prepare results
            results = {
                'matches': matches,
                'clips': clips_info,
                'metadata': {
                    'video_path': video_path,
                    'matching_mode': matching_mode,
                    'target_class': target_class,
                    'similarity_threshold': similarity_threshold,
                    'top_k': top_k,
                    'total_matches': len(matches),
                    'processing_time': processing_time,
                    'timestamp': time.time()
                },
                'performance': {
                    'mode_used': matching_mode,
                    'processing_time_seconds': processing_time,
                    'matches_per_second': len(matches) / processing_time if processing_time > 0 else 0
                }
            }
            
            logger.info(f"Advanced matching completed: {len(matches)} matches in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in advanced image matching: {e}")
            return {
                'matches': [],
                'clips': [],
                'metadata': {
                    'error': str(e),
                    'matching_mode': matching_mode,
                    'processing_time': time.time() - start_time
                },
                'performance': {
                    'error': True,
                    'processing_time_seconds': time.time() - start_time
                }
            }
    
    def _get_default_threshold(self, matching_mode: str) -> float:
        """Get default similarity threshold for matching mode."""
        thresholds = {
            # New simplified modes
            'smart_match': 0.7,        # Hybrid approach - balanced threshold
            'cross_domain': 0.6,       # Lower threshold for cross-domain matching
            'object_focused': 0.75,    # Higher threshold for object matching
            'fast_match': 0.8,         # Higher threshold for fast matching
            # Legacy modes for backward compatibility
            'traditional': getattr(settings, 'TRADITIONAL_SIMILARITY_THRESHOLD', 0.7),
            'hybrid': getattr(settings, 'HYBRID_SIMILARITY_THRESHOLD', 0.7)
        }
        return thresholds.get(matching_mode, 0.7)
    
    def _extract_clips_for_matches(self, video_path: str, matches: List[Dict]) -> List[Dict]:
        """Extract video clips for the matched timestamps."""
        clips_info = []
        
        try:
            for i, match in enumerate(matches):
                timestamp = match['timestamp']
                confidence = match['confidence']
                
                # Calculate clip start and end times
                clip_duration = settings.CLIP_DURATION
                start_time = max(0, timestamp - clip_duration // 2)
                end_time = timestamp + clip_duration // 2
                
                # Extract clip
                clip_path = self.clip_extractor.extract_clip(
                    video_path=video_path,
                    start_time=start_time,
                    end_time=end_time,
                    output_filename=f"match_{i}_{timestamp:.1f}s_conf_{confidence:.3f}.mp4"
                )
                
                if clip_path:
                    clip_info = {
                        'clip_path': str(clip_path),
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time,
                        'match_timestamp': timestamp,
                        'confidence': confidence,
                        'match_index': i
                    }
                    clips_info.append(clip_info)
                    
        except Exception as e:
            logger.error(f"Error extracting clips: {e}")
        
        return clips_info
    
    def batch_process_images(self, 
                           video_path: str, 
                           reference_images: List[Union[np.ndarray, str, Path]],
                           matching_mode: str = None,
                           target_classes: Optional[List[str]] = None,
                           top_k: int = 5,
                           similarity_threshold: float = None) -> List[Dict]:
        """
        Process multiple reference images against the same video.
        
        Args:
            video_path: Path to the video file
            reference_images: List of reference images or paths
            matching_mode: Matching mode to use
            target_classes: List of target classes (one per image, or None)
            top_k: Maximum results per image
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of results for each reference image
        """
        if matching_mode is None:
            matching_mode = self.default_mode
        
        if target_classes is None:
            target_classes = [None] * len(reference_images)
        elif len(target_classes) != len(reference_images):
            logger.warning("Target classes length mismatch. Using None for all.")
            target_classes = [None] * len(reference_images)
        
        logger.info(f"Batch processing {len(reference_images)} images with mode: {matching_mode}")
        
        results = []
        
        for i, (ref_image, target_class) in enumerate(zip(reference_images, target_classes)):
            logger.info(f"Processing image {i+1}/{len(reference_images)}")
            
            result = self.process_image_query(
                video_path=video_path,
                reference_image=ref_image,
                matching_mode=matching_mode,
                target_class=target_class,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                extract_clips=False  # Skip clip extraction for batch processing
            )
            
            result['batch_index'] = i
            results.append(result)
        
        logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def compare_matching_modes(self, 
                             video_path: str, 
                             reference_image: Union[np.ndarray, str, Path],
                             target_class: Optional[str] = None,
                             top_k: int = 5) -> Dict:
        """
        Compare results across different matching modes.
        
        Args:
            video_path: Path to the video file
            reference_image: Reference image or path
            target_class: Target object class
            top_k: Maximum results per mode
            
        Returns:
            Dictionary with results from all modes
        """
        logger.info("Comparing matching modes...")
        
        comparison_results = {
            'modes': {},
            'summary': {
                'best_mode': None,
                'best_score': 0.0,
                'total_unique_matches': 0,
                'processing_times': {}
            }
        }
        
        all_timestamps = set()
        
        for mode in self.matching_modes:
            logger.info(f"Testing mode: {mode}")
            
            result = self.process_image_query(
                video_path=video_path,
                reference_image=reference_image,
                matching_mode=mode,
                target_class=target_class,
                top_k=top_k,
                similarity_threshold=self._get_default_threshold(mode),
                extract_clips=False
            )
            
            comparison_results['modes'][mode] = result
            comparison_results['summary']['processing_times'][mode] = result['metadata'].get('processing_time', 0)
            
            # Track unique timestamps
            for match in result['matches']:
                all_timestamps.add(round(match['timestamp'], 1))
            
            # Update best mode based on average confidence
            if result['matches']:
                avg_confidence = sum(m['confidence'] for m in result['matches']) / len(result['matches'])
                if avg_confidence > comparison_results['summary']['best_score']:
                    comparison_results['summary']['best_mode'] = mode
                    comparison_results['summary']['best_score'] = avg_confidence
        
        comparison_results['summary']['total_unique_matches'] = len(all_timestamps)
        
        logger.info(f"Mode comparison completed. Best mode: {comparison_results['summary']['best_mode']}")
        return comparison_results
    
    def get_processing_statistics(self) -> Dict:
        """Get processing statistics and performance metrics."""
        stats = self.processing_stats.copy()
        
        # Add derived metrics
        if stats['total_processed'] > 0:
            stats['average_matches_per_query'] = stats['successful_matches'] / stats['total_processed']
            stats['average_processing_time'] = stats['processing_time'] / stats['total_processed']
        else:
            stats['average_matches_per_query'] = 0
            stats['average_processing_time'] = 0
        
        # Add mode usage percentages
        total_usage = sum(stats['mode_usage'].values())
        if total_usage > 0:
            stats['mode_usage_percentage'] = {
                mode: (count / total_usage) * 100 
                for mode, count in stats['mode_usage'].items()
            }
        else:
            stats['mode_usage_percentage'] = {mode: 0 for mode in self.matching_modes}
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.processing_stats = {
            'total_processed': 0,
            'successful_matches': 0,
            'processing_time': 0.0,
            'mode_usage': {mode: 0 for mode in self.matching_modes}
        }
        logger.info("Processing statistics reset")
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'image_matcher'):
            self.image_matcher.cleanup()
        
        # Clear statistics
        self.processing_stats.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Phase4AdvancedMatching cleanup completed")