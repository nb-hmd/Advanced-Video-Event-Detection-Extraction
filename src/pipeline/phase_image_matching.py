import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import time
import gc

from ..utils.logger import get_logger
from ..utils.config import settings
from ..utils.memory_manager import memory_manager
from ..services.image_matcher import ImageMatcher
from ..services.clip_extractor import ClipExtractor

logger = get_logger(__name__)

class PhaseImageMatching:
    """
    Image Matching Pipeline Phase
    
    This phase handles image-to-video frame matching using advanced computer vision
    techniques. It integrates seamlessly with the existing video processing pipeline
    while maintaining high performance and accuracy.
    
    Features:
    - Multi-stage matching algorithm
    - Deep learning similarity (OpenCLIP)
    - Structural similarity (SSIM)
    - Feature point matching (ORB/SIFT)
    - Perceptual hashing for fast filtering
    - Memory-optimized processing
    - Result caching for performance
    """
    
    def __init__(self, debug_mode: bool = False):
        self.image_matcher = ImageMatcher(use_gpu=True)
        self.clip_extractor = ClipExtractor()
        self.debug_mode = debug_mode
        self.debug_dir = Path(settings.DATA_DIR) / "debug"
        
        if self.debug_mode:
            self.debug_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("PhaseImageMatching initialized with advanced computer vision algorithms")
    
    def process_video(self, 
                     video_path: str, 
                     reference_image: Union[np.ndarray, str, Path],
                     top_k: int = None,
                     similarity_threshold: float = None,
                     matching_mode: str = 'multi_stage',
                     debug_mode: bool = None) -> List[Dict]:
        """
        Process video for image matching.
        
        Args:
            video_path: Path to the video file
            reference_image: Reference image to match (numpy array or file path)
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold for matches
            matching_mode: 'multi_stage' for maximum accuracy, 'single_stage' for speed
            debug_mode: Enable debug logging and analysis
            
        Returns:
            List of matching results with timestamps and confidence scores
        """
        # Set defaults
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        if similarity_threshold is None:
            similarity_threshold = 0.7  # Higher threshold for image matching
        if debug_mode is not None:
            self.debug_mode = debug_mode
            if self.debug_mode and not self.debug_dir.exists():
                self.debug_dir.mkdir(exist_ok=True)
        
        logger.info(f"Image matching processing: {video_path} with mode: {matching_mode} (debug: {self.debug_mode})")
        start_time = time.time()
        
        try:
            # Log initial memory state
            memory_manager.log_memory_usage("Before image matching processing")
            
            # Validate inputs
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Load and validate reference image
            if isinstance(reference_image, (str, Path)):
                if not Path(reference_image).exists():
                    raise FileNotFoundError(f"Reference image not found: {reference_image}")
                logger.info(f"Using reference image: {reference_image}")
            else:
                logger.info(f"Using reference image array with shape: {reference_image.shape}")
            
            # Perform image matching based on mode
            use_multi_stage = (matching_mode == 'multi_stage')
            
            results = self.image_matcher.match_image_to_video(
                video_path=video_path,
                reference_image=reference_image,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                use_multi_stage=use_multi_stage
            )
            
            # Process and enhance results
            enhanced_results = self._enhance_results(results, video_path)
            
            # Debug analysis
            if self.debug_mode:
                self._log_debug_analysis(enhanced_results, reference_image, matching_mode, similarity_threshold)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self._update_stats(processing_time, len(enhanced_results))
            
            # Log final memory state
            memory_manager.log_memory_usage("After image matching processing")
            
            logger.info(f"Image matching found {len(enhanced_results)} matches in {processing_time:.2f}s")
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in image matching processing: {e}")
            return []
        finally:
            # Cleanup
            memory_manager.aggressive_cleanup()
    
    def _enhance_results(self, results: List[Dict], video_path: str) -> List[Dict]:
        """
        Enhance results with additional metadata and clip extraction.
        
        Args:
            results: Raw matching results
            video_path: Path to the source video
            
        Returns:
            Enhanced results with clip paths and additional metadata
        """
        enhanced_results = []
        
        for i, result in enumerate(results):
            try:
                # Create enhanced result
                enhanced_result = result.copy()
                enhanced_result.update({
                    'phase': 'image_matching',
                    'video_path': video_path,
                    'match_rank': i + 1,
                    'processing_timestamp': time.time()
                })
                
                # Extract clip for this match
                try:
                    clip_info = self.clip_extractor.extract_clip(
                        video_path=video_path,
                        start_time=max(0, result['timestamp'] - 2),  # 2 seconds before
                        end_time=result['timestamp'] + 3,  # 3 seconds after
                        output_name=f"image_match_{i+1}_{result['timestamp']:.2f}s"
                    )
                    
                    if clip_info and clip_info.get('success', False):
                        enhanced_result['clip_path'] = clip_info['output_path']
                        enhanced_result['clip_duration'] = clip_info.get('duration', 5.0)
                    else:
                        logger.warning(f"Failed to extract clip for match at {result['timestamp']:.2f}s")
                        enhanced_result['clip_path'] = None
                        
                except Exception as clip_e:
                    logger.error(f"Error extracting clip for match {i+1}: {clip_e}")
                    enhanced_result['clip_path'] = None
                
                # Add quality assessment
                enhanced_result['quality_score'] = self._assess_match_quality(result)
                
                enhanced_results.append(enhanced_result)
                
            except Exception as e:
                logger.error(f"Error enhancing result {i+1}: {e}")
                # Add basic result even if enhancement fails
                basic_result = result.copy()
                basic_result.update({
                    'phase': 'image_matching',
                    'match_rank': i + 1,
                    'clip_path': None,
                    'quality_score': 0.5
                })
                enhanced_results.append(basic_result)
        
        return enhanced_results
    
    def _assess_match_quality(self, result: Dict) -> float:
        """
        Assess the quality of a match based on multiple factors.
        
        Args:
            result: Match result dictionary
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Base quality from confidence
            quality = result.get('confidence', 0.0)
            
            # Bonus for multi-stage matching with multiple high scores
            if result.get('method') == 'multi_stage_matching':
                clip_sim = result.get('clip_similarity', 0.0)
                ssim_score = result.get('ssim_score', 0.0)
                hist_sim = result.get('histogram_similarity', 0.0)
                feature_matches = result.get('feature_matches', 0)
                
                # Bonus for consistent high scores across methods
                consistency_bonus = 0.0
                high_scores = sum([
                    1 for score in [clip_sim, ssim_score, hist_sim] 
                    if score > 0.8
                ])
                
                if high_scores >= 2:
                    consistency_bonus = 0.1
                elif high_scores >= 3:
                    consistency_bonus = 0.2
                
                # Bonus for feature matches
                feature_bonus = min(feature_matches / 100.0, 0.1)
                
                quality = min(1.0, quality + consistency_bonus + feature_bonus)
            
            return float(quality)
            
        except Exception as e:
            logger.error(f"Error assessing match quality: {e}")
            return 0.5
    
    def _update_stats(self, processing_time: float, num_results: int):
        """
        Update processing statistics.
        
        Args:
            processing_time: Time taken for processing
            num_results: Number of results found
        """
        self.processing_stats['total_processed'] += 1
        
        # Update average processing time
        current_avg = self.processing_stats['average_processing_time']
        total_processed = self.processing_stats['total_processed']
        
        new_avg = ((current_avg * (total_processed - 1)) + processing_time) / total_processed
        self.processing_stats['average_processing_time'] = new_avg
        
        # Update cache stats from image matcher
        cache_stats = self.image_matcher.get_cache_stats()
        # Note: This is a simplified approach - in a real implementation,
        # you'd want to track cache hits/misses more precisely
    
    def _log_debug_analysis(self, 
                          results: List[Dict], 
                          reference_image: Union[np.ndarray, str, Path],
                          matching_mode: str,
                          similarity_threshold: float):
        """
        Log detailed debug analysis.
        
        Args:
            results: Matching results
            reference_image: Reference image used
            matching_mode: Matching mode used
            similarity_threshold: Similarity threshold used
        """
        logger.info("=== IMAGE MATCHING DEBUG ANALYSIS ===")
        logger.info(f"Reference image: {reference_image}")
        logger.info(f"Matching mode: {matching_mode}")
        logger.info(f"Similarity threshold: {similarity_threshold}")
        logger.info(f"Total matches found: {len(results)}")
        
        if results:
            # Analyze confidence distribution
            confidences = [r['confidence'] for r in results]
            logger.info(f"Confidence range: [{min(confidences):.4f}, {max(confidences):.4f}]")
            logger.info(f"Mean confidence: {np.mean(confidences):.4f}")
            logger.info(f"Std confidence: {np.std(confidences):.4f}")
            
            # Top matches analysis
            logger.info("Top 5 matches:")
            for i, result in enumerate(results[:5]):
                timestamp = result['timestamp']
                confidence = result['confidence']
                method = result.get('method', 'unknown')
                quality = result.get('quality_score', 0.0)
                
                logger.info(f"  {i+1}. {timestamp:.2f}s - Confidence: {confidence:.4f}, Quality: {quality:.4f}, Method: {method}")
                
                # Additional details for multi-stage matches
                if method == 'multi_stage_matching':
                    clip_sim = result.get('clip_similarity', 0.0)
                    ssim_score = result.get('ssim_score', 0.0)
                    hist_sim = result.get('histogram_similarity', 0.0)
                    feature_matches = result.get('feature_matches', 0)
                    
                    logger.info(f"     CLIP: {clip_sim:.4f}, SSIM: {ssim_score:.4f}, Hist: {hist_sim:.4f}, Features: {feature_matches}")
        
        # Performance stats
        logger.info(f"Processing stats: {self.processing_stats}")
        
        # Cache stats
        cache_stats = self.image_matcher.get_cache_stats()
        logger.info(f"Cache stats: {cache_stats}")
    
    def get_processing_stats(self) -> Dict:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        stats = self.processing_stats.copy()
        stats.update(self.image_matcher.get_cache_stats())
        return stats
    
    def clear_cache(self):
        """
        Clear all caches.
        """
        self.image_matcher.clear_cache()
        logger.info("Image matching caches cleared")
    
    def set_similarity_thresholds(self, thresholds: Dict[str, float]):
        """
        Update similarity thresholds for different matching methods.
        
        Args:
            thresholds: Dictionary of threshold values
        """
        self.image_matcher.thresholds.update(thresholds)
        logger.info(f"Updated similarity thresholds: {thresholds}")