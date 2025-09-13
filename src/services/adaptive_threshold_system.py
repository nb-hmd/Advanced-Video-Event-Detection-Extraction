import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import torch
import time
import logging
import json
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, deque
import statistics

from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)

class SizeCategory(Enum):
    """Object size categories for adaptive thresholding"""
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

@dataclass
class ThresholdConfig:
    """Configuration for adaptive thresholds"""
    size_category: SizeCategory
    min_area: int
    max_area: Optional[int]
    base_threshold: float
    confidence_boost: float
    context_adjustments: Dict[str, float]

@dataclass
class DetectionContext:
    """Context information for threshold adjustment"""
    motion_detected: bool = False
    high_noise: bool = False
    lighting_condition: str = "normal"  # "low", "normal", "high"
    scene_complexity: str = "medium"  # "low", "medium", "high"
    frame_quality: float = 1.0  # 0-1 scale
    temporal_consistency: float = 1.0  # 0-1 scale

@dataclass
class AdaptiveResult:
    """Result of adaptive threshold calculation"""
    original_confidence: float
    adaptive_threshold: float
    adjusted_confidence: float
    size_category: SizeCategory
    boost_factor: float
    context_adjustments: Dict[str, float]
    reasoning: List[str]

class AdaptiveThresholdSystem:
    """
    Dynamic threshold adjustment system based on object size and context
    
    This system addresses the limitation of fixed thresholds by implementing:
    1. Size-aware threshold calculation with confidence boosting for small objects
    2. Multi-resolution processing framework
    3. Context-based dynamic adjustments
    4. Real-time threshold optimization with feedback loops
    5. Temporal consistency tracking
    
    Features:
    - Dynamic confidence thresholds based on object size
    - Context-aware adjustments (motion, noise, lighting)
    - Multi-scale processing support
    - Performance tracking and optimization
    - Temporal smoothing for video sequences
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Size category definitions (in pixels²)
        self.size_categories = {
            SizeCategory.TINY: ThresholdConfig(
                size_category=SizeCategory.TINY,
                min_area=0,
                max_area=32*32,
                base_threshold=0.05,
                confidence_boost=2.0,
                context_adjustments={
                    'motion': -0.02,  # Lower threshold for moving tiny objects
                    'noise': 0.03,    # Higher threshold in noisy conditions
                    'low_light': 0.02,
                    'high_complexity': 0.01
                }
            ),
            SizeCategory.SMALL: ThresholdConfig(
                size_category=SizeCategory.SMALL,
                min_area=32*32,
                max_area=96*96,
                base_threshold=0.1,
                confidence_boost=1.5,
                context_adjustments={
                    'motion': -0.01,
                    'noise': 0.02,
                    'low_light': 0.015,
                    'high_complexity': 0.01
                }
            ),
            SizeCategory.MEDIUM: ThresholdConfig(
                size_category=SizeCategory.MEDIUM,
                min_area=96*96,
                max_area=256*256,
                base_threshold=0.25,
                confidence_boost=1.0,
                context_adjustments={
                    'motion': 0.0,
                    'noise': 0.01,
                    'low_light': 0.01,
                    'high_complexity': 0.005
                }
            ),
            SizeCategory.LARGE: ThresholdConfig(
                size_category=SizeCategory.LARGE,
                min_area=256*256,
                max_area=None,
                base_threshold=0.4,
                confidence_boost=1.0,
                context_adjustments={
                    'motion': 0.01,
                    'noise': 0.005,
                    'low_light': 0.005,
                    'high_complexity': 0.0
                }
            )
        }
        
        # Multi-scale processing settings
        self.processing_scales = [256, 512, 1024]
        self.scale_weights = {256: 1.2, 512: 1.0, 1024: 0.8}  # Higher weight for smaller scales
        
        # Performance tracking
        self.detection_history = deque(maxsize=1000)
        self.threshold_performance = defaultdict(list)
        self.optimization_enabled = self.config.get('optimization_enabled', True)
        
        # Temporal consistency tracking
        self.temporal_window = 10  # frames
        self.temporal_history = deque(maxsize=self.temporal_window)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_calculations': 0,
            'size_category_counts': defaultdict(int),
            'average_boost_factors': defaultdict(list),
            'context_adjustment_frequency': defaultdict(int)
        }
        
        logger.info("AdaptiveThresholdSystem initialized successfully")
        logger.info(f"Size categories: {list(self.size_categories.keys())}")
        logger.info(f"Processing scales: {self.processing_scales}")
    
    def _get_size_category(self, bbox_area: int) -> SizeCategory:
        """
        Determine size category based on bounding box area
        
        Args:
            bbox_area: Area of bounding box in pixels²
            
        Returns:
            Size category enum
        """
        for category, config in self.size_categories.items():
            if config.max_area is None:
                if bbox_area >= config.min_area:
                    return category
            else:
                if config.min_area <= bbox_area < config.max_area:
                    return category
        
        # Default to large if no match
        return SizeCategory.LARGE
    
    def _calculate_size_boost(self, bbox_area: int, size_category: SizeCategory) -> float:
        """
        Calculate confidence boost factor based on object size
        
        Args:
            bbox_area: Area of bounding box
            size_category: Determined size category
            
        Returns:
            Boost factor (multiplier for confidence)
        """
        config = self.size_categories[size_category]
        base_boost = config.confidence_boost
        
        if size_category in [SizeCategory.TINY, SizeCategory.SMALL]:
            # Inverse relationship: smaller objects get higher boost
            max_area = config.max_area or (32*32 if size_category == SizeCategory.TINY else 96*96)
            size_ratio = bbox_area / max_area
            # Boost decreases as size increases within category
            dynamic_boost = base_boost * (1.5 - size_ratio * 0.5)
            return max(1.0, dynamic_boost)
        
        return base_boost
    
    def _apply_context_adjustments(
        self, 
        base_threshold: float, 
        context: DetectionContext, 
        size_category: SizeCategory
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply context-based threshold adjustments
        
        Args:
            base_threshold: Base threshold value
            context: Detection context information
            size_category: Object size category
            
        Returns:
            Tuple of (adjusted_threshold, applied_adjustments)
        """
        config = self.size_categories[size_category]
        adjusted_threshold = base_threshold
        applied_adjustments = {}
        
        # Motion adjustment
        if context.motion_detected:
            adjustment = config.context_adjustments['motion']
            adjusted_threshold += adjustment
            applied_adjustments['motion'] = adjustment
        
        # Noise adjustment
        if context.high_noise:
            adjustment = config.context_adjustments['noise']
            adjusted_threshold += adjustment
            applied_adjustments['noise'] = adjustment
        
        # Lighting adjustment
        if context.lighting_condition == 'low':
            adjustment = config.context_adjustments['low_light']
            adjusted_threshold += adjustment
            applied_adjustments['low_light'] = adjustment
        
        # Scene complexity adjustment
        if context.scene_complexity == 'high':
            adjustment = config.context_adjustments['high_complexity']
            adjusted_threshold += adjustment
            applied_adjustments['high_complexity'] = adjustment
        
        # Frame quality adjustment
        if context.frame_quality < 0.7:
            quality_adjustment = (0.7 - context.frame_quality) * 0.05
            adjusted_threshold += quality_adjustment
            applied_adjustments['low_quality'] = quality_adjustment
        
        # Temporal consistency adjustment
        if context.temporal_consistency < 0.8:
            temporal_adjustment = (0.8 - context.temporal_consistency) * 0.03
            adjusted_threshold += temporal_adjustment
            applied_adjustments['temporal_inconsistency'] = temporal_adjustment
        
        # Ensure threshold stays within reasonable bounds
        adjusted_threshold = max(0.01, min(0.95, adjusted_threshold))
        
        return adjusted_threshold, applied_adjustments
    
    def _calculate_temporal_consistency(self, current_detections: List[Dict]) -> float:
        """
        Calculate temporal consistency score based on detection history
        
        Args:
            current_detections: Current frame detections
            
        Returns:
            Consistency score (0-1)
        """
        if len(self.temporal_history) < 2:
            return 1.0
        
        try:
            # Compare with previous frames
            consistency_scores = []
            
            for prev_detections in list(self.temporal_history)[-3:]:  # Last 3 frames
                if not prev_detections or not current_detections:
                    consistency_scores.append(0.5)
                    continue
                
                # Calculate overlap ratio
                overlaps = 0
                total_current = len(current_detections)
                
                for curr_det in current_detections:
                    curr_bbox = curr_det.get('bbox', [])
                    if len(curr_bbox) != 4:
                        continue
                    
                    for prev_det in prev_detections:
                        prev_bbox = prev_det.get('bbox', [])
                        if len(prev_bbox) != 4:
                            continue
                        
                        # Calculate IoU
                        iou = self._calculate_iou(curr_bbox, prev_bbox)
                        if iou > 0.3:  # Threshold for considering it the same object
                            overlaps += 1
                            break
                
                consistency = overlaps / total_current if total_current > 0 else 0.5
                consistency_scores.append(consistency)
            
            return statistics.mean(consistency_scores) if consistency_scores else 1.0
            
        except Exception as e:
            logger.warning(f"Temporal consistency calculation failed: {e}")
            return 1.0
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score (0-1)
        """
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            
            # Calculate union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"IoU calculation failed: {e}")
            return 0.0
    
    def calculate_adaptive_threshold(
        self, 
        bbox_area: int, 
        base_confidence: float,
        context: Optional[DetectionContext] = None,
        processing_scale: int = 512
    ) -> AdaptiveResult:
        """
        Calculate adaptive threshold based on object size and context
        
        Args:
            bbox_area: Area of bounding box in pixels²
            base_confidence: Original confidence score
            context: Detection context information
            processing_scale: Scale at which detection was performed
            
        Returns:
            AdaptiveResult with threshold and adjustment details
        """
        with self.lock:
            try:
                self.stats['total_calculations'] += 1
                
                # Determine size category
                size_category = self._get_size_category(bbox_area)
                self.stats['size_category_counts'][size_category] += 1
                
                # Get base configuration
                config = self.size_categories[size_category]
                base_threshold = config.base_threshold
                
                # Apply scale adjustment
                scale_weight = self.scale_weights.get(processing_scale, 1.0)
                scale_adjusted_threshold = base_threshold * scale_weight
                
                # Calculate size-based boost
                boost_factor = self._calculate_size_boost(bbox_area, size_category)
                self.stats['average_boost_factors'][size_category].append(boost_factor)
                
                # Apply context adjustments
                context = context or DetectionContext()
                adjusted_threshold, context_adjustments = self._apply_context_adjustments(
                    scale_adjusted_threshold, context, size_category
                )
                
                # Track context adjustment usage
                for adj_type in context_adjustments:
                    self.stats['context_adjustment_frequency'][adj_type] += 1
                
                # Calculate final adjusted confidence
                adjusted_confidence = base_confidence * boost_factor
                
                # Create reasoning list
                reasoning = [
                    f"Size category: {size_category.value} (area: {bbox_area}px²)",
                    f"Base threshold: {base_threshold:.3f}",
                    f"Scale adjustment: {scale_weight:.2f}x (scale: {processing_scale})",
                    f"Confidence boost: {boost_factor:.2f}x"
                ]
                
                if context_adjustments:
                    reasoning.append(f"Context adjustments: {context_adjustments}")
                
                result = AdaptiveResult(
                    original_confidence=base_confidence,
                    adaptive_threshold=adjusted_threshold,
                    adjusted_confidence=adjusted_confidence,
                    size_category=size_category,
                    boost_factor=boost_factor,
                    context_adjustments=context_adjustments,
                    reasoning=reasoning
                )
                
                # Store for performance tracking
                if self.optimization_enabled:
                    self.detection_history.append({
                        'timestamp': time.time(),
                        'bbox_area': bbox_area,
                        'size_category': size_category,
                        'original_confidence': base_confidence,
                        'adjusted_confidence': adjusted_confidence,
                        'threshold': adjusted_threshold,
                        'boost_factor': boost_factor
                    })
                
                logger.debug(f"Adaptive threshold calculated: {result}")
                return result
                
            except Exception as e:
                logger.error(f"Adaptive threshold calculation failed: {e}")
                # Return fallback result
                return AdaptiveResult(
                    original_confidence=base_confidence,
                    adaptive_threshold=0.25,
                    adjusted_confidence=base_confidence,
                    size_category=SizeCategory.MEDIUM,
                    boost_factor=1.0,
                    context_adjustments={},
                    reasoning=["Fallback due to calculation error"]
                )
    
    def process_multi_scale_detections(
        self,
        detections_by_scale: Dict[int, List[Dict]],
        context: Optional[DetectionContext] = None
    ) -> List[Dict]:
        """
        Process detections from multiple scales with adaptive thresholds
        
        Args:
            detections_by_scale: Dictionary mapping scale to detections
            context: Detection context information
            
        Returns:
            List of processed detections with adaptive thresholds applied
        """
        try:
            all_detections = []
            
            for scale, detections in detections_by_scale.items():
                for detection in detections:
                    bbox = detection.get('bbox', [])
                    if len(bbox) != 4:
                        continue
                    
                    # Calculate bbox area
                    x1, y1, x2, y2 = bbox
                    bbox_area = (x2 - x1) * (y2 - y1)
                    
                    # Get original confidence
                    original_confidence = detection.get('confidence', 0.0)
                    
                    # Calculate adaptive threshold
                    adaptive_result = self.calculate_adaptive_threshold(
                        bbox_area, original_confidence, context, scale
                    )
                    
                    # Check if detection passes adaptive threshold
                    if adaptive_result.adjusted_confidence >= adaptive_result.adaptive_threshold:
                        enhanced_detection = detection.copy()
                        enhanced_detection.update({
                            'original_confidence': original_confidence,
                            'adaptive_confidence': adaptive_result.adjusted_confidence,
                            'adaptive_threshold': adaptive_result.adaptive_threshold,
                            'size_category': adaptive_result.size_category.value,
                            'boost_factor': adaptive_result.boost_factor,
                            'context_adjustments': adaptive_result.context_adjustments,
                            'processing_scale': scale,
                            'adaptive_reasoning': adaptive_result.reasoning
                        })
                        all_detections.append(enhanced_detection)
            
            logger.info(f"Processed {len(all_detections)} detections with adaptive thresholds")
            return all_detections
            
        except Exception as e:
            logger.error(f"Multi-scale detection processing failed: {e}")
            return []
    
    def update_temporal_history(self, detections: List[Dict]):
        """
        Update temporal history for consistency tracking
        
        Args:
            detections: Current frame detections
        """
        with self.lock:
            self.temporal_history.append(detections)
    
    def optimize_thresholds(self) -> Dict[str, Any]:
        """
        Optimize threshold parameters based on performance history
        
        Returns:
            Dictionary with optimization results
        """
        if not self.optimization_enabled or len(self.detection_history) < 100:
            return {'status': 'insufficient_data'}
        
        try:
            optimization_results = {}
            
            # Analyze performance by size category
            for category in SizeCategory:
                category_detections = [
                    d for d in self.detection_history 
                    if d['size_category'] == category
                ]
                
                if len(category_detections) < 10:
                    continue
                
                # Calculate average boost factor effectiveness
                avg_boost = statistics.mean([d['boost_factor'] for d in category_detections])
                
                # Adjust boost factor if needed
                current_boost = self.size_categories[category].confidence_boost
                if abs(avg_boost - current_boost) > 0.1:
                    new_boost = (avg_boost + current_boost) / 2
                    self.size_categories[category].confidence_boost = new_boost
                    optimization_results[f'{category.value}_boost_updated'] = new_boost
            
            logger.info(f"Threshold optimization completed: {optimization_results}")
            return {'status': 'optimized', 'results': optimization_results}
            
        except Exception as e:
            logger.error(f"Threshold optimization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            stats = self.stats.copy()
            
            # Calculate averages
            for category, boosts in stats['average_boost_factors'].items():
                if boosts:
                    stats['average_boost_factors'][category] = statistics.mean(boosts)
            
            # Add current configuration
            stats['current_thresholds'] = {
                category.value: {
                    'base_threshold': config.base_threshold,
                    'confidence_boost': config.confidence_boost,
                    'min_area': config.min_area,
                    'max_area': config.max_area
                }
                for category, config in self.size_categories.items()
            }
            
            # Add temporal info
            stats['temporal_window_size'] = len(self.temporal_history)
            stats['detection_history_size'] = len(self.detection_history)
            
            return stats
    
    def reset_stats(self):
        """
        Reset performance statistics
        """
        with self.lock:
            self.stats = {
                'total_calculations': 0,
                'size_category_counts': defaultdict(int),
                'average_boost_factors': defaultdict(list),
                'context_adjustment_frequency': defaultdict(int)
            }
            self.detection_history.clear()
            self.temporal_history.clear()
            
        logger.info("Adaptive threshold system statistics reset")