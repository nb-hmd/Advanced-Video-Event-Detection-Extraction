import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import deque
import statistics

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
    import torchvision.transforms as transforms
    from torchvision.ops import nms
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.config import settings
from ..utils.memory_manager import memory_manager
from ..utils.model_cache import model_cache

logger = get_logger(__name__)

class ProposalType(Enum):
    """Types of region proposals"""
    RPN = "rpn"
    SALIENCY = "saliency"
    MOTION = "motion"
    EDGE = "edge"
    TEXTURE = "texture"

@dataclass
class RegionProposal:
    """Region proposal structure"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    proposal_type: ProposalType
    area: float
    aspect_ratio: float
    features: Optional[np.ndarray] = None
    motion_vector: Optional[Tuple[float, float]] = None
    saliency_score: float = 0.0
    edge_density: float = 0.0
    texture_score: float = 0.0

@dataclass
class ProposalConfig:
    """Configuration for region proposals"""
    max_proposals_per_frame: int = 100
    nms_threshold: float = 0.3
    min_area: int = 64  # Minimum proposal area
    max_area: int = 10000  # Maximum proposal area
    saliency_weight: float = 0.3
    motion_weight: float = 0.4
    rpn_weight: float = 0.3
    edge_weight: float = 0.2
    texture_weight: float = 0.1

class LightweightRPN(nn.Module):
    """
    Lightweight Region Proposal Network for efficient candidate generation
    """
    
    def __init__(self, in_channels: int = 256, num_anchors: int = 9):
        super().__init__()
        self.num_anchors = num_anchors
        
        # Convolutional layers
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through RPN"""
        x = F.relu(self.conv(x))
        
        # Classification scores
        cls_logits = self.cls_logits(x)
        
        # Bounding box regression
        bbox_pred = self.bbox_pred(x)
        
        return cls_logits, bbox_pred
    
    def generate_anchors(self, feature_map_size: Tuple[int, int], stride: int = 16) -> torch.Tensor:
        """
        Generate anchor boxes for the feature map
        
        Args:
            feature_map_size: (height, width) of feature map
            stride: Stride of the feature map
            
        Returns:
            Anchor boxes tensor
        """
        h, w = feature_map_size
        
        # Anchor scales and ratios optimized for small objects
        scales = [8, 16, 32]  # Smaller scales for tiny objects
        ratios = [0.5, 1.0, 2.0]
        
        anchors = []
        
        for i in range(h):
            for j in range(w):
                cx = (j + 0.5) * stride
                cy = (i + 0.5) * stride
                
                for scale in scales:
                    for ratio in ratios:
                        w_anchor = scale * np.sqrt(ratio)
                        h_anchor = scale / np.sqrt(ratio)
                        
                        x1 = cx - w_anchor / 2
                        y1 = cy - h_anchor / 2
                        x2 = cx + w_anchor / 2
                        y2 = cy + h_anchor / 2
                        
                        anchors.append([x1, y1, x2, y2])
        
        return torch.tensor(anchors, dtype=torch.float32)

class SaliencyDetector:
    """
    Saliency-based region proposal detector
    """
    
    def __init__(self):
        self.saliency_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _load_saliency_model(self):
        """Load saliency detection model"""
        if self.saliency_model is None:
            try:
                # Use OpenCV's saliency detector as fallback
                self.saliency_model = cv2.saliency.StaticSaliencySpectralResidual_create()
                logger.info("Saliency detector loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load saliency detector: {e}")
                self.saliency_model = "fallback"
    
    def detect_salient_regions(
        self, 
        image: np.ndarray, 
        min_area: int = 64,
        max_proposals: int = 50
    ) -> List[RegionProposal]:
        """
        Detect salient regions in the image
        
        Args:
            image: Input image
            min_area: Minimum region area
            max_proposals: Maximum number of proposals
            
        Returns:
            List of salient region proposals
        """
        self._load_saliency_model()
        
        try:
            proposals = []
            
            if self.saliency_model != "fallback":
                # Compute saliency map
                success, saliency_map = self.saliency_model.computeSaliency(image)
                
                if success:
                    # Normalize saliency map
                    saliency_map = (saliency_map * 255).astype(np.uint8)
                    
                    # Apply threshold to get salient regions
                    _, binary_map = cv2.threshold(saliency_map, 127, 255, cv2.THRESH_BINARY)
                    
                    # Find contours
                    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Convert contours to proposals
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < min_area:
                            continue
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Calculate saliency score for this region
                        roi_saliency = saliency_map[y:y+h, x:x+w]
                        saliency_score = np.mean(roi_saliency) / 255.0
                        
                        proposal = RegionProposal(
                            bbox=(x, y, x+w, y+h),
                            confidence=saliency_score,
                            proposal_type=ProposalType.SALIENCY,
                            area=area,
                            aspect_ratio=w/h if h > 0 else 1.0,
                            saliency_score=saliency_score
                        )
                        proposals.append(proposal)
            else:
                # Fallback: Use edge-based saliency
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Find contours from edges
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < min_area:
                        continue
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate edge density as saliency score
                    roi_edges = edges[y:y+h, x:x+w]
                    edge_density = np.sum(roi_edges > 0) / (w * h)
                    
                    proposal = RegionProposal(
                        bbox=(x, y, x+w, y+h),
                        confidence=edge_density,
                        proposal_type=ProposalType.SALIENCY,
                        area=area,
                        aspect_ratio=w/h if h > 0 else 1.0,
                        saliency_score=edge_density
                    )
                    proposals.append(proposal)
            
            # Sort by confidence and limit
            proposals.sort(key=lambda p: p.confidence, reverse=True)
            return proposals[:max_proposals]
            
        except Exception as e:
            logger.error(f"Saliency detection failed: {e}")
            return []

class OpticalFlowTracker:
    """
    Optical flow-based motion tracker for region proposals
    """
    
    def __init__(self):
        self.previous_frame = None
        self.flow_history = deque(maxsize=5)
    
    def track_motion_regions(
        self, 
        previous_frame: np.ndarray, 
        current_frame: np.ndarray,
        min_area: int = 64,
        max_proposals: int = 50
    ) -> List[RegionProposal]:
        """
        Track motion regions using optical flow
        
        Args:
            previous_frame: Previous frame
            current_frame: Current frame
            min_area: Minimum region area
            max_proposals: Maximum number of proposals
            
        Returns:
            List of motion-based region proposals
        """
        try:
            proposals = []
            
            # Convert to grayscale
            prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, None, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Alternative: Dense optical flow
            flow_dense = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate motion magnitude
            magnitude, angle = cv2.cartToPolar(flow_dense[..., 0], flow_dense[..., 1])
            
            # Threshold motion to find moving regions
            motion_threshold = np.percentile(magnitude, 85)  # Top 15% of motion
            motion_mask = (magnitude > motion_threshold).astype(np.uint8) * 255
            
            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours of moving regions
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate motion statistics for this region
                roi_magnitude = magnitude[y:y+h, x:x+w]
                roi_angle = angle[y:y+h, x:x+w]
                
                avg_magnitude = np.mean(roi_magnitude)
                avg_angle = np.mean(roi_angle)
                
                # Motion vector
                motion_x = avg_magnitude * np.cos(avg_angle)
                motion_y = avg_magnitude * np.sin(avg_angle)
                
                # Confidence based on motion strength
                confidence = min(avg_magnitude / (motion_threshold + 1e-6), 1.0)
                
                proposal = RegionProposal(
                    bbox=(x, y, x+w, y+h),
                    confidence=confidence,
                    proposal_type=ProposalType.MOTION,
                    area=area,
                    aspect_ratio=w/h if h > 0 else 1.0,
                    motion_vector=(motion_x, motion_y)
                )
                proposals.append(proposal)
            
            # Sort by confidence and limit
            proposals.sort(key=lambda p: p.confidence, reverse=True)
            return proposals[:max_proposals]
            
        except Exception as e:
            logger.error(f"Motion tracking failed: {e}")
            return []

class ProposalRanker:
    """
    Rank and score region proposals based on multiple criteria
    """
    
    def __init__(self, config: ProposalConfig):
        self.config = config
    
    def rank(self, proposals: List[RegionProposal]) -> List[RegionProposal]:
        """
        Rank proposals based on multiple criteria
        
        Args:
            proposals: List of region proposals
            
        Returns:
            Ranked list of proposals
        """
        try:
            if not proposals:
                return []
            
            # Calculate composite scores
            for proposal in proposals:
                score = 0.0
                
                # Base confidence
                score += proposal.confidence * 0.4
                
                # Type-specific weights
                if proposal.proposal_type == ProposalType.RPN:
                    score += proposal.confidence * self.config.rpn_weight
                elif proposal.proposal_type == ProposalType.SALIENCY:
                    score += proposal.saliency_score * self.config.saliency_weight
                elif proposal.proposal_type == ProposalType.MOTION:
                    score += proposal.confidence * self.config.motion_weight
                
                # Size preference (favor smaller objects)
                size_score = 1.0 - min(proposal.area / 10000, 1.0)  # Normalize by max area
                score += size_score * 0.2
                
                # Aspect ratio preference (avoid extreme ratios)
                aspect_penalty = abs(np.log(proposal.aspect_ratio)) / 2.0
                score -= min(aspect_penalty, 0.3)
                
                # Edge density bonus
                if proposal.edge_density > 0:
                    score += proposal.edge_density * self.config.edge_weight
                
                # Texture score bonus
                if proposal.texture_score > 0:
                    score += proposal.texture_score * self.config.texture_weight
                
                # Update proposal confidence with composite score
                proposal.confidence = max(0.0, min(1.0, score))
            
            # Sort by composite score
            proposals.sort(key=lambda p: p.confidence, reverse=True)
            
            return proposals
            
        except Exception as e:
            logger.error(f"Proposal ranking failed: {e}")
            return proposals

class RegionProposalNetwork:
    """
    Efficient region proposal generation for focused processing
    
    This service focuses computational resources on likely small object regions by:
    1. Lightweight RPN for candidate region generation
    2. Saliency-based region scoring for visual attention
    3. Motion-based region proposals for video sequences
    4. Multi-criteria ranking and filtering
    5. Temporal consistency tracking
    
    Features:
    - Multiple proposal generation methods (RPN, saliency, motion)
    - Intelligent ranking and filtering
    - Temporal consistency across video frames
    - Performance optimization for real-time processing
    - Adaptive proposal generation based on scene complexity
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = ProposalConfig(**(config or {}))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.rpn_model = LightweightRPN()
        self.rpn_model = self.rpn_model.to(self.device)
        self.rpn_model.eval()
        
        self.saliency_detector = SaliencyDetector()
        self.motion_tracker = OpticalFlowTracker()
        self.proposal_ranker = ProposalRanker(self.config)
        
        # Temporal tracking
        self.previous_frame = None
        self.proposal_history = deque(maxsize=10)
        
        # Performance tracking
        self.generation_stats = {
            'total_proposals_generated': 0,
            'proposals_by_type': {ptype.value: 0 for ptype in ProposalType},
            'average_generation_time': 0.0,
            'cache_hits': 0
        }
        
        # Proposal cache
        self.proposal_cache = {}
        self.enable_caching = True
        self.cache_size_limit = 50
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("RegionProposalNetwork initialized successfully")
        logger.info(f"Configuration: {self.config}")
    
    def _create_cache_key(self, frame: np.ndarray) -> str:
        """
        Create cache key for frame-based caching
        """
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        return frame_hash
    
    def _extract_features_for_rpn(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract features for RPN processing
        
        Args:
            image: Input image
            
        Returns:
            Feature tensor
        """
        try:
            # Simple feature extraction using edge detection and gradients
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            resized = cv2.resize(gray, (512, 512))
            
            # Calculate gradients
            grad_x = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
            
            # Combine gradients
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Convert to tensor
            feature_tensor = torch.from_numpy(magnitude).float().unsqueeze(0).unsqueeze(0)
            
            # Expand to 256 channels (simulate backbone features)
            feature_tensor = feature_tensor.repeat(1, 256, 1, 1)
            
            return feature_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Feature extraction for RPN failed: {e}")
            # Return dummy tensor
            return torch.zeros(1, 256, 32, 32).to(self.device)
    
    def _generate_rpn_proposals(
        self, 
        image: np.ndarray, 
        max_proposals: int = 50
    ) -> List[RegionProposal]:
        """
        Generate proposals using lightweight RPN
        
        Args:
            image: Input image
            max_proposals: Maximum number of proposals
            
        Returns:
            List of RPN-based proposals
        """
        try:
            # Extract features
            features = self._extract_features_for_rpn(image)
            
            # Run RPN
            with torch.no_grad():
                cls_logits, bbox_pred = self.rpn_model(features)
            
            # Generate anchors
            feature_map_size = (features.shape[2], features.shape[3])
            anchors = self.rpn_model.generate_anchors(feature_map_size)
            
            # Convert predictions to proposals
            proposals = []
            
            # Flatten predictions
            cls_scores = torch.sigmoid(cls_logits).flatten()
            bbox_deltas = bbox_pred.view(-1, 4)
            
            # Apply bbox deltas to anchors (simplified)
            h, w = image.shape[:2]
            
            for i in range(min(len(anchors), max_proposals * 2)):
                if i >= len(cls_scores):
                    break
                
                confidence = cls_scores[i].item()
                
                # Skip low confidence proposals
                if confidence < 0.1:
                    continue
                
                # Get anchor box
                anchor = anchors[i]
                x1, y1, x2, y2 = anchor.tolist()
                
                # Apply simple clipping
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                area = (x2 - x1) * (y2 - y1)
                
                # Filter by area
                if area < self.config.min_area or area > self.config.max_area:
                    continue
                
                proposal = RegionProposal(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    proposal_type=ProposalType.RPN,
                    area=area,
                    aspect_ratio=(x2-x1)/(y2-y1) if (y2-y1) > 0 else 1.0
                )
                proposals.append(proposal)
            
            # Sort and limit
            proposals.sort(key=lambda p: p.confidence, reverse=True)
            return proposals[:max_proposals]
            
        except Exception as e:
            logger.error(f"RPN proposal generation failed: {e}")
            return []
    
    def _apply_nms_to_proposals(
        self, 
        proposals: List[RegionProposal], 
        nms_threshold: float = None
    ) -> List[RegionProposal]:
        """
        Apply Non-Maximum Suppression to proposals
        
        Args:
            proposals: List of proposals
            nms_threshold: NMS threshold
            
        Returns:
            Filtered proposals
        """
        if not proposals:
            return []
        
        nms_threshold = nms_threshold or self.config.nms_threshold
        
        try:
            # Convert to tensors
            boxes = torch.tensor([p.bbox for p in proposals], dtype=torch.float32)
            scores = torch.tensor([p.confidence for p in proposals], dtype=torch.float32)
            
            # Apply NMS
            keep_indices = nms(boxes, scores, nms_threshold)
            
            # Return filtered proposals
            return [proposals[i] for i in keep_indices.tolist()]
            
        except Exception as e:
            logger.error(f"NMS failed: {e}")
            return proposals
    
    def _calculate_temporal_consistency(
        self, 
        current_proposals: List[RegionProposal]
    ) -> List[RegionProposal]:
        """
        Apply temporal consistency filtering
        
        Args:
            current_proposals: Current frame proposals
            
        Returns:
            Temporally consistent proposals
        """
        if len(self.proposal_history) < 2:
            return current_proposals
        
        try:
            # Compare with previous frames
            consistent_proposals = []
            
            for proposal in current_proposals:
                consistency_score = 0.0
                matches = 0
                
                # Check against recent history
                for prev_proposals in list(self.proposal_history)[-3:]:
                    for prev_proposal in prev_proposals:
                        # Calculate IoU
                        iou = self._calculate_iou(proposal.bbox, prev_proposal.bbox)
                        if iou > 0.3:  # Threshold for temporal consistency
                            consistency_score += iou
                            matches += 1
                            break
                
                # Boost confidence for temporally consistent proposals
                if matches > 0:
                    avg_consistency = consistency_score / matches
                    proposal.confidence *= (1.0 + avg_consistency * 0.2)
                
                consistent_proposals.append(proposal)
            
            return consistent_proposals
            
        except Exception as e:
            logger.error(f"Temporal consistency calculation failed: {e}")
            return current_proposals
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU score
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
    
    async def generate_proposals(
        self, 
        current_frame: np.ndarray,
        previous_frame: Optional[np.ndarray] = None,
        max_proposals: int = None
    ) -> List[RegionProposal]:
        """
        Generate ranked region proposals for efficient processing
        
        Args:
            current_frame: Current video frame
            previous_frame: Previous frame for motion analysis
            max_proposals: Maximum number of proposals to return
            
        Returns:
            List of ranked region proposals
        """
        start_time = time.time()
        max_proposals = max_proposals or self.config.max_proposals_per_frame
        
        try:
            with self.lock:
                # Check cache
                cache_key = self._create_cache_key(current_frame)
                if self.enable_caching and cache_key in self.proposal_cache:
                    self.generation_stats['cache_hits'] += 1
                    logger.debug("Returning cached proposals")
                    return self.proposal_cache[cache_key]
                
                logger.info(f"Generating region proposals (max: {max_proposals})")
                
                all_proposals = []
                
                # 1. RPN-based proposals
                logger.debug("Generating RPN proposals")
                rpn_proposals = self._generate_rpn_proposals(
                    current_frame, max_proposals // 3
                )
                all_proposals.extend(rpn_proposals)
                self.generation_stats['proposals_by_type'][ProposalType.RPN.value] += len(rpn_proposals)
                
                # 2. Saliency-based proposals
                logger.debug("Generating saliency proposals")
                saliency_proposals = self.saliency_detector.detect_salient_regions(
                    current_frame, 
                    min_area=self.config.min_area,
                    max_proposals=max_proposals // 3
                )
                all_proposals.extend(saliency_proposals)
                self.generation_stats['proposals_by_type'][ProposalType.SALIENCY.value] += len(saliency_proposals)
                
                # 3. Motion-based proposals (if previous frame available)
                if previous_frame is not None:
                    logger.debug("Generating motion proposals")
                    motion_proposals = self.motion_tracker.track_motion_regions(
                        previous_frame, 
                        current_frame,
                        min_area=self.config.min_area,
                        max_proposals=max_proposals // 3
                    )
                    all_proposals.extend(motion_proposals)
                    self.generation_stats['proposals_by_type'][ProposalType.MOTION.value] += len(motion_proposals)
                
                logger.debug(f"Generated {len(all_proposals)} raw proposals")
                
                # 4. Rank and filter proposals
                ranked_proposals = self.proposal_ranker.rank(all_proposals)
                
                # 5. Apply NMS to remove duplicates
                filtered_proposals = self._apply_nms_to_proposals(ranked_proposals)
                
                # 6. Apply temporal consistency
                consistent_proposals = self._calculate_temporal_consistency(filtered_proposals)
                
                # 7. Final filtering and limiting
                final_proposals = consistent_proposals[:max_proposals]
                
                # Update temporal history
                self.proposal_history.append(final_proposals)
                
                # Cache results
                if self.enable_caching:
                    if len(self.proposal_cache) >= self.cache_size_limit:
                        # Remove oldest entry
                        oldest_key = next(iter(self.proposal_cache))
                        del self.proposal_cache[oldest_key]
                    
                    self.proposal_cache[cache_key] = final_proposals
                
                # Update statistics
                generation_time = time.time() - start_time
                self.generation_stats['total_proposals_generated'] += len(final_proposals)
                self.generation_stats['average_generation_time'] = (
                    (self.generation_stats['average_generation_time'] * 0.9) + 
                    (generation_time * 0.1)
                )
                
                logger.info(f"Generated {len(final_proposals)} final proposals in {generation_time:.3f}s")
                return final_proposals
                
        except Exception as e:
            logger.error(f"Region proposal generation failed: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            stats = self.generation_stats.copy()
            stats.update({
                'config': {
                    'max_proposals_per_frame': self.config.max_proposals_per_frame,
                    'nms_threshold': self.config.nms_threshold,
                    'min_area': self.config.min_area,
                    'max_area': self.config.max_area
                },
                'cache_size': len(self.proposal_cache),
                'cache_enabled': self.enable_caching,
                'proposal_history_size': len(self.proposal_history),
                'device': str(self.device)
            })
            
            return stats
    
    def clear_cache(self):
        """
        Clear proposal cache and history
        """
        with self.lock:
            self.proposal_cache.clear()
            self.proposal_history.clear()
            self.previous_frame = None
            
        logger.info("Region proposal network cache cleared")
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update configuration parameters
        
        Args:
            new_config: New configuration parameters
        """
        with self.lock:
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"Updated config: {key} = {value}")
            
            # Update ranker config
            self.proposal_ranker.config = self.config