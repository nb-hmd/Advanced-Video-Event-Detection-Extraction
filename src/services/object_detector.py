import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import torch
from PIL import Image
import gc
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import time

# Optional imports for advanced features
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    rembg = None
    REMBG_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    sam_model_registry = None
    SamPredictor = None
    SAM_AVAILABLE = False

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    A = None
    ALBUMENTATIONS_AVAILABLE = False

try:
    import timm
    from torchvision import transforms
    TIMM_AVAILABLE = True
except ImportError:
    timm = None
    transforms = None
    TIMM_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.config import settings
from ..utils.memory_manager import memory_manager
from ..utils.error_handler import error_handler

logger = get_logger(__name__)

class ObjectDetector:
    """
    Advanced Object Detection Service for Background-Independent Matching
    
    This service provides sophisticated object detection and segmentation capabilities
    to enable matching objects/persons across different backgrounds and contexts.
    
    Features:
    1. YOLO-based object detection and classification
    2. Background removal using rembg and SAM
    3. Feature extraction from segmented objects
    4. Object-focused similarity computation
    5. Multi-scale object matching
    6. Person/vehicle/object specific detection
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Model instances
        self.yolo_model = None
        self.sam_predictor = None
        self.background_remover = None
        self.feature_extractor = None
        
        # Configuration
        self.detection_confidence = 0.25
        self.iou_threshold = 0.45
        self.max_detections = 100
        
        # Supported object classes for focused detection (YOLO classes)
        self.target_classes = {
            'person': 0,
            'bicycle': 1,
            'car': 2,
            'motorcycle': 3,
            'airplane': 4,
            'bus': 5,
            'train': 6,
            'truck': 7,
            'boat': 8
        }
        
        # Universal detection capability
        self.unlimited_detection = True
        self.universal_detector = None  # Lazy loaded
        
        # Detection modes
        self.detection_modes = {
            'yolo_only': 'Traditional YOLO detection with predefined classes',
            'universal': 'Unlimited object detection with natural language',
            'hybrid': 'Combined YOLO + universal detection for maximum coverage'
        }
        
        # Cache for processed objects
        self.object_cache = {}
        self.enable_caching = True
        
        logger.info(f"ObjectDetector initialized on {self.device}")
    
    def _lazy_load_yolo(self):
        """Lazy load YOLO model to save memory."""
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available - object detection disabled")
            return False
            
        if self.yolo_model is None:
            try:
                logger.info("Loading YOLO model...")
                # Use YOLOv8n for speed, YOLOv8x for accuracy
                model_size = getattr(settings, 'YOLO_MODEL_SIZE', 'n')  # n, s, m, l, x
                self.yolo_model = YOLO(f'yolov8{model_size}.pt')
                
                if self.use_gpu:
                    self.yolo_model.to(self.device)
                
                logger.info(f"YOLO model loaded successfully: yolov8{model_size}")
                return True
            except Exception as e:
                logger.error(f"Error loading YOLO model: {e}")
                return False
        return True
    
    def _lazy_load_sam(self):
        """Lazy load Segment Anything Model."""
        if self.sam_predictor is None:
            try:
                logger.info("Loading SAM model...")
                # Use SAM-B for balance of speed and accuracy
                sam_checkpoint = "sam_vit_b_01ec64.pth"
                model_type = "vit_b"
                
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device=self.device)
                self.sam_predictor = SamPredictor(sam)
                
                logger.info("SAM model loaded successfully")
            except Exception as e:
                logger.warning(f"SAM model loading failed: {e}. Continuing without SAM.")
                self.sam_predictor = None
    
    def _lazy_load_background_remover(self):
        """Lazy load background removal model."""
        if self.background_remover is None:
            try:
                logger.info("Loading background removal model...")
                self.background_remover = rembg.new_session('u2net')
                logger.info("Background removal model loaded successfully")
            except Exception as e:
                logger.warning(f"Background removal model loading failed: {e}")
                self.background_remover = None
    
    def _lazy_load_feature_extractor(self):
        """Lazy load feature extraction model."""
        if self.feature_extractor is None:
            try:
                logger.info("Loading feature extraction model...")
                # Use EfficientNet for feature extraction
                self.feature_extractor = timm.create_model(
                    'efficientnet_b0', 
                    pretrained=True, 
                    num_classes=0  # Remove classification head
                )
                self.feature_extractor.eval()
                self.feature_extractor.to(self.device)
                
                # Define preprocessing transforms
                self.feature_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                logger.info("Feature extraction model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading feature extraction model: {e}")
                raise
    
    def detect_objects(self, image: np.ndarray, target_classes: Optional[List[str]] = None) -> List[Dict]:
        """
        Detect objects in an image using YOLO.
        
        Args:
            image: Input image as numpy array
            target_classes: List of target class names to focus on
            
        Returns:
            List of detection dictionaries with bbox, confidence, class info
        """
        if not self._lazy_load_yolo():
            logger.warning("YOLO model not available - returning empty detection list")
            return []
        
        try:
            # Run YOLO detection
            results = self.yolo_model(
                image,
                conf=self.detection_confidence,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Extract detection info
                        bbox = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        # Filter by target classes if specified
                        if target_classes and class_name not in target_classes:
                            continue
                        
                        detection = {
                            'bbox': bbox,
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        }
                        detections.append(detection)
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []
    
    def segment_object(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Enhanced object segmentation with multiple fallback methods.
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Segmented object mask or None if segmentation fails
        """
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Ensure valid bbox
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Method 1: Try SAM if available
            if self.sam_predictor is not None:
                try:
                    self.sam_predictor.set_image(image)
                    
                    # Use bbox as prompt
                    input_box = np.array([x1, y1, x2, y2])
                    masks, scores, _ = self.sam_predictor.predict(
                        box=input_box,
                        multimask_output=False
                    )
                    
                    if len(masks) > 0 and masks[0] is not None:
                        return masks[0]
                except Exception as e:
                    logger.warning(f"SAM segmentation failed: {e}")
            
            # Method 2: Background removal with enhanced processing
            if self.background_remover is not None:
                try:
                    # Crop object region with padding
                    padding = 10
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(w, x2 + padding)
                    crop_y2 = min(h, y2 + padding)
                    
                    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # Remove background
                    cropped_pil = Image.fromarray(cropped)
                    result = rembg.remove(cropped_pil, session=self.background_remover)
                    
                    # Convert back to mask
                    result_np = np.array(result)
                    if len(result_np.shape) == 3 and result_np.shape[2] == 4:
                        alpha_mask = result_np[:, :, 3] > 128  # Alpha channel with threshold
                        
                        # Create full image mask
                        full_mask = np.zeros(image.shape[:2], dtype=bool)
                        full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = alpha_mask
                        
                        # Refine mask to original bbox region
                        refined_mask = np.zeros_like(full_mask)
                        refined_mask[y1:y2, x1:x2] = full_mask[y1:y2, x1:x2]
                        
                        return refined_mask
                except Exception as e:
                    logger.warning(f"Background removal failed: {e}")
            
            # Method 3: Enhanced edge-based segmentation
            try:
                cropped = image[y1:y2, x1:x2]
                
                # Convert to grayscale for edge detection
                if len(cropped.shape) == 3:
                    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
                else:
                    gray_crop = cropped
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray_crop, (5, 5), 0)
                
                # Edge detection
                edges = cv2.Canny(blurred, 50, 150)
                
                # Morphological operations to close gaps
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Create mask from contour
                    crop_mask = np.zeros(gray_crop.shape, dtype=np.uint8)
                    cv2.fillPoly(crop_mask, [largest_contour], 255)
                    
                    # Create full image mask
                    full_mask = np.zeros(image.shape[:2], dtype=bool)
                    full_mask[y1:y2, x1:x2] = crop_mask > 0
                    
                    return full_mask
            except Exception as e:
                logger.warning(f"Edge-based segmentation failed: {e}")
            
            # Method 4: Adaptive threshold segmentation
            try:
                cropped = image[y1:y2, x1:x2]
                
                # Convert to grayscale
                if len(cropped.shape) == 3:
                    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
                else:
                    gray_crop = cropped
                
                # Apply adaptive threshold
                thresh = cv2.adaptiveThreshold(
                    gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                
                # Find the largest connected component
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
                
                if num_labels > 1:
                    # Find largest component (excluding background)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    
                    # Create mask
                    crop_mask = (labels == largest_label).astype(np.uint8) * 255
                    
                    # Create full image mask
                    full_mask = np.zeros(image.shape[:2], dtype=bool)
                    full_mask[y1:y2, x1:x2] = crop_mask > 0
                    
                    return full_mask
            except Exception as e:
                logger.warning(f"Adaptive threshold segmentation failed: {e}")
            
            # Method 5: Simple bbox mask fallback
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[y1:y2, x1:x2] = True
            return mask
            
        except Exception as e:
            logger.error(f"All segmentation methods failed: {e}")
            # Return simple bbox mask as final fallback
            try:
                x1, y1, x2, y2 = bbox.astype(int)
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                mask = np.zeros((h, w), dtype=bool)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = True
                return mask
            except:
                return None
    
    def extract_object_features(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Enhanced feature extraction from segmented object with background independence.
        
        Args:
            image: Original image
            mask: Object segmentation mask
            
        Returns:
            Feature vector for the object
        """
        self._lazy_load_feature_extractor()
        
        try:
            # Find bounding box of mask
            coords = np.where(mask)
            if len(coords[0]) == 0:
                return np.zeros(1280)  # EfficientNet-B0 feature size
            
            y1, y2 = coords[0].min(), coords[0].max()
            x1, x2 = coords[1].min(), coords[1].max()
            
            # Ensure valid crop region
            if y2 <= y1 or x2 <= x1:
                return np.zeros(1280)
            
            # Method 1: Masked object with neutral background
            try:
                # Create masked image with neutral gray background
                masked_image = image.copy()
                
                # Use mean color of object as background instead of black
                object_pixels = image[mask]
                if len(object_pixels) > 0:
                    mean_color = np.mean(object_pixels, axis=0)
                    # Use slightly darker version to avoid confusion
                    bg_color = (mean_color * 0.7).astype(np.uint8)
                else:
                    bg_color = np.array([128, 128, 128])  # Neutral gray
                
                masked_image[~mask] = bg_color
                
                # Crop to object region with padding
                padding = max(5, min(10, (x2-x1)//10, (y2-y1)//10))
                crop_y1 = max(0, y1 - padding)
                crop_y2 = min(image.shape[0], y2 + padding)
                crop_x1 = max(0, x1 - padding)
                crop_x2 = min(image.shape[1], x2 + padding)
                
                object_crop = masked_image[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Ensure minimum size for feature extraction
                if object_crop.shape[0] < 32 or object_crop.shape[1] < 32:
                    object_crop = cv2.resize(object_crop, (64, 64), interpolation=cv2.INTER_CUBIC)
                
                # Convert to PIL and preprocess
                object_pil = Image.fromarray(object_crop)
                object_tensor = self.feature_transform(object_pil).unsqueeze(0).to(self.device)
                
                # Extract deep features
                with torch.no_grad():
                    deep_features = self.feature_extractor(object_tensor)
                    deep_features = deep_features.cpu().numpy().flatten()
                
                # Normalize deep features
                deep_features = deep_features / (np.linalg.norm(deep_features) + 1e-8)
                
                return deep_features
                
            except Exception as e:
                logger.warning(f"Deep feature extraction failed: {e}")
            
            # Method 2: Fallback to traditional features
            try:
                # Extract object region
                object_crop = image[y1:y2+1, x1:x2+1]
                mask_crop = mask[y1:y2+1, x1:x2+1]
                
                # Apply mask to crop
                masked_crop = object_crop.copy()
                masked_crop[~mask_crop] = 0
                
                # Extract multiple types of features
                features_list = []
                
                # 1. Color histogram features (background independent)
                if len(object_crop.shape) == 3:
                    for channel in range(3):
                        channel_data = object_crop[:, :, channel][mask_crop]
                        if len(channel_data) > 0:
                            hist, _ = np.histogram(channel_data, bins=32, range=(0, 256))
                            hist = hist.astype(float) / (np.sum(hist) + 1e-8)
                            features_list.append(hist)
                
                # 2. Shape features
                contours, _ = cv2.findContours(mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Hu moments (scale, rotation, translation invariant)
                    moments = cv2.moments(largest_contour)
                    hu_moments = cv2.HuMoments(moments).flatten()
                    hu_moments = np.log(np.abs(hu_moments) + 1e-8)  # Log transform
                    features_list.append(hu_moments)
                    
                    # Aspect ratio and solidity
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    area = cv2.contourArea(largest_contour)
                    hull = cv2.convexHull(largest_contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    
                    shape_features = np.array([aspect_ratio, solidity])
                    features_list.append(shape_features)
                
                # 3. Texture features (LBP)
                if len(masked_crop.shape) == 3:
                    gray_crop = cv2.cvtColor(masked_crop, cv2.COLOR_RGB2GRAY)
                else:
                    gray_crop = masked_crop
                
                # Apply mask to gray image
                gray_crop[~mask_crop] = 0
                
                if np.sum(mask_crop) > 100:  # Enough pixels for LBP
                    try:
                        from skimage import feature
                        radius = 2
                        n_points = 8 * radius
                        lbp = feature.local_binary_pattern(gray_crop, n_points, radius, method='uniform')
                        lbp_hist, _ = np.histogram(lbp[mask_crop], bins=n_points + 2, range=(0, n_points + 2))
                        lbp_hist = lbp_hist.astype(float) / (np.sum(lbp_hist) + 1e-8)
                        features_list.append(lbp_hist)
                    except:
                        pass
                
                # Combine all features
                if features_list:
                    combined_features = np.concatenate(features_list)
                    # Pad or truncate to standard size
                    if len(combined_features) < 1280:
                        padded_features = np.zeros(1280)
                        padded_features[:len(combined_features)] = combined_features
                        combined_features = padded_features
                    else:
                        combined_features = combined_features[:1280]
                    
                    # Normalize
                    combined_features = combined_features / (np.linalg.norm(combined_features) + 1e-8)
                    return combined_features
                
            except Exception as e:
                logger.warning(f"Traditional feature extraction failed: {e}")
            
            # Final fallback: simple statistical features
            try:
                object_pixels = image[mask]
                if len(object_pixels) > 0:
                    # Basic statistical features
                    mean_color = np.mean(object_pixels, axis=0)
                    std_color = np.std(object_pixels, axis=0)
                    
                    # Shape statistics
                    bbox_area = (x2 - x1) * (y2 - y1)
                    mask_area = np.sum(mask)
                    fill_ratio = mask_area / bbox_area if bbox_area > 0 else 0
                    
                    # Combine into feature vector
                    basic_features = np.concatenate([
                        mean_color.flatten(),
                        std_color.flatten(),
                        [fill_ratio, mask_area, bbox_area]
                    ])
                    
                    # Pad to standard size
                    padded_features = np.zeros(1280)
                    padded_features[:len(basic_features)] = basic_features
                    
                    # Normalize
                    padded_features = padded_features / (np.linalg.norm(padded_features) + 1e-8)
                    return padded_features
            
            except Exception as e:
                logger.error(f"Basic feature extraction failed: {e}")
            
            # Ultimate fallback
            return np.zeros(1280)
            
        except Exception as e:
            logger.error(f"Error extracting object features: {e}")
            return np.zeros(1280)
    
    def process_reference_image(self, image: np.ndarray, target_class: Optional[str] = None) -> List[Dict]:
        """
        Process reference image to extract object features.
        
        Args:
            image: Reference image
            target_class: Specific class to focus on (e.g., 'person', 'car')
            
        Returns:
            List of processed objects with features
        """
        # Create cache key
        cache_key = hashlib.md5(image.tobytes()).hexdigest()[:16]
        if target_class:
            cache_key += f"_{target_class}"
        
        if self.enable_caching and cache_key in self.object_cache:
            logger.info("Returning cached reference object features")
            return self.object_cache[cache_key]
        
        logger.info(f"Processing reference image for class: {target_class or 'all'}")
        
        # Detect objects
        target_classes = [target_class] if target_class else None
        detections = self.detect_objects(image, target_classes)
        
        processed_objects = []
        
        for detection in detections:
            # Segment object
            mask = self.segment_object(image, detection['bbox'])
            if mask is None:
                continue
            
            # Extract features
            features = self.extract_object_features(image, mask)
            
            processed_object = {
                'detection': detection,
                'mask': mask,
                'features': features,
                'bbox': detection['bbox'],
                'class_name': detection['class_name'],
                'confidence': detection['confidence']
            }
            processed_objects.append(processed_object)
        
        # Cache results
        if self.enable_caching:
            self.object_cache[cache_key] = processed_objects
        
        logger.info(f"Processed {len(processed_objects)} objects from reference image")
        return processed_objects
    
    def match_objects_in_frame(self, frame: np.ndarray, reference_objects: List[Dict], 
                              similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Match reference objects in a video frame.
        
        Args:
            frame: Video frame to search in
            reference_objects: List of reference object features
            similarity_threshold: Minimum similarity for a match
            
        Returns:
            List of matches with similarity scores
        """
        if not reference_objects:
            return []
        
        # Get target classes from reference objects
        target_classes = list(set(obj['class_name'] for obj in reference_objects))
        
        # Detect objects in frame
        frame_detections = self.detect_objects(frame, target_classes)
        
        matches = []
        
        for frame_detection in frame_detections:
            # Segment object in frame
            frame_mask = self.segment_object(frame, frame_detection['bbox'])
            if frame_mask is None:
                continue
            
            # Extract features from frame object
            frame_features = self.extract_object_features(frame, frame_mask)
            
            # Compare with reference objects
            best_match = None
            best_similarity = 0.0
            
            for ref_obj in reference_objects:
                # Only compare objects of same class
                if ref_obj['class_name'] != frame_detection['class_name']:
                    continue
                
                # Compute feature similarity
                similarity = cosine_similarity(
                    ref_obj['features'].reshape(1, -1),
                    frame_features.reshape(1, -1)
                )[0, 0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = ref_obj
            
            # Check if similarity meets threshold
            if best_similarity >= similarity_threshold:
                match = {
                    'frame_detection': frame_detection,
                    'reference_object': best_match,
                    'similarity': best_similarity,
                    'bbox': frame_detection['bbox'],
                    'class_name': frame_detection['class_name'],
                    'confidence': frame_detection['confidence']
                }
                matches.append(match)
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches
    
    def _lazy_load_universal_detector(self):
        """Lazy load universal detector for unlimited object classes."""
        if self.universal_detector is None:
            try:
                logger.info("Loading UniversalDetector for unlimited object classes...")
                from .universal_detector import UniversalDetector
                self.universal_detector = UniversalDetector(use_gpu=self.use_gpu)
                logger.info("UniversalDetector loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading UniversalDetector: {e}")
                return False
        return True
    
    def detect_unlimited_objects(self, image: np.ndarray, 
                               text_queries: Union[str, List[str]],
                               detection_mode: str = 'universal',
                               confidence_threshold: float = 0.1) -> List[Dict]:
        """
        Detect unlimited object classes using natural language queries.
        
        Args:
            image: Input image as numpy array
            text_queries: Single query or list of object descriptions
            detection_mode: 'universal', 'hybrid', or 'yolo_only'
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of detection dictionaries with unlimited object classes
        """
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        logger.info(f"Detecting unlimited objects with mode: {detection_mode}")
        logger.info(f"Queries: {text_queries}")
        
        all_detections = []
        
        try:
            if detection_mode == 'yolo_only':
                # Use traditional YOLO detection
                yolo_detections = self.detect_objects(image)
                
                # Filter by semantic similarity to queries
                for detection in yolo_detections:
                    class_name = detection.get('class_name', '')
                    for query in text_queries:
                        semantic_score = self._compute_text_similarity(class_name, query)
                        if semantic_score > 0.3:
                            detection['query'] = query
                            detection['semantic_score'] = semantic_score
                            detection['method'] = 'yolo_semantic'
                            all_detections.append(detection)
            
            elif detection_mode == 'universal':
                # Use universal detector for unlimited classes
                if self._lazy_load_universal_detector():
                    universal_detections = self.universal_detector.detect_unlimited_objects(
                        image=image,
                        text_queries=text_queries,
                        detection_mode='hybrid',
                        confidence_threshold=confidence_threshold
                    )
                    all_detections.extend(universal_detections)
                else:
                    logger.warning("Universal detector not available, falling back to YOLO")
                    return self.detect_unlimited_objects(image, text_queries, 'yolo_only', confidence_threshold)
            
            elif detection_mode == 'hybrid':
                # Combine YOLO and universal detection
                yolo_detections = self.detect_unlimited_objects(image, text_queries, 'yolo_only', confidence_threshold)
                universal_detections = self.detect_unlimited_objects(image, text_queries, 'universal', confidence_threshold)
                
                # Merge and deduplicate
                all_detections = self._merge_detections(yolo_detections + universal_detections)
            
            # Filter by confidence threshold
            filtered_detections = [
                det for det in all_detections 
                if det.get('confidence', 0) >= confidence_threshold
            ]
            
            logger.info(f"Unlimited detection completed: {len(filtered_detections)} objects found")
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Error in unlimited object detection: {e}")
            return []
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text strings."""
        try:
            # Simple word-based similarity for fallback
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # Boost for exact matches or substrings
            if text1.lower() in text2.lower() or text2.lower() in text1.lower():
                jaccard_similarity = max(jaccard_similarity, 0.8)
            
            return jaccard_similarity
            
        except Exception as e:
            logger.error(f"Error computing text similarity: {e}")
            return 0.0
    
    def _merge_detections(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Merge overlapping detections from different methods."""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        merged = []
        
        for detection in detections:
            # Check if this detection overlaps significantly with existing ones
            should_add = True
            
            for existing in merged:
                iou = self._compute_iou(detection.get('bbox', []), existing.get('bbox', []))
                if iou > iou_threshold:
                    # If same query and high overlap, skip
                    if detection.get('query') == existing.get('query'):
                        should_add = False
                        break
            
            if should_add:
                merged.append(detection)
        
        return merged
    
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
    
    def get_supported_detection_modes(self) -> Dict[str, str]:
        """Get supported detection modes and their descriptions."""
        return self.detection_modes.copy()
    
    def cleanup(self):
        """Clean up resources."""
        if self.yolo_model is not None:
            del self.yolo_model
            self.yolo_model = None
        
        if self.sam_predictor is not None:
            del self.sam_predictor
            self.sam_predictor = None
        
        if self.background_remover is not None:
            del self.background_remover
            self.background_remover = None
        
        if self.feature_extractor is not None:
            del self.feature_extractor
            self.feature_extractor = None
        
        if self.universal_detector is not None:
            self.universal_detector.cleanup()
            self.universal_detector = None
        
        # Clear cache
        self.object_cache.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("ObjectDetector cleanup completed")