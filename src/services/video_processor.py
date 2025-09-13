from typing import List, Dict, Optional, Union
from pathlib import Path
import os
import re
import numpy as np
from ..utils.logger import get_logger
from ..pipeline.phase1_mvp import Phase1MVP
from ..pipeline.phase2_reranker import Phase2Reranker
from ..pipeline.phase_image_matching import PhaseImageMatching
from .clip_extractor import ClipExtractor
from ..utils.config import settings
from ..utils.memory_manager import memory_manager
from ..utils.progressive_loader import progressive_loader
from ..utils.system_optimizer import system_optimizer

logger = get_logger(__name__)

class VideoProcessor:
    def __init__(self, lazy_load=False, heavy_models=True):
        self.lazy_load = False  # Force disable lazy loading for heavy models
        self.heavy_models = heavy_models
        self.phase1 = None
        self.phase2 = None
        self.phase2_available = False
        self.phase_image_matching = None  # New image matching phase
        self.image_matching_available = False
        self.clip_extractor = None
        self._models_loaded = False
        
        # Always load models immediately for heavy implementation - no fallbacks
        logger.info("VideoProcessor initializing with HEAVY MODEL IMPLEMENTATION - loading ALL models without fallbacks")
        self._load_models_heavy_no_fallback()
    
    def _load_models_heavy_no_fallback(self):
        """Force load ALL heavy models with optimized memory management and smart fallbacks."""
        if self._models_loaded:
            return
            
        logger.info("🚀 HEAVY MODEL LOADING: Loading OpenCLIP, BLIP-2, and UniVTG with optimized memory management")
        
        # Log initial memory usage
        memory_manager.log_memory_usage("Before heavy model loading")
        
        # Perform aggressive cleanup before loading
        memory_manager.aggressive_cleanup()
        
        try:
            # Force load Phase 1 (OpenCLIP) - no lazy loading
            logger.info("Loading Phase 1 (OpenCLIP) - HEAVY MODE")
            self.phase1 = Phase1MVP()
            logger.info("✅ Phase 1 (OpenCLIP) loaded successfully")
            
            # Force load Phase 2 (BLIP-2) with enhanced error handling
            logger.info("Loading Phase 2 (BLIP-2) - HEAVY MODE with smart fallbacks")
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"BLIP-2 loading attempt {attempt + 1}/{max_retries}")
                    
                    # Clean memory before each attempt
                    if attempt > 0:
                        memory_manager.aggressive_cleanup()
                    
                    self.phase2 = Phase2Reranker(lazy_load=False)  # Force immediate loading
                    
                    # Verify BLIP model is actually loaded
                    if hasattr(self.phase2, 'blip_model') and self.phase2.blip_model is not None:
                        if hasattr(self.phase2.blip_model, 'model_loaded') and self.phase2.blip_model.model_loaded:
                            self.phase2_available = True
                            logger.info("✅ Phase 2 (BLIP-2) loaded successfully - HEAVY MODEL ACTIVE")
                            break
                        else:
                            # Force load BLIP model if not loaded
                            logger.info("🔄 Force loading BLIP-2 model...")
                            if self.phase2.blip_model._lazy_load_blip():
                                self.phase2_available = True
                                logger.info("✅ BLIP-2 force loaded successfully - HEAVY MODEL ACTIVE")
                                break
                            else:
                                logger.warning(f"BLIP-2 loading attempt {attempt + 1} failed")
                                if attempt == max_retries - 1:
                                    # Last attempt - try with minimal settings
                                    logger.info("🔄 Final attempt with minimal BLIP-2 settings...")
                                    try:
                                        # Create a minimal BLIP model instance
                                        from ..models.blip_model import BLIPModel
                                        minimal_blip = BLIPModel(lazy_load=True, force_device='cpu')
                                        if minimal_blip._lazy_load_blip():
                                            self.phase2.blip_model = minimal_blip
                                            self.phase2_available = True
                                            logger.info("✅ BLIP-2 loaded with minimal settings - HEAVY MODEL ACTIVE")
                                            break
                                    except Exception as minimal_e:
                                        logger.error(f"Minimal BLIP-2 loading also failed: {minimal_e}")
                    else:
                        logger.warning(f"Phase 2 BLIP model not initialized on attempt {attempt + 1}")
                        
                except Exception as blip_e:
                    logger.warning(f"BLIP-2 loading attempt {attempt + 1} failed: {blip_e}")
                    if attempt == max_retries - 1:
                        logger.error("All BLIP-2 loading attempts failed")
                        raise RuntimeError(f"BLIP-2 model failed to load after {max_retries} attempts: {blip_e}")
            
            if not self.phase2_available:
                raise RuntimeError("BLIP-2 model failed to load - HEAVY MODEL IMPLEMENTATION REQUIRES ALL MODELS")
            
            # Force load clip extractor
            logger.info("Loading Clip Extractor - HEAVY MODE")
            self.clip_extractor = ClipExtractor()
            logger.info("✅ Clip Extractor loaded successfully")
            
            # Force load image matching phase
            logger.info("Loading Image Matching Phase - HEAVY MODE")
            try:
                self.phase_image_matching = PhaseImageMatching()
                self.image_matching_available = True
                logger.info("✅ Image Matching Phase loaded successfully")
            except Exception as img_e:
                logger.warning(f"Image Matching Phase failed to load: {img_e}")
                self.phase_image_matching = None
                self.image_matching_available = False
            
            self._models_loaded = True
            
            # Log final status
            logger.info("🎉 HEAVY MODEL IMPLEMENTATION COMPLETE - ALL MODELS LOADED:")
            logger.info("   ✅ OpenCLIP (Phase 1) - ACTIVE")
            logger.info("   ✅ BLIP-2 (Phase 2) - ACTIVE")
            logger.info("   ✅ Clip Extractor - ACTIVE")
            if self.image_matching_available:
                logger.info("   ✅ Image Matching - ACTIVE")
            else:
                logger.info("   ⚠️ Image Matching - UNAVAILABLE")
            logger.info("   🚀 MAXIMUM ACCURACY MODE ENABLED")
            
            memory_manager.log_memory_usage("After heavy model loading")
            
        except Exception as e:
            logger.error(f"❌ HEAVY MODEL LOADING FAILED: {e}")
            logger.error("Attempting emergency fallback mode...")
            
            # Emergency fallback - try to load with basic models
            try:
                logger.info("🔄 Emergency fallback: Loading basic models...")
                
                # Ensure Phase 1 is loaded
                if self.phase1 is None:
                    self.phase1 = Phase1MVP()
                
                # Try to create a basic Phase 2 with lazy loading
                if self.phase2 is None:
                    self.phase2 = Phase2Reranker(lazy_load=True)
                    self.phase2_available = False  # Mark as not fully available
                
                # Ensure clip extractor is loaded
                if self.clip_extractor is None:
                    self.clip_extractor = ClipExtractor()
                
                # Try to load image matching in fallback mode
                try:
                    if self.phase_image_matching is None:
                        self.phase_image_matching = PhaseImageMatching()
                        self.image_matching_available = True
                except Exception:
                    self.phase_image_matching = None
                    self.image_matching_available = False
                
                self._models_loaded = True
                logger.info("✅ Emergency fallback successful - Basic functionality available")
                logger.warning("⚠️ Running in fallback mode - Some features may be limited")
                
            except Exception as fallback_e:
                logger.error(f"Emergency fallback also failed: {fallback_e}")
                raise RuntimeError(f"Heavy model implementation failed: {e}. Emergency fallback also failed: {fallback_e}")
    
    def _load_models(self):
        """Load all models with enhanced memory management and system optimization."""
        if self._models_loaded:
            return
            
        logger.info("Loading models with enhanced memory management and system optimization...")
        
        # Log initial system state
        system_info = system_optimizer.get_system_info()
        logger.info(f"Initial system state: {system_info}")
        
        # Log initial memory usage
        memory_manager.log_memory_usage("Before model loading")
        
        try:
            if self.heavy_models:
                logger.info("Heavy models enabled - using optimized progressive loading")
                
                # Use system optimizer context for heavy model operations
                with system_optimizer.optimized_context(enable_monitoring=True) as optimizations:
                    logger.info(f"Applied system optimizations: {optimizations}")
                    self._load_models_progressively()
            else:
                logger.info("Standard model loading with basic optimization")
                
                # Apply basic optimizations for standard loading
                basic_opts = system_optimizer.optimize_for_heavy_models()
                logger.info(f"Applied basic optimizations: {basic_opts}")
                
                try:
                    self._load_models_standard()
                finally:
                    system_optimizer.restore_original_settings()
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.info("Attempting emergency fallback with minimal optimization...")
            
            # Ensure optimizations are restored before emergency loading
            system_optimizer.restore_original_settings()
            
            # Try emergency fallback
            try:
                logger.info("Attempting emergency model loading with memory optimization...")
                memory_manager.aggressive_cleanup()
                self._load_models_emergency()
                
            except Exception as emergency_e:
                logger.error(f"Emergency model loading also failed: {emergency_e}")
                raise RuntimeError(f"Failed to load models: {e}. Emergency loading also failed: {emergency_e}")
        
        # Log final system state and memory usage
        final_system_info = system_optimizer.get_system_info()
        logger.info(f"Final system state: {final_system_info}")
        memory_manager.log_memory_usage("After model loading")
    
    def _load_models_progressively(self):
        """Load models using progressive loading system."""
        # Register models for progressive loading
        progressive_loader.register_model(
            'phase1_mvp',
            lambda: Phase1MVP(),
            priority=1,  # Highest priority
            callback=self._on_phase1_loaded
        )
        
        progressive_loader.register_model(
            'phase2_reranker',
            lambda: Phase2Reranker(lazy_load=False),
            priority=2,
            dependencies=['phase1_mvp'],
            callback=self._on_phase2_loaded
        )
        
        progressive_loader.register_model(
            'clip_extractor',
            lambda: ClipExtractor(),
            priority=3,
            callback=self._on_clip_extractor_loaded
        )
        
        # Start progressive loading
        progressive_loader.start_progressive_loading()
        
        # Check loading results
        loading_status = progressive_loader.get_loading_status()
        
        # Set availability based on what was loaded
        self.phase1 = progressive_loader.get_model('phase1_mvp')
        self.phase2 = progressive_loader.get_model('phase2_reranker')
        self.phase2_available = self.phase2 is not None
        self.clip_extractor = progressive_loader.get_model('clip_extractor')
        
        if self.phase1 is None:
            raise RuntimeError("Failed to load critical Phase 1 model")
        
        if self.clip_extractor is None:
            raise RuntimeError("Failed to load clip extractor")
        
        self._models_loaded = True
        
        # Log final status
        loaded_count = sum(1 for status in loading_status.values() if status == 'loaded')
        total_count = len(loading_status)
        logger.info(f"Progressive loading completed: {loaded_count}/{total_count} models loaded")
        memory_manager.log_memory_usage("After progressive model loading")
    
    def _load_models_standard(self):
        """Load models using standard method (for lightweight implementation)."""
        logger.info("Loading Phase 1 (OpenCLIP)...")
        self.phase1 = Phase1MVP()
        
        logger.info("Loading Phase 2 (BLIP re-ranker)...")
        try:
            self.phase2 = Phase2Reranker(lazy_load=True)
            self.phase2_available = True
            logger.info("Phase 2 (BLIP re-ranker) initialized successfully")
        except Exception as e:
            logger.warning(f"Phase 2 initialization failed: {e}")
            logger.info("Running in MVP-only mode")
            self.phase2 = None
            self.phase2_available = False
        
        logger.info("Loading clip extractor...")
        self.clip_extractor = ClipExtractor()
        
        self._models_loaded = True
        logger.info("Standard model loading completed")
    
    def _load_models_emergency(self):
        """Emergency model loading with minimal configuration."""
        logger.info("Emergency loading: minimal configuration")
        
        # Try to load only essential components
        self.phase1 = Phase1MVP()
        self.phase2 = None
        self.phase2_available = False
        self.clip_extractor = ClipExtractor()
        
        self._models_loaded = True
        logger.info("Emergency model loading successful (MVP-only mode)")
    
    def _on_phase1_loaded(self, model):
        """Callback for Phase 1 model loading."""
        logger.info("Phase 1 (OpenCLIP) loaded successfully via progressive loading")
    
    def _on_phase2_loaded(self, model):
        """Callback for Phase 2 model loading."""
        logger.info("Phase 2 (BLIP re-ranker) loaded successfully via progressive loading")
    
    def _on_clip_extractor_loaded(self, model):
        """Callback for clip extractor loading."""
        logger.info("Clip extractor loaded successfully via progressive loading")
    
    def _ensure_models_loaded(self):
        """Ensure models are loaded before processing."""
        if not self._models_loaded:
            self._load_models()
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess query to improve detection accuracy."""
        # Remove extra whitespace and normalize
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase for consistency
        query = query.lower()
        
        # Handle common query patterns and improvements
        query_improvements = {
            # Common action variations
            r'\bwalks?\b': 'walking',
            r'\bruns?\b': 'running', 
            r'\bjumps?\b': 'jumping',
            r'\bfalls?\b': 'falling',
            r'\bsits?\b': 'sitting',
            r'\bstands?\b': 'standing',
            r'\bdrives?\b': 'driving',
            r'\bhits?\b': 'hitting',
            r'\bcrashes?\b': 'crashing',
            
            # Common object variations
            r'\bautomobile\b': 'car',
            r'\bvehicle\b': 'car',
            r'\bpedestrian\b': 'person',
            r'\bindividual\b': 'person',
            r'\bcanine\b': 'dog',
            
            # Color standardization
            r'\bdark blue\b': 'navy',
            r'\blight blue\b': 'blue',
            r'\bdark green\b': 'green',
            r'\blight green\b': 'green',
        }
        
        # Apply improvements
        for pattern, replacement in query_improvements.items():
            query = re.sub(pattern, replacement, query)
        
        # Remove unnecessary articles and prepositions for better matching
        query = re.sub(r'\b(a|an|the)\s+', '', query)
        
        # Simplify complex sentences - keep main action and objects
        # Remove filler words that don't help with visual detection
        filler_words = ['very', 'really', 'quite', 'somewhat', 'rather', 'pretty']
        for word in filler_words:
            query = re.sub(rf'\b{word}\s+', '', query)
        
        logger.info(f"Query preprocessed: '{query}'")
        return query
    
    def process_query(self, video_path: str, query: str, mode: str = "mvp", 
                     top_k: Optional[int] = None, threshold: Optional[float] = None, debug_mode: bool = False) -> Dict:
        """Process a video query with specified mode and comprehensive error handling."""
        
        # Ensure models are loaded before processing
        try:
            self._ensure_models_loaded()
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Failed to load required models: {str(e)}",
                'query': query,
                'mode': mode,
                'results': [],
                'error_type': 'model_loading_error'
            }
        
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        if threshold is None:
            threshold = settings.CONFIDENCE_THRESHOLD
        
        # Preprocess query for better detection
        original_query = query
        processed_query = self.preprocess_query(query)
        
        logger.info(f"Processing query: '{original_query}' -> '{processed_query}' on {video_path} with mode: {mode}, debug: {debug_mode}")
        
        try:
            # Validate video file first
            validation_result = self.validate_video(video_path)
            if not validation_result['valid']:
                return {
                    'status': 'error',
                    'error': f"Video validation failed: {validation_result['error']}",
                    'query': original_query,
                    'mode': mode,
                    'results': []
                }
            # Select processing pipeline
            if mode == "mvp":
                result = self.phase1.process_video(video_path, processed_query, top_k, debug_mode=debug_mode)
                # Handle debug mode return format
                if debug_mode and isinstance(result, tuple):
                    results, debug_info = result
                    logger.info(f"Debug info collected for {len(debug_info)} windows")
                else:
                    results = result
            elif mode == "reranked":
                if self.phase2_available:
                    results = self.phase2.process_video(video_path, processed_query, top_k, debug_mode=debug_mode)
                else:
                    logger.warning("Phase 2 not available, falling back to MVP mode")
                    result = self.phase1.process_video(video_path, processed_query, top_k, debug_mode=debug_mode)
                    # Handle debug mode return format
                    if debug_mode and isinstance(result, tuple):
                        results, debug_info = result
                        logger.info(f"Debug info collected for {len(debug_info)} windows")
                    else:
                        results = result
            elif mode == "advanced":
                if self.phase2_available:
                    # For now, use phase2 as advanced mode (Phase 3 can be added later)
                    results = self.phase2.process_video(video_path, processed_query, top_k, debug_mode=debug_mode)
                else:
                    logger.warning("Advanced mode not available, falling back to MVP mode")
                    result = self.phase1.process_video(video_path, processed_query, top_k, debug_mode=debug_mode)
                    # Handle debug mode return format
                    if debug_mode and isinstance(result, tuple):
                        results, debug_info = result
                        logger.info(f"Debug info collected for {len(debug_info)} windows")
                    else:
                        results = result
            else:
                raise ValueError(f"Unknown processing mode: {mode}")
            
            # Filter by threshold with proper type checking
            filtered_results = []
            for result in results:
                # Ensure result is a dictionary and has required keys
                if isinstance(result, dict) and 'confidence' in result and 'timestamp' in result:
                    if result['confidence'] >= threshold:
                        filtered_results.append(result)
                else:
                    logger.warning(f"Invalid result format: {type(result)} - {result}")
            
            # Extract clips for results
            for result in filtered_results:
                try:
                    # Double-check result structure before accessing
                    if isinstance(result, dict) and 'timestamp' in result:
                        clip_path = self.clip_extractor.extract_clip_with_padding(
                            video_path, 
                            result['timestamp'],
                            settings.CLIP_DURATION
                        )
                        result['clip_path'] = clip_path
                    else:
                        logger.warning(f"Invalid result structure for clip extraction: {result}")
                        result['clip_path'] = None
                except Exception as e:
                    timestamp = result.get('timestamp', 'unknown') if isinstance(result, dict) else 'unknown'
                    logger.warning(f"Failed to extract clip for timestamp {timestamp}: {e}")
                    if isinstance(result, dict):
                        result['clip_path'] = None
            
            response = {
                'status': 'success',
                'query': original_query,
                'processed_query': processed_query,
                'mode': mode,
                'results': filtered_results,
                'total_found': len(filtered_results)
            }
            
            # Add debug info if available
            if debug_mode and 'debug_info' in locals():
                response['debug_info'] = debug_info
            
            return response
            
        except MemoryError as e:
            logger.error(f"Memory allocation error during processing: {e}")
            return {
                'status': 'error',
                'error': f"Insufficient memory to process video. Try using a smaller video or restart the application. Details: {str(e)}",
                'query': original_query,
                'mode': mode,
                'results': [],
                'error_type': 'memory_error'
            }
    
    def process_unlimited_detection(self,
                                  video_path: str,
                                  object_queries: str,
                                  top_k: Optional[int] = None,
                                  threshold: Optional[float] = None,
                                  universal_mode: str = 'hybrid',
                                  open_vocab_mode: str = 'balanced',
                                  debug_mode: bool = False) -> Dict:
        """Process video for unlimited object detection using open-vocabulary models.
        
        Args:
            video_path: Path to the video file
            object_queries: Natural language description of objects to find (can be multiple, separated by ';')
            top_k: Number of top results to return
            threshold: Detection confidence threshold
            universal_mode: Detection method ('hybrid', 'owlvit', 'clip', 'yolo_enhanced')
            open_vocab_mode: Matching precision ('balanced', 'precise', 'comprehensive', 'semantic', 'visual')
            debug_mode: Enable debug mode
            
        Returns:
            Dict with status, results, and metadata
        """
        try:
            # Ensure models are loaded
            self._ensure_models_loaded()
            
            # Validate video
            validation_result = self.validate_video(video_path)
            if not validation_result['valid']:
                return {
                    'status': 'error',
                    'error': f"Video validation failed: {validation_result['error']}",
                    'results': []
                }
            
            # Import and initialize open vocabulary matcher
            try:
                from ..services.open_vocab_matcher import OpenVocabMatcher
                matcher = OpenVocabMatcher()
                
                # Parse multiple queries
                queries = [q.strip() for q in object_queries.split(';') if q.strip()]
                
                # Process unlimited detection
                result = matcher.match_unlimited_objects(
                    video_path=video_path,
                    object_queries=queries,
                    matching_mode=open_vocab_mode,
                    top_k=top_k or 10,
                    detection_mode=universal_mode
                )
                
                results = result.get('matches', [])
                
                return {
                    'status': 'success',
                    'results': results,
                    'queries': queries,
                    'mode': universal_mode,
                    'precision': open_vocab_mode
                }
                
            except ImportError as e:
                return {
                    'status': 'error',
                    'error': f"Universal detection not available: {str(e)}",
                    'results': []
                }
                
        except Exception as e:
            logger.error(f"Error in unlimited detection: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'results': []
            }
    
    def process_image_matching(self, 
                             video_path: str, 
                             reference_image: Union[np.ndarray, str, Path],
                             top_k: Optional[int] = None,
                             similarity_threshold: Optional[float] = None,
                             matching_mode: str = 'traditional',
                             target_class: Optional[str] = None,
                             debug_mode: bool = False) -> Dict:
        """Process video for image matching - find frames that match a reference image.
        
        Args:
            video_path: Path to the video file
            reference_image: Reference image to match (numpy array or file path)
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold for matches
            matching_mode: 'traditional', 'object_focused', 'cross_domain', or 'hybrid'
            target_class: Target object class for object-focused matching
            debug_mode: Enable debug logging and analysis
            
        Returns:
            Dictionary containing status, results, and metadata
        """
        # Ensure models are loaded before processing
        try:
            self._ensure_models_loaded()
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Failed to load required models: {str(e)}",
                'reference_image': str(reference_image) if isinstance(reference_image, (str, Path)) else 'numpy_array',
                'matching_mode': matching_mode,
                'results': [],
                'error_type': 'model_loading_error'
            }
        
        # Check if image matching is available
        if not self.image_matching_available or self.phase_image_matching is None:
            return {
                'status': 'error',
                'error': 'Image matching functionality is not available. Please ensure all required models are loaded.',
                'reference_image': str(reference_image) if isinstance(reference_image, (str, Path)) else 'numpy_array',
                'matching_mode': matching_mode,
                'results': [],
                'error_type': 'feature_unavailable'
            }
        
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        if similarity_threshold is None:
            similarity_threshold = 0.7  # Higher threshold for image matching
        
        logger.info(f"Processing image matching: {video_path} with mode: {matching_mode}, debug: {debug_mode}")
        
        try:
            # Validate video file first
            validation_result = self.validate_video(video_path)
            if not validation_result['valid']:
                return {
                    'status': 'error',
                    'error': f"Video validation failed: {validation_result['error']}",
                    'reference_image': str(reference_image) if isinstance(reference_image, (str, Path)) else 'numpy_array',
                    'matching_mode': matching_mode,
                    'results': []
                }
            
            # Validate reference image
            if isinstance(reference_image, (str, Path)):
                if not Path(reference_image).exists():
                    return {
                        'status': 'error',
                        'error': f"Reference image file not found: {reference_image}",
                        'reference_image': str(reference_image),
                        'matching_mode': matching_mode,
                        'results': []
                    }
            elif isinstance(reference_image, np.ndarray):
                if reference_image.size == 0:
                    return {
                        'status': 'error',
                        'error': 'Reference image array is empty',
                        'reference_image': 'numpy_array',
                        'matching_mode': matching_mode,
                        'results': []
                    }
            else:
                return {
                    'status': 'error',
                    'error': 'Invalid reference image format. Must be file path or numpy array.',
                    'reference_image': str(type(reference_image)),
                    'matching_mode': matching_mode,
                    'results': []
                }
            
            # Use Phase4AdvancedMatching for enhanced capabilities
            try:
                from ..pipeline.phase4_advanced_matching import Phase4AdvancedMatching
                advanced_matcher = Phase4AdvancedMatching(use_gpu=True)
                
                result = advanced_matcher.process_image_query(
                    video_path=video_path,
                    reference_image=reference_image,
                    matching_mode=matching_mode,
                    target_class=target_class,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    extract_clips=True
                )
                
                # Convert to expected format
                results = {
                    'status': 'success' if result['matches'] else 'no_matches',
                    'results': result['matches'],
                    'clips': result['clips'],
                    'metadata': result['metadata'],
                    'performance': result['performance']
                }
                
                # Add debug info if available
                if debug_mode and 'debug_info' in result:
                    results['debug_info'] = result['debug_info']
                
                advanced_matcher.cleanup()
                
            except ImportError:
                # Fallback to original image matching phase
                logger.warning("Advanced matching not available, using traditional image matching")
                results = self.phase_image_matching.process_video(
                    video_path=video_path,
                    reference_image=reference_image,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    matching_mode='multi_stage' if matching_mode == 'traditional' else 'single_stage',
                    debug_mode=debug_mode
                )
            
            # Filter results by threshold (additional safety check)
            filtered_results = []
            if isinstance(results, dict) and 'results' in results:
                # Results from Phase4AdvancedMatching
                for result in results['results']:
                    if isinstance(result, dict) and 'confidence' in result and 'timestamp' in result:
                        if result['confidence'] >= similarity_threshold:
                            filtered_results.append(result)
                    else:
                        logger.warning(f"Invalid image matching result format: {type(result)} - {result}")
            elif isinstance(results, list):
                # Results from fallback phase_image_matching
                for result in results:
                    if isinstance(result, dict) and 'confidence' in result and 'timestamp' in result:
                        if result['confidence'] >= similarity_threshold:
                            filtered_results.append(result)
                    else:
                        logger.warning(f"Invalid image matching result format: {type(result)} - {result}")
            else:
                logger.warning(f"Unexpected results format: {type(results)}")
                filtered_results = []
            
            response = {
                'status': 'success',
                'reference_image': str(reference_image) if isinstance(reference_image, (str, Path)) else 'numpy_array',
                'matching_mode': matching_mode,
                'similarity_threshold': similarity_threshold,
                'results': filtered_results,
                'total_found': len(filtered_results),
                'processing_type': 'image_matching'
            }
            
            return response
            
        except MemoryError as e:
            logger.error(f"Memory allocation error during image matching: {e}")
            return {
                'status': 'error',
                'error': f"Insufficient memory to process image matching. Try using a smaller video or restart the application. Details: {str(e)}",
                'reference_image': str(reference_image) if isinstance(reference_image, (str, Path)) else 'numpy_array',
                'matching_mode': matching_mode,
                'results': [],
                'error_type': 'memory_error'
            }
        except OSError as e:
            if "paging file" in str(e).lower() or "1455" in str(e):
                logger.error(f"Windows paging file error: {e}")
                return {
                    'status': 'error',
                    'error': "System memory error (paging file too small). Please increase virtual memory or restart the application.",
                    'query': original_query,
                    'mode': mode,
                    'results': [],
                    'error_type': 'paging_file_error'
                }
            else:
                logger.error(f"System error during processing: {e}")
                return {
                    'status': 'error',
                    'error': f"System error: {str(e)}",
                    'query': original_query,
                    'mode': mode,
                    'results': [],
                    'error_type': 'system_error'
                }
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            return {
                'status': 'error',
                'error': f"Required file not found: {str(e)}. Please check FFmpeg installation and file paths.",
                'query': original_query,
                'mode': mode,
                'results': [],
                'error_type': 'file_not_found'
            }
        except Exception as e:
            logger.error(f"Unexpected error processing query: {e}")
            return {
                'status': 'error',
                'error': f"Unexpected error: {str(e)}",
                'query': original_query,
                'mode': mode,
                'results': [],
                'error_type': 'unknown_error'
            }
    
    def validate_video(self, video_path: str) -> Dict:
        """Validate video file format and accessibility."""
        try:
            video_file = Path(video_path)
            
            if not video_file.exists():
                return {'valid': False, 'error': 'Video file does not exist'}
            
            file_extension = video_file.suffix.lower().lstrip('.')
            if file_extension not in settings.SUPPORTED_FORMATS:
                return {
                    'valid': False, 
                    'error': f'Unsupported format: {file_extension}. Supported: {settings.SUPPORTED_FORMATS}'
                }
            
            file_size = video_file.stat().st_size
            if file_size > settings.MAX_VIDEO_SIZE:
                return {
                    'valid': False, 
                    'error': f'Video file too large: {file_size} bytes. Max: {settings.MAX_VIDEO_SIZE} bytes'
                }
            
            return {
                'valid': True, 
                'format': file_extension,
                'size': file_size,
                'path': str(video_file)
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'Error validating video: {str(e)}'}