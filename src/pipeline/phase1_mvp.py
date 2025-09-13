import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import cv2
import gc
from ..utils.logger import get_logger
from ..models.openclip_model import OpenCLIPModel
from ..services.frame_extractor import FrameExtractor
from ..utils.config import settings
from ..utils.memory_manager import memory_manager

logger = get_logger(__name__)

class Phase1MVP:
    def __init__(self, debug_mode=False):
        self.clip_model = OpenCLIPModel()
        self.frame_extractor = FrameExtractor()
        self.debug_mode = debug_mode
        self.debug_dir = Path(settings.DATA_DIR) / "debug"
        if self.debug_mode:
            self.debug_dir.mkdir(exist_ok=True)
    
    def process_video(self, video_path: str, query: str, top_k: int = None, debug_mode: bool = None) -> List[Dict]:
        """Process video with OpenCLIP only."""
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        # Override debug mode if specified
        if debug_mode is not None:
            self.debug_mode = debug_mode
            if self.debug_mode and not self.debug_dir.exists():
                self.debug_dir.mkdir(exist_ok=True)
        
        logger.info(f"Phase 1 processing: {video_path} with query: '{query}' (debug: {self.debug_mode})")
        
        # Extract frames and create windows
        frames, timestamps = self.frame_extractor.extract_frames(video_path)
        windows, window_timestamps = self.frame_extractor.create_sliding_windows(frames, timestamps)
        
        logger.info(f"Extracted {len(frames)} frames, created {len(windows)} sliding windows")
        
        # Encode text query
        text_embedding = self.clip_model.encode_text(query)
        
        if self.debug_mode:
            logger.info(f"Text embedding shape: {text_embedding.shape}")
            logger.info(f"Text embedding norm: {np.linalg.norm(text_embedding)}")
        
        # Process windows in memory-efficient chunks
        all_similarities = []
        debug_info = []
        
        # Check initial memory state
        memory_manager.log_memory_usage("Before window processing")
        
        # Process windows in chunks to manage memory
        chunk_size = min(settings.MAX_WINDOWS_PER_BATCH, len(windows))
        
        for chunk_start in range(0, len(windows), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(windows))
            chunk_windows = windows[chunk_start:chunk_end]
            
            logger.info(f"Processing window chunk {chunk_start//chunk_size + 1}/{(len(windows) + chunk_size - 1)//chunk_size} ({chunk_end - chunk_start} windows)")
            
            # Check memory before processing chunk
            if not memory_manager.check_memory_availability():
                logger.warning(f"Low memory detected at chunk {chunk_start//chunk_size + 1}, reducing chunk size")
                # Process remaining windows one by one
                chunk_size = 1
                chunk_end = min(chunk_start + 1, len(windows))
                chunk_windows = windows[chunk_start:chunk_end]
            
            # Process each window in the chunk
            for i in range(chunk_start, chunk_end):
                try:
                    window_idx = i - chunk_start
                    window = chunk_windows[window_idx]
                    
                    # Use middle frame of window for embedding
                    middle_frame = window[len(window) // 2]
                    
                    # Encode frame with memory optimization
                    image_embedding = self.clip_model.encode_images(middle_frame)
                    
                    # Compute similarity
                    similarity = self.clip_model.compute_similarity(image_embedding, text_embedding)[0, 0]
                    all_similarities.append(similarity)
                    
                    # Debug logging and frame saving
                    if self.debug_mode:
                        debug_info.append({
                            'window_index': i,
                            'timestamp': window_timestamps[i],
                            'similarity': float(similarity),
                            'image_embedding_norm': float(np.linalg.norm(image_embedding)),
                            'frame_shape': middle_frame.shape
                        })
                        
                        # Save sample frames for top and bottom similarities
                        if i < 5 or i >= len(windows) - 5:
                            frame_path = self.debug_dir / f"frame_{i:03d}_sim_{similarity:.4f}.jpg"
                            cv2.imwrite(str(frame_path), cv2.cvtColor(middle_frame, cv2.COLOR_RGB2BGR))
                    
                    # Clean up intermediate variables
                    del image_embedding, middle_frame
                    
                    # Periodic memory cleanup
                    if i % settings.MEMORY_CLEANUP_INTERVAL == 0 and i > 0:
                        memory_manager.aggressive_cleanup()
                        
                except MemoryError as e:
                    logger.error(f"Memory error processing window {i}: {e}")
                    # Try to continue with aggressive cleanup
                    memory_manager.aggressive_cleanup()
                    # If we still can't continue, break
                    if not memory_manager.check_memory_availability():
                        logger.error(f"Insufficient memory to continue processing at window {i}")
                        break
                except Exception as e:
                    logger.error(f"Error processing window {i}: {e}")
                    continue
            
            # Cleanup after each chunk
            del chunk_windows
            memory_manager.aggressive_cleanup()
            
            # Log progress
            memory_manager.log_memory_usage(f"After chunk {chunk_start//chunk_size + 1}")
        
        if not all_similarities:
            raise ValueError("No windows could be processed due to memory constraints")
        
        # Debug analysis
        similarities = np.array(all_similarities)
        
        if self.debug_mode:
            self._log_debug_analysis(similarities, debug_info, query, settings.CONFIDENCE_THRESHOLD)
        
        # Log similarity statistics
        logger.info(f"Similarity scores - Min: {similarities.min():.4f}, Max: {similarities.max():.4f}, Mean: {similarities.mean():.4f}, Std: {similarities.std():.4f}")
        logger.info(f"Confidence threshold: {settings.CONFIDENCE_THRESHOLD}")
        logger.info(f"Scores above threshold: {np.sum(similarities >= settings.CONFIDENCE_THRESHOLD)}")
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= settings.CONFIDENCE_THRESHOLD:
                results.append({
                    'timestamp': window_timestamps[idx],
                    'confidence': float(similarities[idx]),
                    'phase': 'phase1_mvp',
                    'window_index': int(idx)
                })
        
        logger.info(f"Phase 1 found {len(results)} candidate events")
        
        # Return debug info if in debug mode
        if self.debug_mode:
            return results, debug_info
        
        return results
    
    def _log_debug_analysis(self, similarities, debug_info, query, threshold):
        """Log detailed debug analysis."""
        logger.info("=== DEBUG ANALYSIS ===")
        logger.info(f"Query: '{query}'")
        logger.info(f"Total windows processed: {len(similarities)}")
        logger.info(f"Similarity range: [{similarities.min():.6f}, {similarities.max():.6f}]")
        logger.info(f"Mean similarity: {similarities.mean():.6f}")
        logger.info(f"Std similarity: {similarities.std():.6f}")
        logger.info(f"Confidence threshold: {threshold}")
        
        # Top 10 similarities
        top_indices = np.argsort(similarities)[::-1][:10]
        logger.info("Top 10 similarity scores:")
        for i, idx in enumerate(top_indices):
            # Safely access debug_info with bounds checking
            if idx < len(debug_info) and isinstance(debug_info[idx], dict):
                info = debug_info[idx]
                timestamp = info.get('timestamp', 0.0)
            else:
                timestamp = 0.0  # Fallback timestamp
            logger.info(f"  {i+1}. Window {idx}: {similarities[idx]:.6f} at {timestamp:.2f}s")
        
        # Bottom 5 similarities
        bottom_indices = np.argsort(similarities)[:5]
        logger.info("Bottom 5 similarity scores:")
        for i, idx in enumerate(bottom_indices):
            # Safely access debug_info with bounds checking
            if idx < len(debug_info) and isinstance(debug_info[idx], dict):
                info = debug_info[idx]
                timestamp = info.get('timestamp', 0.0)
            else:
                timestamp = 0.0  # Fallback timestamp
            logger.info(f"  {i+1}. Window {idx}: {similarities[idx]:.6f} at {timestamp:.2f}s")
        
        # Threshold analysis
        above_threshold = np.sum(similarities >= threshold)
        logger.info(f"Windows above threshold ({threshold}): {above_threshold}/{len(similarities)} ({above_threshold/len(similarities)*100:.1f}%)")
        
        if above_threshold == 0:
            # Suggest lower thresholds
            percentiles = [95, 90, 80, 70, 50]
            logger.info("Suggested thresholds based on percentiles:")
            for p in percentiles:
                thresh = np.percentile(similarities, p)
                count = np.sum(similarities >= thresh)
                logger.info(f"  {p}th percentile ({thresh:.4f}): {count} windows")
        
        logger.info("=== END DEBUG ANALYSIS ===")