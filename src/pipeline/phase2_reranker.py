import numpy as np
from typing import List, Dict
from ..utils.logger import get_logger
from .phase1_mvp import Phase1MVP
from ..models.blip_model import BLIPModel
from ..services.frame_extractor import FrameExtractor

logger = get_logger(__name__)

class Phase2Reranker:
    def __init__(self, lazy_load=True):
        self.phase1 = Phase1MVP()
        self.frame_extractor = FrameExtractor()
        self.lazy_load = lazy_load
        
        if lazy_load:
            self.blip_model = None
            logger.info("Phase2Reranker initialized with lazy BLIP loading")
        else:
            logger.info("🚀 Phase2Reranker HEAVY MODE: Force loading BLIP-2 model immediately")
            self.blip_model = BLIPModel(lazy_load=False)  # Force immediate loading
            logger.info("✅ Phase2Reranker BLIP-2 model loaded in HEAVY MODE")
    
    def _ensure_blip_loaded(self):
        """Ensure BLIP model is loaded before use."""
        if self.blip_model is None:
            logger.info("Lazy loading BLIP model...")
            self.blip_model = BLIPModel()
            logger.info("BLIP model loaded successfully")
    
    def process_video(self, video_path: str, query: str, top_k: int = None, debug_mode: bool = False) -> List[Dict]:
        """Process video with CLIP + BLIP re-ranking."""
        logger.info(f"Phase 2 processing: {video_path} with query: '{query}'")
        
        # Ensure BLIP model is loaded
        self._ensure_blip_loaded()
        
        # Get Phase 1 results
        phase1_result = self.phase1.process_video(video_path, query, top_k * 2 if top_k else 20, debug_mode=debug_mode)  # Get more candidates
        
        # Handle debug mode return format from Phase 1
        if debug_mode and isinstance(phase1_result, tuple):
            phase1_results, debug_info = phase1_result
            logger.info(f"Debug info from Phase 1: {len(debug_info)} windows")
        else:
            phase1_results = phase1_result
        
        if not phase1_results:
            return []
        
        # Extract frames for re-ranking
        frames, timestamps = self.frame_extractor.extract_frames(video_path)
        windows, window_timestamps = self.frame_extractor.create_sliding_windows(frames, timestamps)
        
        # Re-rank using BLIP captions
        reranked_results = []
        
        for result in phase1_results:
            window_idx = result['window_index']
            window = windows[window_idx]
            middle_frame = window[len(window) // 2]
            
            # Generate caption
            caption = self.blip_model.generate_caption(middle_frame)
            
            # Compute caption-query similarity
            caption_similarity = self.blip_model.compute_text_similarity(caption, query)
            
            # Combine CLIP and caption scores
            clip_score = result['confidence']
            combined_score = 0.7 * clip_score + 0.3 * caption_similarity
            
            reranked_results.append({
                'timestamp': result['timestamp'],
                'confidence': float(combined_score),
                'phase': 'phase2_reranked',
                'window_index': window_idx,
                'caption': caption,
                'clip_score': float(clip_score),
                'caption_score': float(caption_similarity)
            })
        
        # Sort by combined score and return top-k
        reranked_results.sort(key=lambda x: x['confidence'], reverse=True)
        final_results = reranked_results[:top_k or len(reranked_results)]
        
        logger.info(f"Phase 2 re-ranked to {len(final_results)} results")
        return final_results