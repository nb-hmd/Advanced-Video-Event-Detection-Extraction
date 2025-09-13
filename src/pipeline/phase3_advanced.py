from typing import List, Dict
from ..utils.logger import get_logger
from .phase2_reranker import Phase2Reranker
from ..models.univtg_model import UniVTGModel

logger = get_logger(__name__)

class Phase3Advanced:
    def __init__(self):
        self.phase2 = Phase2Reranker()
        self.univtg_model = UniVTGModel()
    
    def process_video(self, video_path: str, query: str, top_k: int = None, debug_mode: bool = False) -> List[Dict]:
        """Process video with full pipeline: CLIP + BLIP + UniVTG."""
        logger.info(f"Phase 3 processing: {video_path} with query: '{query}'")
        
        # Get Phase 2 results
        phase2_results = self.phase2.process_video(video_path, query, top_k * 2 if top_k else 20, debug_mode=debug_mode)
        
        if not phase2_results:
            return []
        
        # Refine with UniVTG
        refined_results = self.univtg_model.predict_temporal_boundaries(
            video_path, query, phase2_results
        )
        
        # Additional processing: temporal consistency check
        refined_results = self._apply_temporal_consistency(refined_results)
        
        # Sort by confidence and return
        refined_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Phase 3 completed with {len(refined_results)} refined results")
        return refined_results
    
    def _apply_temporal_consistency(self, results: List[Dict]) -> List[Dict]:
        """Apply temporal consistency checks to remove overlapping or inconsistent detections."""
        if len(results) <= 1:
            return results
        
        # Sort by timestamp
        sorted_results = sorted(results, key=lambda x: x['timestamp'])
        
        # Remove overlapping detections (keep higher confidence)
        filtered_results = []
        for current in sorted_results:
            should_add = True
            
            for existing in filtered_results:
                # Check for temporal overlap
                current_start = current.get('start_time', current['timestamp'] - 2.5)
                current_end = current.get('end_time', current['timestamp'] + 2.5)
                existing_start = existing.get('start_time', existing['timestamp'] - 2.5)
                existing_end = existing.get('end_time', existing['timestamp'] + 2.5)
                
                # Calculate overlap
                overlap_start = max(current_start, existing_start)
                overlap_end = min(current_end, existing_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # If significant overlap (>50% of either segment)
                current_duration = current_end - current_start
                existing_duration = existing_end - existing_start
                
                if (overlap_duration > 0.5 * current_duration or 
                    overlap_duration > 0.5 * existing_duration):
                    
                    # Keep the one with higher confidence
                    if current['confidence'] <= existing['confidence']:
                        should_add = False
                        break
                    else:
                        # Remove the existing one (will be replaced by current)
                        filtered_results.remove(existing)
            
            if should_add:
                filtered_results.append(current)
        
        logger.info(f"Temporal consistency filter: {len(results)} -> {len(filtered_results)} results")
        return filtered_results
    
    def process_with_temporal_grounding(self, video_path: str, query: str, top_k: int = None, debug_mode: bool = False) -> List[Dict]:
        """Alternative processing method using direct temporal grounding."""
        logger.info(f"Phase 3 temporal grounding: {video_path} with query: '{query}'")
        
        try:
            # Extract video features
            video_features = self.univtg_model.extract_video_features(video_path)
            
            # Ground query to video segments
            grounded_segments = self.univtg_model.ground_query_to_video(video_features, query)
            
            # Convert to standard result format
            results = []
            for i, (start_time, end_time, confidence) in enumerate(grounded_segments):
                timestamp = (start_time + end_time) / 2  # Middle of segment
                
                results.append({
                    'timestamp': timestamp,
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': confidence,
                    'phase': 'phase3_temporal_grounding',
                    'duration': end_time - start_time,
                    'segment_id': i
                })
            
            # Sort by confidence and limit results
            results.sort(key=lambda x: x['confidence'], reverse=True)
            if top_k:
                results = results[:top_k]
            
            logger.info(f"Temporal grounding found {len(results)} segments")
            return results
            
        except Exception as e:
            logger.error(f"Error in temporal grounding: {e}")
            # Fallback to regular Phase 3 processing
            return self.process_video(video_path, query, top_k)