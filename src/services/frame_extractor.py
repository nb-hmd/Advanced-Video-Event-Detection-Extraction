import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import gc
from ..utils.logger import get_logger
from ..utils.config import settings
from ..utils.memory_manager import memory_manager

logger = get_logger(__name__)

# Try to import Decord, fallback to OpenCV if not available
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logger.warning("Decord not available, using OpenCV fallback")

class FrameExtractor:
    def __init__(self):
        self.sample_rate = settings.FRAME_SAMPLE_RATE
        self.window_size = settings.WINDOW_SIZE
        self.window_stride = settings.WINDOW_STRIDE
    
    def extract_frames(self, video_path: str) -> Tuple[np.ndarray, List[float]]:
        """Extract frames from video using Decord with OpenCV fallback."""
        # Try Decord first, fallback to OpenCV if it fails
        if DECORD_AVAILABLE:
            try:
                return self._extract_frames_decord(video_path)
            except Exception as e:
                logger.warning(f"Decord failed ({e}), falling back to OpenCV")
                return self._extract_frames_opencv(video_path)
        else:
            return self._extract_frames_opencv(video_path)
    
    def _extract_frames_decord(self, video_path: str) -> Tuple[np.ndarray, List[float]]:
        """Extract frames using Decord library with memory optimization."""
        # Check memory before starting
        memory_manager.log_memory_usage("Before Decord extraction")
        
        if not memory_manager.check_memory_availability():
            logger.warning("Low memory detected, using aggressive sampling")
            # Increase sample rate to reduce memory usage
            original_sample_rate = self.sample_rate
            self.sample_rate = max(self.sample_rate * 2, 5)
            logger.info(f"Increased sample rate from {original_sample_rate} to {self.sample_rate}")
        
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        
        # Validate FPS
        if fps <= 0 or fps > 1000:  # Sanity check
            logger.warning(f"Invalid FPS from Decord: {fps}, attempting to calculate from duration")
            try:
                # Try to get duration and calculate FPS
                duration = total_frames / 30.0  # Fallback assumption
                fps = total_frames / duration if duration > 0 else 30.0
            except:
                fps = 30.0  # Final fallback
            logger.info(f"Using calculated/fallback FPS: {fps}")
        
        # Sample frames with better distribution
        frame_indices = list(range(0, total_frames, self.sample_rate))
        
        # High performance mode - process more frames for better accuracy
        max_frames = min(len(frame_indices), 1000)  # Increased limit for high performance
        if len(frame_indices) > max_frames:
            # Evenly distribute frames across video
            step = len(frame_indices) // max_frames
            frame_indices = frame_indices[::step][:max_frames]
            logger.info(f"Processing {max_frames} frames for high performance mode")
        
        # Ensure we don't exceed total frames
        frame_indices = [idx for idx in frame_indices if idx < total_frames]
        
        if not frame_indices:
            raise ValueError(f"No valid frame indices for video with {total_frames} frames")
        
        try:
            frames = vr.get_batch(frame_indices).asnumpy()
            
            # Process frames for memory optimization
            optimized_frames = []
            for i, frame in enumerate(frames):
                # Resize frame to reduce memory usage
                resized_frame = memory_manager.resize_frame_for_memory(
                    frame, settings.MAX_FRAME_WIDTH, settings.MAX_FRAME_HEIGHT
                )
                
                # Optimize data type
                optimized_frame = memory_manager.optimize_frame_dtype(resized_frame)
                optimized_frames.append(optimized_frame)
                
                # Cleanup every 10 frames
                if i % 10 == 0 and i > 0:
                    memory_manager.aggressive_cleanup()
            
            frames = np.array(optimized_frames)
            
            # Calculate accurate timestamps using actual frame indices
            timestamps = [float(idx) / float(fps) for idx in frame_indices]
            
            memory_manager.log_memory_usage("After Decord extraction")
            logger.info(f"Extracted {len(frames)} frames from {video_path} using Decord (FPS: {fps:.2f}, Duration: {timestamps[-1]:.2f}s)")
            return frames, timestamps
            
        except Exception as e:
            logger.error(f"Memory error during Decord extraction: {e}")
            # Cleanup and retry with even more aggressive settings
            memory_manager.aggressive_cleanup()
            raise MemoryError(f"Insufficient memory for video processing: {e}")
    
    def _extract_frames_opencv(self, video_path: str) -> Tuple[np.ndarray, List[float]]:
        """Extract frames using OpenCV as fallback with memory optimization."""
        # Check memory before starting
        memory_manager.log_memory_usage("Before OpenCV extraction")
        
        if not memory_manager.check_memory_availability():
            logger.warning("Low memory detected, using aggressive sampling")
            # Increase sample rate to reduce memory usage
            original_sample_rate = self.sample_rate
            self.sample_rate = max(self.sample_rate * 2, 5)
            logger.info(f"Increased sample rate from {original_sample_rate} to {self.sample_rate}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            # Get video properties with validation
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Validate and correct FPS
            if fps <= 0 or fps > 1000:  # Sanity check
                logger.warning(f"Invalid FPS from OpenCV: {fps}")
                # Try alternative methods to get FPS
                try:
                    # Method 1: Use frame count and duration if available
                    duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if duration_ms > 0:
                        fps = total_frames / (duration_ms / 1000.0)
                    else:
                        fps = 30.0  # Fallback
                except:
                    fps = 30.0  # Final fallback
                
                logger.info(f"Using corrected FPS: {fps:.2f}")
            
            frames = []
            timestamps = []
            actual_frame_indices = []
            
            # Sample frames with validation and memory limits
            frame_indices = list(range(0, total_frames, self.sample_rate))
            
            # High performance mode - process more frames for better accuracy
            max_frames = min(len(frame_indices), 1000)  # Increased limit for high performance
            if len(frame_indices) > max_frames:
                # Evenly distribute frames across video
                step = len(frame_indices) // max_frames
                frame_indices = frame_indices[::step][:max_frames]
                logger.info(f"Processing {max_frames} frames for high performance mode")
            
            for i, frame_idx in enumerate(frame_indices):
                if frame_idx >= total_frames:
                    logger.warning(f"Frame index {frame_idx} exceeds total frames {total_frames}")
                    break
                
                # Check memory periodically
                if i % 20 == 0 and i > 0:
                    if not memory_manager.check_memory_availability():
                        logger.warning(f"Memory exhausted at frame {i}, stopping extraction")
                        break
                    memory_manager.aggressive_cleanup()
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Verify the position was set correctly
                actual_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    try:
                        # Convert BGR to RGB (OpenCV uses BGR by default)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Resize frame to reduce memory usage
                        resized_frame = memory_manager.resize_frame_for_memory(
                            frame_rgb, settings.MAX_FRAME_WIDTH, settings.MAX_FRAME_HEIGHT
                        )
                        
                        # Optimize data type
                        optimized_frame = memory_manager.optimize_frame_dtype(resized_frame)
                        frames.append(optimized_frame)
                        
                        # Use actual position for more accurate timestamp
                        actual_timestamp = float(actual_pos) / float(fps)
                        timestamps.append(actual_timestamp)
                        actual_frame_indices.append(int(actual_pos))
                        
                    except MemoryError as e:
                        logger.error(f"Memory error processing frame {frame_idx}: {e}")
                        break
                else:
                    logger.warning(f"Failed to read frame at index {frame_idx} (actual pos: {actual_pos})")
            
            cap.release()
            
            if not frames:
                raise ValueError(f"No frames could be extracted from {video_path}")
            
            frames_array = np.array(frames)
            
            # Log extraction summary
            duration = timestamps[-1] if timestamps else 0
            memory_manager.log_memory_usage("After OpenCV extraction")
            logger.info(f"Extracted {len(frames)} frames from {video_path} using OpenCV")
            logger.info(f"Video properties: FPS={fps:.2f}, Total frames={total_frames}, Duration={duration:.2f}s")
            logger.info(f"Frame indices: {actual_frame_indices[:5]}...{actual_frame_indices[-5:] if len(actual_frame_indices) > 5 else ''}")
            
            return frames_array, timestamps
            
        except Exception as e:
            cap.release()
            logger.error(f"Error during OpenCV extraction: {e}")
            memory_manager.aggressive_cleanup()
            if "memory" in str(e).lower() or "allocation" in str(e).lower():
                raise MemoryError(f"Insufficient memory for video processing: {e}")
            raise
    
    def create_sliding_windows(self, frames: np.ndarray, timestamps: List[float]) -> Tuple[np.ndarray, List[float]]:
        """Create sliding windows from frames with accurate timestamp calculation."""
        if len(frames) != len(timestamps):
            raise ValueError(f"Frames and timestamps length mismatch: {len(frames)} vs {len(timestamps)}")
        
        if len(frames) < self.window_size:
            logger.warning(f"Not enough frames ({len(frames)}) for window size ({self.window_size})")
            # Create a single window with all available frames
            if len(frames) > 0:
                window_timestamp = timestamps[len(timestamps) // 2]  # Middle timestamp
                return np.array([frames]), [window_timestamp]
            else:
                return np.array([]), []
        
        windows = []
        window_timestamps = []
        
        for i in range(0, len(frames) - self.window_size + 1, self.window_stride):
            window = frames[i:i + self.window_size]
            
            # Calculate window timestamp more accurately
            # Use the timestamp of the middle frame in the window
            middle_frame_idx = i + self.window_size // 2
            
            # Ensure we don't go out of bounds
            if middle_frame_idx >= len(timestamps):
                middle_frame_idx = len(timestamps) - 1
            
            window_timestamp = timestamps[middle_frame_idx]
            
            windows.append(window)
            window_timestamps.append(window_timestamp)
        
        logger.info(f"Created {len(windows)} sliding windows")
        logger.info(f"Window timestamps range: {window_timestamps[0]:.2f}s to {window_timestamps[-1]:.2f}s")
        
        return np.array(windows), window_timestamps
    
    def save_frame(self, frame: np.ndarray, output_path: str) -> None:
        """Save a single frame as image."""
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))