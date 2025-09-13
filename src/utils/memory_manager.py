import gc
import psutil
import torch
import os
import time
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
from .logger import get_logger

logger = get_logger(__name__)

class MemoryManager:
    """Advanced memory management utility for heavy AI models and video processing."""
    
    def __init__(self, min_available_mb: int = 500):
        self.min_available_mb = min_available_mb
        self.process = psutil.Process()
        self.model_memory_cache = {}
        self.gpu_available = torch.cuda.is_available()
        # Optimized memory requirements for better compatibility
        self.model_requirements = {
            'openclip': {'cpu_mb': 800, 'gpu_mb': 1500},
            'blip2': {'cpu_mb': 400, 'gpu_mb': 1200},  # Reduced for better compatibility
            'univtg': {'cpu_mb': 1000, 'gpu_mb': 2500}   # Optimized performance
        }
        
        if self.gpu_available:
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("No GPU detected, using CPU-only mode")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get comprehensive memory usage information including GPU."""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Process memory
            process_memory = self.process.memory_info()
            
            memory_info = {
                'system_total_gb': system_memory.total / (1024**3),
                'system_available_gb': system_memory.available / (1024**3),
                'system_used_percent': system_memory.percent,
                'process_rss_mb': process_memory.rss / (1024**2),
                'process_vms_mb': process_memory.vms / (1024**2)
            }
            
            # Add GPU memory info if available
            if self.gpu_available:
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_allocated = torch.cuda.memory_allocated(0)
                    gpu_cached = torch.cuda.memory_reserved(0)
                    
                    memory_info.update({
                        'gpu_total_gb': gpu_memory / (1024**3),
                        'gpu_allocated_gb': gpu_allocated / (1024**3),
                        'gpu_cached_gb': gpu_cached / (1024**3),
                        'gpu_available_gb': (gpu_memory - gpu_cached) / (1024**3)
                    })
                except Exception as gpu_e:
                    logger.warning(f"Failed to get GPU memory info: {gpu_e}")
                    memory_info.update({
                        'gpu_total_gb': 0,
                        'gpu_allocated_gb': 0,
                        'gpu_cached_gb': 0,
                        'gpu_available_gb': 0
                    })
            
            return memory_info
            
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {
                'system_total_gb': 0,
                'system_available_gb': 0,
                'system_used_percent': 0,
                'process_rss_mb': 0,
                'process_vms_mb': 0,
                'gpu_total_gb': 0,
                'gpu_allocated_gb': 0,
                'gpu_cached_gb': 0,
                'gpu_available_gb': 0
            }
    
    def check_memory_availability(self) -> bool:
        """Check if sufficient memory is available."""
        memory_info = self.get_memory_info()
        available_mb = memory_info['system_available_gb'] * 1024
        
        if available_mb < self.min_available_mb:
            logger.warning(f"Low memory: {available_mb:.1f}MB available, minimum required: {self.min_available_mb}MB")
            return False
        
        return True
    
    def can_load_model(self, model_name: str, use_gpu: bool = None) -> Tuple[bool, str]:
        """Check if a specific model can be loaded given current memory constraints."""
        if model_name not in self.model_requirements:
            return False, f"Unknown model: {model_name}"
        
        memory_info = self.get_memory_info()
        requirements = self.model_requirements[model_name]
        
        # Determine device preference
        if use_gpu is None:
            use_gpu = self.gpu_available
        
        if use_gpu and self.gpu_available:
            # Check GPU memory
            required_gb = requirements['gpu_mb'] / 1024
            available_gb = memory_info.get('gpu_available_gb', 0)
            
            if available_gb < required_gb:
                # Fallback to CPU if GPU memory insufficient
                logger.warning(f"Insufficient GPU memory for {model_name}: {available_gb:.1f}GB < {required_gb:.1f}GB, trying CPU")
                use_gpu = False
            else:
                return True, f"Can load {model_name} on GPU ({required_gb:.1f}GB required, {available_gb:.1f}GB available)"
        
        if not use_gpu:
            # Check CPU memory
            required_gb = requirements['cpu_mb'] / 1024
            available_gb = memory_info['system_available_gb']
            
            if available_gb < required_gb:
                return False, f"Insufficient CPU memory for {model_name}: {available_gb:.1f}GB < {required_gb:.1f}GB"
            else:
                return True, f"Can load {model_name} on CPU ({required_gb:.1f}GB required, {available_gb:.1f}GB available)"
        
        return False, f"No suitable device found for {model_name}"
    
    def get_optimal_device(self, model_name: str) -> str:
        """Get the optimal device (cpu/cuda) for a model based on memory availability."""
        can_load, message = self.can_load_model(model_name, use_gpu=True)
        
        if can_load and "GPU" in message:
            return "cuda" if torch.cuda.is_available() else "cpu"
        
        can_load_cpu, _ = self.can_load_model(model_name, use_gpu=False)
        if can_load_cpu:
            return "cpu"
        
        # Return CPU as fallback even if memory is tight
        logger.warning(f"Memory constraints detected for {model_name}, using CPU with reduced precision")
        return "cpu"
    
    def optimize_for_model_loading(self, model_name: str) -> None:
        """Optimize system for loading a specific heavy model."""
        logger.info(f"Optimizing system for {model_name} loading...")
        
        # Aggressive cleanup before loading
        self.aggressive_cleanup()
        
        # GPU-specific optimizations
        if self.gpu_available:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Set high performance memory allocation for large models
                if model_name == 'blip2':
                    # Allocate generous memory for BLIP-2 high performance
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'
                elif model_name == 'univtg':
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:3072'
                else:
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'
                
                logger.info(f"GPU memory optimized for {model_name}")
            except Exception as e:
                logger.warning(f"GPU optimization failed: {e}")
        
        # CPU optimizations
        try:
            # Set optimal thread limits for high performance
            if model_name in ['blip2', 'univtg']:
                torch.set_num_threads(min(12, os.cpu_count()))
            else:
                torch.set_num_threads(min(16, os.cpu_count()))
            
            logger.info(f"CPU optimized for {model_name}")
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")
    
    def aggressive_cleanup(self) -> None:
        """Perform aggressive memory cleanup including GPU memory."""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clear GPU memory if available
            if self.gpu_available:
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.debug("GPU memory cache cleared")
                except Exception as gpu_e:
                    logger.warning(f"GPU cleanup failed: {gpu_e}")
            
            # Clear numpy cache if available
            try:
                np.clear_cache()
            except AttributeError:
                pass  # numpy.clear_cache() not available in all versions
            
            # Clear OpenCV cache
            try:
                cv2.setUseOptimized(False)
                cv2.setUseOptimized(True)
            except:
                pass
            
            # Clear PyTorch cache
            try:
                if hasattr(torch, 'clear_autocast_cache'):
                    torch.clear_autocast_cache()
            except:
                pass
            
            logger.debug(f"Memory cleanup completed, collected {collected} objects")
            
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def cache_model_memory_usage(self, model_name: str, memory_usage: Dict[str, float]) -> None:
        """Cache memory usage information for a loaded model."""
        self.model_memory_cache[model_name] = {
            'timestamp': time.time(),
            'memory_usage': memory_usage.copy()
        }
        logger.info(f"Cached memory usage for {model_name}: {memory_usage}")
    
    def get_cached_model_memory(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get cached memory usage for a model if available."""
        if model_name in self.model_memory_cache:
            cache_entry = self.model_memory_cache[model_name]
            # Cache is valid for 1 hour
            if time.time() - cache_entry['timestamp'] < 3600:
                return cache_entry['memory_usage']
            else:
                # Remove stale cache
                del self.model_memory_cache[model_name]
        return None
    
    def monitor_model_loading(self, model_name: str) -> Dict[str, float]:
        """Monitor memory usage during model loading."""
        logger.info(f"Starting memory monitoring for {model_name} loading...")
        
        # Get baseline memory
        baseline = self.get_memory_info()
        
        return {
            'baseline_cpu_gb': baseline['system_available_gb'],
            'baseline_gpu_gb': baseline.get('gpu_available_gb', 0),
            'baseline_process_mb': baseline['process_rss_mb']
        }
    
    def finalize_model_loading(self, model_name: str, baseline: Dict[str, float]) -> Dict[str, float]:
        """Finalize memory monitoring after model loading."""
        current = self.get_memory_info()
        
        usage = {
            'cpu_used_gb': baseline['baseline_cpu_gb'] - current['system_available_gb'],
            'gpu_used_gb': baseline['baseline_gpu_gb'] - current.get('gpu_available_gb', 0),
            'process_increase_mb': current['process_rss_mb'] - baseline['baseline_process_mb']
        }
        
        # Cache the usage information
        self.cache_model_memory_usage(model_name, usage)
        
        logger.info(f"Model {model_name} loaded - Memory usage: CPU: {usage['cpu_used_gb']:.2f}GB, GPU: {usage['gpu_used_gb']:.2f}GB, Process: +{usage['process_increase_mb']:.1f}MB")
        
        return usage
    
    def log_memory_usage(self, context: str = "") -> None:
        """Log comprehensive memory usage including GPU."""
        memory_info = self.get_memory_info()
        context_str = f" ({context})" if context else ""
        
        log_msg = (
            f"Memory usage{context_str}: "
            f"System: {memory_info['system_available_gb']:.1f}GB available "
            f"({memory_info['system_used_percent']:.1f}% used), "
            f"Process: {memory_info['process_rss_mb']:.1f}MB RSS"
        )
        
        if self.gpu_available and memory_info.get('gpu_total_gb', 0) > 0:
            log_msg += (
                f", GPU: {memory_info['gpu_available_gb']:.1f}GB available "
                f"({memory_info['gpu_allocated_gb']:.1f}GB allocated)"
            )
        
        logger.info(log_msg)
    
    def resize_frame_for_memory(self, frame: np.ndarray, max_width: int = 512, max_height: int = 512) -> np.ndarray:
        """Resize frame for high quality processing while preserving aspect ratio."""
        try:
            h, w = frame.shape[:2]
            
            # Calculate scaling factor
            scale_w = max_width / w
            scale_h = max_height / h
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale
            
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Use INTER_AREA for downscaling (better quality, less memory)
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.debug(f"Resized frame from {w}x{h} to {new_w}x{new_h} (scale: {scale:.2f})")
                return resized
            
            return frame
            
        except Exception as e:
            logger.warning(f"Frame resize failed: {e}")
            return frame
    
    def optimize_frame_dtype(self, frame: np.ndarray) -> np.ndarray:
        """Optimize frame data type for memory efficiency."""
        try:
            # Convert to uint8 if not already (most memory efficient for images)
            if frame.dtype != np.uint8:
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    # Assume normalized float values [0, 1]
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
                
                logger.debug(f"Optimized frame dtype to uint8")
            
            return frame
            
        except Exception as e:
            logger.warning(f"Frame dtype optimization failed: {e}")
            return frame
    
    def process_frames_in_chunks(self, frames: np.ndarray, chunk_size: int = 32):
        """Generator to process frames in memory-efficient chunks."""
        total_frames = len(frames)
        
        for i in range(0, total_frames, chunk_size):
            end_idx = min(i + chunk_size, total_frames)
            chunk = frames[i:end_idx]
            
            # Log memory before processing chunk
            if i % (chunk_size * 5) == 0:  # Log every 5 chunks
                self.log_memory_usage(f"Processing chunk {i//chunk_size + 1}")
            
            yield i, chunk
            
            # Cleanup after each chunk
            del chunk
            if i % (chunk_size * 2) == 0:  # Cleanup every 2 chunks
                self.aggressive_cleanup()

# Global memory manager instance
memory_manager = MemoryManager()