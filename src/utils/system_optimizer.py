"""System resource optimization for heavy model operations."""

import os
import gc
import psutil
import threading
import time
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from .logger import get_logger

logger = get_logger(__name__)

class SystemOptimizer:
    """Optimizes system resources for heavy AI model operations."""
    
    def __init__(self):
        self.original_settings = {}
        self.optimization_active = False
        self._monitor_thread = None
        self._stop_monitoring = False
        
    def optimize_for_heavy_models(self) -> Dict[str, Any]:
        """Apply system optimizations for heavy model loading and inference."""
        if self.optimization_active:
            logger.info("System optimization already active")
            return self.original_settings
        
        logger.info("Applying system optimizations for heavy models...")
        
        optimizations = {
            'memory': self._optimize_memory(),
            'cpu': self._optimize_cpu(),
            'environment': self._optimize_environment(),
            'process': self._optimize_process()
        }
        
        self.optimization_active = True
        logger.info(f"System optimizations applied: {optimizations}")
        
        return optimizations
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory settings for heavy model operations."""
        memory_opts = {}
        
        try:
            # Force garbage collection
            collected = gc.collect()
            memory_opts['gc_collected'] = collected
            
            # Set aggressive garbage collection
            original_thresholds = gc.get_threshold()
            self.original_settings['gc_thresholds'] = original_thresholds
            
            # More aggressive GC for heavy models
            gc.set_threshold(100, 10, 10)
            memory_opts['gc_threshold_set'] = True
            
            # Enable automatic garbage collection
            if not gc.isenabled():
                gc.enable()
                memory_opts['gc_enabled'] = True
            
            logger.info(f"Memory optimization completed: {memory_opts}")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            memory_opts['error'] = str(e)
        
        return memory_opts
    
    def _optimize_cpu(self) -> Dict[str, Any]:
        """Optimize CPU settings for heavy model operations."""
        cpu_opts = {}
        
        try:
            # Set process priority to high (if possible)
            current_process = psutil.Process()
            original_priority = current_process.nice()
            self.original_settings['process_priority'] = original_priority
            
            try:
                # Try to set higher priority (lower nice value)
                if os.name == 'nt':  # Windows
                    current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                else:  # Unix-like
                    current_process.nice(-5)
                cpu_opts['priority_increased'] = True
            except (psutil.AccessDenied, OSError) as e:
                logger.warning(f"Could not increase process priority: {e}")
                cpu_opts['priority_warning'] = str(e)
            
            # Set CPU affinity to use all available cores
            try:
                available_cpus = list(range(psutil.cpu_count()))
                current_process.cpu_affinity(available_cpus)
                cpu_opts['cpu_affinity_set'] = len(available_cpus)
            except (psutil.AccessDenied, OSError) as e:
                logger.warning(f"Could not set CPU affinity: {e}")
                cpu_opts['affinity_warning'] = str(e)
            
            logger.info(f"CPU optimization completed: {cpu_opts}")
            
        except Exception as e:
            logger.error(f"CPU optimization failed: {e}")
            cpu_opts['error'] = str(e)
        
        return cpu_opts
    
    def _optimize_environment(self) -> Dict[str, Any]:
        """Optimize environment variables for heavy model operations."""
        env_opts = {}
        
        try:
            # PyTorch optimizations
            torch_vars = {
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
                'TORCH_CUDNN_V8_API_ENABLED': '1',
                'CUDA_LAUNCH_BLOCKING': '0',  # Async CUDA operations
                'TOKENIZERS_PARALLELISM': 'false',  # Avoid tokenizer warnings
            }
            
            # HuggingFace optimizations
            hf_vars = {
                'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
                'HF_HUB_DISABLE_PROGRESS_BARS': '1',
                'TRANSFORMERS_OFFLINE': '0',
                'HF_DATASETS_OFFLINE': '0',
            }
            
            # Memory optimizations
            memory_vars = {
                'MALLOC_TRIM_THRESHOLD_': '100000',
                'MALLOC_MMAP_THRESHOLD_': '100000',
            }
            
            all_vars = {**torch_vars, **hf_vars, **memory_vars}
            
            for var, value in all_vars.items():
                if var not in os.environ:
                    os.environ[var] = value
                    env_opts[f'set_{var}'] = value
                else:
                    env_opts[f'existing_{var}'] = os.environ[var]
            
            logger.info(f"Environment optimization completed: {len(env_opts)} variables processed")
            
        except Exception as e:
            logger.error(f"Environment optimization failed: {e}")
            env_opts['error'] = str(e)
        
        return env_opts
    
    def _optimize_process(self) -> Dict[str, Any]:
        """Optimize current process for heavy model operations."""
        process_opts = {}
        
        try:
            current_process = psutil.Process()
            
            # Get current memory info
            memory_info = current_process.memory_info()
            process_opts['initial_memory_mb'] = memory_info.rss / 1024 / 1024
            
            # Set memory limit if possible (to prevent system freeze)
            try:
                # Set soft memory limit to 80% of available system memory
                available_memory = psutil.virtual_memory().available
                soft_limit = int(available_memory * 0.8)
                
                if hasattr(current_process, 'rlimit'):
                    import resource
                    current_limit = resource.getrlimit(resource.RLIMIT_AS)
                    self.original_settings['memory_limit'] = current_limit
                    
                    # Only set if current limit is higher or unlimited
                    if current_limit[0] == -1 or current_limit[0] > soft_limit:
                        resource.setrlimit(resource.RLIMIT_AS, (soft_limit, current_limit[1]))
                        process_opts['memory_limit_set'] = soft_limit / 1024 / 1024 / 1024  # GB
                
            except (AttributeError, OSError) as e:
                logger.warning(f"Could not set memory limit: {e}")
                process_opts['memory_limit_warning'] = str(e)
            
            # Get CPU count and threads
            process_opts['cpu_count'] = psutil.cpu_count()
            process_opts['cpu_count_logical'] = psutil.cpu_count(logical=True)
            
            logger.info(f"Process optimization completed: {process_opts}")
            
        except Exception as e:
            logger.error(f"Process optimization failed: {e}")
            process_opts['error'] = str(e)
        
        return process_opts
    
    def start_resource_monitoring(self, interval: float = 5.0) -> None:
        """Start continuous resource monitoring during heavy operations."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Resource monitoring already active")
            return
        
        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started resource monitoring with {interval}s interval")
    
    def stop_resource_monitoring(self) -> None:
        """Stop resource monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_monitoring = True
            self._monitor_thread.join(timeout=10)
            logger.info("Stopped resource monitoring")
    
    def _monitor_resources(self, interval: float) -> None:
        """Monitor system resources continuously."""
        logger.info("Resource monitoring started")
        
        while not self._stop_monitoring:
            try:
                # Get system stats
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Get process stats
                current_process = psutil.Process()
                process_memory = current_process.memory_info()
                
                # Log if resources are getting critical
                if memory.percent > 90:
                    logger.warning(f"High memory usage: {memory.percent:.1f}% ({memory.available / 1024**3:.1f}GB available)")
                
                if cpu_percent > 95:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                # Log process memory if it's growing significantly
                process_memory_gb = process_memory.rss / 1024**3
                if process_memory_gb > 8:  # Log if process uses more than 8GB
                    logger.info(f"Process memory usage: {process_memory_gb:.1f}GB")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(interval)
    
    @contextmanager
    def optimized_context(self, enable_monitoring: bool = True):
        """Context manager for optimized heavy model operations."""
        try:
            # Apply optimizations
            optimizations = self.optimize_for_heavy_models()
            
            # Start monitoring if requested
            if enable_monitoring:
                self.start_resource_monitoring()
            
            yield optimizations
            
        finally:
            # Clean up
            if enable_monitoring:
                self.stop_resource_monitoring()
            
            self.restore_original_settings()
    
    def restore_original_settings(self) -> None:
        """Restore original system settings."""
        if not self.optimization_active:
            return
        
        logger.info("Restoring original system settings...")
        
        try:
            # Restore garbage collection thresholds
            if 'gc_thresholds' in self.original_settings:
                gc.set_threshold(*self.original_settings['gc_thresholds'])
            
            # Restore process priority
            if 'process_priority' in self.original_settings:
                try:
                    current_process = psutil.Process()
                    current_process.nice(self.original_settings['process_priority'])
                except (psutil.AccessDenied, OSError) as e:
                    logger.warning(f"Could not restore process priority: {e}")
            
            # Restore memory limit
            if 'memory_limit' in self.original_settings:
                try:
                    import resource
                    resource.setrlimit(resource.RLIMIT_AS, self.original_settings['memory_limit'])
                except (AttributeError, OSError) as e:
                    logger.warning(f"Could not restore memory limit: {e}")
            
            self.optimization_active = False
            self.original_settings.clear()
            
            logger.info("Original system settings restored")
            
        except Exception as e:
            logger.error(f"Error restoring original settings: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            memory = psutil.virtual_memory()
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'cpu_percent': psutil.cpu_percent(interval=1)
            }
            
            process = psutil.Process()
            process_info = {
                'pid': process.pid,
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads()
            }
            
            return {
                'system_memory': {
                    'total_gb': memory.total / 1024**3,
                    'available_gb': memory.available / 1024**3,
                    'used_percent': memory.percent
                },
                'cpu': cpu_info,
                'process': process_info,
                'optimization_active': self.optimization_active
            }
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {'error': str(e)}

# Global system optimizer instance
system_optimizer = SystemOptimizer()