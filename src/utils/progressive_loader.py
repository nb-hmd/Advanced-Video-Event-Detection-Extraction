import time
import threading
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from .logger import get_logger
from .memory_manager import memory_manager
from .model_cache import model_cache

logger = get_logger(__name__)

class ProgressiveModelLoader:
    """Progressive model loading system with real-time memory monitoring."""
    
    def __init__(self, memory_threshold_gb: float = 1.0):
        self.memory_threshold_gb = memory_threshold_gb
        self.loading_queue = []
        self.loaded_models = {}
        self.loading_status = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self.load_callbacks = {}
        
    def register_model(self, model_name: str, load_function: Callable, 
                      priority: int = 1, dependencies: List[str] = None,
                      callback: Callable = None) -> None:
        """Register a model for progressive loading.
        
        Args:
            model_name: Unique identifier for the model
            load_function: Function that loads the model
            priority: Loading priority (1=highest, 5=lowest)
            dependencies: List of model names that must be loaded first
            callback: Optional callback function called after successful loading
        """
        model_info = {
            'name': model_name,
            'load_function': load_function,
            'priority': priority,
            'dependencies': dependencies or [],
            'loaded': False,
            'loading': False,
            'error': None
        }
        
        self.loading_queue.append(model_info)
        self.loading_status[model_name] = 'queued'
        
        if callback:
            self.load_callbacks[model_name] = callback
        
        logger.info(f"Registered model '{model_name}' for progressive loading (priority: {priority})")
    
    def check_cache_status(self) -> Dict[str, bool]:
        """Check which models are available in cache."""
        cache_status = {}
        
        for model_info in self.loading_queue:
            model_name = model_info['name']
            # Generate cache keys for different configurations
            cache_keys = [
                f"{model_name}_cuda",
                f"{model_name}_cpu",
                f"{model_name}_cpu_fallback"
            ]
            
            cached = False
            for cache_key in cache_keys:
                if model_cache.has_model(cache_key):
                    cached = True
                    break
            
            cache_status[model_name] = cached
            
        return cache_status
    
    def start_progressive_loading(self) -> None:
        """Start the progressive model loading process with cache optimization."""
        logger.info("Starting progressive model loading with real-time memory monitoring")
        
        # Start memory monitoring
        self.start_memory_monitoring()
        
        # Sort queue by priority and dependencies
        self._sort_loading_queue()
        
        # Check cache status and optimize loading order
        cache_status = self.check_cache_status()
        logger.info(f"Cache status: {cache_status}")
        
        # Re-sort queue considering cache status (cached models first for faster loading)
        self._optimize_loading_order_with_cache(cache_status)
        
        # Start loading models
        self._load_models_progressively(cache_status)
    
    def _sort_loading_queue(self) -> None:
        """Sort loading queue by priority and resolve dependencies."""
        # Simple topological sort for dependencies
        sorted_queue = []
        remaining = self.loading_queue.copy()
        
        while remaining:
            # Find models with no unresolved dependencies
            ready_models = []
            for model in remaining:
                deps_satisfied = all(
                    dep in [m['name'] for m in sorted_queue] 
                    for dep in model['dependencies']
                )
                if deps_satisfied:
                    ready_models.append(model)
            
            if not ready_models:
                # Circular dependency or missing dependency
                logger.warning("Circular or missing dependencies detected, loading remaining models by priority")
                ready_models = remaining
            
            # Sort ready models by priority
            ready_models.sort(key=lambda x: x['priority'])
            
            # Add to sorted queue and remove from remaining
            for model in ready_models:
                sorted_queue.append(model)
                remaining.remove(model)
        
        self.loading_queue = sorted_queue
        logger.info(f"Sorted {len(self.loading_queue)} models for progressive loading")
    
    def _optimize_loading_order_with_cache(self, cache_status: Dict[str, bool]) -> None:
        """Re-optimize loading order considering cache status."""
        # Separate cached and non-cached models while preserving dependency order
        cached_models = []
        non_cached_models = []
        
        for model_info in self.loading_queue:
            model_name = model_info['name']
            if cache_status.get(model_name, False):
                cached_models.append(model_info)
            else:
                non_cached_models.append(model_info)
        
        # Load cached models first (they're faster), then non-cached
        self.loading_queue = cached_models + non_cached_models
        
        cached_count = len(cached_models)
        total_count = len(self.loading_queue)
        logger.info(f"Optimized loading order: {cached_count} cached models first, {total_count - cached_count} fresh loads")
    
    def _load_models_progressively(self, cache_status: Dict[str, bool]) -> None:
        """Load models one by one with memory monitoring and cache optimization."""
        for model_info in self.loading_queue:
            model_name = model_info['name']
            is_cached = cache_status.get(model_name, False)
            
            try:
                # Check if we should proceed with loading (more lenient for cached models)
                if not self._should_load_model(model_name, is_cached):
                    logger.warning(f"Skipping {model_name} due to memory constraints")
                    self.loading_status[model_name] = 'skipped'
                    continue
                
                cache_info = " (from cache)" if is_cached else " (fresh load)"
                logger.info(f"Loading model: {model_name}{cache_info}")
                self.loading_status[model_name] = 'loading'
                model_info['loading'] = True
                
                # Monitor memory before loading
                baseline = memory_manager.monitor_model_loading(model_name)
                
                # Load the model
                start_time = time.time()
                loaded_model = model_info['load_function']()
                load_time = time.time() - start_time
                
                # Finalize memory monitoring
                usage = memory_manager.finalize_model_loading(model_name, baseline)
                
                # Store loaded model
                self.loaded_models[model_name] = loaded_model
                model_info['loaded'] = True
                model_info['loading'] = False
                self.loading_status[model_name] = 'loaded'
                
                logger.info(f"Successfully loaded {model_name} in {load_time:.2f}s")
                
                # Call callback if provided
                if model_name in self.load_callbacks:
                    try:
                        self.load_callbacks[model_name](loaded_model)
                    except Exception as cb_e:
                        logger.warning(f"Callback for {model_name} failed: {cb_e}")
                
                # Shorter delay for cached models
                delay = 0.2 if is_cached else 1.0
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                model_info['error'] = str(e)
                model_info['loading'] = False
                self.loading_status[model_name] = 'error'
                
                # Decide whether to continue or stop
                if model_info['priority'] <= 2:  # Critical models
                    logger.error(f"Critical model {model_name} failed to load, stopping progressive loading")
                    break
                else:
                    logger.warning(f"Non-critical model {model_name} failed, continuing with other models")
                    continue
        
        logger.info("Progressive model loading completed")
        self._log_loading_summary()
    
    def _should_load_model(self, model_name: str, is_cached: bool = False) -> bool:
        """Check if a model should be loaded based on memory constraints."""
        # Check memory availability with different thresholds for cached vs fresh models
        can_load, message = memory_manager.can_load_model(model_name.lower())
        
        if not can_load:
            logger.warning(f"Memory check failed for {model_name}: {message}")
            
            # Try aggressive cleanup
            memory_manager.aggressive_cleanup()
            time.sleep(2)  # Wait for cleanup to take effect
            
            # Check again with more lenient requirements for cached models
            can_load, message = memory_manager.can_load_model(model_name.lower())
            
            # If still can't load but it's cached, try with reduced memory requirements
            if not can_load and is_cached:
                memory_info = memory_manager.get_memory_info()
                available_gb = memory_info['system_available_gb']
                
                # More lenient threshold for cached models (they typically load faster and use less peak memory)
                if available_gb >= 0.5:  # 500MB minimum for cached models
                    logger.info(f"Allowing cached model {model_name} to load with {available_gb:.1f}GB available")
                    can_load = True
            
        return can_load
    
    def start_memory_monitoring(self) -> None:
        """Start real-time memory monitoring in a separate thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started real-time memory monitoring")
    
    def stop_memory_monitoring(self) -> None:
        """Stop real-time memory monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped real-time memory monitoring")
    
    def _memory_monitor_loop(self) -> None:
        """Memory monitoring loop that runs in a separate thread."""
        while self.monitoring_active:
            try:
                memory_info = memory_manager.get_memory_info()
                
                # Check for memory pressure
                available_gb = memory_info['system_available_gb']
                
                if available_gb < self.memory_threshold_gb:
                    logger.warning(f"Memory pressure detected: {available_gb:.1f}GB available")
                    
                    # Trigger aggressive cleanup
                    memory_manager.aggressive_cleanup()
                    
                    # Log current status
                    memory_manager.log_memory_usage("Memory pressure cleanup")
                
                # Sleep for monitoring interval
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(30)  # Wait longer on error
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        return self.loaded_models.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self.loaded_models
    
    def get_loading_status(self) -> Dict[str, str]:
        """Get the current loading status of all models."""
        return self.loading_status.copy()
    
    def _log_loading_summary(self) -> None:
        """Log a summary of the loading process."""
        total_models = len(self.loading_queue)
        loaded_count = sum(1 for status in self.loading_status.values() if status == 'loaded')
        error_count = sum(1 for status in self.loading_status.values() if status == 'error')
        skipped_count = sum(1 for status in self.loading_status.values() if status == 'skipped')
        
        logger.info(f"Progressive loading summary: {loaded_count}/{total_models} loaded, {error_count} errors, {skipped_count} skipped")
        
        # Log cache statistics
        cache_stats = model_cache.get_cache_stats()
        logger.info(f"Cache statistics: {cache_stats}")
        
        # Log final memory status
        memory_manager.log_memory_usage("Progressive loading completed")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_memory_monitoring()
        
        # Clear loaded models to free memory
        for model_name in list(self.loaded_models.keys()):
            try:
                del self.loaded_models[model_name]
                logger.debug(f"Cleaned up model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup model {model_name}: {e}")
        
        self.loaded_models.clear()
        memory_manager.aggressive_cleanup()
        
        logger.info("Progressive loader cleanup completed")

# Global progressive loader instance
progressive_loader = ProgressiveModelLoader()