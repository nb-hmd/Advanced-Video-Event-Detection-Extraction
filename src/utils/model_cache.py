import os
import time
import pickle
import hashlib
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import threading
from .logger import get_logger
from .memory_manager import memory_manager
from .config import settings

logger = get_logger(__name__)

class ModelCache:
    """Advanced model caching system for efficient memory usage."""
    
    def __init__(self, cache_dir: str = None, max_memory_gb: float = 4.0, 
                 max_cache_age_hours: int = 24):
        self.cache_dir = Path(cache_dir or (Path(settings.DATA_DIR) / "model_cache"))
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.max_memory_gb = max_memory_gb
        self.max_cache_age_hours = max_cache_age_hours
        
        # In-memory cache for loaded models
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Cache metadata
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Thread lock for thread-safe operations
        self._lock = threading.RLock()
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        logger.info(f"Model cache initialized: {self.cache_dir}, max memory: {max_memory_gb}GB")
    
    def get_model(self, model_key: str, model_config: Dict[str, Any] = None) -> Optional[Any]:
        """Get a model from cache.
        
        Args:
            model_key: Unique identifier for the model
            model_config: Configuration used to validate cache validity
            
        Returns:
            Cached model if available and valid, None otherwise
        """
        with self._lock:
            # Check memory cache first
            if model_key in self.memory_cache:
                cache_entry = self.memory_cache[model_key]
                
                # Validate cache entry
                if self._is_cache_valid(model_key, model_config):
                    cache_entry['last_accessed'] = time.time()
                    cache_entry['access_count'] += 1
                    
                    logger.info(f"Model '{model_key}' retrieved from memory cache (access #{cache_entry['access_count']})")
                    return cache_entry['model']
                else:
                    logger.info(f"Memory cache for '{model_key}' is invalid, removing")
                    self._remove_from_memory_cache(model_key)
            
            # Check disk cache
            disk_model = self._load_from_disk_cache(model_key, model_config)
            if disk_model is not None:
                # Load into memory cache if there's space
                if self._can_fit_in_memory_cache(model_key):
                    self._add_to_memory_cache(model_key, disk_model, model_config)
                    logger.info(f"Model '{model_key}' loaded from disk cache to memory")
                else:
                    logger.info(f"Model '{model_key}' loaded from disk cache (memory cache full)")
                
                return disk_model
            
            logger.debug(f"Model '{model_key}' not found in cache")
            return None
    
    def cache_model(self, model_key: str, model: Any, model_config: Dict[str, Any] = None,
                   persist_to_disk: bool = True) -> bool:
        """Cache a model in memory and optionally on disk.
        
        Args:
            model_key: Unique identifier for the model
            model: The model object to cache
            model_config: Configuration associated with the model
            persist_to_disk: Whether to save the model to disk
            
        Returns:
            True if caching was successful, False otherwise
        """
        with self._lock:
            try:
                # Check if we can fit this model in memory
                if not self._can_fit_in_memory_cache(model_key):
                    # Try to make space by evicting old models
                    self._evict_old_models()
                    
                    if not self._can_fit_in_memory_cache(model_key):
                        logger.warning(f"Cannot fit model '{model_key}' in memory cache")
                        if not persist_to_disk:
                            return False
                
                # Add to memory cache if possible
                if self._can_fit_in_memory_cache(model_key):
                    self._add_to_memory_cache(model_key, model, model_config)
                    logger.info(f"Model '{model_key}' cached in memory")
                
                # Save to disk if requested
                if persist_to_disk:
                    success = self._save_to_disk_cache(model_key, model, model_config)
                    if success:
                        logger.info(f"Model '{model_key}' cached to disk")
                    else:
                        logger.warning(f"Failed to cache model '{model_key}' to disk")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache model '{model_key}': {e}")
                return False
    
    def remove_model(self, model_key: str, remove_from_disk: bool = True) -> bool:
        """Remove a model from cache.
        
        Args:
            model_key: Unique identifier for the model
            remove_from_disk: Whether to also remove from disk cache
            
        Returns:
            True if removal was successful, False otherwise
        """
        with self._lock:
            try:
                # Remove from memory cache
                if model_key in self.memory_cache:
                    self._remove_from_memory_cache(model_key)
                    logger.info(f"Model '{model_key}' removed from memory cache")
                
                # Remove from disk cache
                if remove_from_disk:
                    self._remove_from_disk_cache(model_key)
                    logger.info(f"Model '{model_key}' removed from disk cache")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to remove model '{model_key}': {e}")
                return False
    
    def clear_cache(self, clear_disk: bool = False) -> None:
        """Clear all cached models.
        
        Args:
            clear_disk: Whether to also clear disk cache
        """
        with self._lock:
            # Clear memory cache
            for model_key in list(self.memory_cache.keys()):
                self._remove_from_memory_cache(model_key)
            
            logger.info("Memory cache cleared")
            
            # Clear disk cache if requested
            if clear_disk:
                try:
                    for cache_file in self.cache_dir.glob("*.pkl"):
                        cache_file.unlink()
                    
                    for metadata_file in self.cache_dir.glob("*.meta"):
                        metadata_file.unlink()
                    
                    self.cache_metadata.clear()
                    logger.info("Disk cache cleared")
                    
                except Exception as e:
                    logger.error(f"Failed to clear disk cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        with self._lock:
            memory_info = memory_manager.get_memory_info()
            
            # Calculate memory usage of cached models
            total_memory_mb = 0
            for cache_entry in self.memory_cache.values():
                total_memory_mb += cache_entry.get('memory_usage_mb', 0)
            
            # Count disk cache files
            disk_cache_files = len(list(self.cache_dir.glob("*.pkl")))
            
            return {
                'memory_cache_count': len(self.memory_cache),
                'disk_cache_count': disk_cache_files,
                'total_memory_usage_mb': total_memory_mb,
                'max_memory_gb': self.max_memory_gb,
                'system_available_gb': memory_info['system_available_gb'],
                'cache_hit_rate': self._calculate_hit_rate(),
                'oldest_cache_age_hours': self._get_oldest_cache_age()
            }
    
    def cleanup_old_cache(self) -> None:
        """Clean up old cache entries."""
        with self._lock:
            current_time = time.time()
            max_age_seconds = self.max_cache_age_hours * 3600
            
            # Clean memory cache
            expired_keys = []
            for model_key, cache_entry in self.memory_cache.items():
                if current_time - cache_entry['created_at'] > max_age_seconds:
                    expired_keys.append(model_key)
            
            for key in expired_keys:
                self._remove_from_memory_cache(key)
                logger.info(f"Removed expired model '{key}' from memory cache")
            
            # Clean disk cache
            try:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    if current_time - cache_file.stat().st_mtime > max_age_seconds:
                        model_key = cache_file.stem
                        self._remove_from_disk_cache(model_key)
                        logger.info(f"Removed expired model '{model_key}' from disk cache")
            except Exception as e:
                logger.error(f"Failed to cleanup disk cache: {e}")
    
    def _is_cache_valid(self, model_key: str, model_config: Dict[str, Any] = None) -> bool:
        """Check if a cache entry is valid."""
        if model_key not in self.memory_cache:
            return False
        
        cache_entry = self.memory_cache[model_key]
        
        # Check age
        current_time = time.time()
        max_age_seconds = self.max_cache_age_hours * 3600
        if current_time - cache_entry['created_at'] > max_age_seconds:
            return False
        
        # Check config hash if provided
        if model_config is not None:
            config_hash = self._hash_config(model_config)
            if cache_entry.get('config_hash') != config_hash:
                return False
        
        return True
    
    def _can_fit_in_memory_cache(self, model_key: str) -> bool:
        """Check if a model can fit in memory cache."""
        memory_info = memory_manager.get_memory_info()
        available_gb = memory_info['system_available_gb']
        
        # Reserve some memory for system operations
        usable_gb = min(available_gb - 0.5, self.max_memory_gb)
        
        # Calculate current cache usage
        current_usage_gb = sum(
            cache_entry.get('memory_usage_mb', 0) / 1024
            for cache_entry in self.memory_cache.values()
        )
        
        # Estimate model size (conservative estimate)
        estimated_model_size_gb = 0.5  # Default estimate
        if model_key in self.cache_metadata:
            estimated_model_size_gb = self.cache_metadata[model_key].get('size_gb', 0.5)
        
        return (current_usage_gb + estimated_model_size_gb) <= usable_gb
    
    def _add_to_memory_cache(self, model_key: str, model: Any, model_config: Dict[str, Any] = None) -> None:
        """Add a model to memory cache."""
        # Estimate memory usage
        memory_usage_mb = self._estimate_model_memory(model)
        
        cache_entry = {
            'model': model,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'access_count': 1,
            'memory_usage_mb': memory_usage_mb,
            'config_hash': self._hash_config(model_config) if model_config else None
        }
        
        self.memory_cache[model_key] = cache_entry
    
    def _remove_from_memory_cache(self, model_key: str) -> None:
        """Remove a model from memory cache."""
        if model_key in self.memory_cache:
            del self.memory_cache[model_key]
            # Force garbage collection
            memory_manager.aggressive_cleanup()
    
    def _evict_old_models(self) -> None:
        """Evict old models to make space."""
        if not self.memory_cache:
            return
        
        # Sort by last accessed time (oldest first)
        sorted_models = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Evict oldest 25% of models
        evict_count = max(1, len(sorted_models) // 4)
        
        for i in range(evict_count):
            model_key, _ = sorted_models[i]
            self._remove_from_memory_cache(model_key)
            logger.info(f"Evicted model '{model_key}' from memory cache")
    
    def _save_to_disk_cache(self, model_key: str, model: Any, model_config: Dict[str, Any] = None) -> bool:
        """Save a model to disk cache."""
        try:
            cache_file = self.cache_dir / f"{model_key}.pkl"
            metadata_file = self.cache_dir / f"{model_key}.meta"
            
            # Save model
            with open(cache_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata = {
                'created_at': time.time(),
                'config_hash': self._hash_config(model_config) if model_config else None,
                'size_bytes': cache_file.stat().st_size,
                'size_gb': cache_file.stat().st_size / (1024**3)
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            self.cache_metadata[model_key] = metadata
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model '{model_key}' to disk: {e}")
            return False
    
    def _load_from_disk_cache(self, model_key: str, model_config: Dict[str, Any] = None) -> Optional[Any]:
        """Load a model from disk cache."""
        try:
            cache_file = self.cache_dir / f"{model_key}.pkl"
            metadata_file = self.cache_dir / f"{model_key}.meta"
            
            if not cache_file.exists() or not metadata_file.exists():
                return None
            
            # Load and validate metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Check age
            current_time = time.time()
            max_age_seconds = self.max_cache_age_hours * 3600
            if current_time - metadata['created_at'] > max_age_seconds:
                self._remove_from_disk_cache(model_key)
                return None
            
            # Check config hash if provided
            if model_config is not None:
                config_hash = self._hash_config(model_config)
                if metadata.get('config_hash') != config_hash:
                    return None
            
            # Load model
            with open(cache_file, 'rb') as f:
                model = pickle.load(f)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_key}' from disk: {e}")
            return None
    
    def _remove_from_disk_cache(self, model_key: str) -> None:
        """Remove a model from disk cache."""
        try:
            cache_file = self.cache_dir / f"{model_key}.pkl"
            metadata_file = self.cache_dir / f"{model_key}.meta"
            
            if cache_file.exists():
                cache_file.unlink()
            
            if metadata_file.exists():
                metadata_file.unlink()
            
            if model_key in self.cache_metadata:
                del self.cache_metadata[model_key]
                
        except Exception as e:
            logger.error(f"Failed to remove model '{model_key}' from disk: {e}")
    
    def _load_cache_metadata(self) -> None:
        """Load existing cache metadata."""
        try:
            for metadata_file in self.cache_dir.glob("*.meta"):
                model_key = metadata_file.stem
                
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.cache_metadata[model_key] = metadata
            
            logger.info(f"Loaded metadata for {len(self.cache_metadata)} cached models")
            
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate a hash for model configuration."""
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate memory usage of a model in MB."""
        try:
            # Try to get actual memory usage if it's a PyTorch model
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                # Estimate 4 bytes per parameter (float32)
                return (total_params * 4) / (1024 * 1024)
            
            # Fallback to a conservative estimate
            return 100.0  # 100MB default
            
        except Exception:
            return 100.0
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = sum(
            cache_entry['access_count'] 
            for cache_entry in self.memory_cache.values()
        )
        
        if total_accesses == 0:
            return 0.0
        
        # This is a simplified calculation
        # In a real implementation, you'd track hits vs misses
        return min(1.0, len(self.memory_cache) / max(1, total_accesses))
    
    def _get_oldest_cache_age(self) -> float:
        """Get the age of the oldest cache entry in hours."""
        if not self.memory_cache:
            return 0.0
        
        current_time = time.time()
        oldest_time = min(
            cache_entry['created_at'] 
            for cache_entry in self.memory_cache.values()
        )
        
        return (current_time - oldest_time) / 3600

# Global model cache instance
model_cache = ModelCache()