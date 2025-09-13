import traceback
import sys
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime
import json
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

def handle_model_loading_error(func):
    """Decorator to handle model loading errors gracefully."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Model loading error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return None or appropriate fallback
            return None
    return wrapper

def handle_inference_error(func):
    """Decorator to handle inference errors gracefully."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Inference error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty result or appropriate fallback
            return []
    return wrapper

def error_handler(func):
    """General error handler decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return None or raise the exception based on context
            raise e
    return wrapper

class SmallObjectDetectionError(Exception):
    """Base exception for small object detection system."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        self.message = message
        self.error_code = error_code or "SOD_UNKNOWN"
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp
        }

class BackgroundIndependenceError(SmallObjectDetectionError):
    """Error in background independence processing."""
    pass

class AdaptiveThresholdError(SmallObjectDetectionError):
    """Error in adaptive threshold calculation."""
    pass

class SmallObjectModelError(SmallObjectDetectionError):
    """Error in small object model processing."""
    pass

class RegionProposalError(SmallObjectDetectionError):
    """Error in region proposal network."""
    pass

class PerformanceError(SmallObjectDetectionError):
    """Error related to performance monitoring or optimization."""
    pass

class ConfigurationError(SmallObjectDetectionError):
    """Error in system configuration."""
    pass

class ModelLoadError(SmallObjectDetectionError):
    """Error loading AI models."""
    pass

class ErrorHandler:
    """Centralized error handling for small object detection system."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.error_log_path = Path(self.config.get('error_log_path', 'logs/errors.json'))
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Error handling settings
        self.log_errors = self.config.get('log_errors', True)
        self.save_error_details = self.config.get('save_error_details', True)
        self.max_error_logs = self.config.get('max_error_logs', 1000)
        
        # Recovery strategies
        self.auto_recovery_enabled = self.config.get('auto_recovery_enabled', True)
        self.fallback_strategies = self.config.get('fallback_strategies', {})
        
        logger.info(f"Error handler initialized with logging to {self.error_log_path}")
    
    def handle_error(self, 
                    error: Exception, 
                    context: str = None, 
                    details: Dict = None,
                    recovery_action: Callable = None) -> Dict[str, Any]:
        """Handle an error with logging, recovery, and reporting."""
        
        # Create error information
        error_info = {
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context or 'unknown',
            'details': details or {},
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc() if self.save_error_details else None
        }
        
        # Add specific error code if it's our custom error
        if isinstance(error, SmallObjectDetectionError):
            error_info['error_code'] = error.error_code
            error_info['details'].update(error.details)
        
        # Log the error
        if self.log_errors:
            self._log_error(error_info)
        
        # Save to error log file
        if self.save_error_details:
            self._save_error_to_file(error_info)
        
        # Attempt recovery
        recovery_result = None
        if self.auto_recovery_enabled and recovery_action:
            try:
                recovery_result = recovery_action()
                error_info['recovery_attempted'] = True
                error_info['recovery_successful'] = recovery_result is not None
                logger.info(f"Recovery action executed for {error_info['error_type']}")
            except Exception as recovery_error:
                error_info['recovery_attempted'] = True
                error_info['recovery_successful'] = False
                error_info['recovery_error'] = str(recovery_error)
                logger.error(f"Recovery action failed: {recovery_error}")
        
        return {
            'error_handled': True,
            'error_info': error_info,
            'recovery_result': recovery_result
        }
    
    def _log_error(self, error_info: Dict[str, Any]):
        """Log error with appropriate severity level."""
        error_type = error_info['error_type']
        context = error_info['context']
        message = error_info['message']
        
        # Determine log level based on error type
        if error_type in ['ModelLoadError', 'ConfigurationError']:
            logger.critical(f"Critical error in {context}: {message}")
        elif error_type in ['BackgroundIndependenceError', 'SmallObjectModelError']:
            logger.error(f"Processing error in {context}: {message}")
        elif error_type in ['AdaptiveThresholdError', 'RegionProposalError']:
            logger.warning(f"Component error in {context}: {message}")
        else:
            logger.error(f"Error in {context}: {message}")
    
    def _save_error_to_file(self, error_info: Dict[str, Any]):
        """Save error information to JSON log file."""
        try:
            # Load existing errors
            errors = []
            if self.error_log_path.exists():
                with open(self.error_log_path, 'r') as f:
                    errors = json.load(f)
            
            # Add new error
            errors.append(error_info)
            
            # Limit number of stored errors
            if len(errors) > self.max_error_logs:
                errors = errors[-self.max_error_logs:]
            
            # Save back to file
            with open(self.error_log_path, 'w') as f:
                json.dump(errors, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save error to file: {e}")
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time period."""
        try:
            if not self.error_log_path.exists():
                return {'total_errors': 0, 'error_types': {}, 'contexts': {}}
            
            with open(self.error_log_path, 'r') as f:
                errors = json.load(f)
            
            # Filter by time period
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            recent_errors = [
                error for error in errors 
                if datetime.fromisoformat(error['timestamp']).timestamp() >= cutoff_time
            ]
            
            # Calculate statistics
            error_types = {}
            contexts = {}
            recovery_stats = {'attempted': 0, 'successful': 0}
            
            for error in recent_errors:
                # Count error types
                error_type = error['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
                # Count contexts
                context = error['context']
                contexts[context] = contexts.get(context, 0) + 1
                
                # Count recovery attempts
                if error.get('recovery_attempted'):
                    recovery_stats['attempted'] += 1
                    if error.get('recovery_successful'):
                        recovery_stats['successful'] += 1
            
            return {
                'time_period_hours': hours,
                'total_errors': len(recent_errors),
                'error_types': error_types,
                'contexts': contexts,
                'recovery_stats': recovery_stats,
                'recovery_success_rate': (
                    recovery_stats['successful'] / max(recovery_stats['attempted'], 1)
                ) if recovery_stats['attempted'] > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get error statistics: {e}")
            return {'error': str(e)}

# Global error handler instance
error_handler = ErrorHandler()

def handle_errors(context: str = None, 
                 recovery_action: Callable = None,
                 raise_on_error: bool = False,
                 return_on_error: Any = None):
    """Decorator for automatic error handling."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle the error
                result = error_handler.handle_error(
                    error=e,
                    context=context or func.__name__,
                    details={'args': str(args), 'kwargs': str(kwargs)},
                    recovery_action=recovery_action
                )
                
                # Decide what to do next
                if raise_on_error:
                    raise e
                elif result.get('recovery_result') is not None:
                    return result['recovery_result']
                else:
                    return return_on_error
        
        return wrapper
    return decorator

def safe_execute(func: Callable, 
                *args, 
                context: str = None,
                default_return: Any = None,
                **kwargs) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler.handle_error(
            error=e,
            context=context or func.__name__,
            details={'args': str(args), 'kwargs': str(kwargs)}
        )
        return default_return

def create_fallback_strategy(primary_func: Callable, 
                           fallback_func: Callable,
                           context: str = None) -> Callable:
    """Create a function with automatic fallback on error."""
    
    def fallback_wrapper(*args, **kwargs):
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed in {context}, using fallback: {e}")
            error_handler.handle_error(
                error=e,
                context=f"{context}_primary",
                details={'fallback_used': True}
            )
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback function also failed in {context}: {fallback_error}")
                error_handler.handle_error(
                    error=fallback_error,
                    context=f"{context}_fallback"
                )
                raise fallback_error
    
    return fallback_wrapper

# Specific error handling functions for small object detection components

def handle_background_independence_error(func):
    """Decorator for background independence error handling."""
    
    def recovery_action():
        logger.info("Attempting background independence recovery: disabling SAM model")
        return {'sam_disabled': True, 'use_simple_background_removal': True}
    
    return handle_errors(
        context='background_independence',
        recovery_action=recovery_action,
        return_on_error={'background_independent_features': None, 'error': True}
    )(func)

def handle_adaptive_threshold_error(func):
    """Decorator for adaptive threshold error handling."""
    
    def recovery_action():
        logger.info("Attempting adaptive threshold recovery: using default thresholds")
        return {'use_default_thresholds': True, 'adaptive_disabled': True}
    
    return handle_errors(
        context='adaptive_thresholds',
        recovery_action=recovery_action,
        return_on_error={'threshold': 0.3, 'adaptive_applied': False}
    )(func)

def handle_small_object_model_error(func):
    """Decorator for small object model error handling."""
    
    def recovery_action():
        logger.info("Attempting model recovery: switching to fallback model")
        return {'use_fallback_model': True, 'model': 'yolov8_nano'}
    
    return handle_errors(
        context='small_object_models',
        recovery_action=recovery_action,
        return_on_error=[]
    )(func)

def handle_region_proposal_error(func):
    """Decorator for region proposal network error handling."""
    
    def recovery_action():
        logger.info("Attempting RPN recovery: disabling RPN, using full image")
        return {'rpn_disabled': True, 'use_full_image': True}
    
    return handle_errors(
        context='region_proposals',
        recovery_action=recovery_action,
        return_on_error=[]
    )(func)

# Validation functions

def validate_model_path(model_path: str, model_name: str = None) -> bool:
    """Validate that a model file exists and is accessible."""
    try:
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadError(
                f"Model file not found: {model_path}",
                error_code="MODEL_NOT_FOUND",
                details={'model_name': model_name, 'path': str(path)}
            )
        
        if not path.is_file():
            raise ModelLoadError(
                f"Model path is not a file: {model_path}",
                error_code="MODEL_INVALID_PATH",
                details={'model_name': model_name, 'path': str(path)}
            )
        
        # Check file size (should be > 1MB for most models)
        if path.stat().st_size < 1024 * 1024:
            logger.warning(f"Model file seems small ({path.stat().st_size} bytes): {model_path}")
        
        return True
        
    except Exception as e:
        if not isinstance(e, SmallObjectDetectionError):
            error_handler.handle_error(
                error=e,
                context='model_validation',
                details={'model_path': model_path, 'model_name': model_name}
            )
        raise

def validate_configuration(config: Dict[str, Any]) -> bool:
    """Validate system configuration."""
    try:
        required_keys = [
            'SMALL_OBJECT_DETECTION_ENABLED',
            'BACKGROUND_INDEPENDENCE_ENABLED',
            'ADAPTIVE_THRESHOLDS_ENABLED',
            'RPN_ENABLED'
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ConfigurationError(
                f"Missing required configuration keys: {missing_keys}",
                error_code="CONFIG_MISSING_KEYS",
                details={'missing_keys': missing_keys}
            )
        
        # Validate model paths if features are enabled
        if config.get('SMALL_OBJECT_DETECTION_ENABLED'):
            model_paths = {
                'FCOS_RT_MODEL_PATH': config.get('FCOS_RT_MODEL_PATH'),
                'RETINANET_SMALL_MODEL_PATH': config.get('RETINANET_SMALL_MODEL_PATH'),
                'YOLOV8_NANO_MODEL_PATH': config.get('YOLOV8_NANO_MODEL_PATH')
            }
            
            for model_name, model_path in model_paths.items():
                if model_path:
                    validate_model_path(model_path, model_name)
        
        if config.get('BACKGROUND_INDEPENDENCE_ENABLED'):
            sam_path = config.get('SAM_MODEL_PATH')
            if sam_path:
                validate_model_path(sam_path, 'SAM_MODEL')
        
        return True
        
    except Exception as e:
        if not isinstance(e, SmallObjectDetectionError):
            error_handler.handle_error(
                error=e,
                context='configuration_validation',
                details={'config_keys': list(config.keys())}
            )
        raise

def check_system_health() -> Dict[str, Any]:
    """Check overall system health and return status."""
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy',
        'components': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        # Check error statistics
        error_stats = error_handler.get_error_statistics(hours=1)
        if error_stats.get('total_errors', 0) > 10:
            health_status['warnings'].append(f"High error rate: {error_stats['total_errors']} errors in last hour")
            health_status['overall_status'] = 'warning'
        
        # Check critical error types
        critical_errors = ['ModelLoadError', 'ConfigurationError']
        for error_type in critical_errors:
            if error_type in error_stats.get('error_types', {}):
                health_status['errors'].append(f"Critical error detected: {error_type}")
                health_status['overall_status'] = 'error'
        
        # Check recovery success rate
        recovery_rate = error_stats.get('recovery_success_rate', 1.0)
        if recovery_rate < 0.8:
            health_status['warnings'].append(f"Low recovery success rate: {recovery_rate:.1%}")
            if health_status['overall_status'] == 'healthy':
                health_status['overall_status'] = 'warning'
        
        health_status['error_statistics'] = error_stats
        
    except Exception as e:
        health_status['errors'].append(f"Health check failed: {str(e)}")
        health_status['overall_status'] = 'error'
    
    return health_status