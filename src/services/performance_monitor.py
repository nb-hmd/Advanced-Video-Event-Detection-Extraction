import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    processing_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    objects_detected: int = 0
    small_objects_detected: int = 0
    background_independence_success_rate: float = 0.0
    adaptive_threshold_adjustments: int = 0
    rpn_proposals_generated: int = 0
    model_inference_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0

@dataclass
class SystemResources:
    """System resource information."""
    total_memory_gb: float
    available_memory_gb: float
    cpu_count: int
    gpu_available: bool
    gpu_memory_gb: Optional[float] = None
    gpu_name: Optional[str] = None

class PerformanceOptimizer:
    """Performance optimization recommendations and automatic adjustments."""
    
    def __init__(self):
        self.optimization_history = deque(maxlen=100)
        self.current_settings = {}
        
    def analyze_performance(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze performance metrics and provide optimization recommendations."""
        if not metrics:
            return {}
        
        recent_metrics = metrics[-10:]  # Last 10 measurements
        
        avg_processing_time = np.mean([m.processing_time for m in recent_metrics])
        avg_memory_usage = np.mean([m.memory_usage_mb for m in recent_metrics])
        avg_cpu_usage = np.mean([m.cpu_usage_percent for m in recent_metrics])
        
        recommendations = []
        
        # Processing time optimization
        if avg_processing_time > 5.0:  # More than 5 seconds
            recommendations.append({
                'type': 'processing_time',
                'severity': 'high',
                'message': f'High processing time ({avg_processing_time:.2f}s). Consider reducing image resolution or enabling RPN.',
                'suggested_actions': [
                    'Enable Region Proposal Network (RPN) for faster processing',
                    'Reduce input image resolution',
                    'Use lighter models (YOLOv8-nano instead of RetinaNet)',
                    'Enable model caching'
                ]
            })
        
        # Memory usage optimization
        if avg_memory_usage > 2048:  # More than 2GB
            recommendations.append({
                'type': 'memory_usage',
                'severity': 'medium',
                'message': f'High memory usage ({avg_memory_usage:.0f}MB). Consider optimizing batch size.',
                'suggested_actions': [
                    'Reduce batch size for processing',
                    'Enable garbage collection between frames',
                    'Use memory-efficient models',
                    'Clear model cache periodically'
                ]
            })
        
        # CPU usage optimization
        if avg_cpu_usage > 80:
            recommendations.append({
                'type': 'cpu_usage',
                'severity': 'medium',
                'message': f'High CPU usage ({avg_cpu_usage:.1f}%). Consider GPU acceleration.',
                'suggested_actions': [
                    'Enable GPU acceleration if available',
                    'Reduce number of concurrent processes',
                    'Use multi-threading for I/O operations'
                ]
            })
        
        # Small object detection performance
        total_objects = sum(m.objects_detected for m in recent_metrics)
        total_small_objects = sum(m.small_objects_detected for m in recent_metrics)
        
        if total_objects > 0:
            small_object_ratio = total_small_objects / total_objects
            if small_object_ratio < 0.3:  # Less than 30% small objects detected
                recommendations.append({
                    'type': 'small_object_detection',
                    'severity': 'low',
                    'message': f'Low small object detection rate ({small_object_ratio:.1%}). Consider adjusting thresholds.',
                    'suggested_actions': [
                        'Lower confidence thresholds for small objects',
                        'Enable adaptive thresholding',
                        'Use specialized small object models'
                    ]
                })
        
        return {
            'timestamp': time.time(),
            'performance_summary': {
                'avg_processing_time': avg_processing_time,
                'avg_memory_usage_mb': avg_memory_usage,
                'avg_cpu_usage_percent': avg_cpu_usage,
                'total_objects_detected': total_objects,
                'small_object_detection_rate': total_small_objects / max(total_objects, 1)
            },
            'recommendations': recommendations,
            'optimization_score': self._calculate_optimization_score(recent_metrics)
        }
    
    def _calculate_optimization_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate overall optimization score (0-100)."""
        if not metrics:
            return 0.0
        
        scores = []
        
        # Processing time score (lower is better)
        avg_time = np.mean([m.processing_time for m in metrics])
        time_score = max(0, 100 - (avg_time * 10))  # 10s = 0 score
        scores.append(time_score)
        
        # Memory efficiency score
        avg_memory = np.mean([m.memory_usage_mb for m in metrics])
        memory_score = max(0, 100 - (avg_memory / 40))  # 4GB = 0 score
        scores.append(memory_score)
        
        # Detection effectiveness score
        avg_objects = np.mean([m.objects_detected for m in metrics])
        detection_score = min(100, avg_objects * 10)  # 10 objects = 100 score
        scores.append(detection_score)
        
        return np.mean(scores)
    
    def suggest_automatic_optimizations(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Suggest automatic optimizations that can be applied."""
        if not metrics:
            return {}
        
        recent_metrics = metrics[-5:]
        avg_processing_time = np.mean([m.processing_time for m in recent_metrics])
        avg_memory_usage = np.mean([m.memory_usage_mb for m in recent_metrics])
        
        optimizations = {}
        
        # Automatic batch size adjustment
        if avg_memory_usage > 1500:  # High memory usage
            optimizations['batch_size'] = max(1, self.current_settings.get('batch_size', 4) - 1)
        elif avg_memory_usage < 500 and avg_processing_time < 2.0:  # Low usage, good performance
            optimizations['batch_size'] = min(8, self.current_settings.get('batch_size', 4) + 1)
        
        # Automatic model selection
        if avg_processing_time > 3.0:
            optimizations['preferred_model'] = 'yolov8_nano'  # Fastest model
        elif avg_processing_time < 1.0:
            optimizations['preferred_model'] = 'retinanet_small'  # More accurate model
        
        # Automatic threshold adjustment
        avg_objects = np.mean([m.objects_detected for m in recent_metrics])
        if avg_objects < 2:  # Too few detections
            optimizations['confidence_threshold'] = max(0.1, self.current_settings.get('confidence_threshold', 0.3) - 0.05)
        elif avg_objects > 20:  # Too many detections
            optimizations['confidence_threshold'] = min(0.8, self.current_settings.get('confidence_threshold', 0.3) + 0.05)
        
        return optimizations

class PerformanceMonitor:
    """Performance monitoring service for small object detection system."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.is_monitoring = False
        self.monitor_thread = None
        self.optimizer = PerformanceOptimizer()
        
        # Monitoring settings
        self.monitoring_interval = self.config.get('monitoring_interval', 1.0)  # seconds
        self.enable_gpu_monitoring = self.config.get('enable_gpu_monitoring', True)
        self.enable_auto_optimization = self.config.get('enable_auto_optimization', False)
        
        # Performance thresholds
        self.thresholds = {
            'max_processing_time': self.config.get('max_processing_time', 10.0),
            'max_memory_usage_mb': self.config.get('max_memory_usage_mb', 4096),
            'max_cpu_usage_percent': self.config.get('max_cpu_usage_percent', 90),
            'min_background_independence_rate': self.config.get('min_background_independence_rate', 0.85)
        }
        
        # Initialize system info
        self.system_info = self._get_system_info()
        
        logger.info(f"Performance monitor initialized with {len(self.metrics_history)} metrics buffer")
        logger.info(f"System: {self.system_info.cpu_count} CPUs, {self.system_info.total_memory_gb:.1f}GB RAM, GPU: {self.system_info.gpu_available}")
    
    def _get_system_info(self) -> SystemResources:
        """Get system resource information."""
        memory = psutil.virtual_memory()
        
        system_info = SystemResources(
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            cpu_count=psutil.cpu_count(),
            gpu_available=False
        )
        
        # Try to get GPU information
        if self.enable_gpu_monitoring:
            try:
                import torch
                if torch.cuda.is_available():
                    system_info.gpu_available = True
                    system_info.gpu_name = torch.cuda.get_device_name(0)
                    system_info.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except ImportError:
                logger.debug("PyTorch not available for GPU monitoring")
            except Exception as e:
                logger.debug(f"GPU monitoring setup failed: {e}")
        
        return system_info
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                current_metrics = self._collect_system_metrics()
                self.metrics_history.append(current_metrics)
                
                # Check for performance issues
                self._check_performance_thresholds(current_metrics)
                
                # Auto-optimization if enabled
                if self.enable_auto_optimization:
                    self._apply_auto_optimizations()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage_mb = (memory.total - memory.available) / (1024**2)
        
        # GPU metrics
        gpu_usage = None
        gpu_memory = None
        
        if self.system_info.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(0) / (1024**2)  # MB
                    # GPU utilization would need nvidia-ml-py for accurate measurement
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
        
        return PerformanceMetrics(
            timestamp=time.time(),
            processing_time=0.0,  # Will be updated by detection calls
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_percent,
            gpu_usage_percent=gpu_usage,
            gpu_memory_mb=gpu_memory
        )
    
    def record_detection_metrics(self, 
                               processing_time: float,
                               objects_detected: int,
                               small_objects_detected: int = 0,
                               background_independence_success_rate: float = 0.0,
                               adaptive_threshold_adjustments: int = 0,
                               rpn_proposals_generated: int = 0,
                               model_inference_time: float = 0.0,
                               preprocessing_time: float = 0.0,
                               postprocessing_time: float = 0.0):
        """Record detection-specific performance metrics."""
        
        # Get current system metrics
        system_metrics = self._collect_system_metrics()
        
        # Update with detection-specific data
        system_metrics.processing_time = processing_time
        system_metrics.objects_detected = objects_detected
        system_metrics.small_objects_detected = small_objects_detected
        system_metrics.background_independence_success_rate = background_independence_success_rate
        system_metrics.adaptive_threshold_adjustments = adaptive_threshold_adjustments
        system_metrics.rpn_proposals_generated = rpn_proposals_generated
        system_metrics.model_inference_time = model_inference_time
        system_metrics.preprocessing_time = preprocessing_time
        system_metrics.postprocessing_time = postprocessing_time
        
        self.metrics_history.append(system_metrics)
        
        # Log performance if significant
        if processing_time > 2.0 or objects_detected > 10:
            logger.info(f"Detection performance: {processing_time:.2f}s, {objects_detected} objects ({small_objects_detected} small)")
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check if performance metrics exceed thresholds."""
        warnings = []
        
        if metrics.processing_time > self.thresholds['max_processing_time']:
            warnings.append(f"High processing time: {metrics.processing_time:.2f}s > {self.thresholds['max_processing_time']}s")
        
        if metrics.memory_usage_mb > self.thresholds['max_memory_usage_mb']:
            warnings.append(f"High memory usage: {metrics.memory_usage_mb:.0f}MB > {self.thresholds['max_memory_usage_mb']}MB")
        
        if metrics.cpu_usage_percent > self.thresholds['max_cpu_usage_percent']:
            warnings.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}% > {self.thresholds['max_cpu_usage_percent']}%")
        
        if (metrics.background_independence_success_rate > 0 and 
            metrics.background_independence_success_rate < self.thresholds['min_background_independence_rate']):
            warnings.append(f"Low background independence rate: {metrics.background_independence_success_rate:.1%} < {self.thresholds['min_background_independence_rate']:.1%}")
        
        for warning in warnings:
            logger.warning(f"Performance threshold exceeded: {warning}")
    
    def _apply_auto_optimizations(self):
        """Apply automatic optimizations based on performance history."""
        if len(self.metrics_history) < 5:
            return
        
        optimizations = self.optimizer.suggest_automatic_optimizations(list(self.metrics_history))
        
        if optimizations:
            logger.info(f"Applying automatic optimizations: {optimizations}")
            # Here you would apply the optimizations to the detection system
            # This would require integration with the main detection services
    
    def get_performance_summary(self, time_window_minutes: int = 10) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        if not self.metrics_history:
            return {'error': 'No performance data available'}
        
        # Filter metrics by time window
        cutoff_time = time.time() - (time_window_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'error': f'No data available for the last {time_window_minutes} minutes'}
        
        # Calculate statistics
        processing_times = [m.processing_time for m in recent_metrics if m.processing_time > 0]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
        objects_detected = [m.objects_detected for m in recent_metrics]
        small_objects_detected = [m.small_objects_detected for m in recent_metrics]
        
        summary = {
            'time_window_minutes': time_window_minutes,
            'total_measurements': len(recent_metrics),
            'system_info': {
                'cpu_count': self.system_info.cpu_count,
                'total_memory_gb': self.system_info.total_memory_gb,
                'gpu_available': self.system_info.gpu_available,
                'gpu_name': self.system_info.gpu_name
            },
            'performance_stats': {
                'processing_time': {
                    'avg': np.mean(processing_times) if processing_times else 0,
                    'min': np.min(processing_times) if processing_times else 0,
                    'max': np.max(processing_times) if processing_times else 0,
                    'std': np.std(processing_times) if processing_times else 0
                },
                'memory_usage_mb': {
                    'avg': np.mean(memory_usage),
                    'min': np.min(memory_usage),
                    'max': np.max(memory_usage),
                    'std': np.std(memory_usage)
                },
                'cpu_usage_percent': {
                    'avg': np.mean(cpu_usage),
                    'min': np.min(cpu_usage),
                    'max': np.max(cpu_usage),
                    'std': np.std(cpu_usage)
                }
            },
            'detection_stats': {
                'total_objects_detected': sum(objects_detected),
                'total_small_objects_detected': sum(small_objects_detected),
                'avg_objects_per_detection': np.mean(objects_detected) if objects_detected else 0,
                'small_object_detection_rate': sum(small_objects_detected) / max(sum(objects_detected), 1)
            }
        }
        
        # Add optimization analysis
        optimization_analysis = self.optimizer.analyze_performance(recent_metrics)
        summary['optimization_analysis'] = optimization_analysis
        
        return summary
    
    def export_metrics(self, filepath: str, time_window_hours: int = 24):
        """Export performance metrics to a file."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        export_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        # Convert to serializable format
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_window_hours': time_window_hours,
            'system_info': {
                'cpu_count': self.system_info.cpu_count,
                'total_memory_gb': self.system_info.total_memory_gb,
                'gpu_available': self.system_info.gpu_available,
                'gpu_name': self.system_info.gpu_name
            },
            'metrics': [
                {
                    'timestamp': m.timestamp,
                    'processing_time': m.processing_time,
                    'memory_usage_mb': m.memory_usage_mb,
                    'cpu_usage_percent': m.cpu_usage_percent,
                    'gpu_usage_percent': m.gpu_usage_percent,
                    'gpu_memory_mb': m.gpu_memory_mb,
                    'objects_detected': m.objects_detected,
                    'small_objects_detected': m.small_objects_detected,
                    'background_independence_success_rate': m.background_independence_success_rate,
                    'adaptive_threshold_adjustments': m.adaptive_threshold_adjustments,
                    'rpn_proposals_generated': m.rpn_proposals_generated,
                    'model_inference_time': m.model_inference_time,
                    'preprocessing_time': m.preprocessing_time,
                    'postprocessing_time': m.postprocessing_time
                }
                for m in export_metrics
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_metrics)} performance metrics to {filepath}")
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time performance statistics."""
        if not self.metrics_history:
            return {'error': 'No performance data available'}
        
        latest_metric = self.metrics_history[-1]
        
        return {
            'timestamp': latest_metric.timestamp,
            'current_memory_usage_mb': latest_metric.memory_usage_mb,
            'current_cpu_usage_percent': latest_metric.cpu_usage_percent,
            'current_gpu_memory_mb': latest_metric.gpu_memory_mb,
            'last_processing_time': latest_metric.processing_time,
            'last_objects_detected': latest_metric.objects_detected,
            'last_small_objects_detected': latest_metric.small_objects_detected,
            'system_status': 'healthy' if self._is_system_healthy(latest_metric) else 'warning',
            'monitoring_active': self.is_monitoring
        }
    
    def _is_system_healthy(self, metrics: PerformanceMetrics) -> bool:
        """Check if system is operating within healthy parameters."""
        return (metrics.memory_usage_mb < self.thresholds['max_memory_usage_mb'] * 0.8 and
                metrics.cpu_usage_percent < self.thresholds['max_cpu_usage_percent'] * 0.8 and
                metrics.processing_time < self.thresholds['max_processing_time'] * 0.8)