import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test configuration and fixtures

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_image_rgb():
    """Create a sample RGB image for testing."""
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    # Add some patterns
    cv2.rectangle(image, (50, 50), (100, 100), (255, 0, 0), -1)  # Red square
    cv2.circle(image, (150, 150), 30, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(image, (200, 200), (220, 220), (0, 0, 255), -1)  # Small blue square
    return image

@pytest.fixture
def sample_image_grayscale():
    """Create a sample grayscale image for testing."""
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (100, 100), 128, -1)
    cv2.circle(image, (150, 150), 30, 200, -1)
    cv2.rectangle(image, (200, 200), (220, 220), 255, -1)
    return image

@pytest.fixture
def small_objects_image():
    """Create an image with various small objects for testing."""
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Add background texture
    noise = np.random.randint(0, 30, (512, 512, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    # Add small objects of different sizes
    objects = [
        # (x, y, width, height, color)
        (50, 50, 16, 16, (255, 0, 0)),    # Tiny red object
        (100, 100, 24, 24, (0, 255, 0)),  # Small green object
        (200, 200, 32, 32, (0, 0, 255)),  # Small blue object
        (300, 300, 48, 48, (255, 255, 0)), # Medium-small yellow object
        (400, 400, 64, 64, (255, 0, 255)), # Medium-small magenta object
        (450, 50, 12, 12, (0, 255, 255)),  # Very tiny cyan object
    ]
    
    for x, y, w, h, color in objects:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    
    return image

@pytest.fixture
def background_variations():
    """Create multiple images with same objects but different backgrounds."""
    images = []
    
    # Object definition (same across all images)
    object_bbox = (100, 100, 150, 150)
    object_color = (255, 0, 0)  # Red object
    
    # Different background colors
    backgrounds = [
        (50, 50, 50),    # Dark gray
        (200, 200, 200), # Light gray
        (0, 100, 0),     # Dark green
        (100, 0, 100),   # Purple
        (0, 0, 100),     # Dark blue
    ]
    
    for bg_color in backgrounds:
        image = np.full((256, 256, 3), bg_color, dtype=np.uint8)
        cv2.rectangle(image, object_bbox[:2], object_bbox[2:], object_color, -1)
        images.append(image)
    
    return images

@pytest.fixture
def mock_torch():
    """Mock PyTorch for testing without GPU dependencies."""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.device', return_value='cpu'), \
         patch('torch.load') as mock_load, \
         patch('torch.save') as mock_save:
        
        # Mock model loading
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_load.return_value = mock_model
        
        yield {
            'load': mock_load,
            'save': mock_save,
            'model': mock_model
        }

@pytest.fixture
def mock_cv2_operations():
    """Mock OpenCV operations that might be problematic in testing."""
    with patch('cv2.dnn.readNet') as mock_readnet, \
         patch('cv2.dnn.blobFromImage') as mock_blob:
        
        # Mock DNN operations
        mock_net = MagicMock()
        mock_readnet.return_value = mock_net
        mock_blob.return_value = np.random.random((1, 3, 416, 416)).astype(np.float32)
        
        yield {
            'readNet': mock_readnet,
            'blobFromImage': mock_blob,
            'net': mock_net
        }

@pytest.fixture
def sample_detections():
    """Create sample detection results for testing."""
    return [
        {
            'bbox': [10, 10, 50, 50],
            'confidence': 0.8,
            'class_name': 'person',
            'timestamp': 1.5
        },
        {
            'bbox': [100, 100, 130, 130],
            'confidence': 0.6,
            'class_name': 'car',
            'timestamp': 2.3
        },
        {
            'bbox': [200, 200, 216, 216],
            'confidence': 0.4,
            'class_name': 'bird',
            'timestamp': 3.1
        }
    ]

@pytest.fixture
def sample_video_frames():
    """Create a sequence of video frames for testing."""
    frames = []
    
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add moving object
        x = 50 + i * 20
        y = 100
        cv2.rectangle(frame, (x, y), (x + 30, y + 30), (255, 0, 0), -1)
        
        # Add static background elements
        cv2.rectangle(frame, (500, 400), (600, 450), (0, 255, 0), -1)
        
        frames.append(frame)
    
    return frames

@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        'small_object_detection': {
            'enabled': True,
            'min_size': 8,
            'max_size': 128,
            'confidence_threshold': 0.2
        },
        'background_independence': {
            'enabled': True,
            'sam_model_path': 'models/sam_vit_b.pth',
            'target_success_rate': 0.85
        },
        'adaptive_thresholds': {
            'enabled': True,
            'size_categories': {
                'tiny': {'min': 8, 'max': 32, 'threshold': 0.15},
                'small': {'min': 32, 'max': 64, 'threshold': 0.25},
                'medium_small': {'min': 64, 'max': 128, 'threshold': 0.35}
            }
        },
        'region_proposals': {
            'enabled': True,
            'max_proposals': 100,
            'nms_threshold': 0.3,
            'min_area': 64
        }
    }

@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for testing."""
    return {
        'max_processing_time': 10.0,  # seconds
        'min_accuracy': 0.7,
        'min_background_independence_rate': 0.85,
        'max_memory_usage': 2048,  # MB
        'min_fps': 5.0  # frames per second
    }

# Test utilities

def create_test_video_file(frames, output_path, fps=30):
    """Create a test video file from frames."""
    if not frames:
        return False
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return True

def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def assert_detection_quality(detections, expected_count=None, min_confidence=0.3):
    """Assert that detections meet quality requirements."""
    assert isinstance(detections, list), "Detections should be a list"
    
    if expected_count is not None:
        assert len(detections) >= expected_count, f"Expected at least {expected_count} detections, got {len(detections)}"
    
    for detection in detections:
        assert 'bbox' in detection, "Detection should have bbox"
        assert 'confidence' in detection, "Detection should have confidence"
        assert 'class_name' in detection, "Detection should have class_name"
        
        assert detection['confidence'] >= min_confidence, f"Detection confidence {detection['confidence']} below minimum {min_confidence}"
        
        bbox = detection['bbox']
        assert len(bbox) == 4, "Bbox should have 4 coordinates"
        assert bbox[2] > bbox[0], "Bbox width should be positive"
        assert bbox[3] > bbox[1], "Bbox height should be positive"

def assert_small_object_detection(detections, max_size=128):
    """Assert that detections contain small objects."""
    small_objects = []
    
    for detection in detections:
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width <= max_size and height <= max_size:
            small_objects.append(detection)
    
    assert len(small_objects) > 0, "Should detect at least one small object"
    return small_objects

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m "not slow"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "api: marks tests for API endpoints"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid or "end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark API tests
        if "api" in item.nodeid or "endpoint" in item.nodeid:
            item.add_marker(pytest.mark.api)
        
        # Mark GPU tests
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)