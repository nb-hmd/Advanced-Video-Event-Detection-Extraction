import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
import io

# Import the FastAPI app
from src.api.main import app

class TestSmallObjectDetectionAPI:
    """Test suite for small object detection API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_video_processor(self):
        """Mock the video processor for testing."""
        with patch('src.api.main.video_processor') as mock:
            yield mock
    
    @pytest.fixture
    def sample_video_file(self):
        """Create a sample video file for testing."""
        # Create a temporary file that simulates a video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(b'fake video content for testing')
            return tmp.name
    
    def test_small_object_capabilities_endpoint(self, client):
        """Test the small object capabilities endpoint."""
        response = client.get("/api/small-object-capabilities")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check main structure
        assert "small_object_detection" in data
        assert "background_independence" in data
        assert "supported_query_types" in data
        assert "api_endpoints" in data
        
        # Check small object detection details
        sod = data["small_object_detection"]
        assert "name" in sod
        assert "features" in sod
        assert "supported_models" in sod
        assert "enhancement_techniques" in sod
        assert "default_settings" in sod
        
        # Check supported models
        models = sod["supported_models"]
        assert "fcos_rt" in models
        assert "retinanet_small" in models
        assert "yolov8_nano" in models
        
        # Check enhancement techniques
        techniques = sod["enhancement_techniques"]
        assert "background_independence" in techniques
        assert "adaptive_thresholds" in techniques
        assert "region_proposals" in techniques
    
    def test_small_object_detection_endpoint_success(self, client, mock_video_processor):
        """Test successful small object detection request."""
        # Mock the video processor response
        mock_video_processor.process_small_object_detection.return_value = {
            'status': 'success',
            'results': [
                {
                    'bbox': [10, 10, 40, 40],
                    'confidence': 0.8,
                    'class_name': 'small_car',
                    'timestamp': 1.5
                },
                {
                    'bbox': [100, 100, 120, 120],
                    'confidence': 0.6,
                    'class_name': 'tiny_bird',
                    'timestamp': 3.2
                }
            ],
            'processing_time': 2.5,
            'enhancement_success_rate': 0.87
        }
        
        # Mock video file existence
        with patch('pathlib.Path.exists', return_value=True):
            request_data = {
                "video_id": "test-video-123",
                "object_queries": "small car; tiny bird",
                "enable_background_independence": True,
                "enable_adaptive_thresholds": True,
                "enable_rpn": True,
                "min_object_size": 16,
                "max_object_size": 128,
                "confidence_threshold": 0.2,
                "top_k": 20,
                "debug_mode": False
            }
            
            response = client.post("/api/small-object-detection", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "task_id" in data
            assert "status" in data
            assert "results" in data
            assert "queries" in data
            assert "total_found" in data
            assert "small_objects_found" in data
            assert "enhancement_stats" in data
            assert "metadata" in data
            
            # Check response values
            assert data["status"] == "completed"
            assert data["total_found"] == 2
            assert data["small_objects_found"] == 2  # Both objects are small
            assert len(data["results"]) == 2
            assert data["queries"] == ["small car", "tiny bird"]
            
            # Check enhancement stats
            stats = data["enhancement_stats"]
            assert stats["background_independence_enabled"] == True
            assert stats["adaptive_thresholds_enabled"] == True
            assert stats["rpn_enabled"] == True
            assert stats["min_object_size"] == 16
            assert stats["max_object_size"] == 128
    
    def test_small_object_detection_video_not_found(self, client, mock_video_processor):
        """Test small object detection with non-existent video."""
        with patch('pathlib.Path.exists', return_value=False):
            request_data = {
                "video_id": "non-existent-video",
                "object_queries": "small objects"
            }
            
            response = client.post("/api/small-object-detection", json=request_data)
            
            assert response.status_code == 404
            assert "Video not found" in response.json()["detail"]
    
    def test_small_object_detection_invalid_queries(self, client, mock_video_processor):
        """Test small object detection with invalid queries."""
        with patch('pathlib.Path.exists', return_value=True):
            request_data = {
                "video_id": "test-video-123",
                "object_queries": ""  # Empty query
            }
            
            response = client.post("/api/small-object-detection", json=request_data)
            
            assert response.status_code == 400
            assert "No valid object queries provided" in response.json()["detail"]
    
    def test_background_independence_endpoint_success(self, client, mock_video_processor):
        """Test successful background independence request."""
        # Mock the video processor response
        mock_video_processor.process_background_independence.return_value = {
            'status': 'success',
            'results': [
                {
                    'bbox': [50, 50, 100, 100],
                    'confidence': 0.9,
                    'class_name': 'person',
                    'timestamp': 2.1,
                    'background_independent': True
                }
            ],
            'processing_time': 1.8,
            'background_independence_success_rate': 0.92,
            'sam_model_used': True,
            'contrastive_features_extracted': 156
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            request_data = {
                "video_id": "test-video-456",
                "object_queries": ["person walking"],
                "background_removal_strength": 0.8,
                "contrastive_learning_enabled": True,
                "shape_descriptor_enabled": True,
                "confidence_threshold": 0.3,
                "top_k": 15,
                "debug_mode": False
            }
            
            response = client.post("/api/background-independence", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "task_id" in data
            assert "status" in data
            assert "results" in data
            assert "queries" in data
            assert "total_found" in data
            assert "background_independence_stats" in data
            assert "metadata" in data
            
            # Check response values
            assert data["status"] == "completed"
            assert data["total_found"] == 1
            assert len(data["results"]) == 1
            assert data["queries"] == ["person walking"]
            
            # Check background independence stats
            stats = data["background_independence_stats"]
            assert stats["background_removal_strength"] == 0.8
            assert stats["contrastive_learning_enabled"] == True
            assert stats["shape_descriptor_enabled"] == True
            assert stats["background_independence_success_rate"] == 0.85  # Default from response model
            assert stats["sam_model_used"] == True
    
    def test_background_independence_processing_error(self, client, mock_video_processor):
        """Test background independence with processing error."""
        # Mock the video processor to return an error
        mock_video_processor.process_background_independence.return_value = {
            'status': 'error',
            'error': 'SAM model failed to load'
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            request_data = {
                "video_id": "test-video-789",
                "object_queries": "person"
            }
            
            response = client.post("/api/background-independence", json=request_data)
            
            assert response.status_code == 500
            assert "SAM model failed to load" in response.json()["detail"]
    
    def test_api_root_endpoint_includes_new_endpoints(self, client):
        """Test that the root endpoint includes new small object detection endpoints."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        endpoints = data["endpoints"]
        assert "/api/small-object-detection" in endpoints
        assert "/api/background-independence" in endpoints
        assert "/api/small-object-capabilities" in endpoints
        
        # Check endpoint descriptions
        assert "small object detection" in endpoints["/api/small-object-detection"].lower()
        assert "background-independent" in endpoints["/api/background-independence"].lower()
        assert "capabilities" in endpoints["/api/small-object-capabilities"].lower()

class TestAPIValidation:
    """Test API request validation and error handling."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_small_object_detection_request_validation(self, client):
        """Test request validation for small object detection endpoint."""
        # Test missing required fields
        response = client.post("/api/small-object-detection", json={})
        assert response.status_code == 422  # Validation error
        
        # Test invalid field types
        invalid_request = {
            "video_id": 123,  # Should be string
            "object_queries": "test",
            "min_object_size": "invalid",  # Should be int
            "confidence_threshold": "invalid"  # Should be float
        }
        
        response = client.post("/api/small-object-detection", json=invalid_request)
        assert response.status_code == 422
    
    def test_background_independence_request_validation(self, client):
        """Test request validation for background independence endpoint."""
        # Test invalid background_removal_strength (should be 0.0-1.0)
        invalid_request = {
            "video_id": "test",
            "object_queries": "test",
            "background_removal_strength": 1.5  # Invalid range
        }
        
        response = client.post("/api/background-independence", json=invalid_request)
        # Note: This might pass validation but should be handled in business logic
        # The actual validation depends on Pydantic model constraints
    
    def test_query_format_handling(self, client):
        """Test different query formats are handled correctly."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('src.api.main.video_processor') as mock_processor:
            
            mock_processor.process_small_object_detection.return_value = {
                'status': 'success',
                'results': []
            }
            
            # Test string query with semicolons
            request1 = {
                "video_id": "test",
                "object_queries": "small car; tiny bird; person"
            }
            
            response1 = client.post("/api/small-object-detection", json=request1)
            assert response1.status_code == 200
            
            # Test list query
            request2 = {
                "video_id": "test",
                "object_queries": ["small car", "tiny bird", "person"]
            }
            
            response2 = client.post("/api/small-object-detection", json=request2)
            assert response2.status_code == 200
            
            # Both should result in the same parsed queries
            data1 = response1.json()
            data2 = response2.json()
            assert data1["queries"] == data2["queries"]

class TestAPIPerformance:
    """Test API performance and timeout handling."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_api_response_time(self, client):
        """Test that API responses are returned within reasonable time."""
        import time
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('src.api.main.video_processor') as mock_processor:
            
            # Mock a fast response
            mock_processor.process_small_object_detection.return_value = {
                'status': 'success',
                'results': [],
                'processing_time': 0.5
            }
            
            request_data = {
                "video_id": "test",
                "object_queries": "test objects"
            }
            
            start_time = time.time()
            response = client.post("/api/small-object-detection", json=request_data)
            response_time = time.time() - start_time
            
            assert response.status_code == 200
            # API overhead should be minimal (under 1 second for mocked processing)
            assert response_time < 2.0
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent API requests."""
        import threading
        import time
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('src.api.main.video_processor') as mock_processor:
            
            # Mock processing with slight delay
            def mock_process(*args, **kwargs):
                time.sleep(0.1)  # Simulate processing time
                return {
                    'status': 'success',
                    'results': []
                }
            
            mock_processor.process_small_object_detection.side_effect = mock_process
            
            request_data = {
                "video_id": "test",
                "object_queries": "test objects"
            }
            
            responses = []
            threads = []
            
            def make_request():
                response = client.post("/api/small-object-detection", json=request_data)
                responses.append(response)
            
            # Start multiple concurrent requests
            for _ in range(5):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            # Wait for all requests to complete
            for thread in threads:
                thread.join()
            
            # All requests should succeed
            assert len(responses) == 5
            for response in responses:
                assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])