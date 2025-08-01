#!/usr/bin/env python3
"""
Test suite for Destark FIS API
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from main import app

client = TestClient(app)

class TestFISAPI:
    """Test cases for FIS API endpoints"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "description": "Test Environmental Assessment FIS",
            "input_variables": {
                "social": {
                    "min": 0, "max": 10, "step": 0.1,
                    "membership_functions": {
                        "low": {"type": "trapmf", "params": [0, 0, 2, 4]},
                        "medium": {"type": "trapmf", "params": [2, 4, 6, 7]},
                        "high": {"type": "trapmf", "params": [6, 7, 10, 10]}
                    }
                },
                "environmental": {
                    "min": 0, "max": 10, "step": 0.1,
                    "membership_functions": {
                        "low": {"type": "trapmf", "params": [0, 0, 2, 5]},
                        "medium": {"type": "trapmf", "params": [2, 5, 6, 8]},
                        "high": {"type": "trapmf", "params": [6, 8, 10, 10]}
                    }
                },
                "strategic": {
                    "min": 0, "max": 10, "step": 0.1,
                    "membership_functions": {
                        "low": {"type": "trapmf", "params": [0, 0, 3, 5]},
                        "medium": {"type": "trapmf", "params": [3, 5, 7, 8]},
                        "high": {"type": "trapmf", "params": [7, 8, 10, 10]}
                    }
                }
            },
            "output_variable": {
                "name": "priority",
                "min": 0, "max": 10, "step": 0.1,
                "membership_functions": {
                    "very_low": {"type": "trimf", "params": [0, 0, 2.5]},
                    "low": {"type": "trimf", "params": [0, 2.5, 5]},
                    "medium": {"type": "trimf", "params": [2.5, 5, 7.5]},
                    "high": {"type": "trimf", "params": [5, 5.5, 10]},
                    "very_high": {"type": "trimf", "params": [7.5, 10, 10]}
                }
            },
            "rules": []
        }
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Destark FIS API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_upload_endpoint_success(self):
        """Test successful file upload"""
        # Create mock GeoTIFF files
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b'fake_geotiff_data')
            social_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b'fake_geotiff_data')
            environmental_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b'fake_geotiff_data')
            strategic_file = f.name
        
        try:
            with open(social_file, 'rb') as sf, \
                 open(environmental_file, 'rb') as ef, \
                 open(strategic_file, 'rb') as stf:
                
                response = client.post(
                    "/upload",
                    files={
                        "social": ("social.tif", sf, "image/tiff"),
                        "environmental": ("environmental.tif", ef, "image/tiff"),
                        "strategic": ("strategic.tif", stf, "image/tiff")
                    },
                    data={"config": json.dumps(self.test_config)}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "uploaded"
            assert "job_id" in data
            
        finally:
            # Cleanup
            os.unlink(social_file)
            os.unlink(environmental_file)
            os.unlink(strategic_file)
    
    def test_upload_endpoint_missing_files(self):
        """Test upload with missing files"""
        response = client.post("/upload")
        assert response.status_code == 422  # Validation error
    
    def test_upload_endpoint_invalid_config(self):
        """Test upload with invalid JSON config"""
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b'fake_geotiff_data')
            test_file = f.name
        
        try:
            with open(test_file, 'rb') as tf:
                response = client.post(
                    "/upload",
                    files={
                        "social": ("social.tif", tf, "image/tiff"),
                        "environmental": ("environmental.tif", tf, "image/tiff"),
                        "strategic": ("strategic.tif", tf, "image/tiff")
                    },
                    data={"config": "invalid json"}
                )
            
            assert response.status_code == 400
            assert "Invalid JSON configuration" in response.json()["detail"]
            
        finally:
            os.unlink(test_file)
    
    @patch('main.UnifiedRasterFuzzyInferenceSystem')
    def test_process_endpoint_success(self, mock_fis):
        """Test successful processing"""
        # Mock the FIS system
        mock_fis_instance = MagicMock()
        mock_fis.return_value = mock_fis_instance
        
        # Create mock GeoTIFF files
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b'fake_geotiff_data')
            social_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b'fake_geotiff_data')
            environmental_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b'fake_geotiff_data')
            strategic_file = f.name
        
        try:
            with open(social_file, 'rb') as sf, \
                 open(environmental_file, 'rb') as ef, \
                 open(strategic_file, 'rb') as stf:
                
                response = client.post(
                    "/process",
                    files={
                        "social": ("social.tif", sf, "image/tiff"),
                        "environmental": ("environmental.tif", ef, "image/tiff"),
                        "strategic": ("strategic.tif", stf, "image/tiff")
                    },
                    data={"config": json.dumps(self.test_config)}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert "job_id" in data
            assert "statistics" in data
            assert "processing_time" in data
            assert "download_url" in data
            
        finally:
            # Cleanup
            os.unlink(social_file)
            os.unlink(environmental_file)
            os.unlink(strategic_file)
    
    @patch('main.UnifiedRasterFuzzyInferenceSystem')
    def test_process_endpoint_fis_error(self, mock_fis):
        """Test processing with FIS error"""
        # Mock the FIS system to raise an exception
        mock_fis.side_effect = Exception("FIS processing error")
        
        # Create mock GeoTIFF files
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b'fake_geotiff_data')
            social_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b'fake_geotiff_data')
            environmental_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b'fake_geotiff_data')
            strategic_file = f.name
        
        try:
            with open(social_file, 'rb') as sf, \
                 open(environmental_file, 'rb') as ef, \
                 open(strategic_file, 'rb') as stf:
                
                response = client.post(
                    "/process",
                    files={
                        "social": ("social.tif", sf, "image/tiff"),
                        "environmental": ("environmental.tif", ef, "image/tiff"),
                        "strategic": ("strategic.tif", stf, "image/tiff")
                    },
                    data={"config": json.dumps(self.test_config)}
                )
            
            assert response.status_code == 500
            assert "Processing failed" in response.json()["detail"]
            
        finally:
            # Cleanup
            os.unlink(social_file)
            os.unlink(environmental_file)
            os.unlink(strategic_file)
    
    def test_status_endpoint_not_found(self):
        """Test status endpoint with non-existent job"""
        response = client.get("/status/non-existent-job")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    def test_download_endpoint_not_found(self):
        """Test download endpoint with non-existent job"""
        response = client.get("/download/non-existent-job")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    def test_download_endpoint_not_completed(self):
        """Test download endpoint with incomplete job"""
        # First create a job
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b'fake_geotiff_data')
            test_file = f.name
        
        try:
            with open(test_file, 'rb') as tf:
                response = client.post(
                    "/upload",
                    files={
                        "social": ("social.tif", tf, "image/tiff"),
                        "environmental": ("environmental.tif", tf, "image/tiff"),
                        "strategic": ("strategic.tif", tf, "image/tiff")
                    }
                )
            
            job_id = response.json()["job_id"]
            
            # Try to download before processing
            response = client.get(f"/download/{job_id}")
            assert response.status_code == 400
            assert "Job not completed" in response.json()["detail"]
            
        finally:
            os.unlink(test_file)

if __name__ == "__main__":
    pytest.main([__file__]) 