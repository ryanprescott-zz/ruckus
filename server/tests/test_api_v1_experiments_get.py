"""Tests for GET /api/v1/experiments/{experiment_id} endpoint."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ruckus_server.api.v1.routers.experiments import router
from ruckus_server.core.storage.base import ExperimentNotFoundException
from ruckus_common.models import ExperimentSpec


@pytest.fixture
def app():
    """Create test FastAPI app with experiments router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/experiments")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_experiment_manager():
    """Create mock experiment manager."""
    return AsyncMock()


@pytest.fixture
def sample_experiment_spec():
    """Create sample experiment spec for testing."""
    from ruckus_common.models import TaskType
    return ExperimentSpec(
        experiment_id="test-experiment-123",
        name="Test Experiment",
        description="A test experiment for validation",
        models=["test-model"],
        task_type=TaskType.SUMMARIZATION,
        tags=["test", "version-1.0"],
        base_parameters={
            "learning_rate": 0.01,
            "batch_size": 32
        }
    )




class TestGetExperimentSuccess:
    """Test successful get experiment scenarios."""
    
    def test_get_experiment_success(self, client, app, mock_experiment_manager, sample_experiment_spec):
        """Test successful experiment retrieval."""
        # Setup
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.get_experiment.return_value = sample_experiment_spec
        
        # Execute
        response = client.get("/api/v1/experiments/test-experiment-123")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        
        assert "experiment" in response_data
        experiment = response_data["experiment"]
        
        # Verify all ExperimentSpec fields
        assert experiment["experiment_id"] == "test-experiment-123"
        assert experiment["name"] == "Test Experiment"
        assert experiment["description"] == "A test experiment for validation"
        assert experiment["tags"] == {"type": "test", "version": "1.0"}
        assert experiment["parameters"] == {"learning_rate": 0.01, "batch_size": 32}
        
        # Verify experiment manager was called correctly
        mock_experiment_manager.get_experiment.assert_called_once_with("test-experiment-123")
    
    def test_get_experiment_with_special_characters_in_id(self, client, app, mock_experiment_manager, sample_experiment_spec):
        """Test getting experiment with special characters in ID."""
        # Setup
        experiment_id = "test-exp_123.v2"
        sample_experiment_spec.experiment_id = experiment_id
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.get_experiment.return_value = sample_experiment_spec
        
        # Execute
        response = client.get(f"/api/v1/experiments/{experiment_id}")
        
        # Verify
        assert response.status_code == 200
        mock_experiment_manager.get_experiment.assert_called_once_with(experiment_id)
    
    def test_get_experiment_with_complex_parameters(self, client, app, mock_experiment_manager, sample_experiment_spec):
        """Test getting experiment with complex parameter structures."""
        # Setup complex parameters
        sample_experiment_spec.experiment_id = "complex-experiment"
        sample_experiment_spec.name = "Complex Experiment"
        sample_experiment_spec.description = "An experiment with complex parameters"
        sample_experiment_spec.parameters = {
            "model": {
                "type": "neural_network",
                "layers": [
                    {"type": "dense", "units": 128},
                    {"type": "dropout", "rate": 0.2}
                ]
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "validation_split": 0.2
            },
            "metrics": ["accuracy", "loss", "f1_score"]
        }
        
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.get_experiment.return_value = sample_experiment_spec
        
        # Execute
        response = client.get("/api/v1/experiments/complex-experiment")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        experiment = response_data["experiment"]
        
        assert experiment["parameters"]["model"]["type"] == "neural_network"
        assert len(experiment["parameters"]["model"]["layers"]) == 2
        assert experiment["parameters"]["metrics"] == ["accuracy", "loss", "f1_score"]


class TestGetExperimentNotFound:
    """Test experiment not found scenarios."""
    
    def test_get_nonexistent_experiment(self, client, app, mock_experiment_manager):
        """Test getting a non-existent experiment."""
        # Setup
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.get_experiment.side_effect = ExperimentNotFoundException("nonexistent-exp")
        
        # Execute
        response = client.get("/api/v1/experiments/nonexistent-exp")
        
        # Verify
        assert response.status_code == 404
        response_data = response.json()
        assert "detail" in response_data
        assert "nonexistent-exp not found" in response_data["detail"]
        
        mock_experiment_manager.get_experiment.assert_called_once_with("nonexistent-exp")
    
    def test_get_experiment_empty_id(self, client, app, mock_experiment_manager):
        """Test getting experiment with empty ID."""
        # Setup
        app.state.experiment_manager = mock_experiment_manager
        
        # Execute - FastAPI will handle empty path parameter as 404
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 404


class TestGetExperimentServerErrors:
    """Test server error scenarios."""
    
    def test_get_experiment_manager_not_initialized(self, client, app):
        """Test getting experiment when manager is not initialized."""
        # Setup - no experiment manager on app state
        
        # Execute
        response = client.get("/api/v1/experiments/test-experiment")
        
        # Verify
        assert response.status_code == 503
        response_data = response.json()
        assert "detail" in response_data
        assert "not initialized" in response_data["detail"]
    
    def test_get_experiment_manager_none(self, client, app):
        """Test getting experiment when manager is None."""
        # Setup
        app.state.experiment_manager = None
        
        # Execute
        response = client.get("/api/v1/experiments/test-experiment")
        
        # Verify
        assert response.status_code == 503
        response_data = response.json()
        assert "not initialized" in response_data["detail"]
    
    def test_get_experiment_unexpected_error(self, client, app, mock_experiment_manager):
        """Test getting experiment with unexpected error."""
        # Setup
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.get_experiment.side_effect = Exception("Database connection failed")
        
        # Execute
        response = client.get("/api/v1/experiments/test-experiment")
        
        # Verify
        assert response.status_code == 500
        response_data = response.json()
        assert "detail" in response_data
        assert "Internal server error" in response_data["detail"]
    
    def test_get_experiment_value_error(self, client, app, mock_experiment_manager):
        """Test getting experiment with validation error."""
        # Setup
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.get_experiment.side_effect = ValueError("Invalid experiment data")
        
        # Execute
        response = client.get("/api/v1/experiments/test-experiment")
        
        # Verify
        assert response.status_code == 400
        response_data = response.json()
        assert "detail" in response_data
        assert "Invalid experiment data" in response_data["detail"]


class TestGetExperimentEdgeCases:
    """Test edge cases and data transformation."""
    
    def test_get_experiment_with_none_description(self, client, app, mock_experiment_manager, sample_experiment_spec):
        """Test getting experiment with None description."""
        # Setup
        sample_experiment_spec.experiment_id = "test-experiment"
        sample_experiment_spec.name = "Test Experiment"
        sample_experiment_spec.description = None
        
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.get_experiment.return_value = sample_experiment_spec
        
        # Execute
        response = client.get("/api/v1/experiments/test-experiment")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        experiment = response_data["experiment"]
        assert experiment["description"] is None
    
    def test_get_experiment_with_empty_tags(self, client, app, mock_experiment_manager, sample_experiment_spec):
        """Test getting experiment with empty tags."""
        # Setup
        sample_experiment_spec.experiment_id = "test-experiment"
        sample_experiment_spec.name = "Test Experiment"
        sample_experiment_spec.description = "Test description"
        sample_experiment_spec.tags = {}
        
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.get_experiment.return_value = sample_experiment_spec
        
        # Execute
        response = client.get("/api/v1/experiments/test-experiment")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        experiment = response_data["experiment"]
        assert experiment["tags"] == {}
    
    def test_get_experiment_spec_direct_return(self, client, app, mock_experiment_manager, sample_experiment_spec):
        """Test that ExperimentSpec is returned directly without reconstruction."""
        # Setup
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.get_experiment.return_value = sample_experiment_spec
        
        # Execute
        response = client.get(f"/api/v1/experiments/{sample_experiment_spec.experiment_id}")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify the experiment spec is returned correctly
        returned_spec = response_data["experiment"]
        original_spec = sample_experiment_spec.model_dump(mode='json')
        
        assert returned_spec == original_spec