"""Tests for GET /api/v1/experiments/ endpoint (list experiments)."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ruckus_server.api.v1.routers.experiments import router
from ruckus_common.models import ExperimentSpec, TaskType


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
def sample_experiment_specs():
    """Create sample experiment specs for testing."""
    return [
        ExperimentSpec(
            experiment_id="test-experiment-1",
            name="Test Experiment 1",
            description="First test experiment",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION,
            tags=["type-test", "version-1.0"],
            base_parameters={"learning_rate": 0.01, "batch_size": 32}
        ),
        ExperimentSpec(
            experiment_id="test-experiment-2", 
            name="Test Experiment 2",
            description="Second test experiment",
            models=["test-model-2"],
            task_type=TaskType.CLASSIFICATION,
            tags=["type-production", "version-2.0"],
            base_parameters={"learning_rate": 0.001, "batch_size": 64}
        )
    ]


class TestListExperimentsSuccess:
    """Test successful list experiments scenarios."""
    
    def test_list_experiments_success(self, client, app, mock_experiment_manager, sample_experiment_specs):
        """Test successful experiments listing."""
        # Setup
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.return_value = sample_experiment_specs
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        
        assert "experiments" in response_data
        experiments = response_data["experiments"]
        assert len(experiments) == 2
        
        # Verify first experiment
        assert experiments[0]["experiment_id"] == "test-experiment-1"
        assert experiments[0]["name"] == "Test Experiment 1"
        assert experiments[0]["description"] == "First test experiment"
        assert experiments[0]["tags"] == ["type-test", "version-1.0"]
        assert experiments[0]["base_parameters"] == {"learning_rate": 0.01, "batch_size": 32}
        
        # Verify second experiment
        assert experiments[1]["experiment_id"] == "test-experiment-2"
        assert experiments[1]["name"] == "Test Experiment 2"
        assert experiments[1]["description"] == "Second test experiment"
        assert experiments[1]["tags"] == ["type-production", "version-2.0"]
        assert experiments[1]["base_parameters"] == {"learning_rate": 0.001, "batch_size": 64}
        
        # Verify manager was called correctly
        mock_experiment_manager.list_experiments.assert_called_once()
    
    def test_list_experiments_empty(self, client, app, mock_experiment_manager):
        """Test listing when no experiments exist."""
        # Setup
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.return_value = []
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        assert "experiments" in response_data
        assert response_data["experiments"] == []
        
        mock_experiment_manager.list_experiments.assert_called_once()
    
    def test_list_experiments_complex_parameters(self, client, app, mock_experiment_manager):
        """Test listing experiments with complex parameter structures."""
        # Setup
        complex_spec = ExperimentSpec(
            experiment_id="complex-experiment",
            name="Complex Experiment",
            description="An experiment with complex parameters",
            models=["transformer-model"],
            task_type=TaskType.GENERATION,
            tags=["complexity-high", "framework-pytorch"],
            base_parameters={
                "model": {
                    "architecture": "transformer",
                    "layers": [
                        {"type": "attention", "heads": 8, "dim": 512},
                        {"type": "feedforward", "dim": 2048}
                    ],
                    "dropout": 0.1
                },
                "training": {
                    "optimizer": {"name": "adam", "lr": 1e-4, "betas": [0.9, 0.999]},
                    "scheduler": {"type": "cosine", "warmup_steps": 1000},
                    "batch_size": 32
                },
                "data": {
                    "source": "/path/to/data",
                    "preprocessing": ["tokenize", "normalize"],
                    "augmentation": {"enabled": True, "methods": ["random_crop", "flip"]}
                }
            }
        )
        
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.return_value = [complex_spec]
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        experiments = response_data["experiments"]
        assert len(experiments) == 1
        
        experiment = experiments[0]
        assert experiment["experiment_id"] == "complex-experiment"
        assert experiment["base_parameters"]["model"]["architecture"] == "transformer"
        assert experiment["base_parameters"]["training"]["optimizer"]["name"] == "adam"
        assert len(experiment["base_parameters"]["model"]["layers"]) == 2
        assert experiment["base_parameters"]["data"]["augmentation"]["methods"] == ["random_crop", "flip"]
    
    def test_list_experiments_large_dataset(self, client, app, mock_experiment_manager):
        """Test listing many experiments."""
        # Setup
        large_experiment_list = []
        for i in range(100):
            spec = ExperimentSpec(
                experiment_id=f"experiment-{i}",
                name=f"Experiment {i}",
                description=f"Test experiment number {i}",
                models=[f"model-{i}"],
                task_type=TaskType.SUMMARIZATION,
                tags=[f"index-{i}", "batch-large_test"],
                base_parameters={"value": i * 10, "active": i % 2 == 0}
            )
            large_experiment_list.append(spec)
        
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.return_value = large_experiment_list
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        experiments = response_data["experiments"]
        assert len(experiments) == 100
        
        # Spot check first and last
        assert experiments[0]["experiment_id"] == "experiment-0"
        assert experiments[0]["base_parameters"]["value"] == 0
        assert experiments[99]["experiment_id"] == "experiment-99"
        assert experiments[99]["base_parameters"]["value"] == 990


class TestListExperimentsServerErrors:
    """Test server error scenarios."""
    
    def test_list_experiments_manager_not_initialized(self, client, app):
        """Test listing when manager is not initialized."""
        # Setup - no experiment manager on app state
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 503
        response_data = response.json()
        assert "detail" in response_data
        assert "not initialized" in response_data["detail"]
    
    def test_list_experiments_manager_none(self, client, app):
        """Test listing when manager is None."""
        # Setup
        app.state.experiment_manager = None
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 503
        response_data = response.json()
        assert "not initialized" in response_data["detail"]
    
    def test_list_experiments_unexpected_error(self, client, app, mock_experiment_manager):
        """Test listing with unexpected error."""
        # Setup
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.side_effect = Exception("Database connection failed")
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 500
        response_data = response.json()
        assert "detail" in response_data
        assert "Internal server error" in response_data["detail"]
    
    def test_list_experiments_value_error(self, client, app, mock_experiment_manager):
        """Test listing with validation error."""
        # Setup
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.side_effect = ValueError("Invalid query parameters")
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 400
        response_data = response.json()
        assert "detail" in response_data
        assert "Invalid query parameters" in response_data["detail"]


class TestListExperimentsEdgeCases:
    """Test edge cases and data handling."""
    
    def test_list_experiments_with_none_descriptions(self, client, app, mock_experiment_manager):
        """Test listing experiments with None descriptions."""
        # Setup
        spec_with_none = ExperimentSpec(
            experiment_id="none-desc-test",
            name="None Description Test",
            description=None,
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION,
            tags=[],
            base_parameters={}
        )
        
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.return_value = [spec_with_none]
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        experiments = response_data["experiments"]
        assert len(experiments) == 1
        assert experiments[0]["description"] is None
    
    def test_list_experiments_with_empty_containers(self, client, app, mock_experiment_manager):
        """Test listing experiments with empty tags and parameters."""
        # Setup
        empty_spec = ExperimentSpec(
            experiment_id="empty-test",
            name="Empty Containers Test",
            description="Testing empty tags and parameters",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION,
            tags=[],
            base_parameters={}
        )
        
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.return_value = [empty_spec]
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        experiments = response_data["experiments"]
        assert len(experiments) == 1
        assert experiments[0]["tags"] == []
        assert experiments[0]["base_parameters"] == {}
    
    def test_list_experiments_response_format(self, client, app, mock_experiment_manager):
        """Test that response has correct structure."""
        # Setup
        single_spec = ExperimentSpec(
            experiment_id="format-test",
            name="Format Test",
            description="Testing response format",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION,
            tags=["env-test"],
            base_parameters={"test": True}
        )
        
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.return_value = [single_spec]
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify response structure
        assert response.status_code == 200
        response_data = response.json()
        
        # Check top-level structure
        assert isinstance(response_data, dict)
        assert "experiments" in response_data
        assert isinstance(response_data["experiments"], list)
        
        # Check experiment structure
        experiment = response_data["experiments"][0]
        required_fields = ["experiment_id", "name", "description", "tags", "base_parameters"]
        for field in required_fields:
            assert field in experiment
    
    def test_list_experiments_json_serialization(self, client, app, mock_experiment_manager):
        """Test that complex data types are properly JSON serialized."""
        # Setup
        spec_with_types = ExperimentSpec(
            experiment_id="json-test",
            name="JSON Serialization Test",
            description="Testing JSON edge cases",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION,
            tags=["float-3.14159", "boolean-True", "null-None"],
            base_parameters={
                "nested": {"deep": {"value": [1, 2, 3]}},
                "unicode": "测试数据",
                "special_chars": "test@#$%^&*()",
                "numbers": {"int": 42, "float": 3.14, "scientific": 1e-5}
            }
        )
        
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.return_value = [spec_with_types]
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 200
        response_data = response.json()
        experiments = response_data["experiments"]
        experiment = experiments[0]
        
        # Verify all data types are preserved
        assert "float-3.14159" in experiment["tags"]
        assert "boolean-True" in experiment["tags"]
        assert "null-None" in experiment["tags"]
        assert experiment["base_parameters"]["unicode"] == "测试数据"
        assert experiment["base_parameters"]["numbers"]["scientific"] == 1e-5


class TestListExperimentsPerformance:
    """Test performance-related aspects."""
    
    def test_list_experiments_concurrent_requests(self, client, app, mock_experiment_manager):
        """Test handling of concurrent list requests."""
        import threading
        
        # Setup
        test_spec = ExperimentSpec(
            experiment_id="concurrent-test",
            name="Concurrent Test",
            description="Testing concurrent access",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION,
            tags=["test-concurrency"],
            base_parameters={"threads": 10}
        )
        
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.return_value = [test_spec]
        
        def make_request():
            try:
                response = client.get("/api/v1/experiments/")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Execute concurrent requests
        results = []
        errors = []
        threads = []
        for _ in range(10):
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify all requests succeeded
        assert len(results) == 10
        assert all(status == 200 for status in results)
        assert len(errors) == 0
    
    def test_list_experiments_content_type(self, client, app, mock_experiment_manager):
        """Test that proper content type is returned."""
        # Setup
        app.state.experiment_manager = mock_experiment_manager
        mock_experiment_manager.list_experiments.return_value = []
        
        # Execute
        response = client.get("/api/v1/experiments/")
        
        # Verify
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")