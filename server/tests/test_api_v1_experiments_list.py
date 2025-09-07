"""Tests for GET /api/v1/experiments/ endpoint (list experiments)."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ruckus_server.api.v1.routers.experiments import router
from ruckus_common.models import (ExperimentSpec, TaskType, TaskSpec, FrameworkSpec, MetricsSpec, 
                                  LLMGenerationParams, PromptTemplate, PromptMessage, PromptRole, FrameworkName)


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
            name="Test Experiment 1",
            description="First test experiment",
            model="test-model",
            task=TaskSpec(
                name="api_test_task_1",
                type=TaskType.LLM_GENERATION,
                description="API test task 1",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="API test experiment 1.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={"learning_rate": 0.01, "batch_size": 32}
            ),
            metrics=MetricsSpec(
                metrics={"type": "test", "version": "1.0"}
            )
        ),
        ExperimentSpec(
            name="Test Experiment 2",
            description="Second test experiment",
            model="test-model-2",
            task=TaskSpec(
                name="api_test_task_2",
                type=TaskType.LLM_GENERATION,
                description="API test task 2",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="API test experiment 2.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={"learning_rate": 0.001, "batch_size": 64}
            ),
            metrics=MetricsSpec(
                metrics={"type": "production", "version": "2.0"}
            )
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
        assert experiments[0]["id"] == sample_experiment_specs[0].id
        assert experiments[0]["name"] == "Test Experiment 1"
        assert experiments[0]["description"] == "First test experiment"
        assert experiments[0]["model"] == "test-model"
        
        # Verify second experiment
        assert experiments[1]["id"] == sample_experiment_specs[1].id
        assert experiments[1]["name"] == "Test Experiment 2"
        assert experiments[1]["description"] == "Second test experiment"
        assert experiments[1]["model"] == "test-model-2"
        
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
            name="Complex Experiment",
            description="An experiment with complex parameters",
            model="transformer-model",
            task=TaskSpec(
                name="complex_task",
                type=TaskType.LLM_GENERATION,
                description="Complex task with transformer model",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Complex experiment task.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={
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
            ),
            metrics=MetricsSpec(
                metrics={"complexity": "high", "framework": "pytorch"}
            )
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
        assert experiment["id"] == complex_spec.id
        assert experiment["framework"]["params"]["model"]["architecture"] == "transformer"
        assert experiment["framework"]["params"]["training"]["optimizer"]["name"] == "adam"
        assert len(experiment["framework"]["params"]["model"]["layers"]) == 2
        assert experiment["framework"]["params"]["data"]["augmentation"]["methods"] == ["random_crop", "flip"]
    
    def test_list_experiments_large_dataset(self, client, app, mock_experiment_manager):
        """Test listing many experiments."""
        # Setup
        large_experiment_list = []
        for i in range(100):
            spec = ExperimentSpec(
                name=f"Experiment {i}",
                description=f"Test experiment number {i}",
                model=f"model-{i}",
                task=TaskSpec(
                    name=f"large_test_task_{i}",
                    type=TaskType.LLM_GENERATION,
                    description=f"Large test task {i}",
                    params=LLMGenerationParams(
                        prompt_template=PromptTemplate(
                            messages=[
                                PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                                PromptMessage(role=PromptRole.USER, content=f"Large test experiment {i}.")
                            ]
                        )
                    )
                ),
                framework=FrameworkSpec(
                    name=FrameworkName.TRANSFORMERS,
                    params={"value": i * 10, "active": i % 2 == 0}
                ),
                metrics=MetricsSpec(
                    metrics={"index": str(i), "batch": "large_test"}
                )
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
        assert experiments[0]["id"] == large_experiment_list[0].id
        assert experiments[0]["framework"]["params"]["value"] == 0
        assert experiments[99]["id"] == large_experiment_list[99].id
        assert experiments[99]["framework"]["params"]["value"] == 990


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
            name="None Description Test",
            description=None,
            model="test-model",
            task=TaskSpec(
                name="none_desc_task",
                type=TaskType.LLM_GENERATION,
                description="Task with none description",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="None description test.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={}
            ),
            metrics=MetricsSpec(
                metrics={}
            )
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
            name="Empty Containers Test",
            description="Testing empty tags and parameters",
            model="test-model",
            task=TaskSpec(
                name="empty_containers_task",
                type=TaskType.LLM_GENERATION,
                description="Task with empty containers",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Empty containers test.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={}
            ),
            metrics=MetricsSpec(
                metrics={}
            )
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
        assert experiments[0]["metrics"]["metrics"] == {}
        assert experiments[0]["framework"]["params"] == {}
    
    def test_list_experiments_response_format(self, client, app, mock_experiment_manager):
        """Test that response has correct structure."""
        # Setup
        single_spec = ExperimentSpec(
            name="Format Test",
            description="Testing response format",
            model="test-model",
            task=TaskSpec(
                name="format_test_task",
                type=TaskType.LLM_GENERATION,
                description="Format test task",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Format test.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={"test": True}
            ),
            metrics=MetricsSpec(
                metrics={"env": "test"}
            )
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
        required_fields = ["id", "name", "description", "model", "task", "framework", "metrics"]
        for field in required_fields:
            assert field in experiment
    
    def test_list_experiments_json_serialization(self, client, app, mock_experiment_manager):
        """Test that complex data types are properly JSON serialized."""
        # Setup
        spec_with_types = ExperimentSpec(
            name="JSON Serialization Test",
            description="Testing JSON edge cases",
            model="test-model",
            task=TaskSpec(
                name="json_test_task",
                type=TaskType.LLM_GENERATION,
                description="JSON serialization test task",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="JSON serialization test.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={
                    "nested": {"deep": {"value": [1, 2, 3]}},
                    "unicode": "测试数据",
                    "special_chars": "test@#$%^&*()",
                    "numbers": {"int": 42, "float": 3.14, "scientific": 1e-5}
                }
            ),
            metrics=MetricsSpec(
                metrics={"float": "3.14159", "boolean": "True", "null": "None"}
            )
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
        assert experiment["metrics"]["metrics"]["float"] == "3.14159"
        assert experiment["metrics"]["metrics"]["boolean"] == "True"
        assert experiment["metrics"]["metrics"]["null"] == "None"
        assert experiment["framework"]["params"]["unicode"] == "测试数据"
        assert experiment["framework"]["params"]["numbers"]["scientific"] == 1e-5


class TestListExperimentsPerformance:
    """Test performance-related aspects."""
    
    def test_list_experiments_concurrent_requests(self, client, app, mock_experiment_manager):
        """Test handling of concurrent list requests."""
        import threading
        
        # Setup
        test_spec = ExperimentSpec(
            name="Concurrent Test",
            description="Testing concurrent access",
            model="test-model",
            task=TaskSpec(
                name="concurrent_test_task",
                type=TaskType.LLM_GENERATION,
                description="Concurrent test task",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Concurrent test.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={"threads": 10}
            ),
            metrics=MetricsSpec(
                metrics={"test": "concurrency"}
            )
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