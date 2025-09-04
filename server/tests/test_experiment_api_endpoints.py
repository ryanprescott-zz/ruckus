"""Tests for experiment API endpoints in ruckus_server.api.v1.routers.experiments."""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from ruckus_server.core.storage.base import ExperimentAlreadyExistsException, ExperimentNotFoundException, ExperimentHasJobsException
from ruckus_common.models import ExperimentSpec, TaskType


class TestExperimentEndpoints:
    """Tests for experiment-related API endpoints."""

    def test_create_experiment_success(self, test_client_with_experiment_manager, sample_experiment_spec):
        """Test successful experiment creation."""
        # Mock the experiment manager method
        test_client_with_experiment_manager.app.state.experiment_manager.create_experiment = AsyncMock(
            return_value={
                "experiment_id": sample_experiment_spec.experiment_id,
                "created_at": datetime.now(timezone.utc)
            }
        )
        
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            json={"experiment_spec": sample_experiment_spec.model_dump(mode='json')}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "experiment_id" in data
        assert "created_at" in data
        assert data["experiment_id"] == sample_experiment_spec.experiment_id

    def test_create_experiment_invalid_spec(self, test_client_with_experiment_manager):
        """Test experiment creation with invalid ExperimentSpec."""
        # Missing required fields
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            json={
                "experiment_spec": {
                    "name": "Test Experiment"
                    # Missing experiment_id, models, task_type
                }
            }
        )
        
        assert response.status_code == 422  # Validation error

    def test_create_experiment_empty_spec(self, test_client_with_experiment_manager):
        """Test experiment creation with empty experiment spec."""
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            json={"experiment_spec": {}}
        )
        
        assert response.status_code == 422  # Validation error

    def test_create_experiment_already_exists(self, test_client_with_experiment_manager, sample_experiment_spec):
        """Test creating experiment that already exists."""
        test_client_with_experiment_manager.app.state.experiment_manager.create_experiment = AsyncMock(
            side_effect=ExperimentAlreadyExistsException(sample_experiment_spec.experiment_id)
        )
        
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            json={"experiment_spec": sample_experiment_spec.model_dump(mode='json')}
        )
        
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_create_experiment_value_error(self, test_client_with_experiment_manager, sample_experiment_spec):
        """Test experiment creation with validation error."""
        test_client_with_experiment_manager.app.state.experiment_manager.create_experiment = AsyncMock(
            side_effect=ValueError("Invalid experiment data")
        )
        
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            json={"experiment_spec": sample_experiment_spec.model_dump(mode='json')}
        )
        
        assert response.status_code == 400
        assert "Invalid experiment data" in response.json()["detail"]

    def test_create_experiment_server_error(self, test_client_with_experiment_manager, sample_experiment_spec):
        """Test experiment creation with server error."""
        test_client_with_experiment_manager.app.state.experiment_manager.create_experiment = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            json={"experiment_spec": sample_experiment_spec.model_dump(mode='json')}
        )
        
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

    def test_create_experiment_manager_not_initialized(self):
        """Test experiment creation when experiment manager not initialized."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from ruckus_server.api.v1.api import api_router
        
        # Create a simple app without managers
        test_app = FastAPI()
        test_app.include_router(api_router, prefix="/api/v1")
        
        with TestClient(test_app) as client:
            response = client.post(
                "/api/v1/experiments/",
                json={
                    "experiment_spec": {
                        "experiment_id": "test",
                        "name": "Test",
                        "models": ["test-model"],
                        "task_type": "summarization"
                    }
                }
            )
        
        assert response.status_code == 503
        assert "Experiment manager not initialized" in response.json()["detail"]

    def test_create_experiment_complex_spec(self, test_client_with_experiment_manager, experiment_spec_factory):
        """Test creating experiment with complex ExperimentSpec."""
        complex_spec = experiment_spec_factory(
            experiment_id="complex-experiment",
            name="Complex Test Experiment",
            models=["gpt-3.5-turbo", "gpt-4"],
            task_type=TaskType.QUESTION_ANSWERING,
            priority=8,
            timeout_seconds=7200,
            owner="test-user",
            tags=["complex", "multi-model", "qa"]
        )
        
        test_client_with_experiment_manager.app.state.experiment_manager.create_experiment = AsyncMock(
            return_value={
                "experiment_id": complex_spec.experiment_id,
                "created_at": datetime.now(timezone.utc)
            }
        )
        
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            json={"experiment_spec": complex_spec.model_dump(mode='json')}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == "complex-experiment"

    def test_experiment_api_content_type(self, test_client_with_experiment_manager, sample_experiment_spec):
        """Test that experiment API endpoints return proper content type."""
        test_client_with_experiment_manager.app.state.experiment_manager.create_experiment = AsyncMock(
            return_value={
                "experiment_id": sample_experiment_spec.experiment_id,
                "created_at": datetime.now(timezone.utc)
            }
        )
        
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            json={"experiment_spec": sample_experiment_spec.model_dump(mode='json')}
        )
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_experiment_api_error_responses_format(self, test_client_with_experiment_manager, sample_experiment_spec):
        """Test that experiment API error responses have consistent format."""
        test_client_with_experiment_manager.app.state.experiment_manager.create_experiment = AsyncMock(
            side_effect=ValueError("Test error")
        )
        
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            json={"experiment_spec": sample_experiment_spec.model_dump(mode='json')}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], str)

    def test_experiment_request_validation_edge_cases(self, test_client_with_experiment_manager):
        """Test experiment request validation with edge cases."""
        # Test with missing request body
        response = test_client_with_experiment_manager.post("/api/v1/experiments/")
        assert response.status_code == 422
        
        # Test with malformed JSON
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test with wrong field types
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            json={"experiment_spec": "not-an-object"}
        )
        assert response.status_code == 422

    def test_experiment_response_serialization(self, test_client_with_experiment_manager, sample_experiment_spec):
        """Test that experiment responses are properly serialized."""
        created_at = datetime.now(timezone.utc)
        test_client_with_experiment_manager.app.state.experiment_manager.create_experiment = AsyncMock(
            return_value={
                "experiment_id": sample_experiment_spec.experiment_id,
                "created_at": created_at
            }
        )
        
        response = test_client_with_experiment_manager.post(
            "/api/v1/experiments/",
            json={"experiment_spec": sample_experiment_spec.model_dump(mode='json')}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify datetime fields are properly serialized
        assert isinstance(data["experiment_id"], str)
        assert isinstance(data["created_at"], str)
        
        # Verify we can parse the datetime
        parsed_datetime = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
        assert isinstance(parsed_datetime, datetime)

    def test_concurrent_experiment_creation(self, test_client_with_experiment_manager, experiment_spec_factory):
        """Test handling of concurrent experiment creation requests."""
        import threading
        
        results = []
        errors = []
        
        def create_experiment(experiment_id):
            try:
                spec = experiment_spec_factory(experiment_id=experiment_id)
                response = test_client_with_experiment_manager.post(
                    "/api/v1/experiments/",
                    json={"experiment_spec": spec.model_dump(mode='json')}
                )
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Mock successful creation for all
        test_client_with_experiment_manager.app.state.experiment_manager.create_experiment = AsyncMock(
            return_value={
                "experiment_id": "test",
                "created_at": datetime.now(timezone.utc)
            }
        )
        
        # Make multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_experiment, args=(f"experiment-{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)
        assert len(errors) == 0

    # List Experiments Tests
    def test_list_experiments_success(self, test_client_with_experiment_manager):
        """Test successful experiments listing."""
        # Mock the experiment manager method
        mock_experiments = [
            ExperimentSpec(
                experiment_id="exp-1",
                name="Experiment 1",
                description="First experiment",
                models=["model-1"],
                task_type=TaskType.SUMMARIZATION
            ),
            ExperimentSpec(
                experiment_id="exp-2",
                name="Experiment 2", 
                description="Second experiment",
                models=["model-2"],
                task_type=TaskType.QUESTION_ANSWERING
            )
        ]
        
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=mock_experiments
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        data = response.json()
        assert "experiments" in data
        assert len(data["experiments"]) == 2
        assert data["experiments"][0]["experiment_id"] == "exp-1"
        assert data["experiments"][1]["experiment_id"] == "exp-2"

    def test_list_experiments_empty(self, test_client_with_experiment_manager):
        """Test experiments listing when no experiments exist."""
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=[]
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["experiments"] == []

    def test_list_experiments_manager_not_initialized(self):
        """Test experiments listing when experiment manager not initialized."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from ruckus_server.api.v1.api import api_router
        
        # Create a simple app without managers
        test_app = FastAPI()
        test_app.include_router(api_router, prefix="/api/v1")
        
        with TestClient(test_app) as client:
            response = client.get("/api/v1/experiments/")
        
        assert response.status_code == 503
        assert "Experiment manager not initialized" in response.json()["detail"]

    def test_list_experiments_server_error(self, test_client_with_experiment_manager):
        """Test experiments listing with server error."""
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

    def test_list_experiments_value_error(self, test_client_with_experiment_manager):
        """Test experiments listing with validation error."""
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            side_effect=ValueError("Invalid query parameters")
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 400
        assert "Invalid query parameters" in response.json()["detail"]

    def test_list_experiments_large_dataset(self, test_client_with_experiment_manager):
        """Test experiments listing with many experiments."""
        # Create mock data for many experiments
        mock_experiments = []
        for i in range(100):
            mock_experiments.append(
                ExperimentSpec(
                    experiment_id=f"exp-{i}",
                    name=f"Experiment {i}",
                    description=f"Description for experiment {i}",
                    models=[f"model-{i}"],
                    task_type=TaskType.SUMMARIZATION
                )
            )
        
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=mock_experiments
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["experiments"]) == 100
        assert data["experiments"][0]["experiment_id"] == "exp-0"
        assert data["experiments"][99]["experiment_id"] == "exp-99"

    def test_list_experiments_response_format(self, test_client_with_experiment_manager):
        """Test that list experiments response has correct format."""
        mock_experiment = ExperimentSpec(
            experiment_id="format-test",
            name="Format Test Experiment",
            description="Testing response format",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION
        )
        
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=[mock_experiment]
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "experiments" in data
        assert len(data["experiments"]) == 1
        
        experiment = data["experiments"][0]
        assert "experiment_id" in experiment
        assert "name" in experiment
        assert "description" in experiment
        assert "models" in experiment
        assert "task_type" in experiment

    def test_list_experiments_content_type(self, test_client_with_experiment_manager):
        """Test that list experiments endpoint returns proper content type."""
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=[]
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_list_experiments_datetime_serialization(self, test_client_with_experiment_manager):
        """Test that datetime fields are properly serialized in list response."""
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)
        
        mock_experiment = ExperimentSpec(
            experiment_id="datetime-test",
            name="DateTime Test",
            description="Testing datetime serialization",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION
        )
        
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=[mock_experiment]
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        data = response.json()
        
        experiment = data["experiments"][0]
        
        # Verify datetime fields are properly serialized as strings
        assert isinstance(experiment["created_at"], str)
        assert isinstance(experiment["updated_at"], str)
        
        # Verify we can parse the datetime strings
        parsed_created = datetime.fromisoformat(experiment["created_at"].replace('Z', '+00:00'))
        parsed_updated = datetime.fromisoformat(experiment["updated_at"].replace('Z', '+00:00'))
        assert isinstance(parsed_created, datetime)
        assert isinstance(parsed_updated, datetime)

    def test_list_experiments_concurrent_requests(self, test_client_with_experiment_manager):
        """Test handling of concurrent list experiments requests."""
        import threading
        
        results = []
        errors = []
        
        def list_experiments():
            try:
                response = test_client_with_experiment_manager.get("/api/v1/experiments/")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Mock successful listing
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=[
                ExperimentSpec(
                    experiment_id="concurrent-test",
                    name="Concurrent Test",
                    description="Testing concurrent access",
                    models=["test-model"],
                    task_type=TaskType.SUMMARIZATION
                )
            ]
        )
        
        # Make multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=list_experiments)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)
        assert len(errors) == 0

    # Delete Experiment Tests
    def test_delete_experiment_success(self, test_client_with_experiment_manager):
        """Test successful experiment deletion."""
        experiment_id = "test-experiment-123"
        deleted_at = datetime.now(timezone.utc)
        
        # Mock the experiment manager method
        test_client_with_experiment_manager.app.state.experiment_manager.delete_experiment = AsyncMock(
            return_value={
                "experiment_id": experiment_id,
                "deleted_at": deleted_at
            }
        )
        
        response = test_client_with_experiment_manager.delete(f"/api/v1/experiments/{experiment_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert "experiment_id" in data
        assert "deleted_at" in data
        assert data["experiment_id"] == experiment_id

    def test_delete_experiment_not_found(self, test_client_with_experiment_manager):
        """Test deleting non-existent experiment."""
        experiment_id = "non-existent-experiment"
        
        test_client_with_experiment_manager.app.state.experiment_manager.delete_experiment = AsyncMock(
            side_effect=ExperimentNotFoundException(experiment_id)
        )
        
        response = test_client_with_experiment_manager.delete(f"/api/v1/experiments/{experiment_id}")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_experiment_has_jobs(self, test_client_with_experiment_manager):
        """Test deleting experiment that has associated jobs."""
        experiment_id = "experiment-with-jobs"
        job_count = 3
        
        test_client_with_experiment_manager.app.state.experiment_manager.delete_experiment = AsyncMock(
            side_effect=ExperimentHasJobsException(experiment_id, job_count)
        )
        
        response = test_client_with_experiment_manager.delete(f"/api/v1/experiments/{experiment_id}")
        
        assert response.status_code == 409
        assert "associated job(s)" in response.json()["detail"]
        assert str(job_count) in response.json()["detail"]

    def test_delete_experiment_value_error(self, test_client_with_experiment_manager):
        """Test experiment deletion with validation error."""
        experiment_id = "invalid-experiment"
        
        test_client_with_experiment_manager.app.state.experiment_manager.delete_experiment = AsyncMock(
            side_effect=ValueError("Invalid experiment ID format")
        )
        
        response = test_client_with_experiment_manager.delete(f"/api/v1/experiments/{experiment_id}")
        
        assert response.status_code == 400
        assert "Invalid experiment ID format" in response.json()["detail"]

    def test_delete_experiment_server_error(self, test_client_with_experiment_manager):
        """Test experiment deletion with server error."""
        experiment_id = "error-experiment"
        
        test_client_with_experiment_manager.app.state.experiment_manager.delete_experiment = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        
        response = test_client_with_experiment_manager.delete(f"/api/v1/experiments/{experiment_id}")
        
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

    def test_delete_experiment_manager_not_initialized(self):
        """Test experiment deletion when experiment manager not initialized."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from ruckus_server.api.v1.api import api_router
        
        # Create a simple app without managers
        test_app = FastAPI()
        test_app.include_router(api_router, prefix="/api/v1")
        
        with TestClient(test_app) as client:
            response = client.delete("/api/v1/experiments/test-experiment")
        
        assert response.status_code == 503
        assert "Experiment manager not initialized" in response.json()["detail"]

    def test_delete_experiment_empty_id(self, test_client_with_experiment_manager):
        """Test deleting experiment with empty ID."""
        # Empty experiment ID in URL path
        response = test_client_with_experiment_manager.delete("/api/v1/experiments/")
        
        # Should return 404 or 405 (method not allowed) for the base path
        assert response.status_code in [404, 405]

    def test_delete_experiment_special_characters(self, test_client_with_experiment_manager):
        """Test deleting experiment with special characters in ID."""
        experiment_id = "experiment-äöü-🚀"
        deleted_at = datetime.now(timezone.utc)
        
        test_client_with_experiment_manager.app.state.experiment_manager.delete_experiment = AsyncMock(
            return_value={
                "experiment_id": experiment_id,
                "deleted_at": deleted_at
            }
        )
        
        # URL encode the experiment ID
        import urllib.parse
        encoded_id = urllib.parse.quote(experiment_id, safe='')
        response = test_client_with_experiment_manager.delete(f"/api/v1/experiments/{encoded_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == experiment_id

    def test_delete_experiment_response_serialization(self, test_client_with_experiment_manager):
        """Test that delete experiment responses are properly serialized."""
        experiment_id = "serialization-test"
        deleted_at = datetime.now(timezone.utc)
        
        test_client_with_experiment_manager.app.state.experiment_manager.delete_experiment = AsyncMock(
            return_value={
                "experiment_id": experiment_id,
                "deleted_at": deleted_at
            }
        )
        
        response = test_client_with_experiment_manager.delete(f"/api/v1/experiments/{experiment_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify datetime fields are properly serialized
        assert isinstance(data["experiment_id"], str)
        assert isinstance(data["deleted_at"], str)
        
        # Verify we can parse the datetime
        parsed_datetime = datetime.fromisoformat(data["deleted_at"].replace('Z', '+00:00'))
        assert isinstance(parsed_datetime, datetime)

    def test_delete_experiment_content_type(self, test_client_with_experiment_manager):
        """Test that delete experiment endpoint returns proper content type."""
        experiment_id = "content-type-test"
        deleted_at = datetime.now(timezone.utc)
        
        test_client_with_experiment_manager.app.state.experiment_manager.delete_experiment = AsyncMock(
            return_value={
                "experiment_id": experiment_id,
                "deleted_at": deleted_at
            }
        )
        
        response = test_client_with_experiment_manager.delete(f"/api/v1/experiments/{experiment_id}")
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_concurrent_experiment_deletion(self, test_client_with_experiment_manager):
        """Test handling of concurrent experiment deletion requests."""
        import threading
        
        results = []
        errors = []
        
        def delete_experiment(experiment_id):
            try:
                response = test_client_with_experiment_manager.delete(f"/api/v1/experiments/{experiment_id}")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Mock successful deletion for all
        test_client_with_experiment_manager.app.state.experiment_manager.delete_experiment = AsyncMock(
            return_value={
                "experiment_id": "test",
                "deleted_at": datetime.now(timezone.utc)
            }
        )
        
        # Make multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=delete_experiment, args=(f"experiment-{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)
        assert len(errors) == 0

    # List Experiments Tests
    def test_list_experiments_success(self, test_client_with_experiment_manager):
        """Test successful experiments listing."""
        # Mock the experiment manager method
        mock_experiments = [
            ExperimentSpec(
                experiment_id="exp-1",
                name="Experiment 1",
                description="First experiment",
                models=["model-1"],
                task_type=TaskType.SUMMARIZATION
            ),
            ExperimentSpec(
                experiment_id="exp-2",
                name="Experiment 2", 
                description="Second experiment",
                models=["model-2"],
                task_type=TaskType.QUESTION_ANSWERING
            )
        ]
        
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=mock_experiments
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        data = response.json()
        assert "experiments" in data
        assert len(data["experiments"]) == 2
        assert data["experiments"][0]["experiment_id"] == "exp-1"
        assert data["experiments"][1]["experiment_id"] == "exp-2"

    def test_list_experiments_empty(self, test_client_with_experiment_manager):
        """Test experiments listing when no experiments exist."""
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=[]
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["experiments"] == []

    def test_list_experiments_manager_not_initialized(self):
        """Test experiments listing when experiment manager not initialized."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from ruckus_server.api.v1.api import api_router
        
        # Create a simple app without managers
        test_app = FastAPI()
        test_app.include_router(api_router, prefix="/api/v1")
        
        with TestClient(test_app) as client:
            response = client.get("/api/v1/experiments/")
        
        assert response.status_code == 503
        assert "Experiment manager not initialized" in response.json()["detail"]

    def test_list_experiments_server_error(self, test_client_with_experiment_manager):
        """Test experiments listing with server error."""
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

    def test_list_experiments_value_error(self, test_client_with_experiment_manager):
        """Test experiments listing with validation error."""
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            side_effect=ValueError("Invalid query parameters")
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 400
        assert "Invalid query parameters" in response.json()["detail"]

    def test_list_experiments_large_dataset(self, test_client_with_experiment_manager):
        """Test experiments listing with many experiments."""
        # Create mock data for many experiments
        mock_experiments = []
        for i in range(100):
            mock_experiments.append(
                ExperimentSpec(
                    experiment_id=f"exp-{i}",
                    name=f"Experiment {i}",
                    description=f"Description for experiment {i}",
                    models=[f"model-{i}"],
                    task_type=TaskType.SUMMARIZATION
                )
            )
        
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=mock_experiments
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["experiments"]) == 100
        assert data["experiments"][0]["experiment_id"] == "exp-0"
        assert data["experiments"][99]["experiment_id"] == "exp-99"

    def test_list_experiments_response_format(self, test_client_with_experiment_manager):
        """Test that list experiments response has correct format."""
        mock_experiment = ExperimentSpec(
            experiment_id="format-test",
            name="Format Test Experiment",
            description="Testing response format",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION
        )
        
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=[mock_experiment]
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "experiments" in data
        assert len(data["experiments"]) == 1
        
        experiment = data["experiments"][0]
        assert "experiment_id" in experiment
        assert "name" in experiment
        assert "description" in experiment
        assert "models" in experiment
        assert "task_type" in experiment

    def test_list_experiments_content_type(self, test_client_with_experiment_manager):
        """Test that list experiments endpoint returns proper content type."""
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=[]
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_list_experiments_datetime_serialization(self, test_client_with_experiment_manager):
        """Test that datetime fields are properly serialized in list response."""
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)
        
        mock_experiment = ExperimentSpec(
            experiment_id="datetime-test",
            name="DateTime Test",
            description="Testing datetime serialization",
            models=["test-model"],
            task_type=TaskType.SUMMARIZATION
        )
        
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=[mock_experiment]
        )
        
        response = test_client_with_experiment_manager.get("/api/v1/experiments/")
        
        assert response.status_code == 200
        data = response.json()
        
        experiment = data["experiments"][0]
        
        # Verify datetime fields are properly serialized as strings
        assert isinstance(experiment["created_at"], str)
        assert isinstance(experiment["updated_at"], str)
        
        # Verify we can parse the datetime strings
        parsed_created = datetime.fromisoformat(experiment["created_at"].replace('Z', '+00:00'))
        parsed_updated = datetime.fromisoformat(experiment["updated_at"].replace('Z', '+00:00'))
        assert isinstance(parsed_created, datetime)
        assert isinstance(parsed_updated, datetime)

    def test_list_experiments_concurrent_requests(self, test_client_with_experiment_manager):
        """Test handling of concurrent list experiments requests."""
        import threading
        
        results = []
        errors = []
        
        def list_experiments():
            try:
                response = test_client_with_experiment_manager.get("/api/v1/experiments/")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Mock successful listing
        test_client_with_experiment_manager.app.state.experiment_manager.list_experiments = AsyncMock(
            return_value=[
                ExperimentSpec(
                    experiment_id="concurrent-test",
                    name="Concurrent Test",
                    description="Testing concurrent access",
                    models=["test-model"],
                    task_type=TaskType.SUMMARIZATION
                )
            ]
        )
        
        # Make multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=list_experiments)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)
        assert len(errors) == 0