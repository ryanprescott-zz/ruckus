"""Tests for job management API endpoints."""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient
from ruckus_server.core.models import JobInfo
from ruckus_server.api.v1.models import CreateJobRequest, CreateJobResponse, ListJobsResponse
from ruckus_common.models import JobStatus, JobStatusEnum


class TestJobEndpoints:
    """Tests for job-related API endpoints."""

    def test_create_job_success(self, test_client_with_server):
        """Test successful job creation."""
        # Mock job info response
        mock_job_info = JobInfo(
            job_id="test_job_123",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(
                status=JobStatusEnum.ASSIGNED,
                message="Job has been scheduled with the agent"
            )
        )
        
        # Mock the job manager
        test_client_with_server.app.state.job_manager.create_job = AsyncMock(
            return_value=mock_job_info
        )
        
        response = test_client_with_server.post(
            "/api/v1/jobs",
            json={
                "experiment_id": "exp_123",
                "agent_id": "agent_123"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["job_id"] == "test_job_123"
        
        # Verify the job manager was called correctly
        test_client_with_server.app.state.job_manager.create_job.assert_called_once_with(
            experiment_id="exp_123",
            agent_id="agent_123"
        )

    def test_create_job_experiment_not_found(self, test_client_with_server):
        """Test job creation when experiment doesn't exist."""
        # Mock job manager to raise ValueError for non-existent experiment
        test_client_with_server.app.state.job_manager.create_job = AsyncMock(
            side_effect=ValueError("Experiment exp_999 does not exist")
        )
        
        response = test_client_with_server.post(
            "/api/v1/jobs",
            json={
                "experiment_id": "exp_999",
                "agent_id": "agent_123"
            }
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "does not exist" in data["detail"]

    def test_create_job_agent_not_found(self, test_client_with_server):
        """Test job creation when agent doesn't exist."""
        # Mock job manager to raise ValueError for non-existent agent
        test_client_with_server.app.state.job_manager.create_job = AsyncMock(
            side_effect=ValueError("Agent agent_999 does not exist")
        )
        
        response = test_client_with_server.post(
            "/api/v1/jobs",
            json={
                "experiment_id": "exp_123",
                "agent_id": "agent_999"
            }
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "does not exist" in data["detail"]

    def test_create_job_invalid_request(self, test_client_with_server):
        """Test job creation with invalid request data."""
        # Mock the job manager to avoid actual processing
        test_client_with_server.app.state.job_manager.create_job = AsyncMock(
            side_effect=ValueError("Invalid experiment ID")
        )
        
        response = test_client_with_server.post(
            "/api/v1/jobs",
            json={
                "experiment_id": "",  # Empty experiment ID
                "agent_id": "agent_123"
            }
        )
        
        assert response.status_code == 400  # Bad request for invalid data

    def test_create_job_missing_fields(self, test_client_with_server):
        """Test job creation with missing required fields."""
        response = test_client_with_server.post(
            "/api/v1/jobs",
            json={
                "experiment_id": "exp_123"
                # Missing agent_id
            }
        )
        
        assert response.status_code == 422  # Validation error

    def test_create_job_server_error(self, test_client_with_server):
        """Test job creation with unexpected server error."""
        # Mock job manager to raise generic exception
        test_client_with_server.app.state.job_manager.create_job = AsyncMock(
            side_effect=Exception("Unexpected server error")
        )
        
        response = test_client_with_server.post(
            "/api/v1/jobs",
            json={
                "experiment_id": "exp_123",
                "agent_id": "agent_123"
            }
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to create job" in data["detail"]

    def test_get_job_status_success(self, test_client_with_server):
        """Test successful job status retrieval."""
        # Mock job info response
        mock_job_info = JobInfo(
            job_id="test_job_123",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(
                status=JobStatusEnum.RUNNING,
                message="Job is currently running"
            )
        )
        
        # Mock the job manager
        test_client_with_server.app.state.job_manager.get_job_status = AsyncMock(
            return_value=mock_job_info
        )
        
        response = test_client_with_server.get("/api/v1/jobs/test_job_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test_job_123"
        assert data["experiment_id"] == "exp_123"
        assert data["agent_id"] == "agent_123"
        assert data["status"]["status"] == "running"
        
        # Verify the job manager was called correctly
        test_client_with_server.app.state.job_manager.get_job_status.assert_called_once_with("test_job_123")

    def test_get_job_status_not_found(self, test_client_with_server):
        """Test job status retrieval when job doesn't exist."""
        # Mock job manager to return None for non-existent job
        test_client_with_server.app.state.job_manager.get_job_status = AsyncMock(
            return_value=None
        )
        
        response = test_client_with_server.get("/api/v1/jobs/nonexistent_job")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    def test_get_job_status_server_error(self, test_client_with_server):
        """Test job status retrieval with server error."""
        # Mock job manager to raise exception
        test_client_with_server.app.state.job_manager.get_job_status = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        
        response = test_client_with_server.get("/api/v1/jobs/test_job_123")
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to get job status" in data["detail"]

    def test_list_jobs_success(self, test_client_with_server):
        """Test successful job listing."""
        # Mock jobs by agent response
        mock_jobs_by_agent = {
            "agent_123": [
                JobInfo(
                    job_id="job_1",
                    experiment_id="exp_1",
                    agent_id="agent_123",
                    created_time=datetime.now(timezone.utc),
                    status=JobStatus(status=JobStatusEnum.RUNNING, message="Running")
                ),
                JobInfo(
                    job_id="job_2",
                    experiment_id="exp_2",
                    agent_id="agent_123",
                    created_time=datetime.now(timezone.utc),
                    status=JobStatus(status=JobStatusEnum.COMPLETED, message="Completed")
                )
            ],
            "agent_456": [
                JobInfo(
                    job_id="job_3",
                    experiment_id="exp_3",
                    agent_id="agent_456",
                    created_time=datetime.now(timezone.utc),
                    status=JobStatus(status=JobStatusEnum.QUEUED, message="Queued")
                )
            ]
        }
        
        # Mock the job manager
        test_client_with_server.app.state.job_manager.list_job_info = AsyncMock(
            return_value=mock_jobs_by_agent
        )
        
        response = test_client_with_server.get("/api/v1/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "agent_123" in data["jobs"]
        assert "agent_456" in data["jobs"]
        assert len(data["jobs"]["agent_123"]) == 2
        assert len(data["jobs"]["agent_456"]) == 1
        assert data["jobs"]["agent_123"][0]["job_id"] == "job_1"
        assert data["jobs"]["agent_456"][0]["job_id"] == "job_3"
        
        # Verify the job manager was called correctly
        test_client_with_server.app.state.job_manager.list_job_info.assert_called_once()

    def test_list_jobs_empty_result(self, test_client_with_server):
        """Test job listing when no jobs exist."""
        # Mock empty jobs response
        test_client_with_server.app.state.job_manager.list_job_info = AsyncMock(
            return_value={}
        )
        
        response = test_client_with_server.get("/api/v1/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert data["jobs"] == {}

    def test_list_jobs_server_error(self, test_client_with_server):
        """Test job listing with server error."""
        # Mock job manager to raise exception
        test_client_with_server.app.state.job_manager.list_job_info = AsyncMock(
            side_effect=Exception("Storage backend unavailable")
        )
        
        response = test_client_with_server.get("/api/v1/jobs")
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to list jobs" in data["detail"]

    def test_cancel_job_success(self, test_client_with_server):
        """Test successful job cancellation."""
        # Mock the job manager
        test_client_with_server.app.state.job_manager.cancel_job = AsyncMock()
        
        response = test_client_with_server.delete("/api/v1/jobs/test_job_123")
        
        assert response.status_code == 204
        
        # Verify the job manager was called correctly
        test_client_with_server.app.state.job_manager.cancel_job.assert_called_once_with("test_job_123")

    def test_cancel_job_not_found(self, test_client_with_server):
        """Test job cancellation when job doesn't exist."""
        # Mock job manager to raise ValueError for non-existent job
        test_client_with_server.app.state.job_manager.cancel_job = AsyncMock(
            side_effect=ValueError("Job test_job_999 not found")
        )
        
        response = test_client_with_server.delete("/api/v1/jobs/test_job_999")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    def test_cancel_job_agent_not_found(self, test_client_with_server):
        """Test job cancellation when agent doesn't exist."""
        # Mock job manager to raise ValueError for non-existent agent
        test_client_with_server.app.state.job_manager.cancel_job = AsyncMock(
            side_effect=ValueError("Agent agent_999 does not exist")
        )
        
        response = test_client_with_server.delete("/api/v1/jobs/test_job_123")
        
        assert response.status_code == 404
        data = response.json()
        assert "does not exist" in data["detail"]

    def test_cancel_job_invalid_state(self, test_client_with_server):
        """Test job cancellation when job is in invalid state."""
        # Mock job manager to raise ValueError for invalid state
        test_client_with_server.app.state.job_manager.cancel_job = AsyncMock(
            side_effect=ValueError("Cannot cancel completed job")
        )
        
        response = test_client_with_server.delete("/api/v1/jobs/test_job_123")
        
        assert response.status_code == 400
        data = response.json()
        assert "Cannot cancel" in data["detail"]

    def test_cancel_job_server_error(self, test_client_with_server):
        """Test job cancellation with server error."""
        # Mock job manager to raise generic exception
        test_client_with_server.app.state.job_manager.cancel_job = AsyncMock(
            side_effect=Exception("Agent communication failed")
        )
        
        response = test_client_with_server.delete("/api/v1/jobs/test_job_123")
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to cancel job" in data["detail"]


class TestJobEndpointsIntegration:
    """Integration tests for job endpoints."""
    
    def test_create_and_get_job_workflow(self, test_client_with_server):
        """Test complete create and get job workflow."""
        # Mock job creation
        created_job = JobInfo(
            job_id="integration_job_123",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(
                status=JobStatusEnum.ASSIGNED,
                message="Job has been scheduled with the agent"
            )
        )
        
        test_client_with_server.app.state.job_manager.create_job = AsyncMock(
            return_value=created_job
        )
        test_client_with_server.app.state.job_manager.get_job_status = AsyncMock(
            return_value=created_job
        )
        
        # Create job
        create_response = test_client_with_server.post(
            "/api/v1/jobs",
            json={
                "experiment_id": "exp_123",
                "agent_id": "agent_123"
            }
        )
        
        assert create_response.status_code == 201
        job_id = create_response.json()["job_id"]
        
        # Get job status
        get_response = test_client_with_server.get(f"/api/v1/jobs/{job_id}")
        
        assert get_response.status_code == 200
        job_data = get_response.json()
        assert job_data["job_id"] == job_id
        assert job_data["status"]["status"] == "assigned"

    def test_create_list_and_cancel_job_workflow(self, test_client_with_server):
        """Test complete create, list, and cancel job workflow."""
        # Mock job creation and listing
        created_job = JobInfo(
            job_id="workflow_job_123",
            experiment_id="exp_123",
            agent_id="agent_123",
            created_time=datetime.now(timezone.utc),
            status=JobStatus(
                status=JobStatusEnum.RUNNING,
                message="Job is currently running"
            )
        )
        
        jobs_list = {
            "agent_123": [created_job]
        }
        
        test_client_with_server.app.state.job_manager.create_job = AsyncMock(
            return_value=created_job
        )
        test_client_with_server.app.state.job_manager.list_job_info = AsyncMock(
            return_value=jobs_list
        )
        test_client_with_server.app.state.job_manager.cancel_job = AsyncMock()
        
        # Create job
        create_response = test_client_with_server.post(
            "/api/v1/jobs",
            json={
                "experiment_id": "exp_123",
                "agent_id": "agent_123"
            }
        )
        
        assert create_response.status_code == 201
        job_id = create_response.json()["job_id"]
        
        # List jobs
        list_response = test_client_with_server.get("/api/v1/jobs")
        
        assert list_response.status_code == 200
        jobs_data = list_response.json()["jobs"]
        assert "agent_123" in jobs_data
        assert len(jobs_data["agent_123"]) == 1
        assert jobs_data["agent_123"][0]["job_id"] == job_id
        
        # Cancel job
        cancel_response = test_client_with_server.delete(f"/api/v1/jobs/{job_id}")
        
        assert cancel_response.status_code == 204
        
        # Verify cancel was called
        test_client_with_server.app.state.job_manager.cancel_job.assert_called_once_with(job_id)