"""
Unit tests for the orchestrator service.

This module contains tests for the core orchestrator functionality
including experiment, job, and agent management.
"""

import pytest
from uuid import uuid4
from datetime import datetime

from ..core.models import (
    ExperimentCreate, ExperimentUpdate,
    JobCreate, JobUpdate, JobStatus,
    AgentCreate, AgentUpdate, AgentStatus
)


class TestOrchestratorService:
    """Test cases for the OrchestratorService class."""

    @pytest.mark.asyncio
    async def test_create_experiment(self, orchestrator_service):
        """Test creating a new experiment."""
        experiment_data = ExperimentCreate(
            name="Test Experiment",
            description="A test experiment",
            model_name="test-model",
            runtime="transformers",
            platform="cuda",
            task_config={"task": "summarization"},
            data_config={"dataset": "test-data"}
        )
        
        experiment = await orchestrator_service.create_experiment(experiment_data)
        
        assert experiment.name == "Test Experiment"
        assert experiment.description == "A test experiment"
        assert experiment.model_name == "test-model"
        assert experiment.id is not None

    @pytest.mark.asyncio
    async def test_get_experiment(self, orchestrator_service):
        """Test retrieving an experiment by ID."""
        # Create experiment first
        experiment_data = ExperimentCreate(
            name="Test Experiment",
            model_name="test-model",
            runtime="transformers",
            platform="cuda",
            task_config={"task": "summarization"},
            data_config={"dataset": "test-data"}
        )
        created_experiment = await orchestrator_service.create_experiment(experiment_data)
        
        # Retrieve experiment
        retrieved_experiment = await orchestrator_service.get_experiment(created_experiment.id)
        
        assert retrieved_experiment is not None
        assert retrieved_experiment.id == created_experiment.id
        assert retrieved_experiment.name == "Test Experiment"

    @pytest.mark.asyncio
    async def test_get_nonexistent_experiment(self, orchestrator_service):
        """Test retrieving a non-existent experiment."""
        nonexistent_id = uuid4()
        experiment = await orchestrator_service.get_experiment(nonexistent_id)
        assert experiment is None

    @pytest.mark.asyncio
    async def test_list_experiments(self, orchestrator_service):
        """Test listing experiments with pagination."""
        # Create multiple experiments
        for i in range(3):
            experiment_data = ExperimentCreate(
                name=f"Test Experiment {i}",
                model_name="test-model",
                runtime="transformers",
                platform="cuda",
                task_config={"task": "summarization"},
                data_config={"dataset": "test-data"}
            )
            await orchestrator_service.create_experiment(experiment_data)
        
        # List experiments
        experiments = await orchestrator_service.list_experiments(skip=0, limit=10)
        
        assert len(experiments) == 3
        assert all(exp.name.startswith("Test Experiment") for exp in experiments)

    @pytest.mark.asyncio
    async def test_update_experiment(self, orchestrator_service):
        """Test updating an experiment."""
        # Create experiment
        experiment_data = ExperimentCreate(
            name="Original Name",
            model_name="test-model",
            runtime="transformers",
            platform="cuda",
            task_config={"task": "summarization"},
            data_config={"dataset": "test-data"}
        )
        experiment = await orchestrator_service.create_experiment(experiment_data)
        
        # Update experiment
        update_data = ExperimentUpdate(name="Updated Name", description="Updated description")
        updated_experiment = await orchestrator_service.update_experiment(experiment.id, update_data)
        
        assert updated_experiment is not None
        assert updated_experiment.name == "Updated Name"
        assert updated_experiment.description == "Updated description"

    @pytest.mark.asyncio
    async def test_delete_experiment(self, orchestrator_service):
        """Test deleting an experiment."""
        # Create experiment
        experiment_data = ExperimentCreate(
            name="To Delete",
            model_name="test-model",
            runtime="transformers",
            platform="cuda",
            task_config={"task": "summarization"},
            data_config={"dataset": "test-data"}
        )
        experiment = await orchestrator_service.create_experiment(experiment_data)
        
        # Delete experiment
        deleted = await orchestrator_service.delete_experiment(experiment.id)
        assert deleted is True
        
        # Verify deletion
        retrieved = await orchestrator_service.get_experiment(experiment.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_create_job(self, orchestrator_service):
        """Test creating a new job."""
        # Create experiment first
        experiment_data = ExperimentCreate(
            name="Test Experiment",
            model_name="test-model",
            runtime="transformers",
            platform="cuda",
            task_config={"task": "summarization"},
            data_config={"dataset": "test-data"}
        )
        experiment = await orchestrator_service.create_experiment(experiment_data)
        
        # Create job
        job_data = JobCreate(
            experiment_id=experiment.id,
            config={"batch_size": 32}
        )
        job = await orchestrator_service.create_job(job_data)
        
        assert job.experiment_id == experiment.id
        assert job.status == JobStatus.PENDING
        assert job.config == {"batch_size": 32}

    @pytest.mark.asyncio
    async def test_update_job_status(self, orchestrator_service):
        """Test updating job status with timestamps."""
        # Create experiment and job
        experiment_data = ExperimentCreate(
            name="Test Experiment",
            model_name="test-model",
            runtime="transformers",
            platform="cuda",
            task_config={"task": "summarization"},
            data_config={"dataset": "test-data"}
        )
        experiment = await orchestrator_service.create_experiment(experiment_data)
        
        job_data = JobCreate(
            experiment_id=experiment.id,
            config={"batch_size": 32}
        )
        job = await orchestrator_service.create_job(job_data)
        
        # Update to running
        update_data = JobUpdate(status=JobStatus.RUNNING)
        updated_job = await orchestrator_service.update_job(job.id, update_data)
        
        assert updated_job.status == JobStatus.RUNNING
        assert updated_job.started_at is not None
        
        # Update to completed
        update_data = JobUpdate(status=JobStatus.COMPLETED, results={"score": 0.95})
        completed_job = await orchestrator_service.update_job(job.id, update_data)
        
        assert completed_job.status == JobStatus.COMPLETED
        assert completed_job.completed_at is not None
        assert completed_job.results == {"score": 0.95}

    @pytest.mark.asyncio
    async def test_register_agent(self, orchestrator_service):
        """Test registering a new agent."""
        agent_data = AgentCreate(
            name="Test Agent",
            host="localhost",
            port=8001,
            capabilities={"runtime": "transformers", "platform": "cuda"}
        )
        
        agent = await orchestrator_service.register_agent(agent_data)
        
        assert agent.name == "Test Agent"
        assert agent.host == "localhost"
        assert agent.port == 8001
        assert agent.status == AgentStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_agent_heartbeat(self, orchestrator_service):
        """Test updating agent heartbeat."""
        # Register agent
        agent_data = AgentCreate(
            name="Test Agent",
            host="localhost",
            port=8001,
            capabilities={"runtime": "transformers"}
        )
        agent = await orchestrator_service.register_agent(agent_data)
        
        # Update heartbeat
        updated_agent = await orchestrator_service.update_agent_heartbeat(agent.id)
        
        assert updated_agent is not None
        assert updated_agent.last_heartbeat is not None

    @pytest.mark.asyncio
    async def test_assign_job_to_agent(self, orchestrator_service):
        """Test assigning a job to an agent."""
        # Create experiment
        experiment_data = ExperimentCreate(
            name="Test Experiment",
            model_name="test-model",
            runtime="transformers",
            platform="cuda",
            task_config={"task": "summarization"},
            data_config={"dataset": "test-data"}
        )
        experiment = await orchestrator_service.create_experiment(experiment_data)
        
        # Create job
        job_data = JobCreate(
            experiment_id=experiment.id,
            config={"batch_size": 32}
        )
        job = await orchestrator_service.create_job(job_data)
        
        # Register agent
        agent_data = AgentCreate(
            name="Test Agent",
            host="localhost",
            port=8001,
            capabilities={"runtime": "transformers"}
        )
        agent = await orchestrator_service.register_agent(agent_data)
        
        # Assign job to agent
        assigned_job = await orchestrator_service.assign_job_to_agent(job.id, agent.id)
        
        assert assigned_job.agent_id == agent.id
        assert assigned_job.status == JobStatus.RUNNING
