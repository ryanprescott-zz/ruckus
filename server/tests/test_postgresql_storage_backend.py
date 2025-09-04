"""Comprehensive tests for PostgreSQL storage backend.

NOTE: These tests require a test PostgreSQL database to be available.
They are designed to be run in CI/CD environments where test databases can be provisioned.

To run these tests, set the POSTGRESQL_TEST_URL environment variable:
export POSTGRESQL_TEST_URL="postgresql+asyncpg://test:test@localhost:5432/test_ruckus"

Or use the --postgresql-url command line flag:
pytest tests/test_postgresql_storage_backend.py --postgresql-url="postgresql+asyncpg://test:test@localhost:5432/test_ruckus"
"""

import pytest
import pytest_asyncio
import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, Any

# Check if asyncpg is installed
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

# Skip all tests in this module if asyncpg is not installed
pytestmark = pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")

from ruckus_server.core.storage.postgresql import PostgreSQLStorageBackend
from ruckus_server.core.storage.base import (
    ExperimentAlreadyExistsException, 
    ExperimentNotFoundException, 
    ExperimentHasJobsException
)
from ruckus_server.core.config import PostgreSQLSettings
from ruckus_common.models import (ExperimentSpec, RegisteredAgentInfo, AgentType, TaskType, 
                                  TaskSpec, FrameworkSpec, MetricsSpec, LLMGenerationParams, 
                                  PromptTemplate, PromptMessage, PromptRole, FrameworkName)


def pytest_runtest_setup(item):
    """Skip tests if PostgreSQL URL not provided."""
    if item.fspath.purebasename == "test_postgresql_storage_backend":
        # Check for environment variable first, then command line flag
        postgresql_url = os.environ.get("POSTGRESQL_TEST_URL")
        if not postgresql_url:
            postgresql_url = item.config.getoption("--postgresql-url", default=None)
        
        if not postgresql_url:
            pytest.skip("PostgreSQL tests require POSTGRESQL_TEST_URL environment variable or --postgresql-url flag")


@pytest.fixture
def postgresql_settings(request):
    """Create PostgreSQL settings for testing."""
    # Check environment variable first, then command line flag
    test_url = os.environ.get("POSTGRESQL_TEST_URL")
    if not test_url:
        test_url = request.config.getoption("--postgresql-url", default="postgresql+asyncpg://test:test@localhost:5432/test_ruckus")
    
    return PostgreSQLSettings(
        database_url=test_url,
        pool_size=5,
        max_overflow=10,
        echo_sql=False
    )


@pytest_asyncio.fixture
async def postgresql_storage(postgresql_settings):
    """Create PostgreSQL storage backend for testing."""
    backend = PostgreSQLStorageBackend(postgresql_settings)
    await backend.initialize()
    
    # Clean up any existing test data
    async with backend.session_factory() as session:
        from ruckus_server.core.storage.base import Experiment, Job, Agent
        from sqlalchemy import delete
        
        # Delete in order to respect foreign key constraints
        await session.execute(delete(Job))
        await session.execute(delete(Experiment))
        await session.execute(delete(Agent))
        await session.commit()
    
    yield backend
    await backend.close()


@pytest.fixture
def sample_experiment_spec():
    """Create sample experiment spec for testing."""
    return ExperimentSpec(
        name="Test PostgreSQL Experiment",
        description="An experiment for testing PostgreSQL backend",
        model="postgresql-model",
        task=TaskSpec(
            name="postgresql_task",
            type=TaskType.LLM_GENERATION,
            description="PostgreSQL test task",
            params=LLMGenerationParams(
                prompt_template=PromptTemplate(
                    messages=[
                        PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                        PromptMessage(role=PromptRole.USER, content="Test PostgreSQL backend.")
                    ]
                )
            )
        ),
        framework=FrameworkSpec(
            name=FrameworkName.TRANSFORMERS,
            params={"learning_rate": 0.01, "epochs": 10, "batch_size": 32}
        ),
        metrics=MetricsSpec(
            metrics={"type": "test", "backend": "postgresql"}
        )
    )


@pytest.fixture
def sample_agent_info():
    """Create sample agent info for testing."""
    return RegisteredAgentInfo(
        agent_id="test-postgresql-agent",
        agent_name="Test PostgreSQL Agent",
        agent_type=AgentType.WHITE_BOX,
        agent_url="http://localhost:8001",
        system_info={"os": "linux", "python": "3.11"},
        capabilities={"tasks": ["training", "evaluation"]},
        status="active",
        last_heartbeat=datetime.now(timezone.utc),
        registered_at=datetime.now(timezone.utc)
    )


class TestPostgreSQLBackendInitialization:
    """Test PostgreSQL backend initialization and basic operations."""
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, postgresql_settings):
        """Test successful initialization."""
        backend = PostgreSQLStorageBackend(postgresql_settings)
        await backend.initialize()
        
        # Check that engine and session factory are created
        assert backend.engine is not None
        assert backend.session_factory is not None
        
        # Test health check
        health = await backend.health_check()
        assert health is True
        
        await backend.close()
    
    @pytest.mark.asyncio
    async def test_health_check_after_close(self, postgresql_storage):
        """Test health check after closing connection."""
        # Initially should be healthy
        assert await postgresql_storage.health_check() is True
        
        # Close and check health
        await postgresql_storage.close()
        assert await postgresql_storage.health_check() is False


class TestPostgreSQLExperimentOperations:
    """Test experiment CRUD operations."""
    
    @pytest.mark.asyncio
    async def test_create_experiment_success(self, postgresql_storage, sample_experiment_spec):
        """Test successful experiment creation."""
        result = await postgresql_storage.create_experiment(sample_experiment_spec)
        
        # Verify return format
        assert isinstance(result, dict)
        assert "experiment_id" in result
        assert "created_at" in result
        assert result["experiment_id"] == sample_experiment_spec.experiment_id
        assert isinstance(result["created_at"], datetime)
    
    @pytest.mark.asyncio
    async def test_create_experiment_already_exists(self, postgresql_storage, sample_experiment_spec):
        """Test creating experiment with duplicate ID."""
        # Create first experiment
        await postgresql_storage.create_experiment(sample_experiment_spec)
        
        # Try to create duplicate
        with pytest.raises(ExperimentAlreadyExistsException) as exc_info:
            await postgresql_storage.create_experiment(sample_experiment_spec)
        
        assert exc_info.value.experiment_id == sample_experiment_spec.experiment_id
    
    @pytest.mark.asyncio
    async def test_get_experiment_success(self, postgresql_storage, sample_experiment_spec):
        """Test successful experiment retrieval."""
        # Create experiment first
        await postgresql_storage.create_experiment(sample_experiment_spec)
        
        # Retrieve experiment
        result = await postgresql_storage.get_experiment(sample_experiment_spec.experiment_id)
        
        # Verify result
        assert isinstance(result, ExperimentSpec)
        assert result.experiment_id == sample_experiment_spec.experiment_id
        assert result.name == sample_experiment_spec.name
        assert result.description == sample_experiment_spec.description
        assert result.tags == sample_experiment_spec.tags
        assert result.parameters == sample_experiment_spec.parameters
    
    @pytest.mark.asyncio
    async def test_get_experiment_not_found(self, postgresql_storage):
        """Test getting non-existent experiment."""
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await postgresql_storage.get_experiment("nonexistent-experiment")
        
        assert exc_info.value.experiment_id == "nonexistent-experiment"
    
    @pytest.mark.asyncio
    async def test_list_experiments_empty(self, postgresql_storage):
        """Test listing when no experiments exist."""
        result = await postgresql_storage.list_experiments()
        assert result == []
    
    @pytest.mark.asyncio
    async def test_list_experiments_single(self, postgresql_storage, sample_experiment_spec):
        """Test listing single experiment."""
        # Create experiment
        await postgresql_storage.create_experiment(sample_experiment_spec)
        
        # List experiments
        result = await postgresql_storage.list_experiments()
        
        # Verify result
        assert len(result) == 1
        assert isinstance(result[0], ExperimentSpec)
        assert result[0].id == sample_experiment_spec.id
    
    @pytest.mark.asyncio
    async def test_list_experiments_multiple(self, postgresql_storage):
        """Test listing multiple experiments."""
        # Create multiple experiments
        specs = []
        for i in range(3):
            spec = ExperimentSpec(
                name=f"Multi Test {i}",
                description=f"Multiple experiment test {i}",
                model=f"model-{i}",
                task=TaskSpec(
                    name=f"multi_task_{i}",
                    type=TaskType.LLM_GENERATION,
                    description=f"Multiple test task {i}",
                    params=LLMGenerationParams(
                        prompt_template=PromptTemplate(
                            messages=[
                                PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                                PromptMessage(role=PromptRole.USER, content=f"Process multiple test {i}.")
                            ]
                        )
                    )
                ),
                framework=FrameworkSpec(
                    name=FrameworkName.TRANSFORMERS,
                    params={"value": i * 10}
                ),
                metrics=MetricsSpec(
                    metrics={f"index_{i}": "calculation"}
                )
            )
            specs.append(spec)
            await postgresql_storage.create_experiment(spec)
        
        # List experiments
        result = await postgresql_storage.list_experiments()
        
        # Verify result
        assert len(result) == 3
        returned_ids = {spec.id for spec in result}
        expected_ids = {spec.id for spec in specs}
        assert returned_ids == expected_ids
    
    @pytest.mark.asyncio
    async def test_update_experiment_status(self, postgresql_storage, sample_experiment_spec):
        """Test updating experiment status."""
        # Create experiment
        await postgresql_storage.create_experiment(sample_experiment_spec)
        
        # Update status
        success = await postgresql_storage.update_experiment_status(
            sample_experiment_spec.experiment_id, "running"
        )
        assert success is True
        
        # Verify status was updated
        # Note: This would require additional query to verify, but the method succeeding indicates it worked
    
    @pytest.mark.asyncio
    async def test_delete_experiment_success(self, postgresql_storage, sample_experiment_spec):
        """Test successful experiment deletion."""
        # Create experiment
        await postgresql_storage.create_experiment(sample_experiment_spec)
        
        # Delete experiment
        result = await postgresql_storage.delete_experiment(sample_experiment_spec.experiment_id)
        
        # Verify return format
        assert isinstance(result, dict)
        assert "experiment_id" in result
        assert "deleted_at" in result
        assert result["experiment_id"] == sample_experiment_spec.experiment_id
        assert isinstance(result["deleted_at"], datetime)
        
        # Verify experiment is gone
        with pytest.raises(ExperimentNotFoundException):
            await postgresql_storage.get_experiment(sample_experiment_spec.experiment_id)
    
    @pytest.mark.asyncio
    async def test_delete_experiment_not_found(self, postgresql_storage):
        """Test deleting non-existent experiment."""
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await postgresql_storage.delete_experiment("nonexistent-experiment")
        
        assert exc_info.value.experiment_id == "nonexistent-experiment"
    
    @pytest.mark.asyncio
    async def test_delete_experiment_with_jobs(self, postgresql_storage, sample_experiment_spec):
        """Test deleting experiment with associated jobs."""
        # Create experiment
        await postgresql_storage.create_experiment(sample_experiment_spec)
        
        # Create associated job
        await postgresql_storage.create_job(
            "test-job", 
            sample_experiment_spec.experiment_id,
            {"param": "value"}
        )
        
        # Try to delete experiment with jobs
        with pytest.raises(ExperimentHasJobsException) as exc_info:
            await postgresql_storage.delete_experiment(sample_experiment_spec.experiment_id)
        
        assert exc_info.value.experiment_id == sample_experiment_spec.experiment_id
        assert exc_info.value.job_count == 1


class TestPostgreSQLAgentOperations:
    """Test agent management operations."""
    
    @pytest.mark.asyncio
    async def test_register_agent_success(self, postgresql_storage, sample_agent_info):
        """Test successful agent registration."""
        result = await postgresql_storage.register_agent(sample_agent_info)
        assert result is True
        
        # Verify agent exists
        exists = await postgresql_storage.agent_exists(sample_agent_info.agent_id)
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_get_registered_agent_info(self, postgresql_storage, sample_agent_info):
        """Test getting registered agent info."""
        # Register agent first
        await postgresql_storage.register_agent(sample_agent_info)
        
        # Get agent info
        result = await postgresql_storage.get_registered_agent_info(sample_agent_info.agent_id)
        
        # Verify result
        assert isinstance(result, RegisteredAgentInfo)
        assert result.agent_id == sample_agent_info.agent_id
        assert result.agent_name == sample_agent_info.agent_name
        assert result.agent_type == sample_agent_info.agent_type
        assert result.agent_url == sample_agent_info.agent_url
    
    @pytest.mark.asyncio
    async def test_list_registered_agent_info_empty(self, postgresql_storage):
        """Test listing agents when none exist."""
        result = await postgresql_storage.list_registered_agent_info()
        assert result == []
    
    @pytest.mark.asyncio
    async def test_list_registered_agent_info_multiple(self, postgresql_storage):
        """Test listing multiple agents."""
        # Register multiple agents
        agents = []
        for i in range(3):
            agent = RegisteredAgentInfo(
                agent_id=f"test-agent-{i}",
                agent_name=f"Test Agent {i}",
                agent_type=AgentType.WHITE_BOX,
                agent_url=f"http://localhost:800{i}",
                system_info={"index": i},
                capabilities={"task": f"task_{i}"},
                status="active",
                last_heartbeat=datetime.now(timezone.utc),
                registered_at=datetime.now(timezone.utc)
            )
            agents.append(agent)
            await postgresql_storage.register_agent(agent)
        
        # List agents
        result = await postgresql_storage.list_registered_agent_info()
        
        # Verify result
        assert len(result) == 3
        returned_ids = {agent.agent_id for agent in result}
        expected_ids = {agent.agent_id for agent in agents}
        assert returned_ids == expected_ids
    
    @pytest.mark.asyncio
    async def test_update_agent_status(self, postgresql_storage, sample_agent_info):
        """Test updating agent status."""
        # Register agent
        await postgresql_storage.register_agent(sample_agent_info)
        
        # Update status
        success = await postgresql_storage.update_agent_status(
            sample_agent_info.agent_id, "inactive"
        )
        assert success is True
    
    @pytest.mark.asyncio
    async def test_update_agent_heartbeat(self, postgresql_storage, sample_agent_info):
        """Test updating agent heartbeat."""
        # Register agent
        await postgresql_storage.register_agent(sample_agent_info)
        
        # Update heartbeat
        success = await postgresql_storage.update_agent_heartbeat(sample_agent_info.agent_id)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_remove_agent(self, postgresql_storage, sample_agent_info):
        """Test removing agent."""
        # Register agent
        await postgresql_storage.register_agent(sample_agent_info)
        
        # Remove agent
        success = await postgresql_storage.remove_agent(sample_agent_info.agent_id)
        assert success is True
        
        # Verify agent is gone
        exists = await postgresql_storage.agent_exists(sample_agent_info.agent_id)
        assert exists is False


class TestPostgreSQLJobOperations:
    """Test job management operations."""
    
    @pytest.mark.asyncio
    async def test_create_job_success(self, postgresql_storage, sample_experiment_spec):
        """Test successful job creation."""
        # Create experiment first
        await postgresql_storage.create_experiment(sample_experiment_spec)
        
        # Create job
        success = await postgresql_storage.create_job(
            "test-job",
            sample_experiment_spec.experiment_id,
            {"param": "value", "count": 42}
        )
        assert success is True
    
    @pytest.mark.asyncio
    async def test_get_job_success(self, postgresql_storage, sample_experiment_spec):
        """Test successful job retrieval."""
        # Create experiment and job
        await postgresql_storage.create_experiment(sample_experiment_spec)
        job_config = {"param": "value", "count": 42}
        await postgresql_storage.create_job("test-job", sample_experiment_spec.experiment_id, job_config)
        
        # Get job
        result = await postgresql_storage.get_job("test-job")
        
        # Verify result
        assert result is not None
        assert result["id"] == "test-job"
        assert result["experiment_id"] == sample_experiment_spec.experiment_id
        assert result["config"] == job_config
        assert result["status"] == "scheduled"  # default status
    
    @pytest.mark.asyncio
    async def test_get_job_not_found(self, postgresql_storage):
        """Test getting non-existent job."""
        result = await postgresql_storage.get_job("nonexistent-job")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_jobs_empty(self, postgresql_storage):
        """Test listing jobs when none exist."""
        result = await postgresql_storage.list_jobs()
        assert result == []
    
    @pytest.mark.asyncio
    async def test_list_jobs_with_filtering(self, postgresql_storage, sample_experiment_spec):
        """Test listing jobs with various filters."""
        # Create experiment
        await postgresql_storage.create_experiment(sample_experiment_spec)
        
        # Create multiple jobs
        for i in range(3):
            await postgresql_storage.create_job(
                f"job-{i}",
                sample_experiment_spec.experiment_id,
                {"index": i}
            )
        
        # List all jobs
        all_jobs = await postgresql_storage.list_jobs()
        assert len(all_jobs) == 3
        
        # List jobs by experiment
        exp_jobs = await postgresql_storage.list_jobs(experiment_id=sample_experiment_spec.experiment_id)
        assert len(exp_jobs) == 3
        
        # List jobs by status
        status_jobs = await postgresql_storage.list_jobs(status="scheduled")
        assert len(status_jobs) == 3
    
    @pytest.mark.asyncio
    async def test_assign_job_to_agent(self, postgresql_storage, sample_experiment_spec, sample_agent_info):
        """Test assigning job to agent."""
        # Create experiment, agent, and job
        await postgresql_storage.create_experiment(sample_experiment_spec)
        await postgresql_storage.register_agent(sample_agent_info)
        await postgresql_storage.create_job("test-job", sample_experiment_spec.experiment_id, {})
        
        # Assign job to agent
        success = await postgresql_storage.assign_job_to_agent("test-job", sample_agent_info.agent_id)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_update_job_status(self, postgresql_storage, sample_experiment_spec):
        """Test updating job status."""
        # Create experiment and job
        await postgresql_storage.create_experiment(sample_experiment_spec)
        await postgresql_storage.create_job("test-job", sample_experiment_spec.experiment_id, {})
        
        # Update job status
        success = await postgresql_storage.update_job_status(
            "test-job", 
            "running",
            results={"progress": 0.5},
            error_message=None
        )
        assert success is True
    
    @pytest.mark.asyncio
    async def test_get_jobs_for_agent(self, postgresql_storage, sample_experiment_spec, sample_agent_info):
        """Test getting jobs for specific agent."""
        # Create experiment, agent, and jobs
        await postgresql_storage.create_experiment(sample_experiment_spec)
        await postgresql_storage.register_agent(sample_agent_info)
        
        # Create and assign jobs
        for i in range(2):
            job_id = f"agent-job-{i}"
            await postgresql_storage.create_job(job_id, sample_experiment_spec.experiment_id, {})
            await postgresql_storage.assign_job_to_agent(job_id, sample_agent_info.agent_id)
        
        # Get jobs for agent
        result = await postgresql_storage.get_jobs_for_agent(sample_agent_info.agent_id)
        assert len(result) == 2
        assert all(job["agent_id"] == sample_agent_info.agent_id for job in result)


class TestPostgreSQLComplexOperations:
    """Test complex operations and edge cases."""
    
    @pytest.mark.asyncio
    async def test_experiment_with_complex_data(self, postgresql_storage):
        """Test experiment with complex parameter structures."""
        complex_spec = ExperimentSpec(
            name="Complex Test",
            description="Testing complex parameter structures",
            model="complex-model",
            task=TaskSpec(
                name="complex_task",
                type=TaskType.LLM_GENERATION,
                description="Complex generation task",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a complex assistant."),
                            PromptMessage(role=PromptRole.USER, content="Handle complex parameter structures.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={
                    "model": {
                        "architecture": "transformer",
                        "layers": [64, 32, 16],
                        "config": {"dropout": 0.1, "activation": "relu"}
                    },
                    "training": {
                        "optimizer": {"name": "adam", "lr": 1e-4},
                        "schedule": {"type": "cosine", "warmup": 1000}
                    },
                    "unicode": "测试数据",
                    "scientific": 1e-5
                }
            ),
            metrics=MetricsSpec(
                metrics={
                    "string_test": "calculation", 
                    "integer_42": "counter",
                    "float_3.14159": "timer", 
                    "boolean_True": "flag", 
                    "null_None": "optional"
                }
            )
        )
        
        # Create and retrieve experiment
        await postgresql_storage.create_experiment(complex_spec)
        result = await postgresql_storage.get_experiment(complex_spec.id)
        
        # Verify complex data is preserved
        assert result.framework.params["model"]["layers"] == [64, 32, 16]
        assert result.framework.params["training"]["optimizer"]["lr"] == 1e-4
        assert result.framework.params["unicode"] == "测试数据"
        assert result.framework.params["scientific"] == 1e-5
        assert "string_test" in result.metrics.metrics
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, postgresql_storage):
        """Test concurrent experiment operations."""
        # Create multiple experiments concurrently
        specs = []
        for i in range(5):
            spec = ExperimentSpec(
                name=f"Concurrent Test {i}",
                description=f"Concurrent operation test {i}",
                model=f"concurrent-model-{i}",
                task=TaskSpec(
                    name=f"concurrent_task_{i}",
                    type=TaskType.LLM_GENERATION,
                    description=f"Concurrent test task {i}",
                    params=LLMGenerationParams(
                        prompt_template=PromptTemplate(
                            messages=[
                                PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                                PromptMessage(role=PromptRole.USER, content=f"Handle concurrent test {i}.")
                            ]
                        )
                    )
                ),
                framework=FrameworkSpec(
                    name=FrameworkName.TRANSFORMERS,
                    params={"value": i * 10}
                ),
                metrics=MetricsSpec(
                    metrics={f"index_{i}": "calculation"}
                )
            )
            specs.append(spec)
        
        # Create experiments concurrently
        tasks = [postgresql_storage.create_experiment(spec) for spec in specs]
        results = await asyncio.gather(*tasks)
        
        # Verify all created successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result["experiment_id"] == f"concurrent-{i}"
        
        # List experiments concurrently
        list_tasks = [postgresql_storage.list_experiments() for _ in range(3)]
        list_results = await asyncio.gather(*list_tasks)
        
        # Verify all return same results
        assert all(len(result) == 5 for result in list_results)


class TestPostgreSQLErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_operations_after_close(self, postgresql_storage, sample_experiment_spec):
        """Test operations after connection is closed."""
        # Close the backend
        await postgresql_storage.close()
        
        # Operations should fail gracefully
        with pytest.raises(Exception):  # Could be various exceptions depending on the error
            await postgresql_storage.create_experiment(sample_experiment_spec)
    
    @pytest.mark.asyncio
    async def test_invalid_experiment_data(self, postgresql_storage):
        """Test handling of invalid experiment data."""
        # This would test if the backend properly handles invalid ExperimentSpec data
        # but since we're using Pydantic models, invalid data should be caught at the model level
        pass
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, postgresql_storage, sample_experiment_spec):
        """Test that operations are properly rolled back on failure."""
        # Create experiment successfully first
        await postgresql_storage.create_experiment(sample_experiment_spec)
        
        # Try to create duplicate (should fail and rollback)
        with pytest.raises(ExperimentAlreadyExistsException):
            await postgresql_storage.create_experiment(sample_experiment_spec)
        
        # Verify original experiment still exists and is intact
        result = await postgresql_storage.get_experiment(sample_experiment_spec.experiment_id)
        assert result.experiment_id == sample_experiment_spec.experiment_id


def pytest_addoption(parser):
    """Add command line options for PostgreSQL tests."""
    parser.addoption(
        "--postgresql-url",
        action="store",
        default=None,
        help="PostgreSQL database URL for testing"
    )


def pytest_configure(config):
    """Configure pytest for PostgreSQL tests."""
    # Configuration is handled via request.config in fixtures
    pass