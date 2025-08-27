"""Test the end-to-end orchestration workflow."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from ruckus_common.models import (
    ExperimentSpec, ExperimentSubmission, ParameterGrid, ParameterSweep,
    AgentRequirements, TaskType, AgentType, ExpectedOutput, AggregationStrategy,
    RegisteredAgentInfo
)
from ruckus_server.core.orchestrator import ExperimentOrchestrator
from ruckus_server.core.storage.base import StorageBackend


class MockStorageBackend(StorageBackend):
    """Mock storage backend for testing."""
    
    def __init__(self):
        self.agents = []
        self.experiments = {}
        self.jobs = {}
        self.job_specs = {}  # For storing JobSpec objects
        
    async def initialize(self):
        pass
        
    async def close(self):
        pass
        
    async def health_check(self):
        return True
    
    # Agent management (minimal implementation)
    async def register_agent(self, agent_info): return True
    async def update_agent_status(self, agent_id: str, status: str): return True
    async def update_agent_heartbeat(self, agent_id: str): return True
    async def agent_exists(self, agent_id: str): return True
    async def remove_agent(self, agent_id: str): return True
    
    async def get_registered_agent_info(self, agent_id: str):
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    async def list_registered_agent_info(self):
        return self.agents
    
    # Experiment management
    async def create_experiment(self, experiment_id: str, name: str, description: str, config): 
        self.experiments[experiment_id] = {
            "id": experiment_id,
            "name": name, 
            "description": description,
            "config": config
        }
        return True
        
    async def update_experiment_status(self, experiment_id: str, status: str): return True
    async def get_experiment(self, experiment_id: str): return self.experiments.get(experiment_id)
    async def list_experiments(self): return list(self.experiments.values())
    async def delete_experiment(self, experiment_id: str): return True
    
    # Job management
    async def create_job(self, job_id: str, experiment_id: str, config): 
        self.jobs[job_id] = {
            "id": job_id,
            "experiment_id": experiment_id,
            "config": config,
            "status": "queued"
        }
        return True
        
    async def assign_job_to_agent(self, job_id: str, agent_id: str): return True
    async def update_job_status(self, job_id: str, status: str, results=None, error_message=None): return True
    async def get_job(self, job_id: str): return self.jobs.get(job_id)
    async def list_jobs(self, experiment_id=None, agent_id=None, status=None): return list(self.jobs.values())
    async def get_jobs_for_agent(self, agent_id: str, status=None): return []
    
    # JobSpec management for orchestration
    async def create_job_spec(self, job_spec):
        """Store a JobSpec object."""
        self.job_specs[job_spec.job_id] = job_spec
        return True
    
    async def get_job_spec(self, job_id: str):
        """Get a JobSpec object by ID."""
        return self.job_specs.get(job_id)
    
    async def update_job_spec(self, job_spec):
        """Update a JobSpec object."""
        self.job_specs[job_spec.job_id] = job_spec
        return True


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = MockStorageBackend()
    
    # Add some mock registered agents
    storage.agents = [
        RegisteredAgentInfo(
            agent_id="agent-gpu-1",
            agent_name="GPU Agent 1", 
            agent_type=AgentType.WHITE_BOX,
            agent_url="http://gpu-agent-1:8000",
            system_info={
                "gpus": [{"name": "Tesla V100", "memory_mb": 16384}],
                "frameworks": [{"name": "vllm", "version": "0.2.0"}, {"name": "pytorch", "version": "2.0"}],
                "models": [
                    {"name": "llama-7b", "path": "/models/llama-7b"},
                    {"name": "mistral-7b", "path": "/models/mistral-7b"}
                ]
            },
            capabilities={
                "agent_type": "white_box",
                "gpu_count": 1,
                "frameworks": ["vllm", "pytorch"],
                "max_concurrent_jobs": 2
            },
            registered_at=datetime.utcnow()
        ),
        RegisteredAgentInfo(
            agent_id="agent-cpu-1",
            agent_name="CPU Agent 1",
            agent_type=AgentType.GRAY_BOX,
            agent_url="http://cpu-agent-1:8000", 
            system_info={
                "gpus": [],
                "frameworks": [{"name": "transformers", "version": "4.0"}],
                "models": [{"name": "bert-base", "path": "/models/bert-base"}]
            },
            capabilities={
                "agent_type": "gray_box",
                "gpu_count": 0,
                "frameworks": ["transformers"],
                "max_concurrent_jobs": 1
            },
            registered_at=datetime.utcnow()
        )
    ]
    
    return storage


@pytest.fixture
def orchestrator(mock_storage):
    """Create an orchestrator with mock storage."""
    return ExperimentOrchestrator(mock_storage)


@pytest.mark.asyncio
async def test_experiment_planning_workflow(orchestrator):
    """Test the experiment planning and job breakdown workflow."""
    
    # Create an experiment with parameter sweeps
    experiment_spec = ExperimentSpec(
        experiment_id="test-exp-001",
        name="Temperature and Max Tokens Sweep",
        description="Test parameter sweep functionality",
        models=["llama-7b", "mistral-7b"],
        task_type=TaskType.GENERATION,
        parameter_grid=ParameterGrid(
            parameters={
                "temperature": ParameterSweep(name="temperature", values=[0.7, 0.9]),
                "max_tokens": ParameterSweep(name="max_tokens", values=[50, 100])
            },
            samples_per_config=2  # 2 runs per config for statistical significance
        ),
        base_parameters={
            "prompt": "Write a short story about:",
            "seed": 42
        },
        agent_requirements=AgentRequirements(
            required_frameworks=["vllm"],
            min_gpu_count=1,
            required_models=["llama-7b", "mistral-7b"]
        ),
        expected_output=ExpectedOutput(
            required_metrics=["latency", "throughput"],
            optional_metrics=["memory_usage", "gpu_utilization"]
        )
    )
    
    # Submit experiment for planning (dry run)
    submission = ExperimentSubmission(
        spec=experiment_spec,
        dry_run=True,  # Just plan, don't execute
        submit_immediately=False
    )
    
    response = await orchestrator.submit_experiment(submission)
    
    # Verify planning results
    assert response.experiment_id == "test-exp-001"
    assert response.status == "planned"
    assert response.estimated_jobs == 16  # 2 models × 4 param configs × 2 samples = 16 jobs
    assert response.estimated_duration_hours is not None
    assert response.estimated_duration_hours > 0
    
    # Check that experiment was stored
    execution = await orchestrator.get_experiment_execution("test-exp-001")
    assert execution is not None
    assert execution.total_jobs == 16
    assert len(execution.planned_jobs) == 16
    assert execution.status == "planned_only"
    
    print(f"✅ Experiment planned successfully:")
    print(f"   - {response.estimated_jobs} total jobs")
    print(f"   - Estimated duration: {response.estimated_duration_hours:.2f} hours")
    print(f"   - Job breakdown: {len(experiment_spec.models)} models × {len(experiment_spec.parameter_grid.generate_configurations())} configs")


@pytest.mark.asyncio
async def test_agent_compatibility_scoring(orchestrator):
    """Test agent compatibility scoring for job requirements."""
    
    # Get the mock agents
    agents = await orchestrator.storage.list_registered_agent_info()
    gpu_agent = agents[0]  # GPU agent with vllm
    cpu_agent = agents[1]  # CPU agent with transformers only
    
    # Test requirements that match GPU agent perfectly
    gpu_requirements = AgentRequirements(
        required_frameworks=["vllm"],
        min_gpu_count=1,
        required_models=["llama-7b"]
    )
    
    gpu_score = await orchestrator._score_agent_compatibility(gpu_agent, gpu_requirements)
    cpu_score = await orchestrator._score_agent_compatibility(cpu_agent, gpu_requirements)
    
    # GPU agent should score much higher
    assert gpu_score.score > cpu_score.score
    assert gpu_score.score >= 45.0  # Should get points for model, framework, and GPU
    assert cpu_score.score < 10.0   # Should get penalty for missing requirements
    
    # Check scoring details
    assert "llama-7b" in gpu_score.compatible_models
    assert len(cpu_score.missing_requirements) > 0
    
    print(f"✅ Agent compatibility scoring:")
    print(f"   - GPU Agent score: {gpu_score.score:.1f}")
    print(f"     - Compatible models: {gpu_score.compatible_models}")
    print(f"     - Missing requirements: {gpu_score.missing_requirements}")
    print(f"   - CPU Agent score: {cpu_score.score:.1f}")  
    print(f"     - Compatible models: {cpu_score.compatible_models}")
    print(f"     - Missing requirements: {cpu_score.missing_requirements}")


@pytest.mark.asyncio
async def test_job_matching_workflow(orchestrator):
    """Test the job-to-agent matching workflow."""
    
    # Create experiment and get some planned jobs
    experiment_spec = ExperimentSpec(
        experiment_id="test-matching-001",
        name="Agent Matching Test",
        models=["llama-7b"],
        task_type=TaskType.GENERATION,
        parameter_grid=ParameterGrid(
            parameters={
                "temperature": ParameterSweep(name="temperature", values=[0.8])
            }
        ),
        agent_requirements=AgentRequirements(
            required_frameworks=["vllm"],
            required_models=["llama-7b"]
        )
    )
    
    # Plan the experiment
    planned_jobs = await orchestrator._plan_experiment(experiment_spec)
    assert len(planned_jobs) == 1
    
    job_id = planned_jobs[0].job_id
    
    # Test agent matching
    agent_assignments = await orchestrator._match_jobs_to_agents(
        [job_id], experiment_spec.agent_requirements
    )
    
    assert job_id in agent_assignments
    compatible_agents = agent_assignments[job_id]
    assert len(compatible_agents) > 0
    
    # Best agent should be the GPU agent (higher compatibility)
    best_agent = compatible_agents[0]
    assert best_agent.agent_id == "agent-gpu-1"
    assert best_agent.score > 0
    
    print(f"✅ Job matching workflow:")
    print(f"   - Job {job_id} matched to {len(compatible_agents)} compatible agents")
    print(f"   - Best agent: {best_agent.agent_id} (score: {best_agent.score:.1f})")


@pytest.mark.asyncio  
async def test_complete_orchestration_workflow(orchestrator):
    """Test the complete orchestration workflow from submission to job dispatch."""
    
    # Create a simple experiment
    experiment_spec = ExperimentSpec(
        experiment_id="test-complete-001",
        name="Complete Workflow Test",
        models=["llama-7b"],
        task_type=TaskType.GENERATION,
        parameter_grid=ParameterGrid(
            parameters={
                "temperature": ParameterSweep(name="temperature", values=[0.7, 0.9])
            }
        ),
        agent_requirements=AgentRequirements(
            required_frameworks=["vllm"],
            required_models=["llama-7b"]
        )
    )
    
    # Submit for immediate execution
    submission = ExperimentSubmission(
        spec=experiment_spec,
        submit_immediately=True,
        dry_run=False
    )
    
    response = await orchestrator.submit_experiment(submission)
    
    assert response.status == "running"
    assert response.estimated_jobs == 2  # 1 model × 2 temperature values
    
    # Check execution state
    execution = await orchestrator.get_experiment_execution("test-complete-001")
    assert execution is not None
    assert execution.status == "running"
    assert execution.total_jobs == 2
    
    # Give some time for background job dispatch to start
    await asyncio.sleep(0.1)
    
    print(f"✅ Complete orchestration workflow:")
    print(f"   - Experiment submitted and started: {response.experiment_id}")
    print(f"   - Status: {response.status}")
    print(f"   - Jobs planned: {response.estimated_jobs}")
    print(f"   - Background execution started")
    
    # Note: In a real test, we'd wait for jobs to be dispatched and track progress
    # For now, we've validated the core planning and startup workflow


@pytest.mark.asyncio
async def test_result_aggregation_workflow(orchestrator):
    """Test experiment result aggregation when jobs complete."""
    from ruckus_common.models import JobStatus, JobSpec
    
    # Create and submit experiment
    experiment_spec = ExperimentSpec(
        experiment_id="test-results-001",
        name="Result Aggregation Test",
        models=["llama-7b", "mistral-7b"],
        task_type=TaskType.GENERATION,
        parameter_grid=ParameterGrid(
            parameters={
                "temperature": ParameterSweep(name="temperature", values=[0.7, 0.9])
            }
        ),
        agent_requirements=AgentRequirements(
            required_frameworks=["vllm"],
            required_models=["llama-7b", "mistral-7b"]
        )
    )
    
    submission = ExperimentSubmission(spec=experiment_spec, dry_run=True)
    response = await orchestrator.submit_experiment(submission)
    
    # Get planned jobs and manually set experiment to running for testing
    execution = await orchestrator.get_experiment_execution("test-results-001")
    assert execution is not None
    assert len(execution.planned_jobs) == 4  # 2 models × 2 temperatures
    
    # Manually set up execution state for completion testing
    execution.status = "running"
    execution.queued_jobs = execution.planned_jobs.copy()
    execution.running_jobs = {}
    execution.completed_jobs = []
    execution.failed_jobs = []
    
    # Mock job results - simulate jobs completing with performance metrics
    mock_results = [
        {"latency": 150, "throughput": 12.5, "memory_usage": 8.2},
        {"latency": 180, "throughput": 10.8, "memory_usage": 8.5},
        {"latency": 140, "throughput": 13.2, "memory_usage": 7.9},
        {"latency": 170, "throughput": 11.5, "memory_usage": 8.1}
    ]
    
    # Simulate job completions
    for i, job_id in enumerate(execution.planned_jobs):
        # First simulate job being dispatched (move from queued to running)
        execution.queued_jobs.remove(job_id)
        execution.running_jobs[job_id] = "agent-gpu-1"  # Mock agent assignment
        
        # Create mock job spec with results
        job_spec = JobSpec(
            job_id=job_id,
            experiment_id=execution.experiment_id,
            model="llama-7b" if i < 2 else "mistral-7b",
            framework="vllm",
            hardware_target="gpu",
            task_type=TaskType.GENERATION,
            task_config={},
            parameters={"temperature": 0.7 if i % 2 == 0 else 0.9},
            timeout_seconds=3600,
            max_retries=2,
            priority=1,
            status=JobStatus.COMPLETED,
            results=mock_results[i]
        )
        
        # Add to storage mock
        await orchestrator.storage.create_job_spec(job_spec)
        
        # Simulate job completion update
        await orchestrator.handle_job_update(job_id, JobStatus.COMPLETED, mock_results[i])
    
    # Check that experiment completed and results were aggregated
    final_execution = await orchestrator.get_experiment_execution("test-results-001")
    assert final_execution.status == "completed"
    assert final_execution.aggregated_results is not None
    
    # Verify aggregation structure
    aggregated = final_execution.aggregated_results
    assert "job_results" in aggregated
    assert "summary" in aggregated
    assert len(aggregated["job_results"]) == 4
    
    # Check summary statistics
    summary = aggregated["summary"]
    assert summary["status"] == "success"
    assert "metric_summaries" in summary
    assert "model_comparison" in summary
    assert summary["unique_models_tested"] == 2
    assert summary["total_successful_jobs"] == 4
    
    # Verify metric calculations
    metric_summaries = summary["metric_summaries"]
    assert "latency" in metric_summaries
    assert "throughput" in metric_summaries
    
    latency_stats = metric_summaries["latency"]
    expected_mean = (150 + 180 + 140 + 170) / 4  # 160
    assert abs(latency_stats["mean"] - expected_mean) < 0.1
    assert latency_stats["min"] == 140
    assert latency_stats["max"] == 180
    
    print(f"✅ Result aggregation workflow:")
    print(f"   - Experiment completed with {len(aggregated['job_results'])} job results")
    print(f"   - Summary status: {summary['status']}")
    print(f"   - Average latency: {latency_stats['mean']:.1f}ms")
    print(f"   - Models tested: {summary['unique_models_tested']}")
    print(f"   - Model comparison: {list(summary['model_comparison'].keys())}")


if __name__ == "__main__":
    """Run workflow demonstrations."""
    
    async def run_demonstrations():
        print("🎯 RUCKUS Orchestration Workflow Demonstrations\n")
        
        # Setup
        storage = MockStorageBackend()
        storage.agents = [
            RegisteredAgentInfo(
                agent_id="demo-gpu-agent",
                agent_name="Demo GPU Agent",
                agent_type=AgentType.WHITE_BOX,
                agent_url="http://demo-gpu:8000",
                system_info={
                    "gpus": [{"name": "Tesla V100", "memory_mb": 16384}],
                    "frameworks": [{"name": "vllm", "version": "0.2.0"}],
                    "models": [{"name": "llama-7b"}, {"name": "mistral-7b"}]
                },
                capabilities={"gpu_count": 1, "frameworks": ["vllm"]},
                registered_at=datetime.utcnow()
            )
        ]
        
        orchestrator = ExperimentOrchestrator(storage)
        
        # Demo: Parameter Grid Generation
        print("📊 Parameter Grid Generation:")
        param_grid = ParameterGrid(
            parameters={
                "temperature": ParameterSweep(name="temperature", values=[0.7, 0.8, 0.9]),
                "max_tokens": ParameterSweep(name="max_tokens", values=[50, 100]),
                "top_p": ParameterSweep(name="top_p", values=[0.9, 0.95])
            },
            samples_per_config=2
        )
        
        configs = param_grid.generate_configurations()
        print(f"   Generated {len(configs)} parameter configurations")
        print(f"   Example config: {configs[0]}")
        
        # Demo: Experiment Planning
        print(f"\n🎭 Experiment Planning:")
        experiment_spec = ExperimentSpec(
            experiment_id="demo-exp-001",
            name="LLM Parameter Sweep Demo",
            models=["llama-7b", "mistral-7b"], 
            task_type=TaskType.GENERATION,
            parameter_grid=param_grid,
            agent_requirements=AgentRequirements(required_frameworks=["vllm"])
        )
        
        total_jobs = experiment_spec.estimate_job_count()
        print(f"   Experiment will generate {total_jobs} total jobs")
        print(f"   Breakdown: {len(experiment_spec.models)} models × {len(configs)} configs = {total_jobs}")
        
        print(f"\n✨ Orchestration system ready for production use!")
    
    asyncio.run(run_demonstrations())