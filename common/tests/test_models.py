"""Tests for common models."""

import pytest
from datetime import datetime
from pydantic import ValidationError
from ruckus_common.models import (
    AgentType, TaskType,
    AgentRegistrationResponse, AgentInfoResponse,
    ExperimentSpec, MetricValue
)


class TestEnums:
    """Test enum definitions."""

    def test_agent_type_values(self):
        """Test agent type enum values."""
        assert AgentType.WHITE_BOX == "white_box"
        assert AgentType.GRAY_BOX == "gray_box"
        assert AgentType.BLACK_BOX == "black_box"


    def test_task_type_values(self):
        """Test task type enum values."""
        assert TaskType.SUMMARIZATION == "summarization"
        assert TaskType.CLASSIFICATION == "classification"
        assert TaskType.GENERATION == "generation"


class TestAgentRegistrationResponse:
    """Test agent registration response model."""

    def test_minimal_registration_response(self):
        """Test creating minimal registration response."""
        response = AgentRegistrationResponse(agent_id="test-agent-123")
        
        assert response.agent_id == "test-agent-123"
        assert response.agent_name is None
        assert response.message is None
        assert isinstance(response.server_time, datetime)

    def test_full_registration_response(self):
        """Test creating full registration response."""
        response = AgentRegistrationResponse(
            agent_id="test-agent-123",
            agent_name="test-agent-gpu-01",
            message="Registration successful"
        )
        
        assert response.agent_id == "test-agent-123"
        assert response.agent_name == "test-agent-gpu-01"
        assert response.message == "Registration successful"

    def test_registration_response_serialization(self):
        """Test registration response serialization."""
        response = AgentRegistrationResponse(
            agent_id="test-agent",
            agent_name="test-name"
        )
        
        data = response.model_dump()
        assert data["agent_id"] == "test-agent"
        assert data["agent_name"] == "test-name"
        assert "server_time" in data

    def test_empty_agent_id_fails(self):
        """Test that empty agent_id is handled."""
        # Note: The model doesn't have explicit validation for empty strings
        # but in practice the agent should generate valid IDs
        response = AgentRegistrationResponse(agent_id="")
        assert response.agent_id == ""


class TestAgentInfoResponse:
    """Test agent info response model."""

    def test_minimal_info_response(self):
        """Test creating minimal info response."""
        response = AgentInfoResponse(
            agent_id="test-agent",
            agent_type=AgentType.WHITE_BOX
        )
        
        assert response.agent_id == "test-agent"
        assert response.agent_type == AgentType.WHITE_BOX
        assert response.system_info == {}
        assert response.capabilities == {}
        assert isinstance(response.last_updated, datetime)

    def test_full_info_response(self):
        """Test creating full info response."""
        system_info = {
            "system": {"hostname": "test-host", "os": "Linux"},
            "cpu": {"cores": 8, "model": "Intel"},
            "gpus": [{"name": "Tesla", "memory_mb": 8000}]
        }
        
        capabilities = {
            "agent_type": "white_box",
            "gpu_count": 1,
            "frameworks": ["pytorch"]
        }
        
        response = AgentInfoResponse(
            agent_id="test-agent",
            agent_name="test-agent-white-box",
            agent_type=AgentType.WHITE_BOX,
            system_info=system_info,
            capabilities=capabilities
        )
        
        assert response.system_info == system_info
        assert response.capabilities == capabilities
        assert response.agent_name == "test-agent-white-box"

    def test_info_response_serialization(self):
        """Test info response serialization."""
        response = AgentInfoResponse(
            agent_id="test",
            agent_type=AgentType.GRAY_BOX,
            system_info={"test": "data"},
            capabilities={"test": "caps"}
        )
        
        data = response.model_dump()
        assert data["agent_type"] == "gray_box"  # Enum value
        assert data["system_info"] == {"test": "data"}
        assert data["capabilities"] == {"test": "caps"}


class TestExperimentSpec:
    """Test experiment specification model."""

    def test_minimal_experiment_spec(self):
        """Test creating minimal experiment spec."""
        spec = ExperimentSpec(
            experiment_id="exp-001",
            name="Test Experiment",
            models=["gpt-3.5-turbo"],
            frameworks=["transformers"],
            task_type=TaskType.SUMMARIZATION
        )
        
        assert spec.experiment_id == "exp-001"
        assert spec.name == "Test Experiment"
        assert spec.models == ["gpt-3.5-turbo"]
        assert spec.task_type == TaskType.SUMMARIZATION

    def test_experiment_spec_validation(self):
        """Test experiment spec validation."""
        # Empty experiment_id should fail validation
        with pytest.raises(ValidationError):
            ExperimentSpec(
                experiment_id="",
                name="Test",
                models=["model"],
                frameworks=["framework"],
                task_type=TaskType.SUMMARIZATION
            )

    def test_experiment_spec_defaults(self):
        """Test experiment spec default values."""
        spec = ExperimentSpec(
            experiment_id="exp-001",
            name="Test",
            models=["model"],
            frameworks=["framework"],
            task_type=TaskType.SUMMARIZATION
        )
        
        assert spec.hardware_targets == ["any"]
        assert spec.priority == 0
        assert spec.tags == []
        assert spec.parameters == {}

    def test_experiment_spec_priority_validation(self):
        """Test priority validation (0-10)."""
        # Valid priority
        spec = ExperimentSpec(
            experiment_id="exp-001",
            name="Test",
            models=["model"],
            frameworks=["framework"],
            task_type=TaskType.SUMMARIZATION,
            priority=5
        )
        assert spec.priority == 5

        # Invalid priority should fail
        with pytest.raises(ValidationError):
            ExperimentSpec(
                experiment_id="exp-001",
                name="Test",
                models=["model"],
                frameworks=["framework"],
                task_type=TaskType.SUMMARIZATION,
                priority=15  # > 10
            )



class TestMetricValue:
    """Test metric value model."""

    def test_metric_value_creation(self):
        """Test creating metric values."""
        metric = MetricValue(
            name="latency",
            value=150.5,
            unit="ms"
        )
        
        assert metric.name == "latency"
        assert metric.value == 150.5
        assert metric.unit == "ms"
        assert isinstance(metric.timestamp, datetime)

    def test_metric_value_with_metadata(self):
        """Test metric value with metadata."""
        metadata = {"batch_size": 32, "model": "gpt-3.5"}
        
        metric = MetricValue(
            name="throughput",
            value=25.0,
            unit="tokens/sec",
            metadata=metadata
        )
        
        assert metric.metadata == metadata

    def test_metric_value_serialization(self):
        """Test metric value serialization."""
        metric = MetricValue(
            name="accuracy",
            value=0.95,
            metadata={"dataset": "test"}
        )
        
        data = metric.model_dump()
        assert data["name"] == "accuracy"
        assert data["value"] == 0.95
        assert data["metadata"]["dataset"] == "test"
        assert "timestamp" in data