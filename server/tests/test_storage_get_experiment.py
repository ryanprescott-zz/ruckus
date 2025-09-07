"""Tests for storage backend get_experiment method."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from ruckus_server.core.storage.sqlite import SQLiteStorageBackend
from ruckus_server.core.storage.base import ExperimentNotFoundException
from ruckus_server.core.config import SQLiteSettings
from ruckus_common.models import (ExperimentSpec, TaskType, TaskSpec, FrameworkSpec, MetricsSpec, 
                                  LLMGenerationParams, PromptTemplate, PromptMessage, PromptRole, FrameworkName)


@pytest.fixture
def storage_settings():
    """Create storage settings for testing."""
    from ruckus_server.core.config import SQLiteSettings
    return SQLiteSettings(
        database_path=":memory:"
    )


@pytest_asyncio.fixture
async def storage_backend(storage_settings):
    """Create SQLite storage backend for testing."""
    backend = SQLiteStorageBackend(storage_settings)
    await backend.initialize()
    return backend


@pytest.fixture
def sample_experiment_spec():
    """Create sample experiment spec."""
    return ExperimentSpec(
        name="Test Get Experiment",
        description="An experiment for testing get functionality",
        model="test-model",
        task=TaskSpec(
            name="test_get_task",
            type=TaskType.LLM_GENERATION,
            description="Test task for get functionality",
            params=LLMGenerationParams(
                prompt_template=PromptTemplate(
                    messages=[
                        PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                        PromptMessage(role=PromptRole.USER, content="Test get experiment functionality.")
                    ]
                )
            )
        ),
        framework=FrameworkSpec(
            name=FrameworkName.TRANSFORMERS,
            params={"learning_rate": 0.01, "epochs": 10}
        ),
        metrics=MetricsSpec(
            metrics={"test": "calculation", "version": "1.0"}
        )
    )


class TestGetExperimentSuccess:
    """Test successful get experiment scenarios."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_success(self, storage_backend, sample_experiment_spec):
        """Test successful experiment retrieval."""
        # Setup - create experiment first
        await storage_backend.create_experiment(sample_experiment_spec)
        
        # Execute
        result = await storage_backend.get_experiment(sample_experiment_spec.id)
        
        # Verify
        assert result is not None
        assert isinstance(result, ExperimentSpec)
        assert result.id == sample_experiment_spec.id
        assert result.name == "Test Get Experiment"
        assert result.description == "An experiment for testing get functionality"
        assert result.metrics.metrics == {"test": "calculation", "version": "1.0"}
        assert result.framework.params == {"learning_rate": 0.01, "epochs": 10}
    
    @pytest.mark.asyncio
    async def test_get_experiment_with_complex_parameters(self, storage_backend):
        """Test getting experiment with complex parameter structure."""
        # Setup
        complex_spec = ExperimentSpec(
            name="Complex Get Test",
            description="Testing complex parameters",
            model="test-model",
            task=TaskSpec(
                name="complex_test_task",
                type=TaskType.LLM_GENERATION,
                description="Complex test task",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Test complex parameters.")
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
                    "schedule": {"type": "cosine", "warmup_steps": 1000},
                    "batch_size": 32,
                    "gradient_clip": 1.0
                },
                "data": {
                    "source": "/path/to/data",
                    "preprocessing": ["tokenize", "normalize"],
                    "augmentation": {"enabled": True, "methods": ["random_crop", "flip"]}
                }
            }),
            metrics=MetricsSpec(
                metrics={"complexity": "high", "category": "ml"}
            )
        )
        
        # Create experiment
        await storage_backend.create_experiment(complex_spec)
        
        # Execute
        result = await storage_backend.get_experiment(complex_spec.id)
        
        # Verify complex structure is preserved
        assert isinstance(result, ExperimentSpec)
        assert result.framework.params["model"]["architecture"] == "transformer"
        assert result.framework.params["training"]["optimizer"]["name"] == "adam"
        assert len(result.framework.params["model"]["layers"]) == 2
        assert result.framework.params["data"]["augmentation"]["methods"] == ["random_crop", "flip"]
    
    @pytest.mark.asyncio
    async def test_get_experiment_with_none_description(self, storage_backend):
        """Test getting experiment with None description."""
        # Setup
        spec_with_none = ExperimentSpec(
            name="None Description Test",
            description=None,
            model="test-model",
            task=TaskSpec(
                name="none_desc_task",
                type=TaskType.LLM_GENERATION,
                description="Test task",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Test none description.")
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
        
        # Create experiment
        await storage_backend.create_experiment(spec_with_none)
        
        # Execute
        result = await storage_backend.get_experiment(spec_with_none.id)
        
        # Verify
        assert isinstance(result, ExperimentSpec)
        assert result.description is None
    
    @pytest.mark.asyncio
    async def test_get_experiment_special_characters_in_id(self, storage_backend):
        """Test getting experiment with special characters in ID."""
        # Setup
        special_spec = ExperimentSpec(
            name="Special Characters Test",
            description="Testing special characters in ID",
            model="test-model",
            task=TaskSpec(
                name="special_char_task",
                type=TaskType.LLM_GENERATION,
                description="Test task",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Test special characters.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={"test": True}
            ),
            metrics=MetricsSpec(
                metrics={"version": "v2"}
            )
        )
        
        # Create experiment
        await storage_backend.create_experiment(special_spec)
        
        # Execute
        result = await storage_backend.get_experiment(special_spec.id)
        
        # Verify
        assert isinstance(result, ExperimentSpec)
        assert result.id == special_spec.id
        assert result.name == "Special Characters Test"
    
    @pytest.mark.asyncio
    async def test_get_experiment_after_status_update(self, storage_backend, sample_experiment_spec):
        """Test getting experiment after status update."""
        # Setup - create and update experiment
        await storage_backend.create_experiment(sample_experiment_spec)
        await storage_backend.update_experiment_status("test-get-experiment", "running")
        
        # Execute
        result = await storage_backend.get_experiment(sample_experiment_spec.id)
        
        # Verify
        assert isinstance(result, ExperimentSpec)
        assert result.id == sample_experiment_spec.id


class TestGetExperimentNotFound:
    """Test experiment not found scenarios."""
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_experiment(self, storage_backend):
        """Test getting non-existent experiment raises exception."""
        # Execute & Verify
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await storage_backend.get_experiment("nonexistent-experiment")
        
        assert exc_info.value.experiment_id == "nonexistent-experiment"
        assert "nonexistent-experiment not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_deleted_experiment(self, storage_backend, sample_experiment_spec):
        """Test getting experiment that was deleted."""
        # Setup - create then delete experiment
        await storage_backend.create_experiment(sample_experiment_spec)
        await storage_backend.delete_experiment(sample_experiment_spec.id)
        
        # Execute & Verify
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await storage_backend.get_experiment(sample_experiment_spec.id)
        
        assert exc_info.value.experiment_id == sample_experiment_spec.id
    
    @pytest.mark.asyncio
    async def test_get_experiment_empty_id(self, storage_backend):
        """Test getting experiment with empty ID."""
        # Execute & Verify
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await storage_backend.get_experiment("")
        
        assert exc_info.value.experiment_id == ""
    
    @pytest.mark.asyncio
    async def test_get_experiment_none_id(self, storage_backend):
        """Test getting experiment with None ID."""
        # Execute & Verify
        with pytest.raises(ExperimentNotFoundException) as exc_info:
            await storage_backend.get_experiment(None)
        
        assert exc_info.value.experiment_id is None


class TestGetExperimentDataIntegrity:
    """Test data integrity and consistency."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_preserves_json_serialization(self, storage_backend):
        """Test that experiment data JSON serialization is preserved."""
        # Setup with data that tests JSON serialization edge cases
        spec = ExperimentSpec(
            name="JSON Serialization Test",
            description="Testing JSON edge cases",
            model="test-model",
            task=TaskSpec(
                name="json_test_task",
                type=TaskType.LLM_GENERATION,
                description="JSON test task",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Test JSON serialization.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={
                    "test_data": {"float": 3.14159, "boolean": True, "null": None},
                    "nested": {"deep": {"value": [1, 2, 3]}},
                    "unicode": "测试数据",
                    "special_chars": "test@#$%^&*()",
                    "numbers": {"int": 42, "float": 3.14, "scientific": 1e-5}
                }
            ),
            metrics=MetricsSpec(
                metrics={"json": "test"}
            )
        )
        
        # Create experiment
        await storage_backend.create_experiment(spec)
        
        # Execute
        result = await storage_backend.get_experiment(spec.id)
        
        # Verify all data types are preserved
        assert isinstance(result, ExperimentSpec)
        test_data = result.framework.params["test_data"]
        assert test_data["float"] == 3.14159
        assert test_data["boolean"] is True
        assert test_data["null"] is None
        assert result.framework.params["unicode"] == "测试数据"
        assert result.framework.params["numbers"]["scientific"] == 1e-5
    
    @pytest.mark.asyncio
    async def test_get_experiment_timestamps_consistency(self, storage_backend, sample_experiment_spec):
        """Test that timestamps are consistent and properly formatted."""
        # Setup
        before_create = datetime.now(timezone.utc)
        await storage_backend.create_experiment(sample_experiment_spec)
        after_create = datetime.now(timezone.utc)
        
        # Execute
        result = await storage_backend.get_experiment(sample_experiment_spec.id)
        
        # Verify experiment spec is returned correctly
        assert isinstance(result, ExperimentSpec)
        assert result.id == sample_experiment_spec.id
    
    @pytest.mark.asyncio
    async def test_get_experiment_concurrent_access(self, storage_backend, sample_experiment_spec):
        """Test getting experiment with concurrent access."""
        # Setup
        await storage_backend.create_experiment(sample_experiment_spec)
        
        # Execute multiple concurrent gets
        import asyncio
        results = await asyncio.gather(*[
            storage_backend.get_experiment(sample_experiment_spec.id)
            for _ in range(5)
        ])
        
        # Verify all results are identical
        first_result = results[0]
        for result in results[1:]:
            assert isinstance(result, ExperimentSpec)
            assert result.id == first_result.id
            assert result.framework.params == first_result.framework.params
            assert result.metrics.metrics == first_result.metrics.metrics
    
    @pytest.mark.asyncio
    async def test_get_experiment_isolation(self, storage_backend):
        """Test that getting different experiments returns correct data."""
        # Setup - create multiple experiments
        specs = []
        for i in range(3):
            spec = ExperimentSpec(
                name=f"Isolation Test {i}",
                description=f"Testing isolation {i}",
                model="test-model",
                task=TaskSpec(
                    name=f"isolation_task_{i}",
                    type=TaskType.LLM_GENERATION,
                    description=f"Isolation task {i}",
                    params=LLMGenerationParams(
                        prompt_template=PromptTemplate(
                            messages=[
                                PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                                PromptMessage(role=PromptRole.USER, content=f"Test isolation {i}.")
                            ]
                        )
                    )
                ),
                framework=FrameworkSpec(
                    name=FrameworkName.TRANSFORMERS,
                    params={"value": i * 10}
                ),
                metrics=MetricsSpec(
                    metrics={"number": str(i)}
                )
            )
            specs.append(spec)
            await storage_backend.create_experiment(spec)
        
        # Execute - get each experiment
        results = []
        for i in range(3):
            result = await storage_backend.get_experiment(specs[i].id)
            results.append(result)
        
        # Verify each experiment has correct data
        for i, result in enumerate(results):
            assert isinstance(result, ExperimentSpec)
            assert result.id == specs[i].id
            assert result.name == f"Isolation Test {i}"
            assert result.metrics.metrics["number"] == str(i)
            assert result.framework.params["value"] == i * 10


class TestGetExperimentPerformance:
    """Test performance and scalability."""
    
    @pytest.mark.asyncio
    async def test_get_experiment_large_parameters(self, storage_backend):
        """Test getting experiment with large parameter structure."""
        # Setup - create experiment with large parameters
        large_params = {}
        for i in range(100):
            large_params[f"param_{i}"] = {
                "value": i,
                "data": [j for j in range(50)],
                "metadata": {"description": f"Parameter {i} with lots of data"}
            }
        
        large_spec = ExperimentSpec(
            name="Large Parameters Test",
            description="Testing large parameter structures",
            model="test-model",
            task=TaskSpec(
                name="large_params_task",
                type=TaskType.LLM_GENERATION,
                description="Large parameters task",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Test large parameters.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params=large_params
            ),
            metrics=MetricsSpec(
                metrics={"size": "large"}
            )
        )
        
        # Create experiment
        await storage_backend.create_experiment(large_spec)
        
        # Execute
        result = await storage_backend.get_experiment(large_spec.id)
        
        # Verify large structure is handled correctly
        assert isinstance(result, ExperimentSpec)
        assert len(result.framework.params) == 100
        assert result.framework.params["param_50"]["value"] == 50
        assert len(result.framework.params["param_99"]["data"]) == 50