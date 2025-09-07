"""Tests for storage backend list_experiments method."""

import pytest
import pytest_asyncio
from ruckus_server.core.storage.sqlite import SQLiteStorageBackend
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
def sample_experiment_specs():
    """Create sample experiment specs for testing."""
    return [
        ExperimentSpec(
            name="Test List Experiment 1",
            description="First experiment for list testing",
            model="test-model-1",
            task=TaskSpec(
                name="list_task_1",
                type=TaskType.LLM_GENERATION,
                description="List test task 1",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Test list experiment 1.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={"learning_rate": 0.01, "epochs": 10}
            ),
            metrics=MetricsSpec(
                metrics={"test": "classification", "version": "1.0"}
            )
        ),
        ExperimentSpec(
            name="Test List Experiment 2",
            description="Second experiment for list testing",
            model="test-model-2",
            task=TaskSpec(
                name="list_task_2",
                type=TaskType.LLM_GENERATION,
                description="List test task 2",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Test list experiment 2.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={"learning_rate": 0.001, "epochs": 20}
            ),
            metrics=MetricsSpec(
                metrics={"production": "classification", "version": "2.0"}
            )
        ),
        ExperimentSpec(
            name="Test List Experiment 3",
            description="Third experiment for list testing",
            model="test-model-3",
            task=TaskSpec(
                name="list_task_3",
                type=TaskType.LLM_GENERATION,
                description="List test task 3",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Test list experiment 3.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={"learning_rate": 0.005, "epochs": 15}
            ),
            metrics=MetricsSpec(
                metrics={"test": "classification", "version": "1.5"}
            )
        )
    ]


class TestListExperimentsSuccess:
    """Test successful list experiments scenarios."""
    
    @pytest.mark.asyncio
    async def test_list_experiments_empty(self, storage_backend):
        """Test listing when no experiments exist."""
        # Execute
        result = await storage_backend.list_experiments()
        
        # Verify
        assert result == []
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_list_experiments_single(self, storage_backend, sample_experiment_specs):
        """Test listing single experiment."""
        # Setup - create one experiment
        spec = sample_experiment_specs[0]
        await storage_backend.create_experiment(spec)
        
        # Execute
        result = await storage_backend.list_experiments()
        
        # Verify
        assert len(result) == 1
        assert isinstance(result[0], ExperimentSpec)
        returned_spec = result[0]
        
        assert returned_spec.id == spec.id
        assert returned_spec.name == spec.name
        assert returned_spec.description == spec.description
        assert returned_spec.metrics.metrics == spec.metrics.metrics
        assert returned_spec.framework.params == spec.framework.params
    
    @pytest.mark.asyncio
    async def test_list_experiments_multiple(self, storage_backend, sample_experiment_specs):
        """Test listing multiple experiments."""
        # Setup - create all experiments
        for spec in sample_experiment_specs:
            await storage_backend.create_experiment(spec)
        
        # Execute
        result = await storage_backend.list_experiments()
        
        # Verify
        assert len(result) == 3
        assert all(isinstance(spec, ExperimentSpec) for spec in result)
        
        # Extract experiment IDs from results
        returned_ids = {spec.id for spec in result}
        expected_ids = {spec.id for spec in sample_experiment_specs}
        assert returned_ids == expected_ids
        
        # Verify each experiment is correctly returned
        result_by_id = {spec.id: spec for spec in result}
        for original_spec in sample_experiment_specs:
            returned_spec = result_by_id[original_spec.id]
            assert returned_spec.name == original_spec.name
            assert returned_spec.description == original_spec.description
            assert returned_spec.metrics.metrics == original_spec.metrics.metrics
            assert returned_spec.framework.params == original_spec.framework.params
    
    @pytest.mark.asyncio
    async def test_list_experiments_complex_parameters(self, storage_backend):
        """Test listing experiments with complex parameter structures."""
        # Setup
        complex_spec = ExperimentSpec(
            name="Complex List Test",
            description="Testing complex parameters in list",
            model="test-model",
            task=TaskSpec(
                name="complex_list_task",
                type=TaskType.LLM_GENERATION,
                description="Complex list test task",
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
                            {"type": "feedforward", "dim": 2048, "dropout": 0.1}
                        ],
                        "embedding": {"vocab_size": 50000, "dim": 512}
                    },
                    "training": {
                        "optimizer": {
                            "name": "adam",
                            "lr": 1e-4,
                            "betas": [0.9, 0.999],
                            "eps": 1e-8
                        },
                        "scheduler": {
                            "type": "cosine",
                            "warmup_steps": 1000,
                            "total_steps": 100000
                        },
                        "batch_size": 32,
                        "gradient_clip": 1.0
                    },
                    "data": {
                        "source": "/path/to/data",
                        "preprocessing": ["tokenize", "normalize", "truncate"],
                        "augmentation": {
                            "enabled": True,
                            "methods": ["synonym_replacement", "random_insertion"],
                            "probability": 0.1
                        }
                    }
                }
            ),
            metrics=MetricsSpec(
                metrics={"complexity": "high", "framework": "pytorch"}
            )
        )
        
        await storage_backend.create_experiment(complex_spec)
        
        # Execute
        result = await storage_backend.list_experiments()
        
        # Verify
        assert len(result) == 1
        returned_spec = result[0]
        
        # Verify complex structure is preserved
        assert returned_spec.framework.params["model"]["architecture"] == "transformer"
        assert len(returned_spec.framework.params["model"]["layers"]) == 2
        assert returned_spec.framework.params["training"]["optimizer"]["betas"] == [0.9, 0.999]
        assert returned_spec.framework.params["data"]["augmentation"]["methods"] == ["synonym_replacement", "random_insertion"]
    
    @pytest.mark.asyncio
    async def test_list_experiments_after_deletion(self, storage_backend, sample_experiment_specs):
        """Test listing after some experiments are deleted."""
        # Setup - create all experiments
        for spec in sample_experiment_specs:
            await storage_backend.create_experiment(spec)
        
        # Delete one experiment
        await storage_backend.delete_experiment(sample_experiment_specs[1].id)
        
        # Execute
        result = await storage_backend.list_experiments()
        
        # Verify
        assert len(result) == 2
        returned_ids = {spec.id for spec in result}
        expected_ids = {
            sample_experiment_specs[0].id,
            sample_experiment_specs[2].id
        }
        assert returned_ids == expected_ids
        
        # Verify deleted experiment is not in results
        assert sample_experiment_specs[1].id not in returned_ids


class TestListExperimentsDataIntegrity:
    """Test data integrity and consistency."""
    
    @pytest.mark.asyncio
    async def test_list_experiments_preserves_json_types(self, storage_backend):
        """Test that all JSON data types are preserved correctly."""
        # Setup
        spec_with_types = ExperimentSpec(
            name="JSON Types Test",
            description="Testing all JSON data types",
            model="test-model",
            task=TaskSpec(
                name="json_types_task",
                type=TaskType.LLM_GENERATION,
                description="JSON types test task",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Test JSON types.")
                        ]
                    )
                )
            ),
            framework=FrameworkSpec(
                name=FrameworkName.TRANSFORMERS,
                params={
                    "tags_data": {
                        "string": "test_string",
                        "integer": 42,
                        "float": 3.14159,
                        "boolean_true": True,
                        "boolean_false": False,
                        "null_value": None,
                        "nested_object": {"key": "value", "number": 123},
                        "array": [1, 2, 3, "mixed", True]
                    },
                    "unicode": "测试数据",
                    "special_chars": "test@#$%^&*()",
                    "scientific_notation": 1e-5,
                    "negative_number": -100,
                    "large_number": 1234567890,
                    "nested_arrays": [[1, 2], [3, 4], ["a", "b"]],
                    "deeply_nested": {
                        "level1": {
                            "level2": {
                                "level3": {"data": "deep_value", "count": 999}
                            }
                        }
                    }
                }
            ),
            metrics=MetricsSpec(
                metrics={"json_types": "test"}
            )
        )
        
        await storage_backend.create_experiment(spec_with_types)
        
        # Execute
        result = await storage_backend.list_experiments()
        
        # Verify
        assert len(result) == 1
        returned_spec = result[0]
        
        # Verify all data types are preserved
        tags_data = returned_spec.framework.params["tags_data"]
        assert tags_data["string"] == "test_string"
        assert tags_data["integer"] == 42
        assert tags_data["float"] == 3.14159
        assert tags_data["boolean_true"] is True
        assert tags_data["boolean_false"] is False
        assert tags_data["null_value"] is None
        assert tags_data["nested_object"]["key"] == "value"
        assert tags_data["array"] == [1, 2, 3, "mixed", True]
        
        assert returned_spec.framework.params["unicode"] == "测试数据"
        assert returned_spec.framework.params["scientific_notation"] == 1e-5
        assert returned_spec.framework.params["negative_number"] == -100
        assert returned_spec.framework.params["deeply_nested"]["level1"]["level2"]["level3"]["data"] == "deep_value"
    
    @pytest.mark.asyncio
    async def test_list_experiments_with_none_fields(self, storage_backend):
        """Test listing experiments with None values in optional fields."""
        # Setup
        spec_with_none = ExperimentSpec(
            name="None Fields Test",
            description=None,  # None description
            model="test-model",
            task=TaskSpec(
                name="none_fields_task",
                type=TaskType.LLM_GENERATION,
                description="None fields task",
                params=LLMGenerationParams(
                    prompt_template=PromptTemplate(
                        messages=[
                            PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                            PromptMessage(role=PromptRole.USER, content="Test none fields.")
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
        
        await storage_backend.create_experiment(spec_with_none)
        
        # Execute
        result = await storage_backend.list_experiments()
        
        # Verify
        assert len(result) == 1
        returned_spec = result[0]
        assert returned_spec.id == spec_with_none.id
        assert returned_spec.name == "None Fields Test"
        assert returned_spec.description is None
        assert returned_spec.metrics.metrics == {}
        assert returned_spec.framework.params == {}
    
    @pytest.mark.asyncio
    async def test_list_experiments_order_consistency(self, storage_backend, sample_experiment_specs):
        """Test that list order is consistent across multiple calls."""
        # Setup - create experiments
        for spec in sample_experiment_specs:
            await storage_backend.create_experiment(spec)
        
        # Execute multiple times
        results = []
        for _ in range(5):
            result = await storage_backend.list_experiments()
            results.append(result)
        
        # Verify
        assert len(results) == 5
        assert all(len(result) == 3 for result in results)
        
        # Verify order consistency (all calls should return same order)
        first_result_ids = [spec.id for spec in results[0]]
        for result in results[1:]:
            result_ids = [spec.id for spec in result]
            assert result_ids == first_result_ids
    
    @pytest.mark.asyncio
    async def test_list_experiments_concurrent_access(self, storage_backend, sample_experiment_specs):
        """Test concurrent access to list experiments."""
        # Setup - create experiments
        for spec in sample_experiment_specs:
            await storage_backend.create_experiment(spec)
        
        # Execute concurrent list operations
        import asyncio
        tasks = [storage_backend.list_experiments() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify
        assert len(results) == 5
        for result in results:
            assert len(result) == 3
            assert all(isinstance(spec, ExperimentSpec) for spec in result)
            
            returned_ids = {spec.id for spec in result}
            expected_ids = {spec.id for spec in sample_experiment_specs}
            assert returned_ids == expected_ids


class TestListExperimentsPerformance:
    """Test performance-related aspects."""
    
    @pytest.mark.asyncio
    async def test_list_experiments_large_dataset(self, storage_backend):
        """Test listing with many experiments."""
        # Setup - create many experiments
        experiment_count = 100
        for i in range(experiment_count):
            spec = ExperimentSpec(
                name=f"Performance Test {i}",
                description=f"Performance testing experiment number {i}",
                model="test-model",
                task=TaskSpec(
                    name=f"perf_task_{i}",
                    type=TaskType.LLM_GENERATION,
                    description=f"Performance task {i}",
                    params=LLMGenerationParams(
                        prompt_template=PromptTemplate(
                            messages=[
                                PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                                PromptMessage(role=PromptRole.USER, content=f"Performance test {i}.")
                            ]
                        )
                    )
                ),
                framework=FrameworkSpec(
                    name=FrameworkName.TRANSFORMERS,
                    params={
                        "value": i * 10,
                        "config": {"setting": f"value_{i}"},
                        "enabled": i % 2 == 0
                    }
                ),
                metrics=MetricsSpec(
                    metrics={"index": str(i), "batch": "performance_test"}
                )
            )
            await storage_backend.create_experiment(spec)
        
        # Execute
        result = await storage_backend.list_experiments()
        
        # Verify
        assert len(result) == experiment_count
        assert all(isinstance(spec, ExperimentSpec) for spec in result)
        
        # Verify some specific experiments exist with correct values
        result_by_name = {spec.name: spec for spec in result}
        assert "Performance Test 0" in result_by_name
        assert "Performance Test 99" in result_by_name
        
        # Check parameter values for a specific experiment
        perf_test_50 = result_by_name["Performance Test 50"]
        assert perf_test_50.framework.params["value"] == 500
    
    @pytest.mark.asyncio
    async def test_list_experiments_large_parameters(self, storage_backend):
        """Test listing experiment with very large parameter structures."""
        # Setup - create experiment with large parameters
        large_params = {}
        for i in range(50):
            large_params[f"section_{i}"] = {
                "config": {f"param_{j}": f"value_{i}_{j}" for j in range(20)},
                "data": [f"item_{i}_{k}" for k in range(30)],
                "metadata": {
                    "description": f"Large parameter section {i} with lots of data",
                    "tags": [f"tag_{i}_{t}" for t in range(10)],
                    "values": {f"val_{v}": i * v for v in range(15)}
                }
            }
        
        large_spec = ExperimentSpec(
            name="Large Parameters Test",
            description="Testing very large parameter structures",
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
                metrics={"size": "large", "complexity": "extreme"}
            )
        )
        
        await storage_backend.create_experiment(large_spec)
        
        # Execute
        result = await storage_backend.list_experiments()
        
        # Verify
        assert len(result) == 1
        returned_spec = result[0]
        assert returned_spec.id == large_spec.id
        assert len(returned_spec.framework.params) == 50
        assert "section_25" in returned_spec.framework.params
        assert len(returned_spec.framework.params["section_25"]["data"]) == 30


class TestListExperimentsIsolation:
    """Test isolation between different experiments."""
    
    @pytest.mark.asyncio
    async def test_list_experiments_isolation(self, storage_backend):
        """Test that experiments with similar IDs are properly isolated."""
        # Setup - create experiments with similar IDs
        similar_specs = [
            ExperimentSpec(
                name="Isolation Test 1",
                description="First isolation test",
                model="test-model",
                task=TaskSpec(
                    name="iso_task_1",
                    type=TaskType.LLM_GENERATION,
                    description="Isolation task 1",
                    params=LLMGenerationParams(
                        prompt_template=PromptTemplate(
                            messages=[
                                PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                                PromptMessage(role=PromptRole.USER, content="Isolation test 1.")
                            ]
                        )
                    )
                ),
                framework=FrameworkSpec(
                    name=FrameworkName.TRANSFORMERS,
                    params={"value": "first"}
                ),
                metrics=MetricsSpec(
                    metrics={"test_id": "1"}
                )
            ),
            ExperimentSpec(
                name="Isolation Test 10", 
                description="Tenth isolation test",
                model="test-model",
                task=TaskSpec(
                    name="iso_task_10",
                    type=TaskType.LLM_GENERATION,
                    description="Isolation task 10",
                    params=LLMGenerationParams(
                        prompt_template=PromptTemplate(
                            messages=[
                                PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                                PromptMessage(role=PromptRole.USER, content="Isolation test 10.")
                            ]
                        )
                    )
                ),
                framework=FrameworkSpec(
                    name=FrameworkName.TRANSFORMERS,
                    params={"value": "tenth"}
                ),
                metrics=MetricsSpec(
                    metrics={"test_id": "10"}
                )
            ),
            ExperimentSpec(
                name="Isolation Test 11",
                description="Eleventh isolation test",
                model="test-model",
                task=TaskSpec(
                    name="iso_task_11",
                    type=TaskType.LLM_GENERATION,
                    description="Isolation task 11",
                    params=LLMGenerationParams(
                        prompt_template=PromptTemplate(
                            messages=[
                                PromptMessage(role=PromptRole.SYSTEM, content="You are a helpful assistant."),
                                PromptMessage(role=PromptRole.USER, content="Isolation test 11.")
                            ]
                        )
                    )
                ),
                framework=FrameworkSpec(
                    name=FrameworkName.TRANSFORMERS,
                    params={"value": "eleventh"}
                ),
                metrics=MetricsSpec(
                    metrics={"test_id": "11"}
                )
            )
        ]
        
        for spec in similar_specs:
            await storage_backend.create_experiment(spec)
        
        # Execute
        result = await storage_backend.list_experiments()
        
        # Verify
        assert len(result) == 3
        result_by_name = {spec.name: spec for spec in result}
        
        # Verify each experiment has correct data
        assert result_by_name["Isolation Test 1"].framework.params["value"] == "first"
        assert result_by_name["Isolation Test 10"].framework.params["value"] == "tenth"
        assert result_by_name["Isolation Test 11"].framework.params["value"] == "eleventh"
        
        assert result_by_name["Isolation Test 1"].metrics.metrics["test_id"] == "1"
        assert result_by_name["Isolation Test 10"].metrics.metrics["test_id"] == "10"
        assert result_by_name["Isolation Test 11"].metrics.metrics["test_id"] == "11"