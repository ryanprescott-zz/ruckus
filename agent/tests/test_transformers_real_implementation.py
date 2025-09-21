"""Test real Transformers implementation functionality."""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ruckus_agent.adapters.transformers_adapter import TransformersAdapter


class TestTransformersRealImplementation:
    """Test suite for real Transformers functionality (not mocked)."""

    @pytest.fixture
    def adapter(self):
        """Create a fresh TransformersAdapter instance."""
        return TransformersAdapter()

    @pytest.fixture
    def sample_model_path(self):
        """Sample model path for testing."""
        return "/Users/jbisila/models/distilgpt2"

    @pytest.fixture
    def sample_parameters(self):
        """Sample generation parameters."""
        return {
            "temperature": 0.7,
            "max_tokens": 50,
            "top_p": 0.9,
            "top_k": 40
        }

    @pytest.mark.asyncio
    async def test_load_model_success_real_path(self, adapter, sample_model_path):
        """Test loading a real model from filesystem path."""
        # Test that load_model actually loads transformers model and tokenizer
        await adapter.load_model(sample_model_path)

        # Verify model is actually loaded (not just stored as string)
        assert adapter.model is not None, "Model should be loaded, not None"
        assert adapter.tokenizer is not None, "Tokenizer should be loaded, not None"
        assert adapter.model_name == sample_model_path

        # Verify model info indicates successful loading
        info = adapter.get_model_info()
        assert info["loaded"] is True
        assert info["model_name"] == sample_model_path

    @pytest.mark.asyncio
    async def test_load_model_missing_transformers_import(self, adapter):
        """Test error handling when transformers is not installed."""
        # Since imports are inside the function, we need to patch the import inside the module
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'transformers':
                    raise ImportError("No module named 'transformers'")
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            with pytest.raises(ImportError) as exc_info:
                await adapter.load_model("test-model")

            assert "Transformers framework is not installed" in str(exc_info.value)
            assert "pip install transformers torch" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_model_nonexistent_path(self, adapter):
        """Test error handling for nonexistent model path."""
        nonexistent_path = "/nonexistent/model/path"

        with pytest.raises(RuntimeError) as exc_info:
            await adapter.load_model(nonexistent_path)

        assert "Failed to load model" in str(exc_info.value)
        assert "insufficient memory" in str(exc_info.value) or "missing model files" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_real_text_generation(self, adapter, sample_model_path, sample_parameters):
        """Test that generate() produces real model output, not placeholder text."""
        await adapter.load_model(sample_model_path)

        prompt = "The quick brown fox"
        result = await adapter.generate(prompt, sample_parameters)

        # Should NOT be placeholder text
        assert not result.startswith("Generated output for:"), f"Got placeholder text: {result}"
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"

        # Result should be different from input prompt (model should generate new text)
        assert result != prompt, "Generated text should be different from prompt"

    @pytest.mark.asyncio
    async def test_real_tokenization(self, adapter, sample_model_path):
        """Test that tokenize() uses real model tokenizer, not empty list."""
        await adapter.load_model(sample_model_path)

        text = "Hello, how are you today?"
        tokens = await adapter.tokenize(text)

        # Should NOT be empty list or simple range
        assert isinstance(tokens, list), "Tokens should be a list"
        assert len(tokens) > 0, "Should have tokens, not empty list"
        assert all(isinstance(token, int) for token in tokens), "All tokens should be integers"

        # Should be actual tokenization, not placeholder
        simple_split_length = len(text.split())
        assert len(tokens) != simple_split_length or len(tokens) > 1, "Should use real tokenizer, not word splitting"

    @pytest.mark.asyncio
    async def test_accurate_token_counting(self, adapter, sample_model_path):
        """Test that count_tokens() uses real tokenizer, not word count fallback."""
        await adapter.load_model(sample_model_path)

        # Test text with punctuation and subwords to differentiate from word count
        text = "Don't use word-splitting; use real tokenization!"

        token_count = await adapter.count_tokens(text)
        word_count = len(text.split())

        # Token count should be based on real tokenization
        assert isinstance(token_count, int), "Token count should be an integer"
        assert token_count > 0, "Should have positive token count"

        # For most tokenizers, token count differs from word count due to subwords/punctuation
        # This is a heuristic but generally true for transformer tokenizers
        assert token_count != word_count, f"Token count ({token_count}) should differ from word count ({word_count}) for real tokenization"

    @pytest.mark.asyncio
    async def test_generate_without_loaded_model(self, adapter):
        """Test that generate fails appropriately when no model is loaded."""
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.generate("test prompt", {})

        assert "No model loaded" in str(exc_info.value) or "model is None" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tokenize_without_loaded_model(self, adapter):
        """Test that tokenize fails appropriately when no model is loaded."""
        with pytest.raises(RuntimeError) as exc_info:
            await adapter.tokenize("test text")

        assert "No model loaded" in str(exc_info.value) or "tokenizer is None" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_batch_generation_real(self, adapter, sample_model_path, sample_parameters):
        """Test batch generation with real model produces distinct outputs."""
        await adapter.load_model(sample_model_path)

        prompts = [
            "The weather today is",
            "My favorite food is",
            "In the future, technology will"
        ]

        results = await adapter.generate_batch(prompts, sample_parameters)

        assert len(results) == len(prompts), "Should return result for each prompt"

        # Each result should be real generation, not placeholder
        for i, result in enumerate(results):
            assert not result.startswith("Generated output for:"), f"Result {i} is placeholder: {result}"
            assert isinstance(result, str), f"Result {i} should be string"
            assert len(result) > 0, f"Result {i} should not be empty"

    @pytest.mark.asyncio
    async def test_conversation_generation_real_output(self, adapter, sample_model_path, sample_parameters):
        """Test that conversation generation produces real model responses."""
        await adapter.load_model(sample_model_path)

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]

        result = await adapter.generate_with_conversation(conversation, sample_parameters)

        # Verify structure
        assert "conversation" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert "total_tokens" in result
        assert "model_response" in result

        # Verify the response is real, not placeholder
        model_response = result["model_response"]
        assert not model_response.startswith("Generated output for:"), f"Got placeholder response: {model_response}"
        assert len(model_response) > 0, "Model response should not be empty"

        # Token counts should be realistic (not just word counts)
        assert result["input_tokens"] > 0, "Should have positive input token count"
        assert result["output_tokens"] > 0, "Should have positive output token count"
        assert result["total_tokens"] == result["input_tokens"] + result["output_tokens"]

    @pytest.mark.asyncio
    async def test_model_unload_cleanup(self, adapter, sample_model_path):
        """Test that unload_model properly cleans up loaded model and tokenizer."""
        # Load model first
        await adapter.load_model(sample_model_path)
        assert adapter.model is not None
        assert adapter.tokenizer is not None

        # Unload and verify cleanup
        await adapter.unload_model()
        assert adapter.model is None
        assert adapter.tokenizer is None
        assert adapter.model_name is None

        # Model info should reflect unloaded state
        info = adapter.get_model_info()
        assert info["loaded"] is False

    @pytest.mark.asyncio
    async def test_device_detection_after_load(self, adapter, sample_model_path):
        """Test that device information is properly detected after model loading."""
        await adapter.load_model(sample_model_path)

        info = adapter.get_model_info()
        # Device should be detected (cpu, cuda, mps, etc.)
        assert info["device"] is not None, "Device should be detected after model loading"
        assert isinstance(info["device"], str), "Device should be a string identifier"