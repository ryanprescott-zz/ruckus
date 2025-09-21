"""Test model type detection and adaptive prompt formatting."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from ruckus_agent.adapters.transformers_adapter import TransformersAdapter


class TestModelTypeDetection:
    """Test suite for model type detection and prompt adaptation."""

    @pytest.fixture
    def adapter(self):
        """Create a fresh TransformersAdapter instance."""
        return TransformersAdapter()

    def test_detect_base_models(self, adapter):
        """Test detection of base completion models."""
        base_models = [
            "distilgpt2",
            "gpt2-medium",
            "microsoft/DialoGPT-medium",
            "/path/to/gpt-neo-125M",
            "gpt-j-6B",
            "pythia-70m",
            "bloom-560m",
            "facebook/opt-350m"
        ]

        for model_name in base_models:
            adapter.model_name = model_name
            model_type = adapter._detect_model_type()
            assert model_type == "base", f"Model {model_name} should be detected as base, got {model_type}"

    def test_detect_chat_models(self, adapter):
        """Test detection of chat/instruction models."""
        chat_models = [
            "microsoft/DialoGPT-medium-chat",
            "lmsys/vicuna-7b-v1.5",
            "meta-llama/Llama-2-7b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "HuggingFaceH4/zephyr-7b-beta",
            "alpaca-7b-native",
            "tulu-2-dpo-70b"
        ]

        for model_name in chat_models:
            adapter.model_name = model_name
            model_type = adapter._detect_model_type()
            assert model_type == "chat", f"Model {model_name} should be detected as chat, got {model_type}"

    def test_unknown_model_defaults_to_base(self, adapter):
        """Test that unknown models default to base type."""
        unknown_models = [
            "some-random-model",
            "custom/my-model-v1.0",
            ""
        ]

        for model_name in unknown_models:
            adapter.model_name = model_name
            model_type = adapter._detect_model_type()
            assert model_type == "base", f"Unknown model {model_name} should default to base, got {model_type}"

    @pytest.mark.asyncio
    async def test_base_model_prompt_formatting(self, adapter):
        """Test that base models get simple completion prompts."""
        adapter.model_name = "distilgpt2"
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()

        # Mock the generate method to return a simple answer
        async def mock_generate(prompt, parameters):
            # Verify the prompt format for base models
            assert "Question:" in prompt, f"Base model prompt should contain 'Question:', got: {prompt}"
            assert "Answer:" in prompt, f"Base model prompt should contain 'Answer:', got: {prompt}"
            assert "System:" not in prompt, f"Base model prompt should not contain 'System:', got: {prompt}"
            assert "User:" not in prompt, f"Base model prompt should not contain 'User:', got: {prompt}"
            return "Paris"

        adapter.generate = mock_generate
        adapter.count_tokens = AsyncMock(return_value=5)

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]

        result = await adapter.generate_with_conversation(conversation, {"max_tokens": 50})

        assert result["model_response"] == "Paris"
        assert result["conversation"][-1]["content"] == "Paris"

    @pytest.mark.asyncio
    async def test_chat_model_prompt_formatting(self, adapter):
        """Test that chat models get conversational prompts."""
        adapter.model_name = "vicuna-7b-chat"
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()

        # Mock the generate method to return a chat response
        async def mock_generate(prompt, parameters):
            # Verify the prompt format for chat models
            assert "System:" in prompt, f"Chat model prompt should contain 'System:', got: {prompt}"
            assert "User:" in prompt, f"Chat model prompt should contain 'User:', got: {prompt}"
            assert "Assistant:" in prompt, f"Chat model prompt should contain 'Assistant:', got: {prompt}"
            return "The capital of France is Paris."

        adapter.generate = mock_generate
        adapter.count_tokens = AsyncMock(return_value=5)

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]

        result = await adapter.generate_with_conversation(conversation, {"max_tokens": 50})

        assert result["model_response"] == "The capital of France is Paris."
        assert result["conversation"][-1]["content"] == "The capital of France is Paris."

    def test_model_info_includes_type(self, adapter):
        """Test that model info includes detected type."""
        adapter.model_name = "distilgpt2"
        adapter.model = MagicMock()
        adapter.device = "cpu"

        info = adapter.get_model_info()

        assert "model_type" in info
        assert info["model_type"] == "base"
        assert info["model_name"] == "distilgpt2"
        assert info["loaded"] is True
        assert info["device"] == "cpu"

    @pytest.mark.asyncio
    async def test_base_model_with_multiple_user_messages(self, adapter):
        """Test base model handling with multiple user messages."""
        adapter.model_name = "gpt2"
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()

        # Mock the generate method
        async def mock_generate(prompt, parameters):
            # Should use the last user question
            assert "What is 2+2?" in prompt, f"Should use last user question, got: {prompt}"
            return "4"

        adapter.generate = mock_generate
        adapter.count_tokens = AsyncMock(return_value=3)

        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is 2+2?"}
        ]

        result = await adapter.generate_with_conversation(conversation, {"max_tokens": 10})

        assert result["model_response"] == "4"