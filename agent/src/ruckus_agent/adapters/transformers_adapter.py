"""Hugging Face Transformers adapter."""

import logging
from typing import Dict, Any, List, Optional, Union
from .base import ModelAdapter

logger = logging.getLogger(__name__)


class TransformersAdapter(ModelAdapter):
    """Adapter for Hugging Face Transformers models."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        logger.info("TransformersAdapter initialized")

    async def load_model(self, model_name: str, **kwargs) -> None:
        """Load a transformers model."""
        logger.info(f"TransformersAdapter loading model: {model_name}")
        try:
            # Import transformers here to handle import errors gracefully
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
            except ImportError as e:
                raise ImportError(
                    f"❌ Transformers framework is not installed or failed to import: {e}. "
                    f"Please install transformers with: pip install transformers torch"
                )

            # Determine device - prioritize GPU if available
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            logger.info(f"Using device: {self.device}")

            # Load tokenizer
            logger.info(f"Loading tokenizer from: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Add pad token if it doesn't exist (common for GPT models)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            logger.info(f"Loading model from: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
                **kwargs
            )

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            # Store model name
            self.model_name = model_name

            logger.info(f"TransformersAdapter model loaded successfully: {model_name} on {self.device}")

        except ImportError:
            # Re-raise ImportError with our enhanced message
            raise
        except Exception as e:
            logger.error(f"TransformersAdapter failed to load model {model_name}: {e}")
            # Clean up on failure
            self.model = None
            self.tokenizer = None
            self.model_name = None
            self.device = None
            raise RuntimeError(
                f"❌ Failed to load model '{model_name}' with Transformers: {e}. "
                f"This could be due to insufficient memory, incompatible model format, "
                f"or missing model files. Please check the model requirements and availability."
            )

    async def unload_model(self) -> None:
        """Unload the current model."""
        logger.info(f"TransformersAdapter unloading model: {self.model_name}")

        # Clean up model and free GPU memory
        if self.model is not None:
            del self.model

        if self.tokenizer is not None:
            del self.tokenizer

        # Clear CUDA cache if using GPU
        if self.device and self.device.startswith('cuda'):
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache: {e}")

        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        logger.info("TransformersAdapter model unloaded")

    async def generate(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate output from prompt."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("No model loaded. Call load_model() first.")

        logger.debug(f"TransformersAdapter generating for prompt length: {len(prompt)}")
        try:
            import torch

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract generation parameters
            max_tokens = parameters.get('max_tokens', 100)
            temperature = parameters.get('temperature', 0.7)
            top_p = parameters.get('top_p', 1.0)
            top_k = parameters.get('top_k', 50)
            do_sample = temperature > 0.0

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the new tokens (exclude input prompt)
            input_length = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][input_length:]
            result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            logger.debug(f"TransformersAdapter generated output length: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"TransformersAdapter generation failed: {e}")
            raise

    async def generate_batch(self, prompts: List[str], parameters: Dict[str, Any]) -> List[str]:
        """Generate outputs for multiple prompts."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("No model loaded. Call load_model() first.")

        logger.info(f"TransformersAdapter batch generation for {len(prompts)} prompts")
        try:
            import torch

            # For now, use sequential generation (can optimize later with padding)
            results = []
            for prompt in prompts:
                result = await self.generate(prompt, parameters)
                results.append(result)

            logger.info(f"TransformersAdapter batch generation completed: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"TransformersAdapter batch generation failed: {e}")
            raise

    async def tokenize(self, text: str) -> List[int]:
        """Tokenize text."""
        if not self.tokenizer:
            raise RuntimeError("No model loaded. Call load_model() first.")

        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return tokens
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            tokens = await self.tokenize(text)
            return len(tokens)
        except Exception:
            # Fallback to word count if tokenization fails
            logger.warning("Token counting failed, using word count fallback")
            return len(text.split())

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "loaded": self.model is not None,
            "device": self.device,
            "model_type": self._detect_model_type(),
        }

    def _detect_model_type(self) -> str:
        """Detect model type based on model name and architecture."""
        if not self.model_name:
            return "base"

        model_name_lower = self.model_name.lower()

        # Chat/Instruct models (trained for conversation)
        chat_indicators = [
            'chat', 'instruct', 'assistant', 'conversational',
            'alpaca', 'vicuna', 'llama-2-chat', 'llama-3-instruct',
            'mistral-instruct', 'zephyr', 'tulu', 'orca'
        ]

        # Base completion models (trained for text completion)
        base_indicators = [
            'gpt2', 'distilgpt2', 'gpt-neo', 'gpt-j',
            'pythia', 'bloom', 'opt', 'llama-base',
            'mistral-base', 'falcon-base'
        ]

        for indicator in chat_indicators:
            if indicator in model_name_lower:
                return "chat"

        for indicator in base_indicators:
            if indicator in model_name_lower:
                return "base"

        # Default to base for safety (more predictable behavior)
        return "base"

    def get_capabilities(self) -> Dict[str, bool]:
        """Get adapter capabilities."""
        return {
            "streaming": False,
            "batch_processing": True,
            "tokenization": True,
            "quantization": True,
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
        }

    async def generate_with_conversation(self, conversation: List[Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate with full conversation capture for LLM tasks.

        Args:
            conversation: List of PromptMessage objects containing the conversation
            parameters: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Dict containing:
                - conversation: Full conversation including assistant response
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens
                - total_tokens: Total token count
        """
        logger.debug(f"TransformersAdapter generating conversation with {len(conversation)} messages")

        try:
            # Convert conversation to prompt format based on model type
            model_type = self._detect_model_type()
            prompt_parts = []
            input_messages = []

            for msg in conversation:
                # Handle both dict format and PromptMessage objects
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    role = msg.role.value if hasattr(msg.role, 'value') else msg.role
                    content = msg.content
                else:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')

                # Store original message
                input_messages.append({"role": role, "content": content})

            # Generate prompt based on model type
            if model_type == "chat":
                # Chat models: use conversational format
                for msg_data in input_messages:
                    role = msg_data["role"]
                    content = msg_data["content"]
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                prompt_parts.append("Assistant:")
                full_prompt = "\n".join(prompt_parts)
            else:
                # Base models: use simple completion format
                # Extract the user's question and create a simple prompt
                user_questions = [msg["content"] for msg in input_messages if msg["role"] == "user"]
                if user_questions:
                    # Use the last user question for simple completion
                    question = user_questions[-1]
                    # Create a completion-style prompt
                    full_prompt = f"Question: {question}\nAnswer:"
                else:
                    # Fallback to conversational format if no user questions
                    for msg_data in input_messages:
                        role = msg_data["role"]
                        content = msg_data["content"]
                        if role == "system":
                            prompt_parts.append(f"System: {content}")
                        elif role == "user":
                            prompt_parts.append(f"User: {content}")
                        elif role == "assistant":
                            prompt_parts.append(f"Assistant: {content}")
                    prompt_parts.append("Assistant:")
                    full_prompt = "\n".join(prompt_parts)

            logger.debug(f"Generated prompt for Transformers ({model_type} model, length: {len(full_prompt)}): {full_prompt[:100]}...")

            # Count input tokens by summing individual message tokens
            input_tokens = 0
            try:
                for msg_data in input_messages:
                    msg_tokens = await self.count_tokens(msg_data["content"])
                    input_tokens += msg_tokens
            except Exception as e:
                logger.warning(f"Failed to count input tokens: {e}, using word count fallback")
                input_tokens = sum(len(msg["content"].split()) for msg in input_messages)

            # Generate response using existing generate method
            response_text = await self.generate(full_prompt, parameters)

            # Count output tokens
            try:
                output_tokens = await self.count_tokens(response_text)
            except Exception as e:
                logger.warning(f"Failed to count output tokens: {e}, using word count fallback")
                output_tokens = len(response_text.split())

            # Build complete conversation including assistant response
            complete_conversation = input_messages.copy()
            complete_conversation.append({
                "role": "assistant",
                "content": response_text.strip()
            })

            # Calculate total tokens
            total_tokens = input_tokens + output_tokens

            result = {
                "conversation": complete_conversation,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model_response": response_text.strip()  # Quick access to just the response
            }

            logger.debug(f"Transformers conversation generation completed: {input_tokens} input + {output_tokens} output = {total_tokens} total tokens")
            return result

        except Exception as e:
            logger.error(f"TransformersAdapter conversation generation failed: {e}")
            raise
