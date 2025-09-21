"""Model configuration parser for dynamic model setup."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelCapabilities:
    """Model capabilities and supported features."""
    model_type: str  # "base", "chat", "instruct", "code"
    supports_conversation: bool
    supports_system_prompt: bool
    max_context_length: int
    recommended_temperature: float
    recommended_top_p: float
    recommended_top_k: int
    prompt_template: str
    stop_tokens: List[str]
    special_tokens: Dict[str, str]


@dataclass
class ModelConfiguration:
    """Complete model configuration."""
    name: str
    path: str
    architecture: str
    model_type: str
    capabilities: ModelCapabilities
    config: Dict[str, Any]
    tokenizer_config: Dict[str, Any]


class ModelConfigParser:
    """Parser for model configuration files."""

    def __init__(self):
        """Initialize the parser."""
        pass

    def parse_model_config(self, model_path: str) -> Optional[ModelConfiguration]:
        """Parse model configuration from filesystem.

        Args:
            model_path: Path to model directory

        Returns:
            ModelConfiguration if successful, None otherwise
        """
        try:
            model_dir = Path(model_path)
            if not model_dir.exists():
                logger.warning(f"Model directory does not exist: {model_path}")
                return None

            # Load config.json
            config_file = model_dir / "config.json"
            config = {}
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                logger.warning(f"No config.json found in {model_path}")

            # Load tokenizer_config.json
            tokenizer_config_file = model_dir / "tokenizer_config.json"
            tokenizer_config = {}
            if tokenizer_config_file.exists():
                with open(tokenizer_config_file, 'r') as f:
                    tokenizer_config = json.load(f)

            # Determine model type and capabilities
            model_name = model_dir.name
            architecture = config.get("architectures", ["unknown"])[0] if config.get("architectures") else "unknown"
            model_type = self._detect_model_type(model_name, config, tokenizer_config)
            capabilities = self._analyze_capabilities(model_name, config, tokenizer_config, model_type)

            return ModelConfiguration(
                name=model_name,
                path=model_path,
                architecture=architecture,
                model_type=model_type,
                capabilities=capabilities,
                config=config,
                tokenizer_config=tokenizer_config
            )

        except Exception as e:
            logger.error(f"Failed to parse model config for {model_path}: {e}")
            return None

    def _detect_model_type(self, model_name: str, config: Dict[str, Any], tokenizer_config: Dict[str, Any]) -> str:
        """Detect model type from name and configuration.

        Args:
            model_name: Name of the model
            config: Model configuration
            tokenizer_config: Tokenizer configuration

        Returns:
            Model type string
        """
        model_name_lower = model_name.lower()

        # Check for explicit indicators in config
        if "chat" in config.get("model_type", ""):
            return "chat"
        if "instruct" in config.get("model_type", ""):
            return "instruct"

        # Check model name patterns
        chat_indicators = [
            'chat', 'instruct', 'assistant', 'conversational',
            'alpaca', 'vicuna', 'llama-2-chat', 'llama-3-instruct',
            'mistral-instruct', 'zephyr', 'tulu', 'orca'
        ]

        base_indicators = [
            'gpt2', 'distilgpt2', 'gpt-neo', 'gpt-j',
            'pythia', 'bloom', 'opt', 'llama-base',
            'mistral-base', 'falcon-base'
        ]

        code_indicators = [
            'codegen', 'starcoder', 'code-llama', 'codellama',
            'codebert', 'graphcodebert', 'codet5'
        ]

        for indicator in chat_indicators:
            if indicator in model_name_lower:
                return "chat"

        for indicator in code_indicators:
            if indicator in model_name_lower:
                return "code"

        for indicator in base_indicators:
            if indicator in model_name_lower:
                return "base"

        # Check architecture patterns
        architecture = config.get("architectures", [""])[0].lower()
        if "gpt" in architecture and ("chat" in model_name_lower or "instruct" in model_name_lower):
            return "chat"
        elif "gpt" in architecture:
            return "base"
        elif "llama" in architecture and ("chat" in model_name_lower or "instruct" in model_name_lower):
            return "chat"
        elif "llama" in architecture:
            return "base"

        # Default to base
        return "base"

    def _analyze_capabilities(self, model_name: str, config: Dict[str, Any],
                            tokenizer_config: Dict[str, Any], model_type: str) -> ModelCapabilities:
        """Analyze model capabilities based on configuration.

        Args:
            model_name: Name of the model
            config: Model configuration
            tokenizer_config: Tokenizer configuration
            model_type: Detected model type

        Returns:
            ModelCapabilities object
        """
        # Get context length
        max_context = config.get("max_position_embeddings",
                                config.get("n_positions",
                                          config.get("max_sequence_length", 2048)))

        # Determine capabilities based on model type
        if model_type == "chat":
            supports_conversation = True
            supports_system_prompt = True
            prompt_template = self._get_chat_template(tokenizer_config)
            recommended_temperature = 0.7
            recommended_top_p = 0.9
            recommended_top_k = 40
        elif model_type == "instruct":
            supports_conversation = False
            supports_system_prompt = True
            prompt_template = "### Instruction:\n{instruction}\n\n### Response:\n"
            recommended_temperature = 0.7
            recommended_top_p = 0.9
            recommended_top_k = 40
        elif model_type == "code":
            supports_conversation = False
            supports_system_prompt = False
            prompt_template = "# {instruction}\n"
            recommended_temperature = 0.2
            recommended_top_p = 0.95
            recommended_top_k = 50
        else:  # base
            supports_conversation = False
            supports_system_prompt = False
            prompt_template = "Question: {instruction}\nAnswer:"
            recommended_temperature = 0.8
            recommended_top_p = 0.9
            recommended_top_k = 50

        # Get special tokens
        special_tokens = {}
        if tokenizer_config:
            special_tokens = {
                "eos_token": tokenizer_config.get("eos_token", "</s>"),
                "bos_token": tokenizer_config.get("bos_token", "<s>"),
                "pad_token": tokenizer_config.get("pad_token", ""),
                "unk_token": tokenizer_config.get("unk_token", "<unk>")
            }

        # Determine stop tokens
        stop_tokens = []
        if model_type == "chat":
            stop_tokens = ["</s>", "<|endoftext|>", "\n\nUser:", "\n\nAssistant:"]
        elif model_type == "instruct":
            stop_tokens = ["</s>", "<|endoftext|>", "### Instruction:", "### Response:"]
        elif model_type == "code":
            stop_tokens = ["</s>", "<|endoftext|>", "\n\n#", "\n\ndef", "\n\nclass"]
        else:  # base
            stop_tokens = ["</s>", "<|endoftext|>", "\n\n"]

        return ModelCapabilities(
            model_type=model_type,
            supports_conversation=supports_conversation,
            supports_system_prompt=supports_system_prompt,
            max_context_length=max_context,
            recommended_temperature=recommended_temperature,
            recommended_top_p=recommended_top_p,
            recommended_top_k=recommended_top_k,
            prompt_template=prompt_template,
            stop_tokens=stop_tokens,
            special_tokens=special_tokens
        )

    def _get_chat_template(self, tokenizer_config: Dict[str, Any]) -> str:
        """Extract chat template from tokenizer config.

        Args:
            tokenizer_config: Tokenizer configuration

        Returns:
            Chat template string
        """
        # Try to get the chat template from tokenizer config
        chat_template = tokenizer_config.get("chat_template", "")

        if chat_template:
            return chat_template

        # Default chat template
        return "System: {system}\nUser: {user}\nAssistant:"

    def parse_all_models(self, models_directory: str) -> Dict[str, ModelConfiguration]:
        """Parse all models in a directory.

        Args:
            models_directory: Path to directory containing model subdirectories

        Returns:
            Dictionary mapping model names to configurations
        """
        models = {}
        models_dir = Path(models_directory)

        if not models_dir.exists():
            logger.warning(f"Models directory does not exist: {models_directory}")
            return models

        for model_subdir in models_dir.iterdir():
            if model_subdir.is_dir():
                config = self.parse_model_config(str(model_subdir))
                if config:
                    models[config.name] = config
                    logger.info(f"Parsed model: {config.name} (type: {config.model_type})")

        return models