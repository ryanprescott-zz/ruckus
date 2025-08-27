"""Model discovery utilities for scanning and analyzing available models."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.models import ModelInfo

logger = logging.getLogger(__name__)


class ModelDiscovery:
    """Discover and analyze models in mounted directories."""
    
    def __init__(self, models_dir: str = "/ruckus/models"):
        self.models_dir = Path(models_dir)
        logger.info(f"ModelDiscovery initialized for directory: {models_dir}")
    
    async def discover_all_models(self) -> List[ModelInfo]:
        """Discover all models in the models directory."""
        models = []
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory does not exist: {self.models_dir}")
            return models
        
        logger.info(f"Scanning for models in: {self.models_dir}")
        
        # Look for directories that contain config.json (HuggingFace convention)
        for potential_model_dir in self.models_dir.iterdir():
            if potential_model_dir.is_dir():
                config_path = potential_model_dir / "config.json"
                if config_path.exists():
                    try:
                        model_info = await self._analyze_model_directory(potential_model_dir)
                        if model_info:
                            models.append(model_info)
                            logger.info(f"Discovered model: {model_info.name} ({model_info.size_gb:.2f} GB)")
                    except Exception as e:
                        logger.error(f"Failed to analyze model directory {potential_model_dir}: {e}")
        
        logger.info(f"Model discovery complete. Found {len(models)} models")
        return models
    
    async def _analyze_model_directory(self, model_dir: Path) -> Optional[ModelInfo]:
        """Analyze a single model directory and extract metadata."""
        try:
            model_name = model_dir.name
            logger.debug(f"Analyzing model directory: {model_name}")
            
            # Read config.json
            config_path = model_dir / "config.json"
            config_data = {}
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read config.json for {model_name}: {e}")
            
            # Read tokenizer config if available
            tokenizer_config_path = model_dir / "tokenizer_config.json"
            tokenizer_config = {}
            if tokenizer_config_path.exists():
                try:
                    with open(tokenizer_config_path, 'r') as f:
                        tokenizer_config = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read tokenizer_config.json for {model_name}: {e}")
            
            # Scan files and categorize them
            file_analysis = await self._analyze_files(model_dir)
            
            # Determine primary format
            model_format = self._determine_model_format(file_analysis["model_files"])
            
            # Create ModelInfo
            model_info = ModelInfo(
                name=model_name,
                path=str(model_dir),
                size_gb=file_analysis["total_size_gb"],
                format=model_format,
                framework_compatible=self._determine_compatible_frameworks(model_format, config_data),
                
                # HuggingFace config metadata
                model_type=config_data.get("model_type"),
                architecture=config_data.get("architectures", [None])[0] if config_data.get("architectures") else None,
                vocab_size=config_data.get("vocab_size"),
                hidden_size=config_data.get("hidden_size"),
                num_layers=config_data.get("num_hidden_layers") or config_data.get("n_layer"),
                num_attention_heads=config_data.get("num_attention_heads") or config_data.get("n_head"),
                max_position_embeddings=config_data.get("max_position_embeddings"),
                torch_dtype=config_data.get("torch_dtype"),
                
                # Tokenizer info
                tokenizer_type=tokenizer_config.get("tokenizer_class"),
                tokenizer_vocab_size=tokenizer_config.get("vocab_size"),
                
                # File breakdown
                config_files=file_analysis["config_files"],
                model_files=file_analysis["model_files"],
                tokenizer_files=file_analysis["tokenizer_files"],
                other_files=file_analysis["other_files"],
                
                # Additional metadata
                quantization=self._detect_quantization(file_analysis["model_files"], config_data),
                discovered_at=datetime.utcnow()
            )
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to analyze model directory {model_dir}: {e}")
            return None
    
    async def _analyze_files(self, model_dir: Path) -> Dict[str, Any]:
        """Analyze files in a model directory."""
        config_files = []
        model_files = []
        tokenizer_files = []
        other_files = []
        total_size_bytes = 0
        
        # Known file patterns
        config_patterns = ["config.json", "generation_config.json", "model_config.json"]
        tokenizer_patterns = ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "merges.txt", 
                            "special_tokens_map.json", "tokenizer.model", "vocab.json"]
        model_patterns = [".bin", ".safetensors", ".pt", ".pth", ".gguf", ".ggml"]
        
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                file_name = file_path.name
                file_size = file_path.stat().st_size
                total_size_bytes += file_size
                
                # Categorize files
                if any(pattern in file_name for pattern in config_patterns):
                    config_files.append(file_name)
                elif any(pattern in file_name for pattern in tokenizer_patterns):
                    tokenizer_files.append(file_name)
                elif any(file_name.endswith(pattern) for pattern in model_patterns):
                    model_files.append(file_name)
                else:
                    other_files.append(file_name)
        
        return {
            "config_files": config_files,
            "model_files": model_files,
            "tokenizer_files": tokenizer_files,
            "other_files": other_files,
            "total_size_gb": total_size_bytes / (1024 ** 3)
        }
    
    def _determine_model_format(self, model_files: List[str]) -> str:
        """Determine the primary model format based on files present."""
        if any(f.endswith(".safetensors") for f in model_files):
            return "safetensors"
        elif any(f.endswith(".bin") for f in model_files):
            return "pytorch"
        elif any(f.endswith(".gguf") for f in model_files):
            return "gguf"
        elif any(f.endswith(".ggml") for f in model_files):
            return "ggml"
        elif any(f.endswith(".pt") or f.endswith(".pth") for f in model_files):
            return "pytorch"
        else:
            return "unknown"
    
    def _determine_compatible_frameworks(self, model_format: str, config_data: Dict) -> List[str]:
        """Determine which frameworks can potentially load this model."""
        frameworks = []
        
        # Most HuggingFace models work with transformers
        if model_format in ["safetensors", "pytorch"] and config_data:
            frameworks.append("transformers")
            frameworks.append("pytorch")
        
        # VLLM supports many HuggingFace models
        if model_format in ["safetensors", "pytorch"] and config_data.get("model_type"):
            # VLLM supports popular architectures
            supported_types = ["llama", "mistral", "qwen", "gemma", "phi", "falcon"]
            if config_data.get("model_type") in supported_types:
                frameworks.append("vllm")
        
        # GGUF/GGML typically work with llama.cpp variants
        if model_format in ["gguf", "ggml"]:
            frameworks.append("llamacpp")
        
        return frameworks
    
    def _detect_quantization(self, model_files: List[str], config_data: Dict) -> Optional[str]:
        """Detect if model is quantized based on files and config."""
        # Check for GPTQ
        if any("gptq" in f.lower() for f in model_files):
            return "gptq"
        
        # Check for AWQ
        if any("awq" in f.lower() for f in model_files):
            return "awq"
            
        # Check for GGUF quantization levels
        for f in model_files:
            if f.endswith(".gguf"):
                # GGUF files often encode quant level in filename
                if any(quant in f.lower() for quant in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]):
                    return f"gguf_{f.split('.')[-2]}"  # Extract quant type
        
        # Check config for quantization settings
        if config_data.get("quantization_config"):
            quant_method = config_data["quantization_config"].get("quant_method")
            if quant_method:
                return quant_method
        
        return None