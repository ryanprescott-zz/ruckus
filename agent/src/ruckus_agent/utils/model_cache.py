"""Model cache management."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class ModelCache:
    """Manage cached models."""

    def __init__(self, cache_dir: str = "/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()
        logger.info(f"ModelCache initialized with cache directory: {cache_dir}")

    def _load_manifest(self) -> Dict:
        """Load cache manifest."""
        try:
            if self.manifest_file.exists():
                with open(self.manifest_file, 'r') as f:
                    manifest = json.load(f)
                    logger.debug(f"Loaded manifest with {len(manifest.get('models', {}))} models")
                    return manifest
            else:
                logger.debug("No existing manifest found, creating new one")
                return {"models": {}}
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return {"models": {}}

    def _save_manifest(self):
        """Save cache manifest."""
        try:
            with open(self.manifest_file, 'w') as f:
                json.dump(self.manifest, f, indent=2)
            logger.debug("Manifest saved successfully")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            raise

    def list_models(self) -> List[str]:
        """List cached models."""
        return list(self.manifest.get("models", {}).keys())

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to cached model."""
        logger.debug(f"Looking up model path for: {model_name}")
        if model_name in self.manifest.get("models", {}):
            path = self.cache_dir / self.manifest["models"][model_name]["path"]
            if path.exists():
                logger.debug(f"Found cached model at: {path}")
                return path
            else:
                logger.warning(f"Model {model_name} in manifest but path doesn't exist: {path}")
        else:
            logger.debug(f"Model {model_name} not found in cache")
        return None

    def add_model(self, model_name: str, path: str, metadata: Dict = None):
        """Add model to cache."""
        logger.info(f"Adding model to cache: {model_name} at {path}")
        try:
            if "models" not in self.manifest:
                self.manifest["models"] = {}

            self.manifest["models"][model_name] = {
                "path": path,
                "metadata": metadata or {},
            }
            self._save_manifest()
            logger.info(f"Model {model_name} added to cache successfully")
        except Exception as e:
            logger.error(f"Failed to add model {model_name} to cache: {e}")
            raise

    def remove_model(self, model_name: str):
        """Remove model from cache."""
        logger.info(f"Removing model from cache: {model_name}")
        try:
            if model_name in self.manifest.get("models", {}):
                del self.manifest["models"][model_name]
                self._save_manifest()
                logger.info(f"Model {model_name} removed from cache successfully")
            else:
                logger.warning(f"Model {model_name} not found in cache for removal")
        except Exception as e:
            logger.error(f"Failed to remove model {model_name} from cache: {e}")
            raise

    def get_cache_size(self) -> float:
        """Get total cache size in GB."""
        logger.debug("Calculating cache size")
        try:
            total_size = 0
            file_count = 0
            for item in self.cache_dir.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
            
            size_gb = total_size / (1024 ** 3)
            logger.debug(f"Cache size: {size_gb:.2f} GB ({file_count} files)")
            return size_gb
        except Exception as e:
            logger.error(f"Failed to calculate cache size: {e}")
            return 0.0

    def cleanup_old_models(self, keep_n: int = 5):
        """Remove old models keeping only the most recent N."""
        # TODO: Implement LRU cleanup
        pass
