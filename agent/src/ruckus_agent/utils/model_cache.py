"""Model cache management."""

from pathlib import Path
from typing import Dict, List, Optional
import json


class ModelCache:
    """Manage cached models."""

    def __init__(self, cache_dir: str = "/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load cache manifest."""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {"models": {}}

    def _save_manifest(self):
        """Save cache manifest."""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def list_models(self) -> List[str]:
        """List cached models."""
        return list(self.manifest.get("models", {}).keys())

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to cached model."""
        if model_name in self.manifest.get("models", {}):
            path = self.cache_dir / self.manifest["models"][model_name]["path"]
            if path.exists():
                return path
        return None

    def add_model(self, model_name: str, path: str, metadata: Dict = None):
        """Add model to cache."""
        if "models" not in self.manifest:
            self.manifest["models"] = {}

        self.manifest["models"][model_name] = {
            "path": path,
            "metadata": metadata or {},
        }
        self._save_manifest()

    def remove_model(self, model_name: str):
        """Remove model from cache."""
        if model_name in self.manifest.get("models", {}):
            del self.manifest["models"][model_name]
            self._save_manifest()

    def get_cache_size(self) -> float:
        """Get total cache size in GB."""
        total_size = 0
        for item in self.cache_dir.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size / (1024 ** 3)

    def cleanup_old_models(self, keep_n: int = 5):
        """Remove old models keeping only the most recent N."""
        # TODO: Implement LRU cleanup
        pass