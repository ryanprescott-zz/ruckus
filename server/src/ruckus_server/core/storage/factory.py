"""Storage backend factory for creating storage instances."""

import logging
from typing import Dict, Any

from ..config import StorageSettings, StorageBackendType
from .base import StorageBackend
from .postgresql import PostgreSQLStorageBackend
from .sqlite import SQLiteStorageBackend


class StorageFactory:
    """Factory for creating storage backend instances."""
    
    def __init__(self):
        """Initialize the storage factory."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("StorageFactory initialized")
    
    def create_storage_backend(self, storage_settings: StorageSettings) -> StorageBackend:
        """Create a storage backend based on settings.
        
        Args:
            storage_settings: Storage configuration settings
            
        Returns:
            StorageBackend: Configured storage backend instance
            
        Raises:
            ValueError: If storage backend type is not supported
        """
        self.logger.info(f"Creating storage backend of type: {storage_settings.storage_backend}")
        
        try:
            if storage_settings.storage_backend == StorageBackendType.POSTGRESQL:
                self.logger.info("Creating PostgreSQL storage backend")
                backend = PostgreSQLStorageBackend(storage_settings.postgresql)
                self.logger.info("PostgreSQL storage backend created successfully")
                return backend
                
            elif storage_settings.storage_backend == StorageBackendType.SQLITE:
                self.logger.info("Creating SQLite storage backend")
                backend = SQLiteStorageBackend(storage_settings.sqlite)
                self.logger.info("SQLite storage backend created successfully")
                return backend
                
            else:
                error_msg = f"Unsupported storage backend type: {storage_settings.storage_backend}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            self.logger.error(f"Failed to create storage backend: {e}")
            raise
    
    def get_supported_backends(self) -> Dict[str, Any]:
        """Get information about supported storage backends.
        
        Returns:
            Dict containing information about supported backends
        """
        self.logger.debug("Getting supported storage backends information")
        
        supported = {
            "backends": [
                {
                    "type": StorageBackendType.POSTGRESQL,
                    "name": "PostgreSQL",
                    "description": "PostgreSQL database backend with async support",
                    "features": ["transactions", "concurrent_access", "scalability"]
                },
                {
                    "type": StorageBackendType.SQLITE,
                    "name": "SQLite",
                    "description": "SQLite database backend with async support",
                    "features": ["embedded", "file_based", "lightweight"]
                }
            ],
            "default": StorageBackendType.SQLITE
        }
        
        self.logger.debug(f"Supported backends: {[b['type'] for b in supported['backends']]}")
        return supported


# Global factory instance
storage_factory = StorageFactory()
