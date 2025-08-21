"""Storage backend factory for creating storage instances."""

import logging
from typing import Union

from .base import StorageBackend, StorageBackendType
from .postgresql import PostgreSQLStorageBackend
from .sqlite import SQLiteStorageBackend
from ..settings.settings import PostgresStorageSettings, SQLiteStorageSettings


class StorageFactory:
    """Factory for creating storage backend instances."""
    
    @staticmethod
    def create_storage_backend(
        backend_type: StorageBackendType,
        settings: Union[PostgresStorageSettings, SQLiteStorageSettings]
    ) -> StorageBackend:
        """Create a storage backend instance.
        
        Args:
            backend_type: Type of storage backend to create.
            settings: Storage backend specific settings.
            
        Returns:
            Storage backend instance.
            
        Raises:
            ValueError: If backend type is not supported.
        """
        logger = logging.getLogger(__name__)
        
        if backend_type == StorageBackendType.POSTGRESQL:
            if not isinstance(settings, PostgresStorageSettings):
                raise ValueError("PostgreSQL backend requires PostgresStorageSettings")
            logger.info("Creating PostgreSQL storage backend")
            return PostgreSQLStorageBackend(settings)
        
        elif backend_type == StorageBackendType.SQLITE:
            if not isinstance(settings, SQLiteStorageSettings):
                raise ValueError("SQLite backend requires SQLiteStorageSettings")
            logger.info("Creating SQLite storage backend")
            return SQLiteStorageBackend(settings)
        
        else:
            raise ValueError(f"Unsupported storage backend type: {backend_type}")
