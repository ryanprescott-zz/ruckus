"""RUCKUS experiment manager core implementation."""

import asyncio
import logging
import logging.config
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone

import yaml

from .config import ExperimentManagerSettings
from .storage.factory import storage_factory
from .storage.base import StorageBackend, ExperimentAlreadyExistsException, ExperimentNotFoundException, ExperimentHasJobsException
from ruckus_common.models import ExperimentSpec


class ExperimentManager:
    """Experiment management implementation.
    
    Responsible for creating and storing experiments, running experiment jobs 
    by calling agents, polling for experiment job status, and storing experiment 
    job results.
    """
    
    def __init__(self, settings: Optional[ExperimentManagerSettings] = None):
        """Initialize the experiment manager.
        
        Args:
            settings: Experiment manager configuration settings. If None, will load from environment.
        """
        self.settings = settings or ExperimentManagerSettings()
        self.logger = None  # Set up during start()
        
        # Core components
        self.storage_backend: Optional[StorageBackend] = None
        
        # State management
        self._started = False
    
    async def start(self) -> None:
        """Start the experiment manager backend."""
        if self._started:
            return
        
        # Set up logging first
        await self._setup_logging()
        
        self.logger.info("Experiment manager initialized")
        self.logger.info("Starting experiment manager backend...")
        
        try:
            # Initialize storage backend
            await self._setup_storage_backend()
            
            self._started = True
            self.logger.info("Experiment manager backend started")
            
        except Exception as e:
            self.logger.error(f"Failed to start experiment manager backend: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the experiment manager backend."""
        if not self._started:
            return
        
        self.logger.info("Stopping experiment manager backend...")
        
        try:
            # Clean up storage backend
            if self.storage_backend:
                self.logger.info("Cleaning up storage backend...")
                await self.storage_backend.close()
                self.storage_backend = None
            
            self._started = False
            self.logger.info("Experiment manager backend stopped")
            
        except Exception as e:
            self.logger.error(f"Error during experiment manager shutdown: {e}")
    
    async def _setup_logging(self) -> None:
        """Set up logging configuration."""
        # Try to load logging config from file if it exists
        log_config_path = Path(self.settings.log_config_file)
        if log_config_path.exists():
            try:
                with open(log_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logging.config.dictConfig(config)
            except Exception as e:
                # Fall back to basic configuration
                logging.basicConfig(
                    level=self.settings.log_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                print(f"Warning: Failed to load logging config from {log_config_path}: {e}")
        else:
            # Set up basic logging
            logging.basicConfig(
                level=self.settings.log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        self.logger = logging.getLogger(__name__)
    
    async def _setup_storage_backend(self) -> None:
        """Set up and initialize storage backend."""
        self.logger.info("Setting up storage backend...")
        
        # Create storage backend instance
        self.storage_backend = storage_factory.create_storage_backend(self.settings.storage)
        
        # Initialize the backend
        await self.storage_backend.initialize()
        
        self.logger.info(f"Storage backend ({self.settings.storage.storage_backend}) initialized successfully")
    
    async def create_experiment(self, experiment_spec: ExperimentSpec) -> Dict[str, any]:
        """Create and store a new experiment.
        
        Args:
            experiment_spec: ExperimentSpec object containing experiment details
            
        Returns:
            Dict containing the created experiment data with experiment_id and created_at
            
        Raises:
            ExperimentAlreadyExistsException: If experiment with same ID already exists
            RuntimeError: If experiment manager is not started
        """
        if not self._started:
            raise RuntimeError("Experiment manager not started")
        
        self.logger.info(f"Creating experiment {experiment_spec.experiment_id}")
        
        try:
            # Call storage backend to create experiment
            result = await self.storage_backend.create_experiment(experiment_spec)
            
            self.logger.info(f"Experiment {experiment_spec.experiment_id} created successfully")
            return result
            
        except ExperimentAlreadyExistsException as e:
            self.logger.error(f"Failed to create experiment: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to create experiment {experiment_spec.experiment_id}: {e}")
            raise
    
    async def delete_experiment(self, experiment_id: str) -> Dict[str, any]:
        """Delete an existing experiment.
        
        Args:
            experiment_id: ID of the experiment to delete
            
        Returns:
            Dict containing the deleted experiment data with experiment_id and deleted_at
            
        Raises:
            ExperimentNotFoundException: If experiment with given ID doesn't exist
            ExperimentHasJobsException: If experiment has associated jobs
            RuntimeError: If experiment manager is not started
        """
        if not self._started:
            raise RuntimeError("Experiment manager not started")
        
        self.logger.info(f"Deleting experiment {experiment_id}")
        
        try:
            # Call storage backend to delete experiment
            result = await self.storage_backend.delete_experiment(experiment_id)
            
            self.logger.info(f"Experiment {experiment_id} deleted successfully")
            return result
            
        except (ExperimentNotFoundException, ExperimentHasJobsException) as e:
            self.logger.error(f"Failed to delete experiment: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            raise
    
    async def list_experiments(self):
        """List all existing experiments.
        
        Returns:
            List of ExperimentSpec objects
            
        Raises:
            RuntimeError: If experiment manager is not started
        """
        if not self._started:
            raise RuntimeError("Experiment manager not started")
        
        self.logger.info("Listing all experiments")
        
        try:
            # Call storage backend to list experiments
            experiments = await self.storage_backend.list_experiments()
            
            self.logger.info(f"Found {len(experiments)} experiments")
            return experiments
            
        except Exception as e:
            self.logger.error(f"Failed to list experiments: {e}")
            raise
    
    async def get_experiment(self, experiment_id: str):
        """Get an existing experiment by ID.
        
        Args:
            experiment_id: ID of the experiment to retrieve
            
        Returns:
            ExperimentSpec object
            
        Raises:
            ExperimentNotFoundException: If experiment with given ID doesn't exist
            RuntimeError: If experiment manager is not started
        """
        if not self._started:
            raise RuntimeError("Experiment manager not started")
        
        self.logger.info(f"Retrieving experiment {experiment_id}")
        
        try:
            # Call storage backend to get experiment
            experiment_spec = await self.storage_backend.get_experiment(experiment_id)
            
            self.logger.info(f"Experiment {experiment_id} retrieved successfully")
            return experiment_spec
            
        except ExperimentNotFoundException as e:
            self.logger.error(f"Experiment not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to retrieve experiment {experiment_id}: {e}")
            raise