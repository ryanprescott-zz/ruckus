"""TTL-based result cache for agent job results."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

from ruckus_common.models import JobResultType


logger = logging.getLogger(__name__)


@dataclass
class CachedResult:
    """Container for cached job result with metadata."""
    job_id: str
    result: Dict[str, Any]
    result_type: JobResultType
    stored_at: datetime
    result_created_at: datetime


class TTLResultCache:
    """Time-based cache for job results with automatic cleanup."""
    
    def __init__(self, ttl_hours: int = 24, cleanup_interval_minutes: int = 5):
        """Initialize the result cache.
        
        Args:
            ttl_hours: Time-to-live for cached results in hours
            cleanup_interval_minutes: Interval for periodic cleanup in minutes
        """
        self.ttl_hours = ttl_hours
        self.cleanup_interval_minutes = cleanup_interval_minutes
        
        # Storage: job_id -> CachedResult
        self._cache: Dict[str, CachedResult] = {}
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"TTL result cache initialized: TTL={ttl_hours}h, cleanup_interval={cleanup_interval_minutes}m")
    
    async def start(self):
        """Start the background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("TTL result cache background cleanup started")
    
    async def stop(self):
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("TTL result cache background cleanup stopped")
    
    def store(self, job_id: str, result: Dict[str, Any], result_type: JobResultType) -> str:
        """Store a job result in the cache.
        
        Args:
            job_id: Job identifier
            result: Result data to cache
            result_type: Type of result (success, failure, etc.)
            
        Returns:
            Actual job_id used for storage (may have timestamp suffix if duplicate)
        """
        now = datetime.now(timezone.utc)
        
        # Handle potential duplicate job IDs with timestamp suffix
        actual_job_id = job_id
        if job_id in self._cache:
            timestamp = int(now.timestamp())
            actual_job_id = f"{job_id}_{timestamp}"
            logger.warning(f"Duplicate job_id detected, using: {actual_job_id}")
        
        # Add result creation timestamp to the result data
        enhanced_result = {
            **result,
            "result_created_at": now.isoformat(),
            "result_type": result_type.value
        }
        
        # Store in cache
        cached_result = CachedResult(
            job_id=actual_job_id,
            result=enhanced_result,
            result_type=result_type,
            stored_at=now,
            result_created_at=now
        )
        
        self._cache[actual_job_id] = cached_result
        
        logger.info(f"Cached job result: {actual_job_id} (type: {result_type.value})")
        logger.debug(f"Cache size: {len(self._cache)} results")
        
        return actual_job_id
    
    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a job result from cache.
        
        Args:
            job_id: Job identifier to retrieve
            
        Returns:
            Job result data or None if not found/expired
        """
        # Clean expired entries on access
        self._cleanup_expired()
        
        cached_result = self._cache.get(job_id)
        if cached_result is None:
            logger.debug(f"Job result not found in cache: {job_id}")
            return None
        
        logger.debug(f"Retrieved job result from cache: {job_id}")
        return cached_result.result
    
    def list_available_results(self) -> List[Dict[str, Any]]:
        """Get list of all available results with metadata.
        
        Returns:
            List of result metadata (job_id, completed_at, result_type)
        """
        # Clean expired entries before listing
        self._cleanup_expired()
        
        available = []
        for cached_result in self._cache.values():
            available.append({
                "job_id": cached_result.job_id,
                "completed_at": cached_result.result_created_at.isoformat(),
                "result_type": cached_result.result_type.value
            })
        
        logger.debug(f"Listed {len(available)} available results")
        return available
    
    def clear_all(self):
        """Clear all cached results."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared all cached results ({count} items)")
    
    def _cleanup_expired(self):
        """Remove expired results from cache."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours)
        
        expired_ids = [
            job_id for job_id, cached_result in self._cache.items()
            if cached_result.stored_at < cutoff
        ]
        
        for job_id in expired_ids:
            del self._cache[job_id]
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired results")
            logger.debug(f"Cache size after cleanup: {len(self._cache)} results")
    
    async def _periodic_cleanup(self):
        """Background task for periodic cache cleanup."""
        logger.info(f"Starting periodic cleanup task (interval: {self.cleanup_interval_minutes}m)")
        
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                self._cleanup_expired()
            except asyncio.CancelledError:
                logger.debug("Periodic cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                # Continue running despite errors
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for debugging/monitoring.
        
        Returns:
            Cache statistics including size, oldest entry, etc.
        """
        if not self._cache:
            return {
                "size": 0,
                "ttl_hours": self.ttl_hours,
                "cleanup_interval_minutes": self.cleanup_interval_minutes
            }
        
        oldest_entry = min(self._cache.values(), key=lambda x: x.stored_at)
        
        result_types = {}
        for cached_result in self._cache.values():
            result_type = cached_result.result_type.value
            result_types[result_type] = result_types.get(result_type, 0) + 1
        
        return {
            "size": len(self._cache),
            "ttl_hours": self.ttl_hours,
            "cleanup_interval_minutes": self.cleanup_interval_minutes,
            "oldest_entry_age_hours": (
                datetime.now(timezone.utc) - oldest_entry.stored_at
            ).total_seconds() / 3600,
            "result_types": result_types
        }