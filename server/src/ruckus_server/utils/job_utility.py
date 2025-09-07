"""Job utility functions."""

import uuid
from typing import Optional


class JobUtility:
    """Utility class for job-related operations."""
    
    @staticmethod
    def generate_job_id() -> str:
        """Generate a unique job ID.
        
        Returns:
            str: A unique job ID with format 'job_<8-digit-uuid>'
        """
        # Generate a UUID and take the first 8 characters of the hex representation
        unique_id = uuid.uuid4().hex[:8]
        return f"job_{unique_id}"