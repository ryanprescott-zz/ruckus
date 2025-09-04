"""Simple HTTP client for status checks without retries."""

import asyncio
import logging
from typing import Dict, Any, Optional
import httpx


class SimpleHttpClient:
    """Simple HTTP client for quick status checks without retry logic."""
    
    def __init__(self, timeout_seconds: float = 5.0):
        """Initialize the simple HTTP client.
        
        Args:
            timeout_seconds: Timeout for HTTP requests in seconds
        """
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)
    
    async def get_json(self, url: str) -> Optional[Dict[str, Any]]:
        """Make a GET request and return JSON response.
        
        Args:
            url: URL to request
            
        Returns:
            JSON response as dict, or None if request failed
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.logger.debug(f"HTTP GET request to {url} failed: {e}")
            return None
    
    async def post_json(self, url: str, data: Dict[str, Any]) -> bool:
        """Make a POST request with JSON data.
        
        Args:
            url: URL to post to
            data: JSON data to post
            
        Returns:
            True if request succeeded (2xx status), False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=data)
                response.raise_for_status()
                return True
        except Exception as e:
            self.logger.debug(f"HTTP POST request to {url} failed: {e}")
            return False