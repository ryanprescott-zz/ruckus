"""HTTP client with retry and backoff logic."""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import httpx
from datetime import datetime

from ..config import HttpClientSettings


class HttpClientError(Exception):
    """Base exception for HTTP client errors."""
    pass


class ConnectionError(HttpClientError):
    """Exception raised when connection cannot be established."""
    pass


class ServiceUnavailableError(HttpClientError):
    """Exception raised when service returns 503 or similar after retries."""
    pass


class HttpClient:
    """HTTP client with configurable retry and backoff logic."""
    
    def __init__(self, settings: HttpClientSettings):
        """Initialize the HTTP client with settings.
        
        Args:
            settings: HTTP client configuration settings
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        # Create timeout configuration with all parameters explicitly set
        timeout_config = httpx.Timeout(
            connect=self.settings.connection_timeout,
            read=self.settings.read_timeout,
            write=self.settings.connection_timeout,
            pool=self.settings.connection_timeout
        )
        
        self._client = httpx.AsyncClient(timeout=timeout_config)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def get_with_retry(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a GET request with retry and backoff logic.
        
        Args:
            url: The URL to request
            headers: Optional HTTP headers
            
        Returns:
            JSON response as dictionary
            
        Raises:
            ConnectionError: If connection cannot be established
            ServiceUnavailableError: If service is unavailable after retries
            HttpClientError: For other HTTP errors
        """
        if not self._client:
            raise RuntimeError("HttpClient must be used as async context manager")
        
        last_exception = None
        backoff_delay = self.settings.initial_backoff
        
        for attempt in range(self.settings.max_retries + 1):  # +1 for initial attempt
            try:
                self.logger.debug(f"Making GET request to {url} (attempt {attempt + 1})")
                
                response = await self._client.get(url, headers=headers)
                
                # Check if response is successful
                if response.status_code == 200:
                    self.logger.debug(f"GET {url} successful")
                    return response.json()
                
                # Check if status code warrants retry
                if response.status_code in self.settings.retry_status_codes:
                    if attempt < self.settings.max_retries:
                        self.logger.warning(
                            f"GET {url} returned {response.status_code}, retrying in {backoff_delay}s "
                            f"(attempt {attempt + 1}/{self.settings.max_retries + 1})"
                        )
                        await asyncio.sleep(backoff_delay)
                        backoff_delay = min(backoff_delay * 2, self.settings.max_backoff)
                        continue
                    else:
                        self.logger.error(f"GET {url} failed with {response.status_code} after {self.settings.max_retries} retries")
                        raise ServiceUnavailableError(
                            f"Service unavailable: HTTP {response.status_code} after {self.settings.max_retries} retries"
                        )
                else:
                    # Non-retryable error
                    self.logger.error(f"GET {url} failed with non-retryable status {response.status_code}")
                    raise HttpClientError(f"HTTP {response.status_code}: {response.text}")
            
            except httpx.ConnectError as e:
                last_exception = e
                self.logger.error(f"Connection error for GET {url}: {str(e)}")
                # Don't retry connection errors, fail immediately
                raise ConnectionError(f"Cannot connect to {url}: {str(e)}")
            
            except httpx.TimeoutException as e:
                last_exception = e
                if attempt < self.settings.max_retries:
                    self.logger.warning(
                        f"Timeout for GET {url}, retrying in {backoff_delay}s "
                        f"(attempt {attempt + 1}/{self.settings.max_retries + 1})"
                    )
                    await asyncio.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 2, self.settings.max_backoff)
                    continue
                else:
                    self.logger.error(f"GET {url} timed out after {self.settings.max_retries} retries")
                    raise ServiceUnavailableError(f"Request timeout after {self.settings.max_retries} retries")
            
            except (ServiceUnavailableError, HttpClientError, ConnectionError):
                # Re-raise our own exceptions without wrapping
                raise
            
            except Exception as e:
                last_exception = e
                self.logger.error(f"Unexpected error for GET {url}: {str(e)}")
                raise HttpClientError(f"Unexpected error: {str(e)}")
        
        # Should not reach here, but just in case
        if last_exception:
            raise HttpClientError(f"Request failed: {str(last_exception)}")
        else:
            raise HttpClientError("Request failed for unknown reason")