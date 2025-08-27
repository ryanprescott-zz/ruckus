"""Tests for HTTP client implementations."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
import httpx

from ruckus_server.core.clients.http import HttpClient, ConnectionError, ServiceUnavailableError
from ruckus_server.core.clients.simple_http import SimpleHttpClient


class TestHttpClient:
    """Tests for HttpClient with retry logic."""

    @pytest.mark.asyncio
    async def test_successful_get_request(self, http_client_settings):
        """Test successful GET request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = Mock()

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            async with HttpClient(http_client_settings) as client:
                result = await client.get_with_retry("http://test.com/api")

            assert result == {"status": "ok"}
            mock_client.get.assert_called_once_with("http://test.com/api", headers=None)

    @pytest.mark.asyncio
    async def test_connection_error_on_network_failure(self, http_client_settings):
        """Test ConnectionError raised on network failures."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            async with HttpClient(http_client_settings) as client:
                with pytest.raises(ConnectionError) as exc_info:
                    await client.get_with_retry("http://test.com/api")
            
            assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_service_unavailable_on_503(self, http_client_settings):
        """Test ServiceUnavailableError raised on 503 status."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service Unavailable", 
            request=Mock(), 
            response=mock_response
        )

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            async with HttpClient(http_client_settings) as client:
                with pytest.raises(ServiceUnavailableError) as exc_info:
                    await client.get_with_retry("http://test.com/api")
            
            assert "Service unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_on_temporary_failure(self, http_client_settings):
        """Test retry logic on temporary failures."""
        # First call fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", 
            request=Mock(), 
            response=mock_response_fail
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"status": "ok"}
        mock_response_success.raise_for_status = Mock()

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = [mock_response_fail, mock_response_success]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # Use shorter delays for testing
            http_client_settings.initial_backoff = 0.001
            async with HttpClient(http_client_settings) as client:
                result = await client.get_with_retry("http://test.com/api")

            assert result == {"status": "ok"}
            assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, http_client_settings):
        """Test failure after max retries exceeded."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", 
            request=Mock(), 
            response=mock_response
        )

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # Set low retry count and delay for testing
            http_client_settings.max_retries = 2
            http_client_settings.initial_backoff = 0.001
            async with HttpClient(http_client_settings) as client:
                with pytest.raises(ServiceUnavailableError):
                    await client.get_with_retry("http://test.com/api")
            
            # Should be called max_retries + 1 times (initial + retries)
            assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_handling(self, http_client_settings):
        """Test timeout handling."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("Request timeout")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            async with HttpClient(http_client_settings) as client:
                with pytest.raises(ServiceUnavailableError) as exc_info:
                    await client.get_with_retry("http://test.com/api")
            
            assert "Request timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_exponential_backoff(self, http_client_settings):
        """Test exponential backoff between retries."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", 
            request=Mock(), 
            response=mock_response
        )

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            http_client_settings.max_retries = 2
            http_client_settings.initial_backoff = 0.01
            http_client_settings.max_backoff = 0.02
            
            start_time = asyncio.get_event_loop().time()
            
            async with HttpClient(http_client_settings) as client:
                with pytest.raises(ServiceUnavailableError):
                    await client.get_with_retry("http://test.com/api")
            
            end_time = asyncio.get_event_loop().time()
            
            # Should have waited: 0.01 + 0.02 = 0.03 seconds minimum
            assert end_time - start_time >= 0.025


class TestSimpleHttpClient:
    """Tests for SimpleHttpClient without retry logic."""

    @pytest.mark.asyncio
    async def test_successful_get_json(self):
        """Test successful JSON GET request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status = Mock()

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            client = SimpleHttpClient(timeout_seconds=5.0)
            result = await client.get_json("http://test.com/api")

            assert result == {"data": "test"}
            mock_client.get.assert_called_once_with("http://test.com/api")

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self):
        """Test connection error returns None."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            client = SimpleHttpClient(timeout_seconds=5.0)
            result = await client.get_json("http://test.com/api")

            assert result is None

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self):
        """Test HTTP error returns None."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", 
            request=Mock(), 
            response=mock_response
        )

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            client = SimpleHttpClient(timeout_seconds=5.0)
            result = await client.get_json("http://test.com/api")

            assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        """Test timeout returns None."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("Request timeout")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            client = SimpleHttpClient(timeout_seconds=1.0)
            result = await client.get_json("http://test.com/api")

            assert result is None

    @pytest.mark.asyncio
    async def test_json_decode_error_returns_none(self):
        """Test JSON decode error returns None."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = Mock()

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            client = SimpleHttpClient(timeout_seconds=5.0)
            result = await client.get_json("http://test.com/api")

            assert result is None

    @pytest.mark.asyncio
    async def test_timeout_configuration(self):
        """Test timeout configuration is passed to httpx."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {"test": "data"}
            
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            timeout = 10.0
            client = SimpleHttpClient(timeout_seconds=timeout)
            result = await client.get_json("http://test.com/api")

            # Verify the result and that AsyncClient was called with correct timeout
            assert result == {"test": "data"}
            mock_client_class.assert_called_once_with(timeout=timeout)