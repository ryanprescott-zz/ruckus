"""Integration tests for ruckus_server agent functionality."""

import pytest
import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, AsyncMock, Mock

import httpx
from fastapi.testclient import TestClient

from ruckus_server.core.server import RuckusServer
from ruckus_server.core.config import RuckusServerSettings, SQLiteSettings
from ruckus_common.models import AgentType, AgentStatus, AgentStatusEnum
from ruckus_server.api.v1.api import api_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_test_app():
    """Create a fresh FastAPI app instance for testing."""
    test_app = FastAPI(title="RUCKUS Server Test", version="0.1.0")
    test_app.include_router(api_router, prefix="/api/v1")
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return test_app


class TestAgentRegistrationIntegration:
    """Integration tests for agent registration workflow."""

    @pytest.mark.asyncio
    async def test_full_agent_registration_workflow(self, temp_db_path):
        """Test complete agent registration from API to storage."""
        # Create server with temporary database
        sqlite_settings = SQLiteSettings(database_path=temp_db_path)
        server_settings = RuckusServerSettings(
            storage_backend="sqlite",
            sqlite=sqlite_settings
        )
        
        # Start server
        server = RuckusServer(server_settings)
        await server.start()
        
        try:
            # Mock agent responses
            mock_agent_response = {
                "agent_info": {
                    "agent_id": "integration-test-agent",
                    "agent_name": "Integration Test Agent",
                    "agent_type": "white_box",
                    "system_info": {"hostname": "test-host"},
                    "capabilities": {"frameworks": ["pytorch"]},
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            }
            
            with patch('httpx.AsyncClient') as mock_httpx_client_class:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_agent_response
                mock_response.raise_for_status = Mock()
                
                mock_httpx_client = AsyncMock()
                mock_httpx_client.get.return_value = mock_response
                mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
                mock_httpx_client.__aexit__ = AsyncMock(return_value=None)
                mock_httpx_client_class.return_value = mock_httpx_client
                
                # Register agent
                result = await server.register_agent("http://test-agent:8001")
                
                # Verify registration result
                assert "agent_id" in result
                assert result["agent_id"] == "integration-test-agent"
                
                # Verify agent is in database
                agents = await server.list_registered_agent_info()
                assert len(agents) == 1
                assert agents[0].agent_id == "integration-test-agent"
                
                # Verify agent can be retrieved individually
                agent = await server.get_registered_agent_info("integration-test-agent")
                assert agent.agent_id == "integration-test-agent"
                assert agent.agent_url == "http://test-agent:8001"
        
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_agent_status_integration(self, temp_db_path):
        """Test agent status retrieval integration."""
        # Create and start server
        sqlite_settings = SQLiteSettings(database_path=temp_db_path)
        server_settings = RuckusServerSettings(
            storage_backend="sqlite",
            sqlite=sqlite_settings
        )
        
        server = RuckusServer(server_settings)
        await server.start()
        
        try:
            # First register an agent
            mock_info_response = {
                "agent_info": {
                    "agent_id": "status-test-agent",
                    "agent_name": "Status Test Agent",
                    "agent_type": "white_box",
                    "system_info": {},
                    "capabilities": {},
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            }
            
            with patch('httpx.AsyncClient') as mock_httpx_client_class:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_info_response
                mock_response.raise_for_status = Mock()
                
                mock_httpx_client = AsyncMock()
                mock_httpx_client.get.return_value = mock_response
                mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
                mock_httpx_client.__aexit__ = AsyncMock(return_value=None)
                mock_httpx_client_class.return_value = mock_httpx_client
                
                await server.register_agent("http://status-agent:8001")
            
            # Now test status retrieval
            mock_status_response = {
                "agent_id": "status-test-agent",
                "status": "idle",
                "running_jobs": [],
                "queued_jobs": ["job-1"],
                "uptime_seconds": 3600.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            with patch('ruckus_server.core.server.SimpleHttpClient') as mock_simple_client_class:
                mock_simple_client = AsyncMock()
                mock_simple_client.get_json.return_value = mock_status_response
                mock_simple_client_class.return_value = mock_simple_client
                
                # Test individual agent status
                status = await server.get_registered_agent_status("status-test-agent")
                assert status.agent_id == "status-test-agent"
                assert status.status == AgentStatusEnum.IDLE
                assert status.queued_jobs == ["job-1"]
                
                # Test list all agent statuses
                statuses = await server.list_registered_agent_status()
                assert len(statuses) == 1
                assert statuses[0].agent_id == "status-test-agent"
        
        finally:
            await server.stop()

    def test_api_to_storage_integration(self, temp_db_path):
        """Test full API request to storage integration."""
        # Create server with temporary database
        sqlite_settings = SQLiteSettings(database_path=temp_db_path)
        server_settings = RuckusServerSettings(
            storage_backend="sqlite",
            sqlite=sqlite_settings
        )
        
        # Initialize app with server
        server = RuckusServer(server_settings)
        
        # Use asyncio to handle server lifecycle
        async def run_test():
            await server.start()
            try:
                # Create fresh app instance for this test
                test_app = create_test_app()
                test_app.state.server = server
                
                with TestClient(test_app) as client:
                    # Mock agent info response
                    import time
                    agent_id = f"api-integration-agent-{int(time.time())}"
                    mock_response = {
                        "agent_info": {
                            "agent_id": agent_id,
                            "agent_name": "API Integration Agent",
                            "agent_type": "black_box",
                            "system_info": {"os": "linux"},
                            "capabilities": {"gpu_count": 1},
                            "last_updated": datetime.now(timezone.utc).isoformat()
                        }
                    }
                    
                    with patch('httpx.AsyncClient') as mock_httpx_client_class:
                        mock_http_response = Mock()
                        mock_http_response.status_code = 200
                        mock_http_response.json.return_value = mock_response
                        mock_http_response.raise_for_status = Mock()
                        
                        mock_httpx_client = AsyncMock()
                        mock_httpx_client.get.return_value = mock_http_response
                        mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
                        mock_httpx_client.__aexit__ = AsyncMock(return_value=None)
                        mock_httpx_client_class.return_value = mock_httpx_client
                        
                        # Test registration via API
                        response = client.post(
                            "/api/v1/agents/register",
                            json={"agent_url": f"http://{agent_id}:8001"}
                        )
                        assert response.status_code == 200
                        registration_data = response.json()
                        assert registration_data["agent_id"] == agent_id
                        
                        # Test listing via API
                        response = client.get("/api/v1/agents/")
                        assert response.status_code == 200
                        list_data = response.json()
                        assert len(list_data["agents"]) == 1
                        assert list_data["agents"][0]["agent_id"] == agent_id
                        
                        # Test individual retrieval via API
                        response = client.get(f"/api/v1/agents/{agent_id}/info")
                        assert response.status_code == 200
                        info_data = response.json()
                        assert info_data["agent"]["agent_id"] == agent_id
                
                test_app.state.server = None
            finally:
                await server.stop()
        
        asyncio.run(run_test())

    @pytest.mark.asyncio
    async def test_concurrent_agent_registration(self, temp_db_path):
        """Test concurrent agent registrations don't interfere."""
        sqlite_settings = SQLiteSettings(database_path=temp_db_path)
        server_settings = RuckusServerSettings(
            storage_backend="sqlite",
            sqlite=sqlite_settings
        )
        
        server = RuckusServer(server_settings)
        await server.start()
        
        try:
            # Create mock responses for multiple agents
            import time
            base_timestamp = int(time.time())
            def create_mock_response(agent_num):
                return {
                    "agent_info": {
                        "agent_id": f"concurrent-agent-{base_timestamp}-{agent_num}",
                        "agent_name": f"Concurrent Agent {agent_num}",
                        "agent_type": "white_box",
                        "system_info": {},
                        "capabilities": {},
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                }
            
            with patch('httpx.AsyncClient') as mock_httpx_client_class:
                mock_httpx_client = AsyncMock()
                # Configure side_effect to return different responses
                responses = []
                for i in range(5):
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = create_mock_response(i)
                    mock_response.raise_for_status = Mock()
                    responses.append(mock_response)
                mock_httpx_client.get.side_effect = responses
                mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
                mock_httpx_client.__aexit__ = AsyncMock(return_value=None)
                mock_httpx_client_class.return_value = mock_httpx_client
                
                # Register multiple agents concurrently
                tasks = [
                    server.register_agent(f"http://concurrent-agent-{base_timestamp}-{i}:8001")
                    for i in range(5)
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Verify all registrations succeeded
                assert len(results) == 5
                agent_ids = [result["agent_id"] for result in results]
                expected_ids = [f"concurrent-agent-{base_timestamp}-{i}" for i in range(5)]
                assert set(agent_ids) == set(expected_ids)
                
                # Verify all agents are in storage
                stored_agents = await server.list_registered_agent_info()
                assert len(stored_agents) == 5
        
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, temp_db_path):
        """Test error handling across the integration stack."""
        import time
        agent_id = f"error-test-agent-{int(time.time())}"
        
        sqlite_settings = SQLiteSettings(database_path=temp_db_path)
        server_settings = RuckusServerSettings(
            storage_backend="sqlite",
            sqlite=sqlite_settings
        )
        
        server = RuckusServer(server_settings)
        await server.start()
        
        try:
            # Test various error scenarios
            
            # 1. Network error during registration
            with patch('httpx.AsyncClient') as mock_httpx_client_class:
                mock_httpx_client = AsyncMock()
                mock_httpx_client.get.side_effect = httpx.ConnectError("Connection refused")
                mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
                mock_httpx_client.__aexit__ = AsyncMock(return_value=None)
                mock_httpx_client_class.return_value = mock_httpx_client
                
                from ruckus_server.core.clients.http import ConnectionError
                with pytest.raises(ConnectionError):
                    await server.register_agent("http://unreachable:8001")
            
            # 2. Register an agent successfully first
            mock_response = {
                "agent_info": {
                    "agent_id": agent_id,
                    "agent_name": "Error Test Agent",
                    "agent_type": "white_box",
                    "system_info": {},
                    "capabilities": {},
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            }
            
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_http_response = Mock()
                mock_http_response.status_code = 200
                mock_http_response.json.return_value = mock_response
                mock_http_response.raise_for_status = Mock()
                
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_http_response
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client
                
                await server.register_agent(f"http://{agent_id}:8001")
            
            # 3. Try to register same agent again (duplicate error)
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_http_response = Mock()
                mock_http_response.status_code = 200
                mock_http_response.json.return_value = mock_response
                mock_http_response.raise_for_status = Mock()
                
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_http_response
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client
                
                from ruckus_server.core.server import AgentAlreadyRegisteredException
                with pytest.raises(AgentAlreadyRegisteredException):
                    await server.register_agent(f"http://{agent_id}:8001")
            
            # 4. Try to get status for non-existent agent
            from ruckus_server.core.server import AgentNotRegisteredException
            with pytest.raises(AgentNotRegisteredException):
                await server.get_registered_agent_status("non-existent-agent")
            
            # 5. Status retrieval with agent unreachable
            with patch('ruckus_server.core.server.SimpleHttpClient') as mock_simple_client_class:
                mock_simple_client = AsyncMock()
                mock_simple_client.get_json.return_value = None  # Simulate connection failure
                mock_simple_client_class.return_value = mock_simple_client
                
                status = await server.get_registered_agent_status(agent_id)
                assert status.status == AgentStatusEnum.UNAVAILABLE
                assert status.uptime_seconds == 0.0
        
        finally:
            await server.stop()

    def test_database_persistence_integration(self):
        """Test that data persists across server restarts."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            async def test_persistence():
                import time
                agent_id = f"persistent-agent-{int(time.time())}"
                
                # First server instance - register agent
                sqlite_settings1 = SQLiteSettings(database_path=db_path)
                server_settings1 = RuckusServerSettings(
                    storage_backend="sqlite",
                    sqlite=sqlite_settings1
                )
                
                server1 = RuckusServer(server_settings1)
                await server1.start()
                
                mock_response = {
                    "agent_info": {
                        "agent_id": agent_id,
                        "agent_name": "Persistent Agent",
                        "agent_type": "white_box",
                        "system_info": {"test": "data"},
                        "capabilities": {"persistent": True},
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                }
                
                with patch('httpx.AsyncClient') as mock_client_class:
                    mock_http_response = Mock()
                    mock_http_response.status_code = 200
                    mock_http_response.json.return_value = mock_response
                    mock_http_response.raise_for_status = Mock()
                    
                    mock_client = AsyncMock()
                    mock_client.get.return_value = mock_http_response
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client
                    
                    await server1.register_agent(f"http://{agent_id}:8001")
                
                await server1.stop()
                
                # Second server instance - verify agent still exists
                sqlite_settings2 = SQLiteSettings(database_path=db_path)
                server_settings2 = RuckusServerSettings(
                    storage_backend="sqlite",
                    sqlite=sqlite_settings2
                )
                
                server2 = RuckusServer(server_settings2)
                await server2.start()
                
                agents = await server2.list_registered_agent_info()
                assert len(agents) == 1
                assert agents[0].agent_id == agent_id
                assert agents[0].capabilities["persistent"] is True
                
                await server2.stop()
            
            asyncio.run(test_persistence())
        
        finally:
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_health_check_integration(self, temp_db_path):
        """Test health check reflects actual system state."""
        sqlite_settings = SQLiteSettings(database_path=temp_db_path)
        server_settings = RuckusServerSettings(
            storage_backend="sqlite",
            sqlite=sqlite_settings
        )
        
        server = RuckusServer(server_settings)
        
        # Test before starting
        health = await server.health_check()
        assert health["status"] == "unhealthy"
        assert health["storage"] == "not_initialized"
        assert health["agents"] == 0
        
        # Start server and test
        await server.start()
        
        try:
            health = await server.health_check()
            assert health["status"] == "healthy"
            assert health["storage"] == "healthy"
            assert health["agents"] == 0
            
            # Register an agent and test again
            mock_response = {
                "agent_info": {
                    "agent_id": "health-test-agent",
                    "agent_name": "Health Test Agent",
                    "agent_type": "white_box",
                    "system_info": {},
                    "capabilities": {},
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            }
            
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_http_response = Mock()
                mock_http_response.status_code = 200
                mock_http_response.json.return_value = mock_response
                mock_http_response.raise_for_status = Mock()
                
                mock_client = AsyncMock()
                mock_client.get.return_value = mock_http_response
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client
                
                await server.register_agent("http://health-test-agent:8001")
            
            health = await server.health_check()
            assert health["status"] == "healthy"
            assert health["storage"] == "healthy"
            assert health["agents"] == 1
        
        finally:
            await server.stop()

    def test_full_api_workflow_integration(self, temp_db_path):
        """Test complete API workflow from registration to status retrieval."""
        sqlite_settings = SQLiteSettings(database_path=temp_db_path)
        server_settings = RuckusServerSettings(
            storage_backend="sqlite",
            sqlite=sqlite_settings
        )
        
        server = RuckusServer(server_settings)
        
        async def run_workflow():
            await server.start()
            try:
                # Create fresh app instance for this test
                test_app = create_test_app()
                test_app.state.server = server
                
                with TestClient(test_app) as client:
                    # 1. Register agent
                    import time
                    agent_id = f"workflow-agent-{int(time.time())}"
                    mock_info_response = {
                        "agent_info": {
                            "agent_id": agent_id,
                            "agent_name": "Workflow Agent",
                            "agent_type": "gray_box",
                            "system_info": {"workflow": "test"},
                            "capabilities": {"complete": True},
                            "last_updated": datetime.now(timezone.utc).isoformat()
                        }
                    }
                    
                    with patch('httpx.AsyncClient') as mock_client_class:
                        mock_http_response = Mock()
                        mock_http_response.status_code = 200
                        mock_http_response.json.return_value = mock_info_response
                        mock_http_response.raise_for_status = Mock()
                        
                        mock_client = AsyncMock()
                        mock_client.get.return_value = mock_http_response
                        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                        mock_client.__aexit__ = AsyncMock(return_value=None)
                        mock_client_class.return_value = mock_client
                        
                        register_response = client.post(
                            "/api/v1/agents/register",
                            json={"agent_url": f"http://{agent_id}:8001"}
                        )
                        assert register_response.status_code == 200
                    
                    # 2. List agents
                    list_response = client.get("/api/v1/agents/")
                    assert list_response.status_code == 200
                    assert len(list_response.json()["agents"]) == 1
                    
                    # 3. Get agent info
                    info_response = client.get(f"/api/v1/agents/{agent_id}/info")
                    assert info_response.status_code == 200
                    assert info_response.json()["agent"]["agent_id"] == agent_id
                    
                    # 4. Get agent status
                    mock_status_response = {
                        "agent_id": agent_id,
                        "status": "active",
                        "running_jobs": ["job-1", "job-2"],
                        "queued_jobs": [],
                        "uptime_seconds": 7200.0,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    
                    with patch('ruckus_server.core.server.SimpleHttpClient') as mock_simple_client_class:
                        mock_simple_client = AsyncMock()
                        mock_simple_client.get_json.return_value = mock_status_response
                        mock_simple_client_class.return_value = mock_simple_client
                        
                        status_response = client.get(f"/api/v1/agents/{agent_id}/status")
                        assert status_response.status_code == 200
                        status_data = status_response.json()
                        assert status_data["agent"]["status"] == "active"
                        assert status_data["agent"]["running_jobs"] == ["job-1", "job-2"]
                    
                    # 5. List all statuses
                    with patch('ruckus_server.core.server.SimpleHttpClient') as mock_simple_client_class:
                        mock_simple_client = AsyncMock()
                        mock_simple_client.get_json.return_value = mock_status_response
                        mock_simple_client_class.return_value = mock_simple_client
                        
                        list_status_response = client.get("/api/v1/agents/status")
                        assert list_status_response.status_code == 200
                        assert len(list_status_response.json()["agents"]) == 1
                    
                    # 6. Unregister agent
                    unregister_response = client.post(
                        "/api/v1/agents/unregister",
                        json={"agent_id": agent_id}
                    )
                    assert unregister_response.status_code == 200
                    
                    # 7. Verify agent is gone
                    final_list_response = client.get("/api/v1/agents/")
                    assert final_list_response.status_code == 200
                    assert len(final_list_response.json()["agents"]) == 0
                
                test_app.state.server = None
            finally:
                await server.stop()
        
        asyncio.run(run_workflow())