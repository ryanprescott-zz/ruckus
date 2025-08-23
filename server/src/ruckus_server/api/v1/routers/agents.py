"""Agent management endpoints."""

from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional
from datetime import datetime

from ruckus_common.models import AgentCapabilitiesBase
from ..models import RegisterAgentRequest, RegisterAgentResponse, UnregisterAgentRequest, UnregisterAgentResponse
from ruckus_server.core.clients.http import ConnectionError, ServiceUnavailableError
from ruckus_server.core.server import AgentAlreadyRegisteredException, AgentNotRegisteredException

router = APIRouter()


@router.post("/register", response_model=RegisterAgentResponse)
async def register_agent(request_data: RegisterAgentRequest, request: Request):
    """Register a new agent with the server.
    
    Args:
        request_data: RegisterAgentRequest containing agent URL
        request: FastAPI request object to access app state
        
    Returns:
        RegisterAgentResponse containing registered agent information
        
    Raises:
        HTTPException: 400 for invalid URLs, 404 for connection errors, 503 for unavailable agents
    """
    server = request.app.state.server
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        # Call the RuckusServer register_agent method
        registration_result = await server.register_agent(request_data.agent_url)
        
        # Return success response
        return RegisterAgentResponse(
            agent_id=registration_result["agent_id"],
            registered_at=registration_result["registered_at"]
        )
        
    except AgentAlreadyRegisteredException as e:
        # Agent is already registered - return 409 conflict
        raise HTTPException(
            status_code=409,
            detail=f"Agent {e.agent_id} is already registered (registered at {e.registered_at})"
        )
    except ConnectionError as e:
        # Agent is unreachable - return 404 with details
        raise HTTPException(
            status_code=404,
            detail=f"Agent not found or unreachable at {request_data.agent_url}: {str(e)}"
        )
    except ServiceUnavailableError as e:
        # Agent returned 503 or similar after retries
        raise HTTPException(
            status_code=503,
            detail=f"Agent at {request_data.agent_url} is temporarily unavailable: {str(e)}"
        )
    except ValueError as e:
        # Invalid agent data or other validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/unregister", response_model=UnregisterAgentResponse)
async def unregister_agent(request_data: UnregisterAgentRequest, request: Request):
    """Unregister an agent from the server.
    
    Args:
        request_data: UnregisterAgentRequest containing agent ID
        request: FastAPI request object to access app state
        
    Returns:
        UnregisterAgentResponse containing unregistered agent information
        
    Raises:
        HTTPException: 404 if agent not found, 503 if server not initialized, 500 for other errors
    """
    server = request.app.state.server
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        # Call the RuckusServer unregister_agent method
        unregistration_result = await server.unregister_agent(request_data.agent_id)
        
        # Return success response
        return UnregisterAgentResponse(
            agent_id=unregistration_result["agent_id"],
            unregistered_at=unregistration_result["unregistered_at"]
        )
        
    except AgentNotRegisteredException as e:
        # Agent is not registered - return 404
        raise HTTPException(
            status_code=404,
            detail=f"No agent with ID {e.agent_id} is registered"
        )
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/")
async def list_agents():
    """List all registered agents."""
    # TODO: Implement agent listing
    return {
        "agents": [],
        "total": 0,
    }


@router.get("/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent details."""
    # TODO: Implement agent retrieval
    raise HTTPException(status_code=404, detail="Agent not found")


@router.post("/{agent_id}/heartbeat")
async def agent_heartbeat(agent_id: str, status: dict):
    """Receive agent heartbeat."""
    # TODO: Implement heartbeat handling
    return {"acknowledged": True}


@router.delete("/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister an agent."""
    # TODO: Implement agent removal
    return {"unregistered": True}