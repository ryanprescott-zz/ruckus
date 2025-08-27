"""Agent management endpoints."""

from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional
from datetime import datetime

from ruckus_common.models import AgentCapabilitiesBase
from ..models import RegisterAgentRequest, RegisterAgentResponse, UnregisterAgentRequest, UnregisterAgentResponse, ListAgentInfoResponse, GetAgentInfoResponse, ListAgentStatusResponse, GetAgentStatusResponse
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


@router.get("/", response_model=ListAgentInfoResponse)
async def list_agents(request: Request):
    """List all registered agents.
    
    Args:
        request: FastAPI request object to access app state
        
    Returns:
        ListAgentInfoResponse containing all registered agents
        
    Raises:
        HTTPException: 503 if server not initialized, 500 for other errors
    """
    server = request.app.state.server
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        # Get all registered agent info from the server
        agents = await server.list_registered_agent_info()
        
        # Return response with list of agents
        return ListAgentInfoResponse(agents=agents)
        
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{agent_id}/info", response_model=GetAgentInfoResponse)
async def get_agent_info(agent_id: str, request: Request):
    """Get specific agent information.
    
    Args:
        agent_id: ID of the agent to retrieve
        request: FastAPI request object to access app state
        
    Returns:
        GetAgentInfoResponse containing registered agent information
        
    Raises:
        HTTPException: 404 if agent not found, 503 if server not initialized, 500 for other errors
    """
    server = request.app.state.server
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        # Get registered agent info from the server
        agent_info = await server.get_registered_agent_info(agent_id)
        
        # Return response with agent info
        return GetAgentInfoResponse(agent=agent_info)
        
    except AgentNotRegisteredException as e:
        # Agent is not registered - return 404
        raise HTTPException(
            status_code=404,
            detail=f"No agent with ID {e.agent_id} is registered"
        )
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/status", response_model=ListAgentStatusResponse)
async def list_agent_status(request: Request):
    """Get status of all registered agents.
    
    Args:
        request: FastAPI request object to access app state
        
    Returns:
        ListAgentStatusResponse containing status of all registered agents
        
    Raises:
        HTTPException: 503 if server not initialized, 500 for other errors
    """
    server = request.app.state.server
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        # Get status of all registered agents from the server
        agent_statuses = await server.list_registered_agent_status()
        
        # Return response with list of agent statuses
        return ListAgentStatusResponse(agents=agent_statuses)
        
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{agent_id}/status", response_model=GetAgentStatusResponse)
async def get_agent_status(agent_id: str, request: Request):
    """Get status of a specific registered agent.
    
    Args:
        agent_id: ID of the agent to get status for
        request: FastAPI request object to access app state
        
    Returns:
        GetAgentStatusResponse containing status of the specified agent
        
    Raises:
        HTTPException: 404 if agent not found, 503 if server not initialized, 500 for other errors
    """
    server = request.app.state.server
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        # Get status of the specified agent from the server
        agent_status = await server.get_registered_agent_status(agent_id)
        
        # Return response with agent status
        return GetAgentStatusResponse(agent=agent_status)
        
    except AgentNotRegisteredException as e:
        # Agent is not registered - return 404
        raise HTTPException(
            status_code=404,
            detail=f"No agent with ID {e.agent_id} is registered"
        )
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


