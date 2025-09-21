"""Agent management endpoints."""

from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional
from datetime import datetime

from ruckus_common.models import AgentCapabilitiesBase
from ..models import RegisterAgentRequest, RegisterAgentResponse, UnregisterAgentRequest, UnregisterAgentResponse, ListAgentInfoResponse, GetAgentInfoResponse, ListAgentStatusResponse, GetAgentStatusResponse, CheckAgentCompatibilityRequest, CheckAgentCompatibilityResponse
from ruckus_server.core.clients.http import ConnectionError, ServiceUnavailableError
from ruckus_server.core.agent_manager import AgentAlreadyRegisteredException, AgentNotRegisteredException

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
    agent_manager = request.app.state.agent_manager
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    
    try:
        # Call the AgentManager register_agent method
        registration_result = await agent_manager.register_agent(request_data.agent_url)
        
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
    agent_manager = request.app.state.agent_manager
    job_manager = request.app.state.job_manager

    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    if not job_manager:
        raise HTTPException(status_code=503, detail="Job manager not initialized")

    try:
        # Call the RuckusServer unregister_agent method
        unregistration_result = await agent_manager.unregister_agent(request_data.agent_id)

        # Clean up job manager references for this agent
        await job_manager.cleanup_agent_references(request_data.agent_id)

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
    agent_manager = request.app.state.agent_manager
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    
    try:
        # Get all registered agent info from the server
        agents = await agent_manager.list_registered_agent_info()
        
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
    agent_manager = request.app.state.agent_manager
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    
    try:
        # Get registered agent info from the server
        agent_info = await agent_manager.get_registered_agent_info(agent_id)
        
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
    agent_manager = request.app.state.agent_manager
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    
    try:
        # Get status of all registered agents from the server
        agent_statuses = await agent_manager.list_registered_agent_status()
        
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
    agent_manager = request.app.state.agent_manager
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    
    try:
        # Get status of the specified agent from the server
        agent_status = await agent_manager.get_registered_agent_status(agent_id)
        
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




@router.post("/compatibility/check", response_model=CheckAgentCompatibilityResponse)
async def check_agent_compatibility(request_data: CheckAgentCompatibilityRequest, request: Request):
    """Check which agents are compatible with an experiment specification.
    
    This endpoint allows the UI to determine which agents can run a specific experiment type.
    It performs static capability analysis based on agent registration data without
    communicating with the agents themselves.
    
    Args:
        request_data: CheckAgentCompatibilityRequest containing experiment spec and optional agent filter
        request: FastAPI request object to access app state
        
    Returns:
        CheckAgentCompatibilityResponse with detailed compatibility information for each agent
        
    Raises:
        HTTPException: 503 if services not initialized, 500 for other errors
    """
    agent_manager = request.app.state.agent_manager
    if not agent_manager:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    
    # Check if job_manager is available for compatibility checking
    job_manager = getattr(request.app.state, "job_manager", None)
    if not job_manager:
        raise HTTPException(status_code=503, detail="Job manager not initialized")
    
    try:
        # Get list of agents to check
        if request_data.agent_ids:
            # Check specific agents
            agents_to_check = []
            for agent_id in request_data.agent_ids:
                try:
                    agent = await agent_manager.get_registered_agent_info(agent_id)
                    agents_to_check.append(agent)
                except AgentNotRegisteredException:
                    # Skip agents that dont exist - dont fail the whole request
                    continue
        else:
            # Check all registered agents
            agents_to_check = await agent_manager.list_registered_agent_info()
        
        # Perform compatibility checking for each agent
        compatibility_results = []
        compatible_count = 0
        
        for agent in agents_to_check:
            try:
                compatibility = job_manager._check_agent_compatibility(agent, request_data.experiment_spec)
                compatibility_results.append(compatibility)
                
                if compatibility.can_run:
                    compatible_count += 1
                    
            except Exception as e:
                # If compatibility check fails for an agent, include it with error
                from ruckus_common.models import AgentCompatibility
                error_compatibility = AgentCompatibility(
                    agent_id=agent.agent_id,
                    agent_name=agent.agent_name or agent.agent_id,
                    can_run=False,
                    missing_requirements=[f"Compatibility check failed: {str(e)}"],
                    warnings=["Agent compatibility could not be determined"]
                )
                compatibility_results.append(error_compatibility)
        
        # Build response
        response = CheckAgentCompatibilityResponse(
            compatibility_results=compatibility_results,
            experiment_name=request_data.experiment_spec.name,
            total_agents_checked=len(compatibility_results),
            compatible_agents_count=compatible_count
        )
        
        return response
        
    except Exception as e:
        # Unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

