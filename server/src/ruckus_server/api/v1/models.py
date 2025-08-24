"""Pydantic models for API v1 endpoints."""

from pydantic import BaseModel, Field, field_validator
from urllib.parse import urlparse
import re

from datetime import datetime
from typing import List
from ruckus_common.models import RegisteredAgentInfo, AgentStatus


class RegisterAgentRequest(BaseModel):
    """Request model for registering an agent."""
    
    agent_url: str = Field(..., description="Base URL of the agent to register")
    
    @field_validator('agent_url')
    @classmethod
    def validate_agent_url(cls, v: str) -> str:
        """Validate that agent_url is a valid URL."""
        if not v:
            raise ValueError("agent_url cannot be empty")
        
        # Parse the URL
        try:
            parsed = urlparse(v)
        except Exception as e:
            raise ValueError(f"Invalid URL format: {str(e)}")
        
        # Check for required components
        if not parsed.scheme:
            raise ValueError("URL must include a scheme (http or https)")
        
        if parsed.scheme not in ('http', 'https'):
            raise ValueError("URL scheme must be http or https")
        
        if not parsed.netloc:
            raise ValueError("URL must include a hostname")
        
        # Check for valid hostname format
        hostname_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$'
        hostname = parsed.netloc.split(':')[0]  # Remove port if present
        
        if not re.match(hostname_pattern, hostname) and hostname != 'localhost':
            raise ValueError("Invalid hostname format")
        
        return v


class RegisterAgentResponse(BaseModel):
    """Response model for agent registration."""
    
    agent_id: str = Field(..., description="ID of the registered agent")
    registered_at: datetime = Field(..., description="Timestamp when the agent was registered")


class UnregisterAgentRequest(BaseModel):
    """Request model for unregistering an agent."""
    
    agent_id: str = Field(..., description="ID of the agent to unregister", min_length=1)


class UnregisterAgentResponse(BaseModel):
    """Response model for agent unregistration."""
    
    agent_id: str = Field(..., description="ID of the unregistered agent")
    unregistered_at: datetime = Field(..., description="Timestamp when the agent was unregistered")


class ListAgentInfoResponse(BaseModel):
    """Response model for getting all registered agent information."""
    
    agents: List[RegisteredAgentInfo] = Field(..., description="List of all registered agents")


class GetAgentInfoResponse(BaseModel):
    """Response model for getting specific agent information."""
    
    agent: RegisteredAgentInfo = Field(..., description="Registered agent information")


class ListAgentStatusResponse(BaseModel):
    """Response model for getting all agent status information."""
    
    agents: List[AgentStatus] = Field(..., description="List of agent status information")


class GetAgentStatusResponse(BaseModel):
    """Response model for getting specific agent status information."""
    
    agent: AgentStatus = Field(..., description="Agent status information")