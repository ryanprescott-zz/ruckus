"""Pydantic models for API v1 endpoints."""

from pydantic import BaseModel, Field, field_validator
from urllib.parse import urlparse
import re

from ...core.models import RegisteredAgentInfo


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
    
    agent_info: RegisteredAgentInfo = Field(..., description="Registered agent information")