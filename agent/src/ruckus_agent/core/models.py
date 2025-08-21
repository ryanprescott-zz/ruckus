"""Agent-specific models."""

from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Agent configuration model."""
    
    server_url: str = "http://localhost:8000"
    agent_id: str = "default-agent"
    poll_interval: int = 30
