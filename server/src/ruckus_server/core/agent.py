"""Agent protocol utilities for communicating with agents."""

import logging
from urllib.parse import urljoin
from typing import Dict, Any

from .config import AgentSettings, HttpClientSettings
from .clients.http import HttpClient, ConnectionError, ServiceUnavailableError
from ruckus_common.models import RegisteredAgentInfo
from ruckus_common.models import AgentInfoResponse


class AgentProtocolUtility:
    """Utility class for agent protocol operations."""
    
    def __init__(self, agent_settings: AgentSettings, http_client_settings: HttpClientSettings):
        """Initialize the agent protocol utility.
        
        Args:
            agent_settings: Agent configuration settings
            http_client_settings: HTTP client configuration settings
        """
        self.agent_settings = agent_settings
        self.http_client_settings = http_client_settings
        self.logger = logging.getLogger(__name__)
    
    def build_info_url(self, agent_base_url: str) -> str:
        """Build the full URL for an agent's info endpoint.
        
        Args:
            agent_base_url: Base URL of the agent
            
        Returns:
            Full URL to the agent's info endpoint
        """
        # Ensure base URL ends with / for proper joining
        if not agent_base_url.endswith('/'):
            agent_base_url += '/'
        
        # Remove leading / from endpoint path if present
        endpoint_path = self.agent_settings.info_endpoint_path.lstrip('/')
        
        full_url = urljoin(agent_base_url, endpoint_path)
        self.logger.debug(f"Built info URL: {full_url}")
        return full_url
    
    async def get_agent_info(self, agent_base_url: str) -> AgentInfoResponse:
        """Get agent info from the agent's /info endpoint.
        
        Args:
            agent_base_url: Base URL of the agent
            
        Returns:
            AgentInfoResponse object with agent information
            
        Raises:
            ConnectionError: If connection to agent fails
            ServiceUnavailableError: If agent returns 503 after retries
        """
        info_url = self.build_info_url(agent_base_url)
        
        self.logger.info(f"Fetching agent info from {info_url}")
        
        async with HttpClient(self.http_client_settings) as http_client:
            try:
                response_data = await http_client.get_with_retry(info_url)
                
                # Parse response into AgentInfoResponse model
                agent_info_response = AgentInfoResponse(**response_data)
                self.logger.info(f"Successfully retrieved info for agent {agent_info_response.agent_info.agent_id}")
                return agent_info_response
                
            except ConnectionError as e:
                self.logger.error(f"Failed to connect to agent at {info_url}: {str(e)}")
                raise
            except ServiceUnavailableError as e:
                self.logger.error(f"Agent at {info_url} is unavailable: {str(e)}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error getting agent info from {info_url}: {str(e)}")
                raise ConnectionError(f"Failed to get agent info: {str(e)}")
    
    def create_registered_agent_info(self, agent_info_response: AgentInfoResponse, agent_base_url: str) -> RegisteredAgentInfo:
        """Create a RegisteredAgentInfo object from AgentInfoResponse and URL.
        
        Args:
            agent_info_response: Agent info response from the agent
            agent_base_url: Base URL of the agent
            
        Returns:
            RegisteredAgentInfo object ready for storage
        """
        agent_info = agent_info_response.agent_info
        registered_info = RegisteredAgentInfo(
            agent_id=agent_info.agent_id,
            agent_name=agent_info.agent_name,
            agent_type=agent_info.agent_type,
            system_info=agent_info.system_info,
            capabilities=agent_info.capabilities,
            last_updated=agent_info.last_updated,
            agent_url=agent_base_url
        )
        
        self.logger.debug(f"Created RegisteredAgentInfo for agent {registered_info.agent_id}")
        return registered_info