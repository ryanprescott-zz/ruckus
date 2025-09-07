"""Agent protocol utilities for communicating with agents."""

import logging
from urllib.parse import urljoin
from typing import Dict, Any

from .config import AgentSettings, HttpClientSettings
from .clients.http import HttpClient, ConnectionError, ServiceUnavailableError
from ruckus_common.models import (
    RegisteredAgentInfo,
    AgentInfoResponse,
    AgentStatus,
    ExecuteJobRequest,
    ExperimentSpec,
    JobStatus,
    JobStatusEnum,
)


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
    
    async def get_agent_status(self, agent_base_url: str) -> AgentStatus:
        """Get agent status from the agent's /status endpoint.
        
        Args:
            agent_base_url: Base URL of the agent
            
        Returns:
            AgentStatus object from the agent
            
        Raises:
            ConnectionError: If unable to connect to agent
            ServiceUnavailableError: If agent is temporarily unavailable
        """
        # Build the status URL
        if not agent_base_url.endswith('/'):
            agent_base_url += '/'
        
        status_url = urljoin(agent_base_url, "api/v1/status")
        self.logger.debug(f"Getting agent status from {status_url}")
        
        async with HttpClient(self.http_client_settings) as client:
            try:
                response = await client.get(status_url)
                agent_status = AgentStatus(**response)
                self.logger.debug(f"Got agent status: {agent_status.status}")
                return agent_status
            except ServiceUnavailableError:
                self.logger.warning(f"Agent at {status_url} is temporarily unavailable")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error getting agent status from {status_url}: {str(e)}")
                raise ConnectionError(f"Failed to get agent status: {str(e)}")
    
    async def execute_experiment(self, agent_base_url: str, experiment_spec: ExperimentSpec, job_id: str) -> Dict[str, Any]:
        """Execute an experiment on an agent.
        
        Args:
            agent_base_url: Base URL of the agent
            experiment_spec: The experiment specification to execute
            job_id: The job ID for this execution
            
        Returns:
            Response from the agent's /jobs endpoint
            
        Raises:
            ConnectionError: If unable to connect to agent
            ServiceUnavailableError: If agent is temporarily unavailable
        """
        # Build the jobs URL (new endpoint)
        if not agent_base_url.endswith('/'):
            agent_base_url += '/'
        
        jobs_url = urljoin(agent_base_url, "api/v1/jobs")
        self.logger.debug(f"Executing experiment on agent at {jobs_url}")
        
        # Create the ExecuteJobRequest
        execute_request = ExecuteJobRequest(
            experiment_spec=experiment_spec,
            job_id=job_id
        )
        
        async with HttpClient(self.http_client_settings) as client:
            try:
                # Use model_dump with serialization mode to handle datetime objects
                request_data = execute_request.model_dump(mode='json')
                response = await client.post(jobs_url, request_data)
                self.logger.debug(f"Experiment execution initiated: {response}")
                return response
            except ServiceUnavailableError:
                self.logger.warning(f"Agent at {jobs_url} is temporarily unavailable")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error executing experiment on {jobs_url}: {str(e)}")
                raise ConnectionError(f"Failed to execute experiment: {str(e)}")
    
    async def get_job_status(self, agent_base_url: str, job_id: str) -> JobStatus:
        """Get job status from the agent's /status/{job_id} endpoint.
        
        Args:
            agent_base_url: Base URL of the agent
            job_id: The job ID to get status for
            
        Returns:
            JobStatus object for the specified job
            
        Raises:
            ConnectionError: If unable to connect to agent
            ServiceUnavailableError: If agent is temporarily unavailable
        """
        # Build the job status URL
        if not agent_base_url.endswith('/'):
            agent_base_url += '/'
        
        status_url = urljoin(agent_base_url, f"api/v1/status/{job_id}")
        self.logger.debug(f"Getting job status from {status_url}")
        
        async with HttpClient(self.http_client_settings) as client:
            try:
                response = await client.get(status_url)
                # The response should have status, timestamp, and message
                job_status = JobStatus(**response)
                self.logger.debug(f"Got job status: {job_status.status}")
                return job_status
            except ServiceUnavailableError:
                self.logger.warning(f"Agent at {status_url} is temporarily unavailable")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error getting job status from {status_url}: {str(e)}")
                raise ConnectionError(f"Failed to get job status: {str(e)}")
    
    async def get_experiment_results(self, agent_base_url: str, job_id: str) -> Dict[str, Any]:
        """Get experiment results from the agent's /results/{job_id} endpoint.
        
        Args:
            agent_base_url: Base URL of the agent
            job_id: The job ID to get results for
            
        Returns:
            Experiment results from the agent
            
        Raises:
            ConnectionError: If unable to connect to agent
            ServiceUnavailableError: If agent is temporarily unavailable
        """
        # Build the results URL
        if not agent_base_url.endswith('/'):
            agent_base_url += '/'
        
        results_url = urljoin(agent_base_url, f"api/v1/results/{job_id}")
        self.logger.debug(f"Getting experiment results from {results_url}")
        
        async with HttpClient(self.http_client_settings) as client:
            try:
                response = await client.get(results_url)
                self.logger.debug(f"Got experiment results for job {job_id}")
                return response
            except ServiceUnavailableError:
                self.logger.warning(f"Agent at {results_url} is temporarily unavailable")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error getting experiment results from {results_url}: {str(e)}")
                raise ConnectionError(f"Failed to get experiment results: {str(e)}")
    
    async def cancel_experiment(self, agent_base_url: str, job_id: str) -> bool:
        """Cancel an experiment running on an agent.
        
        Args:
            agent_base_url: Base URL of the agent
            job_id: The job ID to cancel
            
        Returns:
            True if cancellation was successful (200 status), False otherwise
            
        Raises:
            ConnectionError: If unable to connect to agent
        """
        # Build the cancel URL
        if not agent_base_url.endswith('/'):
            agent_base_url += '/'
        
        cancel_url = urljoin(agent_base_url, f"api/v1/jobs/{job_id}")
        self.logger.debug(f"Cancelling job {job_id} at {cancel_url}")
        
        async with HttpClient(self.http_client_settings) as client:
            try:
                # DELETE request to cancel the job
                await client.delete(cancel_url)
                self.logger.info(f"Successfully cancelled job {job_id} on agent")
                return True
            except Exception as e:
                self.logger.error(f"Failed to cancel job {job_id} on agent at {cancel_url}: {str(e)}")
                return False