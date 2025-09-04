/**
 * API service layer for communicating with the Ruckus server
 */

import type {
  RegisterAgentRequest,
  RegisterAgentResponse,
  UnregisterAgentRequest,
  UnregisterAgentResponse,
  ListAgentInfoResponse,
  GetAgentInfoResponse,
  ListAgentStatusResponse,
  GetAgentStatusResponse,
  CreateExperimentRequest,
  CreateExperimentResponse,
  DeleteExperimentResponse,
  ListExperimentsResponse,
  GetExperimentResponse,
} from '../types/api';

// Configuration
const DEFAULT_SERVER_URL = 'http://localhost:8000';
const API_BASE_PATH = '/api/v1';

// Get server URL from environment variables
function getServerUrl(): string {
  // In Vite, environment variables are prefixed with VITE_ and available on import.meta.env
  const envServerUrl = import.meta.env.VITE_RUCKUS_SERVER_URL;
  
  if (envServerUrl) {
    return envServerUrl;
  }
  
  // Fallback to default
  return DEFAULT_SERVER_URL;
}

export class RuckusApiClient {
  private baseUrl: string;

  constructor(serverUrl?: string) {
    const resolvedServerUrl = serverUrl || getServerUrl();
    this.baseUrl = `${resolvedServerUrl}${API_BASE_PATH}`;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      return response.json();
    } catch (error) {
      console.error(`API request failed for ${url}:`, error);
      
      // Check if it's a connection error
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('CONNECTION_ERROR: Cannot connect to server');
      }
      
      // Check for network errors
      if (error instanceof Error && (
        error.message.includes('NetworkError') ||
        error.message.includes('Failed to fetch') ||
        error.message.includes('ERR_NETWORK') ||
        error.message.includes('ERR_INTERNET_DISCONNECTED')
      )) {
        throw new Error('CONNECTION_ERROR: Cannot connect to server');
      }
      
      throw error;
    }
  }

  // Agent registration endpoints
  async registerAgent(request: RegisterAgentRequest): Promise<RegisterAgentResponse> {
    return this.request<RegisterAgentResponse>('/agents/register', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async unregisterAgent(request: UnregisterAgentRequest): Promise<UnregisterAgentResponse> {
    return this.request<UnregisterAgentResponse>('/agents/unregister', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Agent information endpoints
  async listAgents(): Promise<ListAgentInfoResponse> {
    return this.request<ListAgentInfoResponse>('/agents/');
  }

  async getAgentInfo(agentId: string): Promise<GetAgentInfoResponse> {
    return this.request<GetAgentInfoResponse>(`/agents/${agentId}/info`);
  }

  // Agent status endpoints
  async listAgentStatus(): Promise<ListAgentStatusResponse> {
    return this.request<ListAgentStatusResponse>('/agents/status');
  }

  async getAgentStatus(agentId: string): Promise<GetAgentStatusResponse> {
    return this.request<GetAgentStatusResponse>(`/agents/${agentId}/status`);
  }

  // Experiment endpoints
  async listExperiments(): Promise<ListExperimentsResponse> {
    return this.request<ListExperimentsResponse>('/experiments/');
  }

  async getExperiment(experimentId: string): Promise<GetExperimentResponse> {
    return this.request<GetExperimentResponse>(`/experiments/${experimentId}`);
  }

  async createExperiment(request: CreateExperimentRequest): Promise<CreateExperimentResponse> {
    return this.request<CreateExperimentResponse>('/experiments/', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async deleteExperiment(experimentId: string): Promise<DeleteExperimentResponse> {
    return this.request<DeleteExperimentResponse>(`/experiments/${experimentId}`, {
      method: 'DELETE',
    });
  }
}

// Default API client instance using environment configuration
export const apiClient = new RuckusApiClient();