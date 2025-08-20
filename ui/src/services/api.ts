/**
 * API service configuration for the Ruckus UI.
 * 
 * This module configures axios instances for communicating
 * with the orchestrator and agent services.
 */

import axios from 'axios';

// Base URL for the orchestrator API
const ORCHESTRATOR_BASE_URL = process.env.REACT_APP_ORCHESTRATOR_URL || 'http://localhost:8000';

// Create axios instance for orchestrator API
export const orchestratorApi = axios.create({
  baseURL: `${ORCHESTRATOR_BASE_URL}/api/v1`,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth tokens (if needed in the future)
orchestratorApi.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling common errors
orchestratorApi.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token');
      // Redirect to login if needed
    }
    return Promise.reject(error);
  }
);

// Helper function to create agent API instance for a specific agent
export const createAgentApi = (agentUrl: string) => {
  return axios.create({
    baseURL: `${agentUrl}/api/v1`,
    timeout: 5000,
    headers: {
      'Content-Type': 'application/json',
    },
  });
};

export default orchestratorApi;
