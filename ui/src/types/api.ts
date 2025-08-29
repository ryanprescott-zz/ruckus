/**
 * TypeScript interfaces for Ruckus API models
 * Based on the Python models in common/src/ruckus_common/models.py 
 * and server/src/ruckus_server/api/v1/models.py
 */

// Enums
export const AgentType = {
  WHITE_BOX: "white_box",
  GRAY_BOX: "gray_box", 
  BLACK_BOX: "black_box"
} as const;

export type AgentType = typeof AgentType[keyof typeof AgentType];

export const AgentStatusEnum = {
  ACTIVE: "active",
  IDLE: "idle", 
  ERROR: "error",
  OFFLINE: "offline",
  UNAVAILABLE: "unavailable"
} as const;

export type AgentStatusEnum = typeof AgentStatusEnum[keyof typeof AgentStatusEnum];

// Base interfaces
export interface AgentInfo {
  agent_id: string;
  agent_name?: string;
  agent_type: AgentType;
  system_info: Record<string, any>;
  capabilities: Record<string, any>;
  last_updated: string; // ISO timestamp
}

export interface RegisteredAgentInfo extends AgentInfo {
  agent_url: string;
  registered_at: string; // ISO timestamp
}

export interface AgentStatus {
  agent_id: string;
  status: AgentStatusEnum;
  running_jobs: string[];
  queued_jobs: string[];
  uptime_seconds: number;
  timestamp: string; // ISO timestamp
}

// API Request/Response models
export interface RegisterAgentRequest {
  agent_url: string;
}

export interface RegisterAgentResponse {
  agent_id: string;
  registered_at: string; // ISO timestamp
}

export interface UnregisterAgentRequest {
  agent_id: string;
}

export interface UnregisterAgentResponse {
  agent_id: string;
  unregistered_at: string; // ISO timestamp
}

export interface ListAgentInfoResponse {
  agents: RegisteredAgentInfo[];
}

export interface GetAgentInfoResponse {
  agent: RegisteredAgentInfo;
}

export interface ListAgentStatusResponse {
  agents: AgentStatus[];
}

export interface GetAgentStatusResponse {
  agent: AgentStatus;
}

// Combined interface for the UI table
export interface AgentTableRow {
  id: string;
  name: string;
  status: AgentStatusEnum;
  jobs: string; // Will be populated later from jobs endpoints
  uptime: string; // Formatted uptime string
  lastStatusChange: string; // Formatted timestamp
  unregister: string; // For the unregister action column
  agent: RegisteredAgentInfo; // Full agent data for details panel
}