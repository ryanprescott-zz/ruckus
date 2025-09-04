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

// Experiment related types
export const TaskType = {
  LLM_GENERATION: "llm_generation"
} as const;

export type TaskType = typeof TaskType[keyof typeof TaskType];

export const FrameworkName = {
  PYTORCH: "pytorch",
  TRANSFORMERS: "transformers", 
  VLLM: "vllm",
  TENSORRT: "tensorrt",
  ONNX: "onnx",
  TRITON: "triton",
  UNKNOWN: "unknown"
} as const;

export type FrameworkName = typeof FrameworkName[keyof typeof FrameworkName];

export const PromptRole = {
  SYSTEM: "system",
  USER: "user", 
  ASSISTANT: "assistant"
} as const;

export type PromptRole = typeof PromptRole[keyof typeof PromptRole];

export interface PromptMessage {
  role: PromptRole;
  content: string;
}

export interface PromptTemplate {
  messages: PromptMessage[];
  extra_body?: Record<string, any>;
}

export interface LLMGenerationParams {
  prompt_template: PromptTemplate;
}

export interface TaskSpec {
  name: string;
  type: TaskType;
  description?: string;
  params: any; // This can be LLMGenerationParams or other types
}

export interface FrameworkSpec {
  name: FrameworkName;
  params: any;
}

export interface MetricsSpec {
  metrics: Record<string, any>;
}

export interface ExperimentSpec {
  name: string;
  description?: string;
  model: string;
  task: TaskSpec;
  framework: FrameworkSpec;
  metrics: MetricsSpec;
  created_at?: string;
  updated_at?: string;
  // Computed field - always present in API responses
  id: string;
}

// API Request/Response models for experiments
export interface CreateExperimentSpec {
  name: string;
  description?: string;
  model: string;
  task: TaskSpec;
  framework: FrameworkSpec;
  metrics: MetricsSpec;
}

export interface CreateExperimentRequest {
  experiment_spec: CreateExperimentSpec;
}

export interface CreateExperimentResponse {
  experiment_id: string;
  created_at: string;
}

export interface DeleteExperimentResponse {
  experiment_id: string;
  deleted_at: string;
}

export interface ListExperimentsResponse {
  experiments: ExperimentSpec[];
}

export interface GetExperimentResponse {
  experiment: ExperimentSpec;
}

export interface ExperimentTableRow {
  id: string;
  name: string;
  jobs: string; // Will be populated later
  created: string; // Formatted timestamp
  remove: string; // For the remove action column
  experiment: ExperimentSpec; // Full experiment data
}