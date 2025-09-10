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

export const JobStatusEnum = {
  QUEUED: "queued",
  ASSIGNED: "assigned", 
  RUNNING: "running",
  COMPLETED: "completed",
  FAILED: "failed",
  CANCELLED: "cancelled",
  TIMEOUT: "timeout"
} as const;

export type JobStatusEnum = typeof JobStatusEnum[keyof typeof JobStatusEnum];

// Base interfaces
export interface AgentInfo {
  agent_id: string;
  agent_name?: string;
  agent_type: AgentType;
  system_info: {
    models?: Record<string, {
      name: string;
      [key: string]: any;
    }>;
    frameworks?: Array<{
      name: string;
      version?: string;
      [key: string]: any;
    }>;
    [key: string]: any;
  };
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

// Job interfaces
export interface JobStatus {
  status: JobStatusEnum;
  message?: string;
  timestamp: string; // ISO timestamp
}

export interface JobInfo {
  job_id: string;
  experiment_id: string;
  agent_id: string;
  created_time: string; // ISO timestamp
  status: JobStatus;
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
  LLM_GENERATION: "llm_generation",
  GPU_BENCHMARK: "gpu_benchmark",
  MEMORY_BENCHMARK: "memory_benchmark", 
  COMPUTE_BENCHMARK: "compute_benchmark"
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

// Job API Response models
export interface ListJobsResponse {
  jobs: Record<string, JobInfo[]>; // Dictionary keyed by agent_id
}

export interface CreateJobRequest {
  experiment_id: string;
  agent_id: string;
}

export interface CreateJobResponse {
  job_id: string;
}

export interface ExperimentTableRow {
  id: string;
  name: string;
  jobs: string; // Will be populated later
  created: string; // Formatted timestamp
  remove: string; // For the remove action column
  experiment: ExperimentSpec; // Full experiment data
}

export interface JobTableRow {
  job_id: string;
  experiment_id: string;
  agent_id: string;
  status: string; // Status enum as string for display
  updated: string; // Formatted timestamp
  cancel: string; // For the cancel action column
  jobInfo: JobInfo; // Full job data
}

// Results-related types
export interface ExperimentResult {
  job_id: string;
  experiment_id: string;
  agent_id: string;
  status: JobStatusEnum;
  started_at: string; // ISO timestamp
  completed_at?: string; // ISO timestamp
  duration_seconds?: number;
  output?: any;
  metrics: Record<string, any>;
  model_actual?: string;
  framework_version?: string;
  hardware_info: Record<string, any>;
  artifacts: string[];
  error?: string;
  error_type?: string;
  traceback?: string;
}

export interface ExperimentResultTableRow {
  experiment_id: string;
  job_id: string;
  agent_id: string;
  status: string; // Status enum as string for display
  export: string; // For the export action column
  experimentResult: ExperimentResult; // Full experiment result data
}

export interface ListExperimentResultsResponse {
  results: ExperimentResult[];
}

export interface GetExperimentResultResponse {
  result: ExperimentResult;
}

// Agent Capability Matching types
export interface AgentCompatibility {
  agent_id: string;
  agent_name: string;
  can_run: boolean;
  available_capabilities: string[];
  missing_requirements: string[];
  supported_features: string[];
  warnings: string[];
  hardware_summary: Record<string, any>;
  framework_versions: Record<string, string>;
  compatible_models: string[];
  estimated_queue_time_seconds?: number;
  last_capability_check: string; // ISO timestamp
}

export interface CheckAgentCompatibilityRequest {
  experiment_spec: ExperimentSpec;
  agent_ids?: string[]; // Optional: filter specific agents
}

export interface CheckAgentCompatibilityResponse {
  compatibility_results: AgentCompatibility[];
  experiment_name: string;
  total_agents_checked: number;
  compatible_agents_count: number;
  checked_at: string; // ISO timestamp
}

export interface AgentCompatibilityMatrixResponse {
  agents: Record<string, {
    agent_name: string;
    experiment_compatibility: Record<string, {
      compatible: boolean;
      hardware_summary?: Record<string, any>;
      available_capabilities?: string[];
      missing_requirements?: string[];
      warnings?: string[];
    }>;
  }>;
  experiment_types: string[];
  total_agents: number;
  checked_at: string; // ISO timestamp
}