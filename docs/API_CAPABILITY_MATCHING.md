# Agent Capability Matching API

This document provides complete API documentation for the new agent capability matching endpoints, with real request/response examples for UI integration.

## Overview

The capability matching system enables the UI to:
- Show users which agents can run specific experiment types
- Display rich agent capability information (GPU specs, frameworks, models)
- Support the three main user workflows: hardware comparison, model deployment, framework testing

## Authentication & Base URL

```
Base URL: http://server:8000/api/v1/agents
Authentication: Not required (internal API)
```

## Endpoints

### 1. Check Agent Compatibility

**Endpoint**: `POST /compatibility/check`

**Purpose**: Check which agents can run a specific experiment specification

#### Request Format
```http
POST /api/v1/agents/compatibility/check
Content-Type: application/json

{
  "experiment_spec": {
    "name": "GPU Benchmark Test",
    "capability_requirements": {
      "gpu_requirements": {
        "min_gpu_count": 1,
        "min_memory_mb": 4096
      },
      "required_capabilities": ["gpu_monitoring"],
      "required_metrics": ["memory_bandwidth", "flops_fp32"],
      "optional_metrics": ["gpu_temperature"]
    }
  },
  "agent_ids": ["agent-rtx-4090", "agent-rtx-3080"] // Optional: filter specific agents
}
```

#### Response Format
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "compatibility_results": [
    {
      "agent_id": "agent-rtx-4090",
      "agent_name": "RTX 4090 Workstation", 
      "can_run": true,
      "available_capabilities": ["gpu", "gpu_monitoring", "model_loading"],
      "missing_requirements": [],
      "supported_features": ["memory_bandwidth", "flops_fp32", "gpu_temperature"],
      "warnings": [],
      "hardware_summary": {
        "gpus": ["NVIDIA GeForce RTX 4090 (24.0GB)"]
      },
      "framework_versions": {
        "transformers": "4.21.0",
        "pytorch": "2.0.1+cu118"
      },
      "compatible_models": ["llama-7b", "llama-13b"],
      "estimated_queue_time_seconds": null,
      "last_capability_check": "2024-01-15T10:30:00Z"
    },
    {
      "agent_id": "agent-cpu-server",
      "agent_name": "CPU Server",
      "can_run": false,
      "available_capabilities": ["memory_monitoring", "tokenization"],
      "missing_requirements": ["GPU required for GPU benchmarks"],
      "supported_features": [],
      "warnings": ["No GPU detected - cannot run hardware benchmarks"],
      "hardware_summary": {
        "gpus": []
      },
      "framework_versions": {
        "transformers": "4.21.0"
      },
      "compatible_models": [],
      "estimated_queue_time_seconds": null,
      "last_capability_check": "2024-01-15T10:30:00Z"
    }
  ],
  "experiment_name": "GPU Benchmark Test",
  "total_agents_checked": 2,
  "compatible_agents_count": 1,
  "checked_at": "2024-01-15T10:30:00Z"
}
```

### 2. Get Compatibility Matrix

**Endpoint**: `GET /compatibility/matrix`

**Purpose**: Get overview of which agents can run standard experiment types

#### Request Format
```http
GET /api/v1/agents/compatibility/matrix
```

#### Response Format
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "agents": {
    "agent-rtx-4090": {
      "agent_name": "RTX 4090 Workstation",
      "experiment_compatibility": {
        "gpu_benchmark": {
          "compatible": true,
          "hardware_summary": {
            "gpus": ["NVIDIA GeForce RTX 4090 (24.0GB)"]
          },
          "available_capabilities": ["gpu", "gpu_monitoring"],
          "missing_requirements": [],
          "warnings": []
        },
        "llm_inference": {
          "compatible": true,
          "hardware_summary": {
            "gpus": ["NVIDIA GeForce RTX 4090 (24.0GB)"]
          },
          "available_capabilities": ["model_loading", "tokenization"],
          "missing_requirements": [],
          "warnings": []
        }
      }
    },
    "agent-cpu-server": {
      "agent_name": "CPU Server",
      "experiment_compatibility": {
        "gpu_benchmark": {
          "compatible": false,
          "missing_requirements": ["GPU required for GPU benchmarks"],
          "warnings": ["No GPU detected"]
        },
        "llm_inference": {
          "compatible": false,
          "missing_requirements": ["No models loaded - required for model experiments"],
          "warnings": []
        }
      }
    }
  },
  "experiment_types": ["gpu_benchmark", "llm_inference"],
  "total_agents": 2,
  "checked_at": "2024-01-15T10:30:00Z"
}
```

## UI Integration Examples

### Example 1: Hardware Comparison Workflow

**User Goal**: "Run GPU benchmark on all compatible agents"

```typescript
// 1. Check which agents can run GPU benchmarks
const gpuBenchmarkSpec = {
  name: "GPU Memory Bandwidth Test",
  capability_requirements: {
    gpu_requirements: {
      min_gpu_count: 1,
      min_memory_mb: 2048
    },
    required_capabilities: ["gpu_monitoring"],
    required_metrics: ["memory_bandwidth", "flops_fp32"]
  }
};

const response = await fetch('/api/v1/agents/compatibility/check', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({experiment_spec: gpuBenchmarkSpec})
});

const {compatibility_results} = await response.json();

// 2. Filter compatible agents and show hardware details
const compatibleAgents = compatibility_results.filter(agent => agent.can_run);

// 3. Display agent selection UI
compatibleAgents.forEach(agent => {
  console.log(`✅ ${agent.agent_name}`);
  console.log(`   GPU: ${agent.hardware_summary.gpus[0]}`);
  console.log(`   Features: ${agent.supported_features.join(', ')}`);
});
```

**UI Display**:
```
Select agents for GPU benchmark:
☑️ RTX 4090 Workstation - NVIDIA GeForce RTX 4090 (24.0GB) 
☑️ RTX 3080 Gaming PC - NVIDIA GeForce RTX 3080 (10.0GB)
☑️ MacBook M2 - Apple M2 Pro GPU (16.0GB)
❌ CPU Server - No GPU available
```

### Example 2: Model Deployment Workflow

**User Goal**: "Test llama-7b inference on agents that have the model loaded"

```typescript
// 1. Check LLM inference compatibility
const llamaInferenceSpec = {
  name: "Llama-7B Performance Test",
  model: "llama-7b", 
  capability_requirements: {
    model_requirements: {
      required_frameworks: ["transformers"],
      min_gpu_memory_gb: 8.0
    },
    required_capabilities: ["model_loading", "tokenization"],
    required_metrics: ["inference_time", "throughput"]
  }
};

const response = await fetch('/api/v1/agents/compatibility/check', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({experiment_spec: llamaInferenceSpec})
});

// 2. Show agents with model loaded
const {compatibility_results} = await response.json();
const modelReadyAgents = compatibility_results.filter(agent => 
  agent.can_run && agent.compatible_models.includes('llama-7b')
);

// 3. Display model deployment options
modelReadyAgents.forEach(agent => {
  console.log(`📦 ${agent.agent_name}`);
  console.log(`   Framework: ${agent.framework_versions.transformers}`);
  console.log(`   Models: ${agent.compatible_models.join(', ')}`);
});
```

### Example 3: Framework Comparison Workflow

**User Goal**: "Compare transformers vs VLLM performance"

```typescript
// 1. Get compatibility matrix to see framework availability
const matrixResponse = await fetch('/api/v1/agents/compatibility/matrix');
const {agents} = await matrixResponse.json();

// 2. Group agents by framework availability
const frameworkGroups = {
  transformers: [],
  vllm: [],
  both: []
};

Object.entries(agents).forEach(([agentId, agentInfo]) => {
  const hasTransformers = agentInfo.experiment_compatibility.llm_inference?.compatible;
  const hasVLLM = agentInfo.agent_name.includes('4090'); // Simplified check
  
  if (hasTransformers && hasVLLM) {
    frameworkGroups.both.push({agentId, ...agentInfo});
  } else if (hasTransformers) {
    frameworkGroups.transformers.push({agentId, ...agentInfo});
  }
});

// 3. Show framework comparison options
console.log('Framework Comparison Options:');
console.log(`Transformers only: ${frameworkGroups.transformers.length} agents`);
console.log(`VLLM available: ${frameworkGroups.both.length} agents`);
```

## Error Handling

### Common Error Responses

#### 503 Service Unavailable
```json
{
  "detail": "Job manager not initialized"
}
```

**UI Action**: Show "Service temporarily unavailable" message, retry after delay

#### 500 Internal Server Error  
```json
{
  "detail": "Internal server error: compatibility check failed"
}
```

**UI Action**: Log error, show "Unable to check compatibility" message

### Handling Partial Failures

If individual agent compatibility checks fail, they're included with error information:

```json
{
  "agent_id": "agent-problematic",
  "agent_name": "Problematic Agent",
  "can_run": false,
  "missing_requirements": ["Compatibility check failed: GPU detection error"],
  "warnings": ["Agent compatibility could not be determined"]
}
```

**UI Action**: Show agent as "Status Unknown" with warning icon

## UI Component Recommendations

### Agent Selection Component
```typescript
interface AgentCompatibility {
  agent_id: string;
  agent_name: string; 
  can_run: boolean;
  hardware_summary: {gpus: string[]};
  missing_requirements: string[];
  warnings: string[];
}

function AgentSelector({compatibility_results}: {compatibility_results: AgentCompatibility[]}) {
  const compatibleAgents = compatibility_results.filter(a => a.can_run);
  const incompatibleAgents = compatibility_results.filter(a => !a.can_run);
  
  return (
    <div>
      <h3>Compatible Agents ({compatibleAgents.length})</h3>
      {compatibleAgents.map(agent => (
        <AgentCard key={agent.agent_id} agent={agent} selectable />
      ))}
      
      <details>
        <summary>Incompatible Agents ({incompatibleAgents.length})</summary>
        {incompatibleAgents.map(agent => (
          <AgentCard key={agent.agent_id} agent={agent} disabled />
        ))}
      </details>
    </div>
  );
}
```

### Hardware Summary Display
```typescript
function AgentCard({agent}: {agent: AgentCompatibility}) {
  const gpuInfo = agent.hardware_summary.gpus[0] || "No GPU";
  const status = agent.can_run ? "✅ Compatible" : "❌ Incompatible";
  
  return (
    <div className={`agent-card ${agent.can_run ? 'compatible' : 'incompatible'}`}>
      <h4>{agent.agent_name}</h4>
      <p>{status}</p>
      <p>Hardware: {gpuInfo}</p>
      {agent.missing_requirements.length > 0 && (
        <ul className="missing-requirements">
          {agent.missing_requirements.map(req => <li key={req}>{req}</li>)}
        </ul>
      )}
    </div>
  );
}
```

## Performance Considerations

- **Caching**: Compatibility results are static until agent capabilities change
- **Pagination**: Not needed - typical deployments have <50 agents  
- **Concurrent Requests**: Safe - no agent communication required
- **Response Time**: ~10-50ms per agent (static analysis only)

## Next Steps for UI Integration

1. **Update Types**: Add `AgentCompatibility` interface to `ui/src/types/api.ts`
2. **Add API Client**: Create compatibility checking functions in `ui/src/services/api.ts`
3. **Enhance Agent Tab**: Show hardware summaries and capabilities
4. **Update Experiment Creation**: Filter agent selection based on compatibility
5. **Add Hardware Comparison**: Multi-select UI for compatible agents