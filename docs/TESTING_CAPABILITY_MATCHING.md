# Testing Agent Capability Matching

This document describes how to test the agent capability matching system both with mock data (unit tests) and real hardware (integration tests).

## Overview

The capability matching system enables users to:
1. **Hardware Comparison**: Run GPU benchmarks across different hardware configurations
2. **Model Deployment**: Test LLM inference on agents with specific models loaded
3. **Framework Testing**: Compare performance across different ML runtime environments

## Unit Tests (Mock Data)

### Running Unit Tests
```bash
# From project root
conda activate ruckus
cd server
python -m pytest tests/test_capability_matching.py -v
```

### What Unit Tests Cover
- ✅ Compatibility logic with various agent configurations
- ✅ GPU requirement checking (memory, vendor, tensor cores)
- ✅ Model availability validation 
- ✅ Framework version detection
- ✅ Missing requirement reporting
- ✅ Binary compatibility decisions (`can_run: true/false`)

### Mock Agent Configurations Tested
1. **RTX 4090 Workstation**: High-end NVIDIA GPU, 24GB VRAM, all frameworks
2. **RTX 3080 Gaming PC**: Mid-range NVIDIA GPU, 10GB VRAM, most frameworks
3. **CPU Server**: No GPU, transformers + pytorch CPU-only
4. **MacBook M2**: Apple Silicon GPU, 16GB unified memory, MPS support

## Integration Tests (Real Hardware)

### Prerequisites for Real Hardware Testing

#### 1. Agent Setup
You need at least 2-3 agents running on different hardware configurations:

```bash
# On each agent machine
conda activate ruckus
cd agent
python -m ruckus_agent.main --host 0.0.0.0 --port 8080
```

#### 2. Server Setup
```bash
# On server machine
conda activate ruckus  
cd server
python -m ruckus_server.main
```

### Integration Test Scenarios

#### Scenario 1: GPU Hardware Comparison

**Goal**: Compare GPU benchmark performance across different hardware

**Test Steps**:
1. Register agents with different GPUs:
   ```bash
   curl -X POST http://server:8000/api/v1/agents/register \
     -H "Content-Type: application/json" \
     -d '{"agent_url": "http://gpu-workstation:8080"}'
   ```

2. Check GPU benchmark compatibility:
   ```bash
   curl -X POST http://server:8000/api/v1/agents/compatibility/check \
     -H "Content-Type: application/json" \
     -d '{
       "experiment_spec": {
         "name": "GPU Benchmark Test",
         "capability_requirements": {
           "gpu_requirements": {
             "min_gpu_count": 1,
             "min_memory_mb": 2048
           },
           "required_capabilities": ["gpu_monitoring"],
           "required_metrics": ["memory_bandwidth", "flops_fp32"]
         }
       }
     }'
   ```

3. **Expected Results**:
   - Agents with NVIDIA/AMD GPUs ≥2GB: `can_run: true`
   - CPU-only agents: `can_run: false`
   - Rich hardware details in `hardware_summary`

#### Scenario 2: Model Deployment Testing

**Goal**: Find agents that can run specific models

**Test Steps**:
1. Load models on some agents:
   ```bash
   # On agent with sufficient GPU memory
   # Model loading happens during agent startup based on configuration
   ```

2. Test LLM inference compatibility:
   ```bash
   curl -X POST http://server:8000/api/v1/agents/compatibility/check \
     -H "Content-Type: application/json" \
     -d '{
       "experiment_spec": {
         "name": "Llama-7B Inference",
         "model": "llama-7b",
         "capability_requirements": {
           "model_requirements": {
             "required_frameworks": ["transformers"],
             "min_gpu_memory_gb": 8.0
           },
           "required_capabilities": ["model_loading", "tokenization"]
         }
       }
     }'
   ```

3. **Expected Results**:
   - Agents with llama-7b loaded + transformers: `can_run: true`
   - Agents missing model or framework: `can_run: false`  
   - `compatible_models` list populated
   - `framework_versions` showing actual versions

#### Scenario 3: Framework Comparison

**Goal**: Test same model across different ML frameworks

**Test Steps**:
1. Get framework compatibility matrix:
   ```bash
   curl -X GET http://server:8000/api/v1/agents/compatibility/matrix
   ```

2. **Expected Results**:
   ```json
   {
     "agents": {
       "agent-1": {
         "agent_name": "GPU Workstation",
         "experiment_compatibility": {
           "gpu_benchmark": {"compatible": true, "hardware_summary": {...}},
           "llm_inference": {"compatible": true, "framework_versions": {...}}
         }
       }
     },
     "experiment_types": ["gpu_benchmark", "llm_inference"],
     "total_agents": 3
   }
   ```

### Real Hardware Test Configuration

#### Agent Configuration Files
Create different agent configs for testing:

**`agent-gpu-config.yaml`** (High-end GPU):
```yaml
agent:
  name: "RTX 4090 Workstation"
  type: "white_box"
  
models:
  - name: "llama-7b"
    path: "/models/llama-7b-hf" 
    framework: "transformers"
    
frameworks:
  transformers:
    version: "4.21.0"
    device: "cuda"
  vllm:
    version: "0.2.1" 
    device: "cuda"
```

**`agent-cpu-config.yaml`** (CPU-only):
```yaml  
agent:
  name: "CPU Server"
  type: "black_box"
  
frameworks:
  transformers:
    version: "4.21.0"
    device: "cpu"
```

### Validation Criteria

#### ✅ Success Criteria
- [ ] Agents register successfully with correct capability detection
- [ ] GPU agents marked compatible for GPU experiments  
- [ ] CPU agents correctly excluded from GPU experiments
- [ ] Agents with models marked compatible for inference experiments
- [ ] Framework versions detected and reported accurately
- [ ] `hardware_summary` contains realistic GPU specifications
- [ ] Missing requirements clearly explain incompatibility

#### ❌ Failure Indicators  
- Agents with GPUs marked incompatible for GPU experiments
- Framework versions showing as "unknown" when frameworks are installed
- `can_run: true` for agents missing obvious requirements
- Server crashes or 500 errors during compatibility checking
- Memory requirements not properly validated

### Performance Testing

#### Load Testing Compatibility Checks
```bash
# Test with many agents
for i in {1..10}; do
  curl -X POST http://server:8000/api/v1/agents/compatibility/check \
    -H "Content-Type: application/json" \
    -d '{"experiment_spec": {...}}' &
done
wait
```

#### Expected Performance
- Compatibility checks should complete in <100ms per agent
- No agent communication required (static analysis only)  
- Concurrent requests should not interfere with each other

## Debugging Failed Tests

### Common Issues

#### 1. Agent Registration Problems
```bash
# Check agent status
curl http://server:8000/api/v1/agents/

# Verify agent info endpoint
curl http://agent:8080/api/v1/info
```

#### 2. Capability Detection Issues
```bash
# Check what capabilities agent reported
curl http://server:8000/api/v1/agents/{agent_id}/info

# Look for missing GPU/framework/model information
```

#### 3. Compatibility Logic Bugs
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check server logs for capability matching details
tail -f server/logs/ruckus-server.log | grep compatibility
```

### Test Data Validation

Use the mock test data as reference for what real agent registration should look like:
- `server/tests/test_capability_matching.py` - Contains realistic agent configurations
- Agents should report similar `system_info` structure
- GPU specifications should include memory, vendor, capabilities
- Framework versions should be properly detected

## Continuous Integration

### Automated Testing Pipeline
```bash
# Unit tests (run on every commit)
python -m pytest server/tests/test_capability_matching.py

# Integration tests (run nightly with real hardware)
# TODO: Implement integration test suite with docker containers
```

### Future Improvements
- [ ] Docker containers for reproducible integration testing  
- [ ] Automated agent deployment with different configurations
- [ ] Performance benchmarking for compatibility checking
- [ ] Regression testing for compatibility logic changes