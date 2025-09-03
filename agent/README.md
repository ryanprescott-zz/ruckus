# RUCKUS Agent

Worker agent component for the RUCKUS distributed benchmarking system. The agent automatically detects system information and provides detailed data through REST APIs.

## Features

### Core Agent Capabilities
- **System Detection**: Automatically detects CPU, GPU, memory, and available ML frameworks
- **Extensible Storage**: Abstract storage backend (in-memory by default, extensible to PostgreSQL/SQLite)
- **RESTful API**: Provides `/info` and other endpoints for system communication

### Multi-Run Job System (NEW)
- **Statistical Job Execution**: Execute jobs multiple times (`runs_per_job` parameter) for reliable statistics
- **Cold Start Analysis**: Separate tracking of model loading time vs. inference time
- **Performance Statistics**: Automatic calculation of mean ± std, outlier detection across runs
- **Enhanced Result Models**: `SingleRunResult`, `MetricStatistics`, and `MultiRunJobResult` for rich data

### Advanced GPU Detection & Benchmarking (NEW)
- **Multi-Layer GPU Detection**:
  - Primary: pynvml for comprehensive NVIDIA GPU information
  - Fallback: PyTorch CUDA detection for cross-platform support
  - Secondary: nvidia-smi XML parsing as backup
- **Real-Time GPU Metrics**:
  - Temperature, power usage, utilization rates
  - Memory allocation and bandwidth testing
  - Compute capability and tensor core generation detection
- **GPU Benchmarking System**:
  - Memory bandwidth testing with adaptive tensor sizes
  - Multi-precision FLOPS benchmarking (FP32, FP16, BF16, FP8)
  - Tensor core capability assessment
  - Live performance monitoring during job execution

### Comprehensive Testing Suite (NEW)
- **150+ Test Cases**: Extensive test coverage including new multi-run and GPU features
- **Hardware Mocking**: Complete mock system for GPU detection and benchmarking testing
- **End-to-End Integration**: Full agent-server communication testing
- **Statistical Validation**: Tests for metric aggregation and outlier detection

## Installation

### Prerequisites
- Python 3.12+
- Conda environment (recommended)

### Setup
```bash
# From the project root
conda env create -f environment.yml
conda activate ruckus

# Install common package (dependency)
cd common && pip install -e .

# Install agent package
cd ../agent && pip install -e .
```

## Usage

### Running the Agent
```bash
# Start agent server (default port 8081)
python -m ruckus_agent.main

# Or use the installed command
ruckus-agent
```

### Testing the Agent
```bash
# System info endpoint  
curl http://localhost:8081/api/v1/info

# Health check
curl http://localhost:8081/health
```

## API Endpoints

### `/api/v1/info` (GET)
Returns detailed system information including new GPU benchmarking data:
```json
{
  "agent_id": "agent-656b6586",
  "agent_type": "white_box",
  "system_info": {
    "system": {"hostname": "gpu-benchmark-agent", "os": "Linux"},
    "cpu": {"cores_physical": 16, "model": "AMD EPYC 7542"},
    "gpus": [{
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24576,
      "memory_free_mb": 22760,
      "compute_capability": [8, 9],
      "tensor_core_generation": 4,
      "temperature_c": 45,
      "power_usage_w": 25,
      "utilization_gpu": 15,
      "benchmark_results": {
        "memory_bandwidth": {"peak_bandwidth_gb_s": 800.0},
        "compute_performance": {
          "FP16": {"tflops": 120.0, "avg_time_ms": 4.0},
          "BF16": {"tflops": 110.0, "avg_time_ms": 4.5}
        },
        "tensor_core_capabilities": {
          "generation": 4,
          "supported_precisions": ["FP32", "FP16", "BF16", "FP8"]
        }
      }
    }],
    "frameworks": [
      {"name": "pytorch", "version": "2.0.1", "gpu_support": true},
      {"name": "transformers", "version": "4.21.0", "gpu_support": true}
    ],
    "metrics": ["latency", "throughput", "memory_usage", "gpu_utilization"]
  }
}
```

### Multi-Run Job Execution Examples

#### Single Run Job (Traditional)
```json
{
  "job_id": "single-run-001",
  "runs_per_job": 1,
  "model": "llama-7b",
  "task_type": "summarization"
}
```
Returns: `JobResult` with single execution metrics

#### Multi-Run Job (Statistical Analysis)
```json
{
  "job_id": "multi-run-001", 
  "runs_per_job": 5,
  "model": "llama-7b",
  "task_type": "summarization"
}
```
Returns: `MultiRunJobResult` with:
- Individual run data for all 5 executions
- Cold start data (first run with model loading time)
- Statistical summary (mean ± std, outliers) for warm runs
- Complete raw data for further analysis

## Configuration

Agent behavior can be configured via environment variables:
- `RUCKUS_AGENT_HOST`: Server host (default: 0.0.0.0)
- `RUCKUS_AGENT_PORT`: Server port (default: 8081) 
- `RUCKUS_AGENT_AGENT_TYPE`: Agent type (white_box, gray_box, black_box)
- `RUCKUS_AGENT_MAX_CONCURRENT_JOBS`: Max concurrent jobs (default: 1)

## Development

### Running Tests
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_agent.py

# Run with verbose output
pytest -v
```

### Storage Backends
The agent uses an abstract storage system. To implement a new backend:

```python
from ruckus_agent.core.storage import AgentStorage

class MyStorage(AgentStorage):
    async def store_system_info(self, system_info):
        # Your implementation
        pass
    # ... implement other abstract methods

# Use custom storage
agent = Agent(settings, storage=MyStorage())
```

## Architecture

- **Agent Core** (`core/agent.py`): Main agent orchestration
- **Storage Layer** (`core/storage.py`): Pluggable data persistence
- **System Detection** (`core/detector.py`): Hardware/software capability detection
- **REST API** (`api/v1/api.py`): HTTP endpoints for server communication
- **Configuration** (`core/config.py`): Environment-based configuration