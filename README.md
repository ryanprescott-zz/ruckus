# RUCKUS

RUCKUS is a distributed benchmarking and evaluation system designed to assess the performance of machine learning models across heterogeneous hardware configurations. The system addresses a critical challenge in ML deployment: understanding how different models perform when deployed on vastly different infrastructure, from edge devices like Raspberry Pis to high-end datacenter GPUs like H100 clusters.

## Architecture

RUCKUS consists of four independent subprojects:

1. **ruckus-server** - The central orchestrator that manages experiments, schedules jobs, and aggregates results
2. **ruckus-agent** - Worker components that execute benchmarks on target hardware
3. **ruckus-common** - Shared protocol definitions and models used by server and agent
4. **ruckus-ui** - Web-based user interface for experiment management and visualization

## Key Features

### Core Capabilities
- **Heterogeneous Hardware Support**: Run benchmarks across diverse hardware from edge devices to datacenter GPUs
- **Adaptive Capabilities**: Automatically adapts to available hardware and software capabilities
- **Multiple Access Modes**: Supports white-box, gray-box, and black-box model access
- **Distributed Architecture**: Scalable design with independent, containerized components
- **Auto-Discovery**: Agents automatically detect system capabilities (CPU, GPU, frameworks, monitoring tools)
- **Self-Registration**: Agents generate unique IDs and announce themselves to the server

### Multi-Run Job System (NEW)
- **Statistical Reliability**: Execute multiple runs per job for robust performance measurements
- **Cold Start Separation**: First run tracks model loading time separately from warm runs
- **Statistical Analysis**: Automatic computation of mean ± std, outlier detection, and raw data retention
- **Configurable Runs**: Set `runs_per_job` parameter (1-100) for desired statistical confidence

### Advanced GPU Detection & Benchmarking (NEW)
- **Comprehensive GPU Detection**: 
  - pynvml integration for NVIDIA GPUs with live metrics (temperature, utilization, power)
  - PyTorch fallback for cross-platform GPU detection (CUDA, Apple Silicon MPS)
  - Automatic tensor core generation mapping (1st gen Volta → 4th gen Hopper)
- **Real-Time GPU Benchmarking**:
  - Memory bandwidth testing with adaptive tensor sizes based on VRAM
  - Multi-precision FLOPS testing (FP32, FP16, BF16, INT8, FP8)
  - Tensor core capability detection with precision support mapping
  - Live GPU metrics during benchmark execution

### Comprehensive Metrics Collection
- **Performance Metrics**: Latency, throughput, model loading time
- **Resource Metrics**: Memory usage, GPU utilization, power consumption
- **Quality Metrics**: ROUGE scores, accuracy, custom evaluation metrics
- **Statistical Metrics**: Mean, std deviation, outlier identification across multiple runs

## Development Setup

### Prerequisites

- Python 3.12+
- Conda (recommended) or Python venv

### Environment Setup

1. **Create and activate conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate ruckus
   ```

   Or with pip/venv:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install packages in development mode:**
   ```bash
   # Install common package first (required by server and agent)
   cd common && pip install -e .
   
   # Install agent package
   cd ../agent && pip install -e .
   
   # Install server package
   cd ../server && pip install -e .
   ```

### Quick Start

**Run the agent:**
```bash
conda activate ruckus
python -m ruckus_agent.main
```

**Test agent endpoints:**
```bash
# Registration endpoint - agent announces itself
curl http://localhost:8081/api/v1/register

# System info endpoint - detailed hardware/software capabilities
curl http://localhost:8081/api/v1/info

# Health check
curl http://localhost:8081/health
```

**Run the server** (when implemented):
```bash
conda activate ruckus
python -m ruckus_server.main
```

## Project Structure

```
ruckus/
├── server/          # Orchestrator subsystem (ruckus-server package)
├── agent/           # Agent subsystem (ruckus-agent package)  
├── common/          # Shared subsystem (ruckus-common package)
├── ui/              # Web UI (React/TypeScript)
├── scripts/         # Utility scripts
├── docker/          # Docker configurations
└── docs/            # Documentation
```

## Development

Each subproject maintains its own dependencies and can be developed independently. The server and agent depend on the common package for shared protocol definitions.

See individual subproject README files for specific development instructions.

## Documentation

- [Project Overview](docs/project_overview.md) - Detailed system architecture and design
- [Project Structure](docs/project_structure.md) - Directory layout and organization

## License

[License information to be added]
