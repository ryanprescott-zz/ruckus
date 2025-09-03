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

## Hardware Detection & Enum Reference

RUCKUS uses comprehensive Pydantic enums for hardware detection, providing standardized values for UI dropdowns and validation. Below is the complete reference for all supported hardware types and their mappings:

### Operating Systems (`OSType`)
- `LINUX` - Linux distributions (Ubuntu, CentOS, RHEL, etc.)
- `WINDOWS` - Microsoft Windows  
- `DARWIN` - Apple macOS
- `UNKNOWN` - Unrecognized or other operating systems

### CPU Architectures (`CPUArchitecture`)
- `X86_64` / `AMD64` - 64-bit x86 processors (Intel, AMD)
- `ARM64` / `AARCH64` - 64-bit ARM processors (Apple Silicon, server ARM)
- `ARM` - 32-bit ARM processors
- `I386` - 32-bit x86 processors (legacy)
- `UNKNOWN` - Unrecognized CPU architecture

### GPU Vendors (`GPUVendor`)
- `NVIDIA` - NVIDIA graphics cards (GeForce, Quadro, Tesla, A-series)
- `AMD` - AMD graphics cards (Radeon, Instinct)
- `INTEL` - Intel integrated and discrete graphics
- `APPLE` - Apple Silicon integrated GPU (M1, M2, M3)
- `UNKNOWN` - Unrecognized or other GPU vendors

### NVIDIA Tensor Core Generations (`TensorCoreGeneration`)
- `FIRST_GEN` - Volta architecture (Tesla V100, Titan V)
- `SECOND_GEN` - Turing architecture (RTX 20 series: RTX 2060/2070/2080/2090)
- `THIRD_GEN` - Ampere architecture (RTX 30 series: RTX 3060/3070/3080/3090, A100/A6000)  
- `FOURTH_GEN` - Ada Lovelace/Hopper (RTX 40 series: RTX 4060/4070/4080/4090, H100)
- `NONE` - No tensor cores (older GPUs or non-NVIDIA)

### GPU Precision Support (`PrecisionType`)
- `FP64` - Double precision (64-bit floating point)
- `FP32` - Single precision (32-bit floating point) - **Universal support**
- `TF32` - TensorFloat-32 (19-bit precision) - **Ampere+ only**
- `FP16` - Half precision (16-bit floating point) - **Volta+**
- `BF16` - Brain Float 16 (Google format) - **Ampere+**
- `INT8` - 8-bit integer - **Turing+**  
- `FP8` - 8-bit floating point - **Hopper+ only**

### GPU Detection Methods (`DetectionMethod`)
- `PYNVML` - NVIDIA Management Library (most comprehensive for NVIDIA)
- `NVIDIA_SMI` - NVIDIA System Management Interface fallback
- `PYTORCH_CUDA` - PyTorch CUDA detection
- `PYTORCH_MPS` - PyTorch Metal Performance Shaders (Apple Silicon)
- `PYTORCH_ROCM` - PyTorch ROCm (AMD GPUs)
- `UNKNOWN` - Unrecognized detection method

### ML Frameworks (`FrameworkName`)
- `PYTORCH` - PyTorch deep learning framework
- `TRANSFORMERS` - Hugging Face Transformers library
- `VLLM` - vLLM high-performance inference engine
- `TENSORRT` - NVIDIA TensorRT optimization library
- `ONNX` - Open Neural Network Exchange format
- `TRITON` - NVIDIA Triton Inference Server
- `UNKNOWN` - Unrecognized or other framework

### System Monitoring Tools (`HookType`)
- `GPU_MONITOR` - GPU monitoring tools (nvidia-smi, rocm-smi)
- `CPU_MONITOR` - CPU monitoring tools (htop, top)
- `MEMORY_MONITOR` - Memory monitoring tools (free, vmstat)
- `DISK_MONITOR` - Disk I/O monitoring (iotop, iostat)
- `PROCESS_MONITOR` - Process monitoring (ps, pidstat)
- `NETWORK_MONITOR` - Network monitoring (netstat, iftop)
- `PROFILER` - Performance profilers (nsight, perf)
- `UNKNOWN` - Unrecognized monitoring tool

### Metric Collection Methods (`MetricCollectionMethod`)
- `TIMER` - Built-in timing measurements
- `CALCULATION` - Computed/derived metrics
- `NVIDIA_SMI` - NVIDIA GPU metrics via nvidia-smi
- `PYNVML` - NVIDIA GPU metrics via Python NVML
- `PSUTIL` - System resource metrics via psutil
- `PYTORCH` - PyTorch built-in performance metrics
- `CUSTOM` - Custom collection logic
- `EXTERNAL_API` - External service API calls
- `UNKNOWN` - Unrecognized collection method

### Real-World GPU Examples

**NVIDIA Consumer (GeForce)**
- RTX 4090: `FOURTH_GEN` tensor cores, `FP8` precision support
- RTX 3080: `THIRD_GEN` tensor cores, `BF16` precision support  
- RTX 2080: `SECOND_GEN` tensor cores, `FP16` precision support

**NVIDIA Professional (Quadro/Tesla)**
- H100: `FOURTH_GEN` tensor cores, full `FP8` support
- A100: `THIRD_GEN` tensor cores, `BF16` + sparsity support
- V100: `FIRST_GEN` tensor cores, `FP16` support only

**Apple Silicon**
- M1/M2/M3: `APPLE` vendor, `PYTORCH_MPS` detection, custom acceleration

These enums ensure consistent hardware identification across agent registration, job assignment, and UI display. The system automatically detects capabilities and maps them to these standardized values.

## Documentation

- [Project Overview](docs/project_overview.md) - Detailed system architecture and design
- [Project Structure](docs/project_structure.md) - Directory layout and organization

## License

[License information to be added]
