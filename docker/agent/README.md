# RUCKUS Agent Docker Configuration

This directory contains Docker configuration for running RUCKUS agents in containerized environments with GPU support.

## Prerequisites

1. **NVIDIA GPU Drivers**: Installed on host system
2. **Docker with GPU Support**: Docker Engine with nvidia-container-toolkit
3. **Downloaded Models**: HuggingFace models downloaded to a host directory
4. **Sufficient Resources**: GPU memory and system RAM for target models

### Quick GPU Setup Test
```bash
# Verify GPU support works in Docker
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

## Configuration

1. **Copy environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file**:
   - Set `MODELS_HOST_PATH` to your models directory
   - Set `ORCHESTRATOR_URL` to your RUCKUS server
   - Adjust other settings as needed

3. **Download models** to your configured `MODELS_HOST_PATH`:
   ```bash
   # Example: Download a model using huggingface-hub
   mkdir -p /path/to/your/models
   cd /path/to/your/models
   git clone https://huggingface.co/microsoft/DialoGPT-medium
   ```

## Usage

### Build and Run
```bash
# Build the agent image
docker-compose build

# Run the agent
docker-compose up -d

# View logs
docker-compose logs -f ruckus-agent

# Check agent health
curl http://localhost:8081/health

# Test agent registration
curl http://localhost:8081/api/v1/register

# View discovered models and system info
curl http://localhost:8081/api/v1/info
```

### Development Mode
```bash
# Set debug mode in .env
echo "DEBUG=true" >> .env
echo "LOG_LEVEL=DEBUG" >> .env

# Rebuild and restart
docker-compose up --build -d
```

## Directory Structure

```
docker/agent/
├── Dockerfile              # Multi-stage build with CUDA support
├── docker-compose.yml      # Service definition with GPU config
├── .env                     # Environment variables (customize this)
└── README.md               # This file
```

## Volume Mounts

- **`/ruckus/models`**: Read-only mount for HuggingFace models
- **`/ruckus/logs`**: Agent logs (optional)
- **`/ruckus/config`**: Configuration files (optional)

## Environment Variables

Key configuration options:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_HOST_PATH` | `./models` | Host path containing HuggingFace models |
| `ORCHESTRATOR_URL` | - | URL of RUCKUS server |
| `AGENT_PORT` | `8081` | Port to expose agent API |
| `MAX_CONCURRENT_JOBS` | `1` | Maximum concurrent benchmark jobs |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |

## Troubleshooting

### GPU Not Detected
```bash
# Check nvidia-smi works in container
docker-compose exec ruckus-agent nvidia-smi

# Verify GPU environment variables
docker-compose exec ruckus-agent env | grep CUDA
```

### Models Not Found
```bash
# Check models directory is mounted correctly
docker-compose exec ruckus-agent ls -la /ruckus/models

# Verify models have correct structure
docker-compose exec ruckus-agent find /ruckus/models -name "config.json"
```

### Agent Registration Issues
```bash
# Check agent can reach server
docker-compose exec ruckus-agent curl -v ${ORCHESTRATOR_URL}/health

# Check agent logs for registration errors
docker-compose logs ruckus-agent | grep -i registration
```

## Performance Tuning

### Memory Limits
Uncomment and adjust memory limits in `docker-compose.yml` based on your models and hardware.

### GPU Memory Management
For multiple models or large models, consider:
- Setting `CUDA_VISIBLE_DEVICES` to specific GPU indices
- Adjusting vLLM memory settings
- Using model quantization

### Concurrent Jobs
Adjust `MAX_CONCURRENT_JOBS` based on:
- Available GPU memory
- Model sizes
- System capabilities