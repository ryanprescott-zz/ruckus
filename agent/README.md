# RUCKUS Agent

Worker agent component for the RUCKUS distributed benchmarking system. The agent automatically detects system capabilities, registers with the server, and provides detailed system information through REST APIs.

## Features

- **Automatic Registration**: Generates unique agent ID and registers with server
- **System Detection**: Automatically detects CPU, GPU, memory, and available ML frameworks
- **Extensible Storage**: Abstract storage backend (in-memory by default, extensible to PostgreSQL/SQLite)
- **RESTful API**: Provides `/register` and `/info` endpoints for server communication
- **Comprehensive Testing**: 29 tests with mocking for reliable CI/CD

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
# Registration endpoint
curl http://localhost:8081/api/v1/register

# System info endpoint  
curl http://localhost:8081/api/v1/info

# Health check
curl http://localhost:8081/health
```

## API Endpoints

### `/api/v1/register` (GET)
Returns agent registration information:
```json
{
  "agent_id": "agent-656b6586",
  "agent_name": "agent-656b6586-white_box", 
  "message": "Agent registered successfully",
  "server_time": "2025-08-21T22:02:01.315925"
}
```

### `/api/v1/info` (GET)
Returns detailed system information:
```json
{
  "agent_id": "agent-656b6586",
  "agent_type": "white_box",
  "system_info": {
    "system": {"hostname": "...", "os": "Darwin"},
    "cpu": {"cores_physical": 8, "model": "..."},
    "gpus": [...],
    "frameworks": [...],
    "metrics": [...]
  },
  "capabilities": {
    "gpu_count": 0,
    "frameworks": [],
    "monitoring_available": true
  }
}
```

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