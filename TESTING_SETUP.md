# RUCKUS End-to-End Testing Setup Guide

## Prerequisites 

### Hardware Requirements
- **NVIDIA GPU** (for proper GPU metrics collection)
- **8GB+ RAM** (for model loading)
- **10GB+ disk space** (for models and dependencies)

### Software Requirements
- **Python 3.10-3.12** (tested with 3.12)
- **NVIDIA drivers** and **nvidia-smi** (for GPU monitoring)
- **Git** (to clone the repository)
- **conda** or **venv** (for environment isolation)

## Step 1: Environment Setup

```bash
# Clone and navigate to RUCKUS
git clone <your-repo-url>
cd ruckus

# Create conda environment (recommended)
conda create -n ruckus-test python=3.12 -y
conda activate ruckus-test

# OR create venv environment
python -m venv ruckus-test
source ruckus-test/bin/activate  # Linux/Mac
# ruckus-test\Scripts\activate   # Windows
```

## Step 2: Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# OR: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ML frameworks
pip install transformers accelerate datasets

# Install RUCKUS packages in development mode
cd common && pip install -e . && cd ..
cd agent && pip install -e . && cd ..
cd server && pip install -e . && cd ..
```

## Step 3: Download Test Model

```bash
# Create models directory
mkdir -p ~/models

# Download distilgpt2 model
python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
print('Downloading distilgpt2 model...')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
os.makedirs('~/models/distilgpt2', exist_ok=True)
model.save_pretrained('~/models/distilgpt2')
tokenizer.save_pretrained('~/models/distilgpt2')
print('Model saved to ~/models/distilgpt2')
"
```

**✅ Expected Output:** Should see model download progress and "Model saved" message.

## Step 4: Create Configuration Files

### Server Configuration (`server.env`)
```bash
cat > server.env << 'EOF'
# RUCKUS Server Configuration
RUCKUS_SERVER_HOST=0.0.0.0
RUCKUS_SERVER_PORT=8000
RUCKUS_SERVER_DEBUG=true

# Storage - SQLite for testing
RUCKUS_STORAGE_STORAGE_BACKEND=sqlite
RUCKUS_SQLITE_DATABASE_URL=sqlite+aiosqlite:///./data/ruckus.db

# Agent Management
RUCKUS_AGENT_AGENT_TIMEOUT=120
RUCKUS_AGENT_HEARTBEAT_INTERVAL=30

# Scheduler
RUCKUS_SCHEDULER_MAX_CONCURRENT_JOBS=10
RUCKUS_SCHEDULER_JOB_POLL_INTERVAL=5
RUCKUS_SCHEDULER_SCHEDULER_INTERVAL=2

# Logging
RUCKUS_LOG_LEVEL=INFO
EOF
```

### Agent Configuration (`agent.env`)
```bash
cat > agent.env << 'EOF'
# RUCKUS Agent Configuration
RUCKUS_AGENT_AGENT_ID=gpu-agent-1
RUCKUS_AGENT_AGENT_NAME=GPU Test Agent
RUCKUS_AGENT_HOST=0.0.0.0
RUCKUS_AGENT_PORT=8081
RUCKUS_AGENT_DEBUG=true

# Model path (adjust to your models directory)
RUCKUS_AGENT_MODEL_PATH=/home/username/models
# RUCKUS_AGENT_MODEL_PATH=~/models  # Alternative

# Framework settings
RUCKUS_AGENT_ENABLE_VLLM=false
RUCKUS_AGENT_ENABLE_TRANSFORMERS=true
RUCKUS_AGENT_ENABLE_GPU_MONITORING=true

# Job execution
RUCKUS_AGENT_JOB_MAX_EXECUTION_HOURS=2.0
RUCKUS_AGENT_RESULT_CACHE_TTL_HOURS=24

# Logging
RUCKUS_AGENT_LOG_LEVEL=INFO
RUCKUS_AGENT_LOG_FORMAT=text
EOF
```

## Step 5: Create Required Directories

```bash
mkdir -p data static
```

## Step 6: Start the Server

### Terminal 1: Start RUCKUS Server
```bash
# Set Python path and start server
export PYTHONPATH=/path/to/ruckus/server/src:$PYTHONPATH
python -c "
from dotenv import load_dotenv
load_dotenv('server.env')
import uvicorn
from ruckus_server.main import app
uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
"
```

**✅ Expected Output:**
```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Agent manager initialized
INFO:     Agent manager backend started
INFO:     Experiment manager backend started
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Verify Server is Running
```bash
curl http://localhost:8000/api/v1/
# Expected: {"version": "v1", "endpoints": ["/agents", "/jobs", "/experiments"]}
```

## Step 7: Start the Agent

### Terminal 2: Start RUCKUS Agent
```bash
# Set Python path and start agent
export PYTHONPATH=/path/to/ruckus/agent/src:$PYTHONPATH
python -c "
from dotenv import load_dotenv
load_dotenv('agent.env')
import uvicorn
from ruckus_agent.main import app
uvicorn.run(app, host='0.0.0.0', port=8081, log_level='info')
"
```

**✅ Expected Output:**
```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8081
```

### Verify Agent is Running and Detects GPU
```bash
curl http://localhost:8081/api/v1/info | jq .
```

**✅ Expected Output:** Should show:
- Agent info with system details
- **GPU detection** with nvidia-smi details (memory, utilization, etc.)
- **Model discovery** showing distilgpt2 model
- **Framework capabilities** with CUDA support

## Step 8: Register Agent with Server

```bash
curl -X POST http://localhost:8000/api/v1/agents/register \
  -H "Content-Type: application/json" \
  -d '{"agent_url": "http://localhost:8081"}' | jq .
```

**✅ Expected Output:**
```json
{
  "agent_id": "gpu-agent-1", 
  "registered_at": "2025-01-XX..."
}
```

## Step 9: Test Job Execution

### Submit Test Job
```bash
curl -X POST http://localhost:8081/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "gpu-test-1",
    "experiment_id": "exp-gpu-test",
    "model": "distilgpt2",
    "framework": "transformers",
    "task_type": "llm_generation", 
    "task_config": {
      "prompt": "The future of AI is",
      "max_tokens": 50,
      "temperature": 0.8
    },
    "required_metrics": ["latency", "throughput", "gpu_utilization", "gpu_memory"],
    "timeout_seconds": 300,
    "runs_per_job": 1
  }' | jq .
```

**✅ Expected Output:**
```json
{"job_id": "gpu-test-1", "status": "accepted"}
```

### Monitor Job Execution

**During Execution** (should show job is running):
```bash
curl -s http://localhost:8081/api/v1/status | jq .
```

**✅ Expected Output (while running):**
```json
{
  "agent_id": "gpu-agent-1",
  "status": "busy",
  "current_job_id": "gpu-test-1", 
  "current_experiment_id": "exp-gpu-test",
  ...
}
```

**After Completion** (should show results available):
```bash
curl -s http://localhost:8081/api/v1/status | jq .
```

**✅ Expected Output (after completion):**
```json
{
  "agent_id": "gpu-agent-1",
  "status": "idle",
  "current_job_id": null,
  "available_results": [
    {
      "job_id": "gpu-test-1",
      "completed_at": "2025-01-XX...",
      "result_type": "success"
    }
  ]
}
```

### Retrieve Results
```bash
curl -s http://localhost:8081/api/v1/results/gpu-test-1 | jq .
```

**✅ Expected Output (SUCCESS case):**
```json
{
  "job_id": "gpu-test-1",
  "experiment_id": "exp-gpu-test", 
  "status": "completed",
  "result_type": "success",
  "started_at": "2025-01-XX...",
  "completed_at": "2025-01-XX...",
  "duration_seconds": 15.234,
  "outputs": {
    "generated_text": "The future of AI is bright and full of possibilities...",
    "token_count": 50
  },
  "metrics": {
    "latency": 0.234,
    "throughput": 213.45,
    "gpu_utilization": 85.2,
    "gpu_memory": 2048
  },
  "system_metrics": {
    "peak_gpu_memory_mb": 2048,
    "average_gpu_utilization": 82.1
  }
}
```

## Step 10: Test Error Scenarios

### Test Timeout (should timeout after 30 seconds)
```bash
curl -X POST http://localhost:8081/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "timeout-test",
    "experiment_id": "exp-timeout",
    "model": "distilgpt2", 
    "framework": "transformers",
    "task_type": "llm_generation",
    "task_config": {"prompt": "Hello", "max_tokens": 1000},
    "timeout_seconds": 30,
    "runs_per_job": 1
  }'
```

**✅ Expected Behavior:** 
- Job should timeout after 30 seconds
- Status should show "timeout" 
- Results should have `"result_type": "execution"`

## Step 11: Swagger UI Testing

### Access Swagger Documentation
1. **Server Swagger:** http://localhost:8000/docs
2. **Agent Swagger:** http://localhost:8081/docs

### Key Endpoints to Test via Swagger:

**Server Endpoints:**
- `GET /api/v1/` - API info
- `POST /api/v1/agents/register` - Register agent
- `GET /api/v1/agents/` - List agents

**Agent Endpoints:**
- `GET /api/v1/info` - Agent system info
- `GET /api/v1/status` - Current status  
- `POST /api/v1/execute` - Execute job
- `GET /api/v1/results/{job_id}` - Get results
- `DELETE /api/v1/jobs/{job_id}` - Cancel job

## What to Look For

### ✅ Success Indicators:
1. **GPU Detection:** nvidia-smi output in agent info
2. **Model Loading:** Model loads successfully on GPU
3. **Real Inference:** Generated text output with proper metrics
4. **GPU Metrics:** Real GPU utilization and memory usage
5. **Performance:** Faster execution than CPU (should be <30s vs 2+ minutes)
6. **Status Tracking:** Proper status transitions (idle → busy → idle)
7. **Result Persistence:** Results available after completion

### ⚠️ Common Issues:
1. **CUDA not available:** Check PyTorch CUDA installation
2. **Model path issues:** Verify `RUCKUS_AGENT_MODEL_PATH` is correct
3. **Permission errors:** Ensure nvidia-smi is accessible
4. **Port conflicts:** Make sure ports 8000/8081 are available
5. **Memory issues:** Monitor GPU memory usage

## Cleanup

```bash
# Stop the servers (Ctrl+C in both terminals)
# Deactivate environment
conda deactivate
# Or: deactivate  # for venv

# Optional: Remove test data
rm -rf data/ *.env
```

This setup provides a complete end-to-end test environment for validating the RUCKUS job execution system with real GPU hardware!