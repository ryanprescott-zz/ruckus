# RUCKUS Local Development Setup

This guide shows how to run the RUCKUS system locally for development and testing, including the new hardware benchmark experiments.

## Prerequisites

- **Python 3.12** with conda
- **Node.js** and npm (for the UI)
- **Git** (repository already cloned)
- **macOS** (tested on MacBook Air M1/M2)

## Quick Start

### 1. Environment Setup

```bash
# Navigate to project root
cd /Users/jbisila/Documents/misc/ruckus

# Activate the ruckus conda environment
source ~/.bash_profile && conda activate ruckus
```

### 2. Install Dependencies (One-time setup)

```bash
# Install server dependencies
cd server && pip install -e . && cd ..

# Install agent dependencies
cd agent && pip install -e . && cd ..

# Install common dependencies
cd common && pip install -e . && cd ..

# Install UI dependencies
cd ui && npm install && cd ..
```

### 3. Create Configuration Files (One-time setup)

#### Server Configuration (`server.env`)
```bash
cat > server.env << 'EOF'
RUCKUS_SERVER_HOST=localhost
RUCKUS_SERVER_PORT=8000
RUCKUS_SERVER_DEBUG=true
RUCKUS_SERVER_MODEL_PATH=/Users/jbisila/models
RUCKUS_STORAGE_STORAGE_BACKEND=sqlite
RUCKUS_SQLITE_DATABASE_URL=sqlite+aiosqlite:///./datastore/ruckus.db
RUCKUS_AGENT_AGENT_TIMEOUT=120
RUCKUS_AGENT_HEARTBEAT_INTERVAL=30
RUCKUS_SCHEDULER_MAX_CONCURRENT_JOBS=10
RUCKUS_SCHEDULER_JOB_POLL_INTERVAL=5
RUCKUS_SCHEDULER_SCHEDULER_INTERVAL=2
RUCKUS_LOG_LEVEL=INFO
EOF
```

#### Agent Configuration (`agent.env`)
```bash
cat > agent.env << 'EOF'
RUCKUS_AGENT_AGENT_ID=macbook-agent-1
RUCKUS_AGENT_AGENT_NAME=MacBook Air Test Agent
RUCKUS_AGENT_AGENT_TYPE=white_box
RUCKUS_AGENT_HOST=127.0.0.1
RUCKUS_AGENT_PORT=8081
RUCKUS_AGENT_DEBUG=true
RUCKUS_AGENT_MODEL_PATH=/Users/jbisila/models
RUCKUS_AGENT_ENABLE_VLLM=false
RUCKUS_AGENT_ENABLE_TRANSFORMERS=true
RUCKUS_AGENT_ENABLE_GPU_MONITORING=true
RUCKUS_AGENT_RUCKUS_SERVER_URL=http://127.0.0.1:8000
RUCKUS_AGENT_JOB_MAX_EXECUTION_HOURS=1
RUCKUS_AGENT_RUNS_PER_JOB=3
RUCKUS_AGENT_RESULT_CACHE_TTL_HOURS=24
RUCKUS_AGENT_RESULT_CACHE_CLEANUP_INTERVAL_MINUTES=60
RUCKUS_LOG_LEVEL=INFO
EOF
```

#### UI Configuration (`ui/.env`)
```bash
cat > ui/.env << 'EOF'
VITE_RUCKUS_SERVER_URL=http://localhost:8000
EOF
```

### 4. Create Required Directories (One-time setup)

```bash
mkdir -p server/datastore
mkdir -p /Users/jbisila/models
```

## Running the Services

You'll need **4 terminal windows/tabs**. Each terminal should start from the project root:

```bash
cd /Users/jbisila/Documents/misc/ruckus
source ~/.bash_profile && conda activate ruckus
```

### Terminal 1: Server
```bash
cd server
PYTHONPATH=src:../common/src uvicorn ruckus_server.main:app --host localhost --port 8000 --log-level info
```

**Expected output:**
```
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

### Terminal 2: Agent
```bash
cd agent
PYTHONPATH=src:../common/src uvicorn ruckus_agent.main:app --host 127.0.0.1 --port 8081 --log-level info
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8081 (Press CTRL+C to quit)
```

### Terminal 3: UI Development Server
```bash
cd ui
npm run dev
```

**Expected output:**
```
Local:   http://localhost:5173/
```

### Terminal 4: Testing/Debug Terminal
Keep this terminal free for running API tests, checking logs, etc.

## Service URLs

- **UI (Frontend)**: http://localhost:5173
- **Server (API)**: http://localhost:8000
- **Agent (API)**: http://127.0.0.1:8081
- **Server API Docs**: http://localhost:8000/docs
- **Agent API Docs**: http://127.0.0.1:8081/docs

## Testing the Setup

### 1. Health Check
```bash
# Test server
curl http://localhost:8000/health

# Test agent
curl http://127.0.0.1:8081/health
```

### 2. Register Agent
In the UI at http://localhost:5173:
1. Go to "Agents" tab
2. Register agent with URL: `http://127.0.0.1:8081`

Or via API:
```bash
curl -X POST http://localhost:8000/api/v1/agents/register \
  -H "Content-Type: application/json" \
  -d '{"agent_url": "http://127.0.0.1:8081"}'
```

### 3. Test Hardware Benchmark Experiments
1. Go to "Experiments" tab in UI
2. Click "New Experiment"
3. Try different task types:
   - `LLM_GENERATION` - Shows prompt message forms
   - `GPU_BENCHMARK` - Shows GPU benchmark parameters ✨ **NEW**
   - `MEMORY_BENCHMARK` - Shows memory test parameters ✨ **NEW**
   - `COMPUTE_BENCHMARK` - Shows compute test parameters ✨ **NEW**

## Hardware Benchmark Features

### GPU Benchmark Parameters
- **Test Memory Bandwidth**: Enable/disable memory bandwidth testing
- **Test Compute FLOPS**: Enable/disable FLOPS testing
- **Test Tensor Cores**: Enable tensor core performance testing
- **Max Memory Usage (%)**: Limit GPU memory usage (10-95%)
- **Benchmark Duration (seconds)**: How long to run tests (5-300s)

### Memory Benchmark Parameters
- **Test Sizes (MB)**: Comma-separated list (e.g., "64, 256, 1024")
- **Test Patterns**: Access patterns (e.g., "sequential, random")
- **Iterations per Size**: Number of iterations per test size (1-100)

### Compute Benchmark Parameters
- **Precision Types**: Comma-separated list (e.g., "fp32, fp16")
- **Matrix Sizes**: Comma-separated list (e.g., "1024, 2048, 4096")
- **Include Tensor Operations**: Enable specialized tensor operations

## MacBook Air Considerations

- **No NVIDIA GPU**: Hardware benchmarks will run in CPU/MPS mode
- **Performance Numbers**: Will be CPU-based, not GPU-based
- **Functionality**: UI workflow and backend logic work perfectly
- **Benchmarking**: Uses PyTorch CPU/MPS operations instead of CUDA

## Troubleshooting

### Server Won't Start
- Check if port 8000 is in use: `lsof -i :8000`
- Verify conda environment: `which python`
- Check server logs for import errors

### Agent Won't Start
- Check if port 8081 is in use: `lsof -i :8081`
- Model directory warnings are normal for hardware benchmarks
- Verify agent can reach server: `curl http://localhost:8000/health`

### UI Can't Connect to Server
- Verify server is running: `curl http://localhost:8000/health`
- Check UI console for CORS errors
- Ensure `.env` file exists in `ui/` directory

### Environment Variable Issues
- Don't include comments in `.env` files
- Use clean files without `#` comments to avoid shell parsing errors
- Set variables manually if `export $(cat file.env | xargs)` fails

### Port Conflicts
If you need different ports:
- **Server**: Change `RUCKUS_SERVER_PORT` and update `VITE_RUCKUS_SERVER_URL`
- **Agent**: Change `RUCKUS_AGENT_PORT` and update agent registration URL
- **UI**: Change in `vite.config.ts` or use `npm run dev -- --port 3000`

## Development Workflow

1. **Start Services**: Server → Agent → UI
2. **Register Agent**: Via UI or API
3. **Create Experiments**: Test different task types
4. **Run Jobs**: Execute experiments and check results
5. **Debug**: Use Terminal 4 for API testing and log checking

## Stopping Services

- **Server/Agent**: Ctrl+C in their respective terminals
- **UI**: Ctrl+C in the UI terminal
- **Clean shutdown**: Stop UI first, then agent, then server

## Next Steps

- **GPU Testing**: Test on machine with NVIDIA GPU for full hardware metrics
- **Model Setup**: Add actual models to `/Users/jbisila/models` for LLM experiments
- **Production**: See main README for production deployment setup