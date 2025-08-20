# Ruckus

Ruckus is an evaluation and benchmarking application designed to support evaluation of large language models.

## Features

- Create test suites for LLM evaluation
- Support for multiple LLM runtimes (vllm, transformers, pytorch)
- Multi-platform support (CUDA, CPU)
- Web-based UI for test management
- Distributed agent architecture for scalable testing
- Comprehensive benchmarking and scoring

## Architecture

- **UI**: React/TypeScript web interface
- **Orchestrator**: FastAPI service for test coordination
- **Agents**: Distributed FastAPI services for test execution
- **Database**: SQLite (upgradeable to PostgreSQL)

## Development Setup

1. Create conda environment:
```bash
conda env create -f environments/environment.yml
conda activate ruckus
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Run tests:
```bash
pytest
```

5. Start the orchestrator:
```bash
cd src/ruckus/orchestrator
python main.py
```

6. Start an agent:
```bash
cd src/ruckus/agent
python main.py
```

7. Start the UI:
```bash
cd ui
npm install
npm start
```

## Project Structure

```
ruckus/
├── ui/                     # React/TypeScript UI
├── src/ruckus/            # Python source code
│   ├── orchestrator/      # Orchestration service
│   └── agent/            # Agent service
├── environments/          # Conda environment files
└── pyproject.toml        # Python project configuration
```
