# RUCKUS

RUCKUS is a distributed benchmarking and evaluation system designed to assess the performance of machine learning models across heterogeneous hardware configurations. The system addresses a critical challenge in ML deployment: understanding how different models perform when deployed on vastly different infrastructure, from edge devices like Raspberry Pis to high-end datacenter GPUs like H100 clusters.

## Architecture

RUCKUS consists of four independent subprojects:

1. **ruckus-server** - The central orchestrator that manages experiments, schedules jobs, and aggregates results
2. **ruckus-agent** - Worker components that execute benchmarks on target hardware
3. **ruckus-common** - Shared protocol definitions and models used by server and agent
4. **ruckus-ui** - Web-based user interface for experiment management and visualization

## Key Features

- **Heterogeneous Hardware Support**: Run benchmarks across diverse hardware from edge devices to datacenter GPUs
- **Adaptive Capabilities**: Automatically adapts to available hardware and software capabilities
- **Multiple Access Modes**: Supports white-box, gray-box, and black-box model access
- **Distributed Architecture**: Scalable design with independent, containerized components
- **Comprehensive Metrics**: Collects performance, quality, and resource utilization metrics

## Quick Start

Each subproject is independently buildable and deployable:

```bash
# Build common package first (required by server and agent)
cd common && pip install -e .

# Build and run server
cd server && pip install -e .
python -m ruckus_server

# Build and run agent
cd agent && pip install -e .
python -m ruckus_agent
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
