ruckus/
├── orchestrator/           # Central coordination service
├── agent/                  # Worker that runs benchmarks
├── common/                 # Shared code between orchestrator and agent
├── dashboard/              # Web UI for monitoring
├── data/                   # Test data and configurations
├── docker/                 # Container definitions
├── scripts/                # Utility and demo scripts
├── tests/                  # Test suites
├── configs/                # Configuration files
├── docs/                   # Documentation
├── .env.example            # Environment variable template
├── docker-compose.yml      # Local development setup
├── requirements.txt        # Python dependencies
├── Makefile               # Common commands
└── README.md              # Project documentation
Detailed Component Breakdown
orchestrator/ - The Brain

orchestrator/
├── __init__.py
├── main.py                 # FastAPI application entry point
├── config.py               # Pydantic settings and configuration
├── api/                    # REST API layer
│   ├── __init__.py
│   ├── routes/             # API endpoint definitions
│   │   ├── __init__.py
│   │   ├── experiments.py  # POST /experiments, GET /experiments/{id}
│   │   ├── agents.py       # POST /agents/register, GET /agents
│   │   ├── jobs.py         # GET /jobs, POST /jobs/{id}/update
│   │   └── results.py      # GET /results/{experiment_id}
│   └── dependencies.py     # FastAPI dependency injection
├── core/                   # Business logic
│   ├── __init__.py
│   ├── scheduler.py        # Job scheduling and distribution logic
│   ├── agent_manager.py    # Agent pool and capability management
│   ├── result_aggregator.py # Combine and normalize results
│   └── experiment_manager.py # Experiment lifecycle management
├── db/                     # Database layer
│   ├── __init__.py
│   ├── database.py         # SQLAlchemy engine and session setup
│   ├── models.py           # ORM models (Experiment, Job, Agent, Result)
│   ├── repositories.py     # Data access patterns
│   └── migrations/         # Alembic migrations (if needed)
├── services/               # External service integrations
│   ├── __init__.py
│   ├── metrics_service.py  # Metric calculation and normalization
│   └── storage_service.py  # File/artifact storage
└── utils/                  # Helper utilities
    ├── __init__.py
    └── logging.py          # Structured logging setup
agent/ - The Worker

agent/
├── __init__.py
├── main.py                 # FastAPI app for agent
├── config.py               # Agent configuration
├── api/                    # Agent API endpoints
│   ├── __init__.py
│   └── routes.py           # POST /execute, GET /capabilities, GET /health
├── core/                   # Core agent logic
│   ├── __init__.py
│   ├── executor.py         # Job execution orchestration
│   ├── capability_manager.py # Detect and advertise capabilities
│   └── progress_reporter.py # Send updates to orchestrator
├── adapters/               # Model framework adapters
│   ├── __init__.py
│   ├── base.py            # Abstract base adapter
│   ├── transformers_adapter.py # Hugging Face transformers
│   ├── vllm_adapter.py    # vLLM integration
│   ├── pytorch_adapter.py # Raw PyTorch
│   └── blackbox_adapter.py # Generic API adapter
├── metrics/                # Metric collectors
│   ├── __init__.py
│   ├── base.py            # Base metric collector
│   ├── performance.py     # Latency, throughput
│   ├── resource.py        # Memory, GPU usage
│   └── quality.py         # Task-specific quality metrics
├── tasks/                  # Task implementations
│   ├── __init__.py
│   ├── base.py            # Base task interface
│   └── wikipedia_summarization.py # Initial task
└── utils/                  # Agent utilities
    ├── __init__.py
    ├── hardware_detection.py # Detect GPUs, memory, etc.
    └── model_cache.py      # Local model management
common/ - Shared Components

common/
├── __init__.py
├── protocol.py             # Wire protocol definitions
├── models.py               # Pydantic models used by both services
│   ├── # ExperimentSpec, JobSpec, AgentCapabilities
│   ├── # MetricResult, TaskConfig, etc.
├── constants.py            # Shared constants and enums
│   ├── # JobStatus, JobStage, MetricType
└── utils.py                # Shared utility functions
dashboard/ - Web Interface

dashboard/
├── static/                 # Static web files
│   ├── index.html         # Main dashboard page
│   ├── css/
│   │   └── dashboard.css  # Styling
│   ├── js/
│   │   ├── dashboard.js   # Main application logic
│   │   ├── api.js        # API client for polling
│   │   └── charts.js     # Result visualizations
│   └── img/              # Images/icons if needed
└── templates/             # If using server-side rendering
data/ - Test Data and Configs

data/
├── datasets/              # Test datasets
│   ├── wikipedia/         # Wikipedia articles for testing
│   │   ├── short/        # 1-2k token articles
│   │   ├── medium/       # 5-10k token articles
│   │   └── long/         # 15k+ token articles
│   └── references/       # Ground truth for quality metrics
├── experiments/          # Experiment configurations
│   ├── quick_test.yaml  # 5-minute smoke test
│   ├── standard.yaml    # Standard benchmark suite
│   └── stress_test.yaml # Heavy load testing
└── results/             # Result storage (git-ignored)
docker/ - Container Definitions

docker/
├── orchestrator/
│   ├── Dockerfile         # Orchestrator container
│   └── entrypoint.sh     # Startup script
├── agent/
│   ├── Dockerfile.base   # Base agent image
│   ├── Dockerfile.gpu    # GPU-enabled agent
│   └── Dockerfile.cpu    # CPU-only agent
└── scripts/              # Docker helper scripts
    ├── build_all.sh
    └── push_images.sh
scripts/ - Automation and Utilities

scripts/
├── setup/                # Setup scripts
│   ├── init_db.py       # Initialize database
│   ├── seed_data.py     # Load test data
│   └── download_models.py # Pre-download models
├── demo/                 # Demo automation
│   ├── run_demo.py      # Full demo script
│   ├── create_experiment.py # Create sample experiment
│   └── show_results.py  # Display results nicely
├── utils/               # Utility scripts
│   ├── health_check.py  # System health verification
│   ├── cleanup.py       # Clean up resources
│   └── export_results.py # Export to various formats
└── dev/                 # Development helpers
    ├── generate_data.py # Create test data
    └── test_agent.py    # Test agent connectivity
configs/ - Configuration Templates

configs/
├── orchestrator/
│   ├── base.yaml        # Base orchestrator config
│   ├── development.yaml # Dev overrides
│   └── production.yaml  # Prod settings
├── agent/
│   ├── white_box.yaml   # Full control agent
│   ├── gray_box.yaml    # API-based agent
│   └── black_box.yaml   # Minimal agent
├── tasks/
│   ├── wikipedia_summarization.yaml
│   └── task_template.yaml
└── metrics/
    └── metric_definitions.yaml
tests/ - Test Suites

tests/
├── unit/                # Unit tests
│   ├── orchestrator/
│   ├── agent/
│   └── common/
├── integration/         # Integration tests
│   ├── test_job_flow.py
│   └── test_agent_registration.py
├── e2e/                # End-to-end tests
│   └── test_full_experiment.py
├── fixtures/           # Test fixtures
└── conftest.py        # Pytest configuration