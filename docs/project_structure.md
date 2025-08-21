ruckus/
├── server/                    # Orchestrator subsystem
│   ├── pyproject.toml        # Independent package: ruckus-server
│   ├── src/
│   │   └── ruckus_server/
│   │       ├── __init__.py
│   │       ├── api/
│   │       ├── core/
│   │       ├── services/
│   │       └── main.py
│   └── tests/
│
├── agent/                     # Agent subsystem
│   ├── pyproject.toml        # Independent package: ruckus-agent
│   ├── src/
│   │   └── ruckus_agent/
│   │       ├── __init__.py
│   │       ├── adapters/
│   │       ├── core/
│   │       │   ├── models.py      # Agent-specific models
│   │       │   ├── detector.py    # System detection
│   │       │   └── agent.py
│   │       └── main.py
│   └── tests/
│
├── common/                    # Shared subsystem
│   ├── pyproject.toml        # Independent package: ruckus-common
│   ├── src/
│   │   └── ruckus_common/
│   │       ├── __init__.py
│   │       ├── protocol.py   # Wire protocol
│   │       ├── models.py     # Shared models
│   │       └── constants.py
│   └── tests/
│
├── ui/                        # Keep UI separate as is
├── scripts/                   # Top-level scripts
├── docker/                    # Docker configurations
├── docs/                      # Documentation
└── README.md                  # Project overview