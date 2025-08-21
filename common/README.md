# RUCKUS Common

The `ruckus-common` package provides shared protocol definitions, data models, and constants used by both the RUCKUS server and agent components.

## Building the Package

To build a wheel file for distribution:

```bash
# From the common/ directory
pip install build
python -m build
```

This will create wheel and source distribution files in the `dist/` directory:
- `dist/ruckus_common-0.1.0-py3-none-any.whl`
- `dist/ruckus-common-0.1.0.tar.gz`

## Installing in Other Projects

### Development Installation

For development, install the package in editable mode from the source directory:

```bash
# From the server/ or agent/ directory
pip install -e ../common
```

This creates a link to the source code, so changes to the common package are immediately available.

### Production Installation

Install from the built wheel file:

```bash
# From the server/ or agent/ directory
pip install ../common/dist/ruckus_common-0.1.0-py3-none-any.whl
```

### Installing from PyPI (Future)

Once published to PyPI, install with:

```bash
pip install ruckus-common
```

## Package Contents

- `protocol.py` - Wire protocol definitions for server-agent communication
- `models.py` - Shared data models (Experiment, Job, Agent)
- `constants.py` - System constants and configuration defaults

## Usage

```python
from ruckus_common.protocol import JobRequest, JobResponse, AgentRegistration
from ruckus_common.models import Experiment, Job, Agent
from ruckus_common.constants import API_VERSION, DEFAULT_SERVER_PORT
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
isort src/
```

### Type Checking

```bash
mypy src/
```
