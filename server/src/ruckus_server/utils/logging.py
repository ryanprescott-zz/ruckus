"""Logging utilities for server."""

import logging
import sys
from datetime import datetime

from ..core.config import settings


def setup_logging():
    """Setup server logging."""
    level = getattr(logging, settings.log_level.upper())

    # Configure formatting
    if settings.log_format == "json":
        import json

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                })

        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Setup handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure logger
    logger = logging.getLogger("ruckus_server")
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


logger = setup_logging()