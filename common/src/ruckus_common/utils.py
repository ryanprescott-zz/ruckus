"""Shared utility functions for RUCKUS."""

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import uuid
import psutil


# ID Generation
def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    unique_id = uuid.uuid4().hex[:8]
    if prefix:
        return f"{prefix}-{unique_id}"
    return unique_id


def generate_experiment_id() -> str:
    """Generate unique experiment ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    unique = uuid.uuid4().hex[:6]
    return f"exp-{timestamp}-{unique}"


def generate_job_id(experiment_id: str, model: str, framework: str) -> str:
    """Generate deterministic job ID."""
    content = f"{experiment_id}:{model}:{framework}:{time.time()}"
    hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"job-{hash_val}"


# Hashing and Checksums
def calculate_hash(data: Union[str, bytes, Dict]) -> str:
    """Calculate SHA256 hash of data."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def calculate_checksum(file_path: Union[str, Path]) -> str:
    """Calculate checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# Time Utilities
def utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.utcnow()


def timestamp_str() -> str:
    """Get ISO format timestamp string."""
    return datetime.utcnow().isoformat() + "Z"


def parse_duration(duration_str: str) -> int:
    """Parse duration string to seconds (e.g., '1h', '30m', '45s')."""
    units = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
    }

    if duration_str[-1] in units:
        value = int(duration_str[:-1])
        unit = duration_str[-1]
        return value * units[unit]

    return int(duration_str)


# Network Utilities
def get_local_ip() -> str:
    """Get local IP address."""
    try:
        # Create a socket to external address (doesn't actually connect)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def get_hostname() -> str:
    """Get system hostname."""
    return socket.gethostname()


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


# File System Utilities
def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_files(directory: Union[str, Path], pattern: str) -> List[Path]:
    """Find files matching pattern in directory."""
    directory = Path(directory)
    return list(directory.glob(pattern))


def get_file_size_mb(path: Union[str, Path]) -> float:
    """Get file size in MB."""
    return Path(path).stat().st_size / (1024 * 1024)


def cleanup_old_files(directory: Union[str, Path], days: int = 7) -> int:
    """Remove files older than specified days."""
    directory = Path(directory)
    cutoff = datetime.utcnow() - timedelta(days=days)
    removed = 0

    for file in directory.iterdir():
        if file.is_file():
            modified = datetime.fromtimestamp(file.stat().st_mtime)
            if modified < cutoff:
                file.unlink()
                removed += 1

    return removed


# System Information
def get_system_info() -> Dict[str, Any]:
    """Get basic system information."""
    return {
        "hostname": get_hostname(),
        "ip_address": get_local_ip(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024 ** 3),
        "disk_gb": psutil.disk_usage('/').total / (1024 ** 3),
        "python_version": os.sys.version,
    }


def get_resource_usage() -> Dict[str, float]:
    """Get current resource usage."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
    }


# Data Formatting
def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format duration to human readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def truncate_string(s: str, max_length: int = 100) -> str:
    """Truncate string with ellipsis."""
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


# Validation Utilities
def validate_model_name(model_name: str) -> bool:
    """Validate model name format."""
    # Basic validation - can be extended
    if not model_name or len(model_name) < 3:
        return False

    # Check for valid characters
    import re
    pattern = r'^[a-zA-Z0-9/_.-]+$'
    return bool(re.match(pattern, model_name))


def validate_url(url: str) -> bool:
    """Validate URL format."""
    from urllib.parse import urlparse
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


# Configuration Utilities
def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep merge two configuration dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def load_yaml_config(path: Union[str, Path]) -> Dict:
    """Load YAML configuration file."""
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml_config(config: Dict, path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    import yaml
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)


# Retry Utilities
def retry_with_backoff(
        func,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        max_wait: float = 60.0
) -> Any:
    """Execute function with exponential backoff retry."""
    wait_time = 1.0
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(min(wait_time, max_wait))
                wait_time *= backoff_factor

    raise last_exception


# Logging Utilities
def setup_logging(
        level: str = "INFO",
        format_type: str = "json",
        log_file: Optional[str] = None
) -> None:
    """Setup logging configuration."""
    import logging
    import sys

    # Configure based on format type
    if format_type == "json":
        import json

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                }
                if record.exc_info:
                    log_obj["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_obj)

        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
    )

    for handler in handlers:
        handler.setFormatter(formatter)