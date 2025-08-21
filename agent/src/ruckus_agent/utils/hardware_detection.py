"""Hardware detection utilities."""

import platform
import subprocess
from typing import Dict, Any, List, Optional


def detect_hardware() -> Dict[str, Any]:
    """Comprehensive hardware detection."""
    return {
        "cpu": detect_cpu(),
        "gpus": detect_gpus(),
        "memory": detect_memory(),
        "storage": detect_storage(),
        "network": detect_network(),
    }


def detect_cpu() -> Dict[str, Any]:
    """Detect CPU information."""
    import psutil

    return {
        "model": platform.processor(),
        "architecture": platform.machine(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
    }


def detect_gpus() -> List[Dict[str, Any]]:
    """Detect all GPUs."""
    gpus = []

    # NVIDIA GPUs
    gpus.extend(detect_nvidia_gpus())

    # AMD GPUs
    gpus.extend(detect_amd_gpus())

    return gpus


def detect_nvidia_gpus() -> List[Dict[str, Any]]:
    """Detect NVIDIA GPUs."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpus.append({
                        "vendor": "nvidia",
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_mb": int(parts[2]),
                    })
    except Exception:
        pass

    return gpus


def detect_amd_gpus() -> List[Dict[str, Any]]:
    """Detect AMD GPUs."""
    gpus = []
    # TODO: Implement AMD GPU detection using rocm-smi
    return gpus


def detect_memory() -> Dict[str, Any]:
    """Detect memory information."""
    import psutil

    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024 ** 3),
        "available_gb": mem.available / (1024 ** 3),
        "used_gb": mem.used / (1024 ** 3),
        "percent": mem.percent,
    }


def detect_storage() -> Dict[str, Any]:
    """Detect storage information."""
    import psutil

    disk = psutil.disk_usage('/')
    return {
        "total_gb": disk.total / (1024 ** 3),
        "used_gb": disk.used / (1024 ** 3),
        "free_gb": disk.free / (1024 ** 3),
        "percent": disk.percent,
    }


def detect_network() -> Dict[str, Any]:
    """Detect network capabilities."""
    import socket

    hostname = socket.gethostname()
    try:
        ip_address = socket.gethostbyname(hostname)
    except Exception:
        ip_address = "127.0.0.1"

    return {
        "hostname": hostname,
        "ip_address": ip_address,
    }