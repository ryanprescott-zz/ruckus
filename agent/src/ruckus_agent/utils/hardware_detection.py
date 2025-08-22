"""Hardware detection utilities."""

import logging
import platform
import subprocess
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def detect_hardware() -> Dict[str, Any]:
    """Comprehensive hardware detection."""
    logger.info("Starting comprehensive hardware detection")
    try:
        result = {
            "cpu": detect_cpu(),
            "gpus": detect_gpus(),
            "memory": detect_memory(),
            "storage": detect_storage(),
            "network": detect_network(),
        }
        logger.info("Hardware detection completed successfully")
        return result
    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        raise


def detect_cpu() -> Dict[str, Any]:
    """Detect CPU information."""
    logger.debug("Detecting CPU information")
    try:
        import psutil

        result = {
            "model": platform.processor(),
            "architecture": platform.machine(),
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
        }
        logger.debug(f"CPU detected: {result['model']}, {result['cores_physical']} physical cores")
        return result
    except Exception as e:
        logger.error(f"CPU detection failed: {e}")
        raise


def detect_gpus() -> List[Dict[str, Any]]:
    """Detect all GPUs."""
    logger.debug("Detecting GPUs")
    try:
        gpus = []

        # NVIDIA GPUs
        nvidia_gpus = detect_nvidia_gpus()
        gpus.extend(nvidia_gpus)
        logger.debug(f"Found {len(nvidia_gpus)} NVIDIA GPUs")

        # AMD GPUs
        amd_gpus = detect_amd_gpus()
        gpus.extend(amd_gpus)
        logger.debug(f"Found {len(amd_gpus)} AMD GPUs")

        logger.info(f"Total GPUs detected: {len(gpus)}")
        return gpus
    except Exception as e:
        logger.error(f"GPU detection failed: {e}")
        raise


def detect_nvidia_gpus() -> List[Dict[str, Any]]:
    """Detect NVIDIA GPUs."""
    logger.debug("Detecting NVIDIA GPUs")
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
                    gpu_info = {
                        "vendor": "nvidia",
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_mb": int(parts[2]),
                    }
                    gpus.append(gpu_info)
                    logger.debug(f"NVIDIA GPU found: {gpu_info['name']} ({gpu_info['memory_mb']}MB)")
        else:
            logger.debug("nvidia-smi command failed or returned non-zero exit code")
    except FileNotFoundError:
        logger.debug("nvidia-smi not found - no NVIDIA GPUs detected")
    except Exception as e:
        logger.warning(f"NVIDIA GPU detection failed: {e}")

    return gpus


def detect_amd_gpus() -> List[Dict[str, Any]]:
    """Detect AMD GPUs."""
    logger.debug("Detecting AMD GPUs")
    gpus = []
    # TODO: Implement AMD GPU detection using rocm-smi
    logger.debug("AMD GPU detection not yet implemented")
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
