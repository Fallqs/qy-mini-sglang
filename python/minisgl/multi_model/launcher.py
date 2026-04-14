from __future__ import annotations

import atexit
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List

from minisgl.utils import init_logger

from .config import MultiModelConfig, ModelInstanceConfig, load_config
from .gateway import InstanceStatus, run_gateway

logger = init_logger(__name__, "MultiModelLauncher")


def _serialize_arg(name: str, value) -> List[str]:
    flag = f"--{name.replace('_', '-')}"
    if isinstance(value, bool):
        return [flag] if value else []
    if isinstance(value, list):
        return [flag, ",".join(str(item) for item in value)]
    return [flag, str(value)]


def _build_command(instance: ModelInstanceConfig) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "minisgl",
        "--model",
        instance.model_path,
        "--host",
        instance.host,
        "--port",
        str(instance.port),
    ]
    for key, value in instance.extra_args.items():
        cmd.extend(_serialize_arg(key, value))
    return cmd


@dataclass
class ManagedInstance:
    config: ModelInstanceConfig
    process: subprocess.Popen[str]
    status: InstanceStatus


def _start_instance(instance: ModelInstanceConfig) -> ManagedInstance:
    cmd = _build_command(instance)
    logger.info("Starting model '%s' on %s:%s", instance.model_name, instance.host, instance.port)
    process = subprocess.Popen(cmd)
    status = InstanceStatus(
        model_name=instance.model_name,
        base_url=instance.base_url,
        pid=process.pid,
    )
    return ManagedInstance(config=instance, process=process, status=status)


def launch_from_config(config_path: str) -> None:
    config = load_config(config_path)
    managed = [_start_instance(instance) for instance in config.models]

    def _shutdown(*_) -> None:
        for item in managed:
            if item.process.poll() is None:
                item.process.terminate()
        for item in managed:
            try:
                item.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                item.process.kill()

    atexit.register(_shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    statuses: Dict[str, List[InstanceStatus]] = {}
    for item in managed:
        statuses.setdefault(item.config.model_name, []).append(item.status)

    try:
        run_gateway(config, statuses)
    finally:
        _shutdown()
