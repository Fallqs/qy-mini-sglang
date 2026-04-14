from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 1920


@dataclass(frozen=True)
class ModelInstanceConfig:
    model_name: str
    model_path: str
    port: int
    host: str = "127.0.0.1"
    extra_args: Dict[str, Any] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def distributed_port(self) -> int:
        value = self.extra_args.get("distributed_port")
        if value is None:
            return self.port + 1
        if not isinstance(value, int):
            raise ValueError("'distributed_port' in extra_args must be an integer")
        return value


@dataclass(frozen=True)
class MultiModelConfig:
    gateway: GatewayConfig
    models: List[ModelInstanceConfig]


def _validate_ports(config: MultiModelConfig) -> None:
    used_ports: Dict[int, str] = {
        config.gateway.port: f"gateway:{config.gateway.host}:{config.gateway.port}"
    }

    for model in config.models:
        for port, label in [
            (model.port, f"model:{model.model_name}:api"),
            (model.distributed_port, f"model:{model.model_name}:distributed"),
        ]:
            existing = used_ports.get(port)
            if existing is not None:
                raise ValueError(f"Port conflict on {port}: {label} conflicts with {existing}")
            used_ports[port] = label


def _require_str(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string for '{key}'")
    return value


def _require_int(data: Dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer for '{key}'")
    return value


def load_config(path: str | Path) -> MultiModelConfig:
    config_path = Path(path)
    content = re.sub(r"^\s*//.*$", "", config_path.read_text(), flags=re.MULTILINE)
    payload = json.loads(content)

    gateway_raw = payload.get("gateway", {})
    if not isinstance(gateway_raw, dict):
        raise ValueError("'gateway' must be an object")
    gateway = GatewayConfig(
        host=gateway_raw.get("host", GatewayConfig.host),
        port=gateway_raw.get("port", GatewayConfig.port),
    )

    models_raw = payload.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise ValueError("'models' must be a non-empty list")

    models: List[ModelInstanceConfig] = []
    for item in models_raw:
        if not isinstance(item, dict):
            raise ValueError("Each model entry must be an object")
        extra_args = item.get("extra_args", {})
        if not isinstance(extra_args, dict):
            raise ValueError("'extra_args' must be an object when provided")
        models.append(
            ModelInstanceConfig(
                model_name=_require_str(item, "model_name"),
                model_path=_require_str(item, "model_path"),
                host=item.get("host", "127.0.0.1"),
                port=_require_int(item, "port"),
                extra_args=extra_args,
            )
        )

    config = MultiModelConfig(gateway=gateway, models=models)
    _validate_ports(config)
    return config
