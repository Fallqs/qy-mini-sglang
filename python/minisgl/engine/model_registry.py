from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.utils import init_logger, torch_dtype

from .offload import BlockSpec, discover_model_blocks

if TYPE_CHECKING:
    from minisgl.models import ModelConfig
    from .runtime import ExecutionRuntime

logger = init_logger(__name__)


class ModelHandle:
    """Manages the lifecycle of one model instance, including GPU/CPU swapping."""

    def __init__(self, model_path: str, model_config: ModelConfig, runtime: ExecutionRuntime, use_dummy_weight: bool = False):
        self.model_path = model_path
        self.model_config = model_config
        self.runtime = runtime
        self.use_dummy_weight = use_dummy_weight
        self.active_model: torch.nn.Module | None = None
        self.offloaded_model: torch.nn.Module | None = None
        self.offload_count = 0
        self.activation_count = 0
        self.block_specs: list[BlockSpec] = []
        self.block_count = 0

    @property
    def is_active(self) -> bool:
        return self.active_model is not None

    def _to_pinned_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        cpu_tensor = tensor.detach().to("cpu", copy=True)
        if cpu_tensor.device.type == "cpu" and not cpu_tensor.is_pinned():
            return cpu_tensor.pin_memory()
        return cpu_tensor

    def activate(self) -> torch.nn.Module:
        if self.active_model is not None:
            return self.active_model

        logger.info(f"Activating model {self.model_path}")
        set_rope_device(self.runtime.device)
        model = self.offloaded_model
        if model is not None:
            state_dict = {
                k: v.to(self.runtime.device, non_blocking=True)
                for k, v in model.state_dict().items()
            }
            model.load_state_dict(state_dict)
            self.offloaded_model = None
        else:
            with torch.device("meta"), torch_dtype(self.runtime.dtype):
                model = create_model(self.model_config)

            if self.use_dummy_weight:
                state_dict = {
                    k: torch.randn_like(v, device=self.runtime.device)
                    for k, v in model.state_dict().items()
                }
            else:
                state_dict = {
                    k: v.to(self.runtime.dtype)
                    for k, v in load_weight(self.model_path, self.runtime.device)
                }

            model.load_state_dict(state_dict)

        self.active_model = model
        if not self.block_specs:
            self.block_specs = discover_model_blocks(model)
            self.block_count = len(self.block_specs)
            if self.block_specs:
                logger.info(
                    "Discovered %d offloadable blocks for model %s",
                    len(self.block_specs),
                    self.model_path,
                )
        self.activation_count += 1
        return model

    def deactivate(self) -> None:
        if self.active_model is None:
            return
        logger.info(f"Deactivating model for config {self.model_config.model_type}")
        state_dict_cpu = {
            k: self._to_pinned_cpu(v)
            for k, v in self.active_model.state_dict().items()
        }
        self.active_model.load_state_dict(state_dict_cpu)
        self.offloaded_model = self.active_model
        self.active_model = None
        self.offload_count += 1
        torch.cuda.empty_cache()


class ModelRegistry:
    """Registry of model handles indexed by tenant_id."""

    def __init__(self, runtime: ExecutionRuntime):
        self.runtime = runtime
        self.handles: Dict[str, ModelHandle] = {}

    def get_or_create(self, tenant_id: str, model_path: str, model_config: ModelConfig, use_dummy_weight: bool = False) -> ModelHandle:
        if tenant_id not in self.handles:
            self.handles[tenant_id] = ModelHandle(model_path, model_config, self.runtime, use_dummy_weight)
        return self.handles[tenant_id]
