from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from minisgl.layers import OPList
from minisgl.utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class BlockSpec:
    """A model block that can become a future offloading unit."""

    name: str
    index: int
    kind: str


@dataclass(frozen=True)
class TenantOffloadState:
    tenant_id: str
    is_active: bool
    idle_seconds: float
    activation_count: int
    offload_count: int


class OffloadPolicy:
    """Decides which inactive tenant models should be offloaded."""

    def __init__(self, idle_seconds: float, max_active_models: int | None):
        self.idle_seconds = idle_seconds
        self.max_active_models = max_active_models

    def select_tenants_to_offload(
        self,
        states: Sequence[TenantOffloadState],
        *,
        keep_tenant_id: str | None = None,
    ) -> list[str]:
        active_states = [state for state in states if state.is_active]
        if not active_states:
            return []

        forced_budget = self.max_active_models
        need_forced_eviction = forced_budget is not None and len(active_states) > forced_budget
        candidates = [state for state in active_states if state.tenant_id != keep_tenant_id]
        if not candidates:
            return []

        selected: list[str] = []
        for state in candidates:
            if state.idle_seconds >= self.idle_seconds:
                selected.append(state.tenant_id)

        if need_forced_eviction:
            selected_set = set(selected)
            remaining = [state for state in candidates if state.tenant_id not in selected_set]
            remaining.sort(
                key=lambda state: (
                    state.idle_seconds,
                    -state.offload_count,
                    state.activation_count,
                    state.tenant_id,
                ),
                reverse=True,
            )
            active_count_after_idle = len(active_states) - len(selected)
            for state in remaining:
                if active_count_after_idle <= forced_budget:
                    break
                selected.append(state.tenant_id)
                active_count_after_idle -= 1

        return selected


class LayerOffloadManager:
    """Keeps only a small window of transformer blocks resident on GPU."""

    def __init__(
        self,
        model: object,
        block_specs: Sequence[BlockSpec],
        *,
        device: torch.device,
        max_resident_blocks: int,
    ):
        self.model = model
        self.block_specs = list(block_specs)
        self.device = device
        self.max_resident_blocks = max(1, max_resident_blocks)
        self.blocks = [
            _resolve_attr_path(model, spec.name)
            for spec in self.block_specs
        ]
        self.cpu_cache = [self._capture_cpu_state(block) for block in self.blocks]
        self._resident = [False] * len(self.blocks)
        self._resident_order: deque[int] = deque()
        self._last_prepared: int | None = None

        for idx in range(min(self.max_resident_blocks, len(self.blocks))):
            self._resident[idx] = True
            self._resident_order.append(idx)
        for idx in range(self.max_resident_blocks, len(self.blocks)):
            self._evict_block(idx, compact=False)

    def prepare(self, block_idx: int) -> None:
        self._last_prepared = block_idx
        self._ensure_loaded(block_idx)
        if self.max_resident_blocks >= 2 and block_idx + 1 < len(self.blocks):
            self._ensure_loaded(block_idx + 1, is_prefetch=True)
        self._compact_resident(block_idx)

    def _ensure_loaded(self, block_idx: int, *, is_prefetch: bool = False) -> None:
        if self._resident[block_idx]:
            self._touch(block_idx)
            return
        block = self.blocks[block_idx]
        assert block is not None
        state_dict = {
            name: tensor.to(self.device, non_blocking=True)
            for name, tensor in self.cpu_cache[block_idx].items()
        }
        block.load_state_dict(state_dict)
        self._resident[block_idx] = True
        self._touch(block_idx)

    def _evict_block(self, block_idx: int, *, compact: bool = True) -> None:
        if not self._resident[block_idx]:
            return
        block = self.blocks[block_idx]
        assert block is not None
        state_dict = {
            name: tensor
            for name, tensor in self.cpu_cache[block_idx].items()
        }
        block.load_state_dict(state_dict)
        self._resident[block_idx] = False
        try:
            self._resident_order.remove(block_idx)
        except ValueError:
            pass
        if compact and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compact_resident(self, current_idx: int) -> None:
        protected = {current_idx}
        if current_idx + 1 < len(self.blocks):
            protected.add(current_idx + 1)
        while sum(self._resident) > self.max_resident_blocks:
            victim = next(
                (idx for idx in self._resident_order if idx not in protected),
                None,
            )
            if victim is None:
                break
            self._evict_block(victim)

    def _touch(self, block_idx: int) -> None:
        try:
            self._resident_order.remove(block_idx)
        except ValueError:
            pass
        self._resident_order.append(block_idx)

    def resident_blocks(self) -> list[int]:
        return [idx for idx, is_resident in enumerate(self._resident) if is_resident]

    def _capture_cpu_state(self, block: object | None) -> dict[str, torch.Tensor]:
        if block is None:
            return {}
        state_dict_fn = getattr(block, "state_dict", None)
        if state_dict_fn is None:
            return {}
        state_dict = state_dict_fn()
        result: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            cpu_tensor = tensor.detach().to("cpu", copy=True)
            try:
                if not cpu_tensor.is_pinned():
                    cpu_tensor = cpu_tensor.pin_memory()
            except RuntimeError:
                # CPU-only environments cannot always allocate pinned memory.
                pass
            result[name] = cpu_tensor
        return result


def discover_model_blocks(model: object) -> list[BlockSpec]:
    """Best-effort discovery of transformer blocks without model-specific code."""

    for path, kind in _candidate_block_paths():
        block_list = _resolve_attr_path(model, path)
        if isinstance(block_list, OPList):
            return [
                BlockSpec(name=f"{path}.{idx}", index=idx, kind=kind)
                for idx, _ in enumerate(block_list.op_list)
            ]
        if isinstance(block_list, list):
            return [
                BlockSpec(name=f"{path}.{idx}", index=idx, kind=kind)
                for idx, _ in enumerate(block_list)
            ]
    return []


def _candidate_block_paths() -> Iterable[tuple[str, str]]:
    yield "model.layers", "decoder_layer"
    yield "layers", "decoder_layer"
    yield "decoder.layers", "decoder_layer"


def _resolve_attr_path(root: object, path: str) -> object | None:
    cur = root
    for part in path.split("."):
        if part.isdigit():
            index = int(part)
            if isinstance(cur, OPList):
                if 0 <= index < len(cur.op_list):
                    cur = cur.op_list[index]
                    continue
                return None
            if isinstance(cur, list):
                if 0 <= index < len(cur):
                    cur = cur[index]
                    continue
                return None
            return None
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur
