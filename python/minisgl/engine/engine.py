from __future__ import annotations

import time
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.distributed import set_tp_info
from minisgl.utils import div_even, init_logger, is_sm100_supported, is_sm90_supported, torch_dtype

from .config import EngineConfig
from .graph import mem_GB
from .kv_pool import KVPoolManager
from .model_registry import ModelRegistry
from .offload import OffloadPolicy, TenantOffloadState
from .runtime import ExecutionRuntime
from .sample import BatchSamplingArgs
from .tenant import TenantConfig, TenantContext

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class MultiTenantEngine:
    """Multi-tenant inference engine supporting heterogeneous models and shared KV pools."""

    def __init__(self, base_config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=base_config.tp_info.rank, size=base_config.tp_info.size)
        _adjust_config(base_config)

        self.base_config = base_config
        self.runtime = ExecutionRuntime(base_config)

        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # Determine total KV pages across all tenants
        total_num_pages = _determine_total_num_pages(init_free_memory, base_config)

        self.pool_mgr = KVPoolManager(
            device=self.runtime.device,
            total_num_pages=total_num_pages,
            page_size=base_config.page_size,
            dtype=base_config.dtype,
            base_model_config=base_config.model_config,
        )
        self.model_registry = ModelRegistry(self.runtime)
        self.tenants: Dict[str, TenantContext] = {}
        self.default_tenant_id: str | None = None
        self.last_active_tenant_id: str | None = None
        self.offload_policy = OffloadPolicy(
            idle_seconds=base_config.offload_idle_seconds,
            max_active_models=base_config.max_active_models,
        )

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after engine init: {mem_GB(post_free_memory)}")

    @property
    def device(self) -> torch.device:
        return self.runtime.device

    @property
    def stream(self) -> torch.cuda.Stream:
        return self.runtime.stream

    @property
    def tp_cpu_group(self) -> torch.distributed.ProcessGroup:
        return self.runtime.tp_cpu_group

    def add_tenant(
        self,
        tenant_id: str,
        config: EngineConfig | None = None,
        num_pages: int | None = None,
    ) -> TenantContext:
        if tenant_id in self.tenants:
            raise ValueError(f"Tenant {tenant_id} already exists")

        cfg = config or self.base_config
        _adjust_config(cfg)
        tenant_config = TenantConfig.from_engine_config(tenant_id, cfg)

        if num_pages is None:
            num_pages = _determine_tenant_num_pages(cfg, self.pool_mgr.total_num_pages)

        tenant = TenantContext(
            config=tenant_config,
            runtime=self.runtime,
            pool_mgr=self.pool_mgr,
            model_registry=self.model_registry,
            num_pages=num_pages,
        )
        self.tenants[tenant_id] = tenant
        if self.default_tenant_id is None:
            self.default_tenant_id = tenant_id
        return tenant

    def get_tenant(self, tenant_id: str | None = None) -> TenantContext:
        if tenant_id is None:
            tenant_id = self.default_tenant_id
        if tenant_id is None or tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")
        return self.tenants[tenant_id]

    def _active_tenants(self) -> list[TenantContext]:
        return [tenant for tenant in self.tenants.values() if tenant.model_handle.is_active]

    def maybe_offload_inactive_tenants(self, keep_tenant_id: str | None = None) -> list[str]:
        if not self.base_config.enable_parameter_offloading:
            return []

        now = time.monotonic()
        states = [
            TenantOffloadState(
                tenant_id=tenant.config.tenant_id,
                is_active=tenant.model_handle.is_active,
                idle_seconds=now - tenant.last_used_at,
                activation_count=tenant.model_handle.activation_count,
                offload_count=tenant.model_handle.offload_count,
            )
            for tenant in self.tenants.values()
        ]
        offloaded: list[str] = []
        for tenant_id in self.offload_policy.select_tenants_to_offload(
            states,
            keep_tenant_id=keep_tenant_id,
        ):
            tenant = self.tenants[tenant_id]
            idle_time = now - tenant.last_used_at
            logger.info(
                "Offloading tenant %s after %.2fs idle time (%d blocks discovered)",
                tenant.config.tenant_id,
                idle_time,
                len(tenant.block_specs),
            )
            tenant.deactivate()
            offloaded.append(tenant_id)
        return offloaded

    def forward_batch(
        self, batch: Batch, args: BatchSamplingArgs, tenant_id: str | None = None
    ) -> ForwardOutput:
        tenant = self.get_tenant(tenant_id)
        tenant.ensure_active()
        self.last_active_tenant_id = tenant.config.tenant_id
        assert torch.cuda.current_stream() == self.runtime.stream
        with tenant.bind(batch):
            if tenant.graph_runner.can_use_cuda_graph(batch):
                logits = tenant.graph_runner.replay(batch)
            else:
                logits = tenant.model_handle.active_model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = tenant.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.runtime.stream)
        tenant.touch()
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        for tenant in self.tenants.values():
            if tenant.graph_runner is not None:
                tenant.graph_runner.destroy_cuda_graphs()
        self.runtime.shutdown()

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.runtime.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.runtime.device)
        free_memory = get_free_memory(self.runtime.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.runtime.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory


# Backward-compatible alias: single-tenant engine
class Engine(MultiTenantEngine):
    """Backward-compatible single-tenant engine."""

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.add_tenant("default", config)
        # Activate immediately to create CUDA graphs (backward compat)
        default = self.get_tenant("default")
        default.ensure_active()
        # Expose default tenant attributes for old code
        self.page_table = default.page_table
        self.num_pages = default.num_pages
        self.max_seq_len = default.max_seq_len
        self.attn_backend = default.attn_backend
        self.sampler = default.sampler
        self.graph_runner = default.graph_runner
        self.dummy_req = default.dummy_req
        self.kv_cache = default.kv_pool.pool

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        return super().forward_batch(batch, args, tenant_id="default")


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _determine_total_num_pages(init_free_memory: int, config: EngineConfig) -> int:
    """Determine total number of pages available for all tenants."""
    # We need an estimate of cache size per page. Use base config model.
    cache_per_page = (
        2
        * config.model_config.head_dim
        * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
        * config.page_size
        * config.dtype.itemsize
        * config.model_config.num_layers
    )
    num_pages = config.num_page_override
    if num_pages is None:
        # We don't know model memory yet since models are loaded lazily.
        # Reserve 50% of free memory for models, use the rest for KV.
        available_memory = int(0.5 * init_free_memory)
        num_pages = available_memory // cache_per_page

    assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
    num_tokens = num_pages * config.page_size
    real_kv_size = num_pages * cache_per_page
    logger.info(f"Allocating {num_tokens} total tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
    return num_pages


def _determine_tenant_num_pages(config: EngineConfig, total_num_pages: int) -> int:
    """Determine number of pages for a single tenant."""
    # For now, divide evenly. Later this can be quota-based.
    # Default to using all pages (single tenant behavior).
    return total_num_pages


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
