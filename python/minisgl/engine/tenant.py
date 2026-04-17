from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Context, Req
from minisgl.kvcache import create_prefix_cache
from minisgl.moe import create_moe_backend
from minisgl.utils import init_logger

from .graph import GraphRunner
from .kv_pool import KVPoolManager, VirtualKVPool
from .model_registry import ModelRegistry
from .sample import Sampler

if TYPE_CHECKING:
    from minisgl.engine.config import EngineConfig
    from minisgl.models import ModelConfig
    from .runtime import ExecutionRuntime

logger = init_logger(__name__)


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


@dataclass(frozen=True)
class TenantConfig:
    """Per-tenant configuration."""

    tenant_id: str
    model_path: str
    model_config: ModelConfig
    dtype: torch.dtype
    max_running_req: int
    attention_backend: str
    moe_backend: str
    cuda_graph_bs: list[int] | None
    cuda_graph_max_bs: int | None
    page_size: int
    max_seq_len: int
    cache_type: str = "radix"
    use_dummy_weight: bool = False

    @classmethod
    def from_engine_config(cls, tenant_id: str, config: EngineConfig) -> TenantConfig:
        return cls(
            tenant_id=tenant_id,
            model_path=config.model_path,
            model_config=config.model_config,
            dtype=config.dtype,
            max_running_req=config.max_running_req,
            attention_backend=config.attention_backend,
            moe_backend=config.moe_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            page_size=config.page_size,
            max_seq_len=config.max_seq_len,
            use_dummy_weight=config.use_dummy_weight,
        )


class TenantContext:
    """All mutable state belonging to a single tenant."""

    def __init__(
        self,
        config: TenantConfig,
        runtime: ExecutionRuntime,
        pool_mgr: KVPoolManager,
        model_registry: ModelRegistry,
        num_pages: int,
    ):
        self.config = config
        self.runtime = runtime

        # Model handle
        self.model_handle = model_registry.get_or_create(
            config.tenant_id, config.model_path, config.model_config, config.use_dummy_weight
        )

        # KV pool (allocator may cap num_pages based on global budget)
        self.kv_pool: VirtualKVPool = pool_mgr.register_pool(config.tenant_id, config.model_config, config.page_size)
        self.kv_pool.allocate(num_pages)
        self.num_pages = self.kv_pool.num_pages
        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)

        # Page table
        self.page_table = torch.zeros(
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=runtime.device,
        )

        # Prefix cache
        self.prefix_cache = create_prefix_cache(
            device=runtime.device, type=config.cache_type, page_size=config.page_size
        )

        # Context object for forward passes
        self.ctx = Context(config.page_size)
        self.ctx.kv_cache = self.kv_pool.pool
        self.ctx.page_table = self.page_table

        # Attention / MoE backends call get_global_ctx() during init;
        # temporarily push our context so they can find it.
        from minisgl.core import pop_global_ctx, push_global_ctx

        push_global_ctx(self.ctx)
        try:
            self.attn_backend = create_attention_backend(config.attention_backend, config.model_config)
            self.ctx.attn_backend = self.attn_backend

            self.moe_backend = None
            if config.model_config.is_moe:
                self.moe_backend = create_moe_backend(config.moe_backend)
                self.ctx.moe_backend = self.moe_backend
        finally:
            pop_global_ctx()

        # Sampler
        self.sampler = Sampler(runtime.device, config.model_config.vocab_size)

        # Dummy req for graph padding
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)

        # Graph runner (lazily initialized on first activation)
        self.graph_runner: GraphRunner | None = None

    def ensure_active(self) -> None:
        """Load model weights and capture CUDA graphs if not already active."""
        self.model_handle.activate()
        if self.graph_runner is None:
            self.graph_runner = GraphRunner(
                stream=self.runtime.stream,
                device=self.runtime.device,
                model=self.model_handle.active_model,
                attn_backend=self.attn_backend,
                cuda_graph_bs=self.config.cuda_graph_bs,
                cuda_graph_max_bs=self.config.cuda_graph_max_bs,
                free_memory=get_free_memory(self.runtime.device),
                max_seq_len=_align_up_32(self.max_seq_len),
                vocab_size=self.config.model_config.vocab_size,
                dummy_req=self.dummy_req,
                ctx=self.ctx,
            )

    def bind(self, batch):
        """Context manager that pushes this tenant's Context onto the global stack and sets the active batch."""
        from contextlib import contextmanager
        from minisgl.core import pop_global_ctx, push_global_ctx

        @contextmanager
        def _cm():
            push_global_ctx(self.ctx)
            try:
                with self.ctx.forward_batch(batch):
                    yield self.ctx
            finally:
                pop_global_ctx()

        return _cm()

    def deactivate(self) -> None:
        """Offload model weights to free GPU memory for other tenants."""
        self.model_handle.deactivate()
        # We keep CUDA graphs around; they will be recreated on next activation
        self.graph_runner = None


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]
