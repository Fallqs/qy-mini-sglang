from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from minisgl.distributed import get_tp_info
from minisgl.kvcache import BaseKVCachePool, MHAKVCache
from minisgl.utils import div_even, init_logger

from .segment_list import SegmentList

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

logger = init_logger(__name__)


@dataclass
class _TenantPageInfo:
    page_size: int
    num_pages: int
    m_i: int          # fine units per token (local_kv_heads * head_dim)
    num_layers: int
    seg_list: SegmentList
    page_ref_counts: List[int]


class GlobalFineAllocator:
    """Allocates fine-grained units globally and synchronizes per-tenant page tables."""

    def __init__(self, total_fine_units: int, device: torch.device):
        self.total_fine_units = total_fine_units
        self._tenants: Dict[str, _TenantPageInfo] = {}
        self.device = device

    def register_tenant(self, tenant_id: str, model_config: ModelConfig, page_size: int, num_pages: int) -> int:
        if tenant_id in self._tenants:
            return self._tenants[tenant_id].num_pages
        tp_info = get_tp_info()
        local_kv_heads = div_even(model_config.num_kv_heads, tp_info.size, allow_replicate=True)
        m_i = local_kv_heads * model_config.head_dim
        page_fu = page_size * m_i
        max_pages = self.total_fine_units // page_fu
        capped_num_pages = min(num_pages, max_pages)
        if capped_num_pages < num_pages:
            logger.warning(
                f"Tenant {tenant_id}: num_pages capped from {num_pages} to {capped_num_pages} "
                f"due to global memory budget"
            )

        seg_list = SegmentList(capped_num_pages)
        ref_counts = [0] * capped_num_pages

        # Initialize dirty state from existing tenants' allocations
        for tinfo in self._tenants.values():
            t_page_fu = tinfo.page_size * tinfo.m_i
            for b, length in tinfo.seg_list.to_dirty_list():
                fu_start = b * t_page_fu
                fu_len = length * t_page_fu
                ps = fu_start // page_fu
                pe = (fu_start + fu_len - 1) // page_fu + 1
                ps = max(ps, 0)
                pe = min(pe, capped_num_pages)
                for p in range(ps, pe):
                    ref_counts[p] += 1
                    seg_list.mark_dirty(p, 1)

        self._tenants[tenant_id] = _TenantPageInfo(
            page_size=page_size,
            num_pages=capped_num_pages,
            m_i=m_i,
            num_layers=model_config.num_layers,
            seg_list=seg_list,
            page_ref_counts=ref_counts,
        )
        return capped_num_pages

    def available_pages(self, tenant_id: str) -> int:
        return self._tenants[tenant_id].seg_list.total_clean

    def allocate(self, tenant_id: str, needed_pages: int) -> torch.Tensor:
        info = self._tenants[tenant_id]
        if info.seg_list.total_clean < needed_pages:
            raise RuntimeError(
                f"Out of pages for tenant {tenant_id}: need {needed_pages}, "
                f"have {info.seg_list.total_clean}."
            )

        # Allocate whole pages from the tenant's own segment list (always aligned)
        segs = info.seg_list.allocate(needed_pages)
        page_fu = info.page_size * info.m_i

        # Sync to all other tenants
        for b, length in segs:
            fu_start = b * page_fu
            fu_len = length * page_fu

            for other_id, other in self._tenants.items():
                if other_id == tenant_id:
                    continue
                other_page_fu = other.page_size * other.m_i
                ps = fu_start // other_page_fu
                pe = (fu_start + fu_len - 1) // other_page_fu + 1
                ps = max(ps, 0)
                pe = min(pe, other.num_pages)
                for p in range(ps, pe):
                    other.page_ref_counts[p] += 1
                    other.seg_list.mark_dirty(p, 1)

        # Build token indices for requesting tenant
        tokens: List[int] = []
        for b, length in segs:
            base_token = b * info.page_size
            tokens.extend(range(base_token, base_token + length * info.page_size))

        return torch.tensor(tokens, dtype=torch.int32, device=self.device)

    def free(self, tenant_id: str, indices: torch.Tensor) -> None:
        info = self._tenants[tenant_id]
        if len(indices) == 0:
            return

        arr = indices.cpu().tolist()
        arr.sort()

        ranges_fu: List[Tuple[int, int]] = []
        start = arr[0]
        prev = arr[0]
        for idx in arr[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                ranges_fu.append((start * info.m_i, (prev - start + 1) * info.m_i))
                start = idx
                prev = idx
        ranges_fu.append((start * info.m_i, (prev - start + 1) * info.m_i))

        page_fu = info.page_size * info.m_i
        for b, length in ranges_fu:
            # Mark requestor's own pages clean directly
            ps = b // page_fu
            pe = (b + length - 1) // page_fu + 1
            ps = max(ps, 0)
            pe = min(pe, info.num_pages)
            for p in range(ps, pe):
                info.seg_list.mark_clean(p, 1)

            # Sync other tenants
            for tid, tinfo in self._tenants.items():
                if tid == tenant_id:
                    continue
                t_page_fu = tinfo.page_size * tinfo.m_i
                ps = b // t_page_fu
                pe = (b + length - 1) // t_page_fu + 1
                ps = max(ps, 0)
                pe = min(pe, tinfo.num_pages)
                for p in range(ps, pe):
                    tinfo.page_ref_counts[p] -= 1
                    if tinfo.page_ref_counts[p] == 0:
                        tinfo.seg_list.mark_clean(p, 1)


class VirtualKVPool:
    """Lazy-allocated KV pool for a specific model using shared global memory."""

    def __init__(
        self,
        tenant_id: str,
        model_config: ModelConfig,
        page_size: int,
        dtype: torch.dtype,
        allocator: GlobalFineAllocator,
        pool_mgr: KVPoolManager,
    ):
        self.tenant_id = tenant_id
        self.model_config = model_config
        self.page_size = page_size
        self.dtype = dtype
        self.allocator = allocator
        self.pool_mgr = pool_mgr
        self._pool: BaseKVCachePool | None = None
        self._num_pages = 0

    @property
    def is_allocated(self) -> bool:
        return self._pool is not None

    @property
    def pool(self) -> BaseKVCachePool:
        assert self._pool is not None, "VirtualKVPool not allocated yet"
        return self._pool

    @property
    def num_pages(self) -> int:
        return self._num_pages

    def allocate(self, num_pages: int) -> None:
        if self._pool is not None:
            return
        capped_num_pages = self.allocator.register_tenant(
            self.tenant_id, self.model_config, self.page_size, num_pages
        )
        self._num_pages = capped_num_pages

        tp_info = get_tp_info()
        local_kv_heads = div_even(self.model_config.num_kv_heads, tp_info.size, allow_replicate=True)
        m_i = local_kv_heads * self.model_config.head_dim
        num_layers = self.model_config.num_layers

        self.pool_mgr._ensure_layers(num_layers)
        total_fine_units = self.pool_mgr.fine_units_per_layer
        # Exact adaptive view: only cover the tokens we were actually granted.
        # capped_num_pages * page_size * m_i <= max_pages * page_size * m_i
        #                        <= total_fine_units, so this never exceeds the tensor.
        num_tokens_global = self._num_pages * self.page_size

        k_buffers: List[torch.Tensor] = []
        v_buffers: List[torch.Tensor] = []
        for layer_id in range(num_layers):
            layer_tensor = self.pool_mgr.global_kv[layer_id]
            # layer_tensor shape: (2, total_fine_units, 1)
            # Attention backends expect 4-D cache: (tokens, page_size, heads, dim)
            k_view = layer_tensor[0, : num_tokens_global * m_i, 0].view(
                num_tokens_global, 1, local_kv_heads, self.model_config.head_dim
            )
            v_view = layer_tensor[1, : num_tokens_global * m_i, 0].view(
                num_tokens_global, 1, local_kv_heads, self.model_config.head_dim
            )
            k_buffers.append(k_view)
            v_buffers.append(v_view)

        self._pool = MHAKVCache(k_buffers=k_buffers, v_buffers=v_buffers)
        logger.info(
            f"Tenant {self.tenant_id}: registered shared KV pool with "
            f"{num_pages} pages ({num_pages * self.page_size} tokens), m_i={m_i}"
        )

    def destroy(self) -> None:
        if self._pool is None:
            return
        del self._pool
        self._pool = None
        self._num_pages = 0


class KVPoolManager:
    """Manages total KV memory budget across tenants with a shared global pool."""

    def __init__(
        self,
        device: torch.device,
        total_num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        base_model_config: ModelConfig | None = None,
    ):
        self.device = device
        self.page_size = page_size
        self.dtype = dtype

        if base_model_config is not None:
            tp_info = get_tp_info()
            local_kv_heads = div_even(
                base_model_config.num_kv_heads, tp_info.size, allow_replicate=True
            )
            E_base = local_kv_heads * base_model_config.head_dim
            self.fine_units_per_layer = total_num_pages * page_size * E_base
            self.max_layers = base_model_config.num_layers
        else:
            self.fine_units_per_layer = total_num_pages * page_size
            self.max_layers = 1

        self.total_num_pages = total_num_pages
        self.allocator = GlobalFineAllocator(self.fine_units_per_layer, device)
        self.global_kv: List[torch.Tensor] = []
        self._ensure_layers(self.max_layers)
        self.pools: Dict[str, VirtualKVPool] = {}

    def _ensure_layers(self, num_layers: int) -> None:
        while len(self.global_kv) < num_layers:
            self.global_kv.append(
                torch.empty((2, self.fine_units_per_layer, 1), dtype=self.dtype, device=self.device)
            )

    def register_pool(
        self,
        tenant_id: str,
        model_config: ModelConfig,
        page_size: int | None = None,
    ) -> VirtualKVPool:
        pool = VirtualKVPool(
            tenant_id=tenant_id,
            model_config=model_config,
            page_size=page_size if page_size is not None else self.page_size,
            dtype=self.dtype,
            allocator=self.allocator,
            pool_mgr=self,
        )
        self.pools[tenant_id] = pool
        return pool

    def get_pool(self, tenant_id: str) -> VirtualKVPool:
        return self.pools[tenant_id]
