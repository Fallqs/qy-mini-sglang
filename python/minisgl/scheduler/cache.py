from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import (
    BaseCacheHandle,
    BaseKVCachePool,
    BaseSpillBackend,
    HierarchicalCacheHandle,
    MatchResult,
    create_prefix_cache,
)
from minisgl.utils import div_ceil, init_logger

if TYPE_CHECKING:
    from minisgl.engine.kv_pool import GlobalFineAllocator
    from .utils import PendingReq

logger = init_logger(__name__)


class CacheManager:
    def __init__(
        self,
        tenant_id: str,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        type: str,
        allocator: GlobalFineAllocator,
    ):
        self.tenant_id = tenant_id
        self.allocator = allocator
        self.prefix_cache = create_prefix_cache(device=page_table.device, type=type, page_size=page_size)
        self.device = page_table.device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + self.allocator.available_pages(self.tenant_id) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        self.unlock(old_handle)
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:
            self._free(page_indices[new_handle.cached_len :])
        else:
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self, strict: bool = True) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        free_pages = self.allocator.available_pages(self.tenant_id)
        if strict:
            if free_pages + cache_pages != self.num_pages:
                raise RuntimeError(
                    "CacheManager integrity check failed:"
                    f" free_pages({free_pages}) +"
                    f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
                )
        elif free_pages + cache_pages > self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({free_pages}) +"
                f" cache_pages({cache_pages}) > num_pages({self.num_pages})"
            )

    @contextmanager
    def lazy_free_region(self):
        lazy_indices_list: List[torch.Tensor] = []

        def lazy_free(indices: torch.Tensor) -> None:
            if len(indices) > 0:
                lazy_indices_list.append(indices)

        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            if lazy_indices_list:
                all_indices = torch.cat(lazy_indices_list)
                self.allocator.free(self.tenant_id, all_indices)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        free_pages = self.allocator.available_pages(self.tenant_id)
        if needed_pages > free_pages:
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.allocator.free(self.tenant_id, evicted)
            free_pages = self.allocator.available_pages(self.tenant_id)
            assert free_pages >= needed_pages, "Eviction did not free enough space."
        tokens = self.allocator.allocate(self.tenant_id, needed_pages)
        return tokens[::self.page_size]

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.allocator.free(self.tenant_id, indices)

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


class HierarchicalCacheManager(CacheManager):
    """
    Extends CacheManager with tiered spill-to-MoonCake support.

    * Read path:  ``match_req`` checks GPU radix cache first, then queries the
      spill backend.  Spilled prefixes are loaded back into GPU pages
      synchronously so the scheduler sees a unified cache handle.
    * Write path: ``maybe_offload`` stores newly-computed KV chunks to the
      spill backend asynchronously.  This is typically called by the scheduler
      after a prefill batch finishes.
    """

    def __init__(
        self,
        tenant_id: str,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        allocator: GlobalFineAllocator,
        kv_pool: BaseKVCachePool,
        spill_backend: BaseSpillBackend | None = None,
        async_queue=None,
    ):
        # Bypass CacheManager.__init__ so we can pass spill_backend to the
        # prefix cache directly (the registry factory does not accept it).
        self.tenant_id = tenant_id
        self.allocator = allocator
        from minisgl.kvcache import HierarchicalPrefixCache
        self.prefix_cache = HierarchicalPrefixCache(
            device=page_table.device,
            page_size=page_size,
            spill_backend=spill_backend,
        )
        self.device = page_table.device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self.kv_pool = kv_pool
        self.spill = spill_backend
        self._async_queue = async_queue

    def match_req(self, req: PendingReq) -> MatchResult:
        result = super().match_req(req)
        gpu_handle = result.cuda_handle
        spilled_handle = result.spilled_handle

        if spilled_handle is None or spilled_handle.cached_len == 0:
            return result

        # Load spilled prefix back into GPU pages.
        loaded_len = self._load_spilled_prefix(
            req.input_ids[gpu_handle.cached_len : gpu_handle.cached_len + spilled_handle.cached_len],
            spilled_handle.cached_len,
        )

        if loaded_len == 0:
            return MatchResult(cuda_handle=gpu_handle)

        total_len = gpu_handle.cached_len + loaded_len
        gpu_indices = (
            gpu_handle.get_matched_indices()
            if gpu_handle.cached_len > 0
            else torch.empty(0, dtype=torch.int32, device=self.device)
        )
        combined_indices = torch.cat([
            gpu_indices,
            self._fresh_indices[:loaded_len],
        ])
        combined_handle = HierarchicalCacheHandle(
            cached_len=total_len,
            indices=combined_indices,
        )
        return MatchResult(cuda_handle=combined_handle)

    def try_expand_from_spilled(
        self, req: PendingReq, gpu_handle: BaseCacheHandle, table_idx: int
    ) -> Tuple[BaseCacheHandle, int] | None:
        """
        Attempt to extend a GPU-resident prefix by loading additional tokens
        from the spill backend.  Returns a new (handle, cached_len) pair or
        ``None`` if no expansion happened.

        This is called by ``PrefillAdder`` after ``table_idx`` has been
        allocated so that the newly loaded pages can be written into the page
        table immediately.
        """
        if self.spill is None:
            return None

        remaining = req.input_ids[gpu_handle.cached_len : req.input_len - 1]
        if len(remaining) == 0:
            return None

        spill_len = self.spill.match_prefix(remaining)
        if spill_len == 0:
            return None

        loaded_len = self._load_spilled_prefix(
            req.input_ids[gpu_handle.cached_len : gpu_handle.cached_len + spill_len],
            spill_len,
        )
        if loaded_len == 0:
            return None

        total_len = gpu_handle.cached_len + loaded_len
        gpu_indices = (
            gpu_handle.get_matched_indices()
            if gpu_handle.cached_len > 0
            else torch.empty(0, dtype=torch.int32, device=self.device)
        )
        combined_indices = torch.cat([
            gpu_indices,
            self._fresh_indices[:loaded_len],
        ])
        # Update page table for the newly loaded portion.
        self.page_table[table_idx, gpu_handle.cached_len : total_len].copy_(
            combined_indices[gpu_handle.cached_len : total_len]
        )
        return HierarchicalCacheHandle(
            cached_len=total_len, indices=combined_indices
        ), total_len

    def maybe_offload(self, req: Req) -> None:
        """
        Asynchronously store newly-computed KV chunks to the spill backend.
        Call this after a prefill forward completes.
        """
        if self.spill is None or self._async_queue is None:
            return

        # Only offload the portion that was actually computed during this
        # request's prefill phase.  We use the request's cache_handle to
        # determine what was already cached before vs what is new.
        old_cached_len = req.cache_handle.cached_len if req.cache_handle else 0
        new_ids = req.input_ids[old_cached_len : req.cached_len]
        new_indices = self.page_table[req.table_idx, old_cached_len : req.cached_len]

        if len(new_ids) == 0:
            return

        self._async_queue.enqueue(
            token_ids=new_ids,
            gpu_indices=new_indices,
            backend=self.spill,
            kv_pool=self.kv_pool,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_spilled_prefix(self, token_ids: torch.Tensor, spill_len: int) -> int:
        """Allocate GPU pages and synchronously load *spill_len* tokens."""
        assert self.spill is not None
        needed_pages = div_ceil(spill_len, self.page_size)
        free_pages = self.allocator.available_pages(self.tenant_id)
        if needed_pages > free_pages:
            # We need more GPU memory; evict from radix cache first.
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.allocator.free(self.tenant_id, evicted)

        allocated = self.allocator.allocate(self.tenant_id, needed_pages)
        gpu_indices = self._page_to_token(allocated)[:spill_len]

        try:
            self.spill.load_prefix(token_ids[:spill_len], gpu_indices, self.kv_pool)
        except Exception:
            logger.error("Failed to load spilled prefix; falling back to recompute.")
            self.allocator.free(self.tenant_id, allocated)
            return 0

        self._fresh_indices = gpu_indices
        return spill_len


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
