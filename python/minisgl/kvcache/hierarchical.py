from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from minisgl.utils import div_ceil, init_logger

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo
from .radix_cache import RadixPrefixCache

if TYPE_CHECKING:
    from .mooncake_backend import BaseSpillBackend

logger = init_logger(__name__)


@dataclass(frozen=True)
class HierarchicalCacheHandle(BaseCacheHandle):
    """Unified handle that may contain both GPU-resident and freshly-loaded indices."""

    cached_len: int
    indices: torch.Tensor

    def get_matched_indices(self) -> torch.Tensor:
        return self.indices


class HierarchicalPrefixCache(BasePrefixCache):
    """
    Two-tier prefix cache: GPU-resident radix cache (L1) + spill backend (L2/L3).

    The spill backend (e.g. MoonCake Store) holds evicted or explicitly-offloaded
    KV pages.  On ``match_prefix()`` we first walk the hot radix tree; any
    remaining suffix is checked against the spill backend.  On ``evict()`` the
    caller is responsible for copying data to the spill backend before the
    returned indices are freed.
    """

    def __init__(
        self,
        device: torch.device,
        page_size: int = 1,
        spill_backend: BaseSpillBackend | None = None,
    ):
        super().__init__()
        self.device = device
        self.page_size = page_size
        self.spill = spill_backend
        self.gpu_cache = RadixPrefixCache(device=device, page_size=page_size)

    # ------------------------------------------------------------------ #
    # Delegated GPU-cache operations
    # ------------------------------------------------------------------ #

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        # HierarchicalCacheHandle contains freshly-loaded data that is not
        # (yet) tracked by the GPU radix tree, so locking is a no-op.
        if isinstance(handle, HierarchicalCacheHandle):
            return
        self.gpu_cache.lock_handle(handle, unlock=unlock)

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return self.gpu_cache.insert_prefix(input_ids, indices)

    def evict(self, size: int) -> torch.Tensor:
        return self.gpu_cache.evict(size)

    def reset(self) -> None:
        self.gpu_cache.reset()
        if self.spill is not None:
            self.spill.reset()

    @property
    def size_info(self) -> SizeInfo:
        return self.gpu_cache.size_info

    def check_integrity(self) -> None:
        self.gpu_cache.check_integrity()

    # ------------------------------------------------------------------ #
    # Tiered match
    # ------------------------------------------------------------------ #

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        gpu_result = self.gpu_cache.match_prefix(input_ids)
        gpu_handle = gpu_result.cuda_handle
        gpu_len = gpu_handle.cached_len

        if self.spill is None or gpu_len >= len(input_ids):
            return MatchResult(cuda_handle=gpu_handle)

        remaining = input_ids[gpu_len:]
        spill_len = self.spill.match_prefix(remaining)

        if spill_len == 0:
            return MatchResult(cuda_handle=gpu_handle)

        # Build a spilled handle that carries token metadata so the
        # HierarchicalCacheManager can load it back into GPU pages.
        spilled = _SpilledCacheHandle(
            cached_len=spill_len,
            token_ids=input_ids[gpu_len : gpu_len + spill_len],
        )
        return MatchResult(cuda_handle=gpu_handle, spilled_handle=spilled)


@dataclass(frozen=True)
class _SpilledCacheHandle(BaseCacheHandle):
    """Opaque handle for prefix data that lives in the spill backend."""

    cached_len: int
    token_ids: torch.Tensor

    def get_matched_indices(self) -> torch.Tensor:
        raise RuntimeError(
            "Spilled cache handle has no GPU indices; "
            "it must be loaded by HierarchicalCacheManager first."
        )
