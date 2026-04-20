from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .async_transfer import AsyncTransferQueue
from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)
from .hierarchical import HierarchicalCacheHandle, HierarchicalPrefixCache
from .mha_pool import MHAKVCache
from .mooncake_backend import (
    BaseSpillBackend,
    MoonCakeKVBackend,
    NoopSpillBackend,
    build_model_fingerprint,
    mooncake_available,
)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device, page_size: int = 1) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache.from_model_config(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device, page_size: int = 1):
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device, page_size: int = 1):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device, page_size=page_size)


@SUPPORTED_CACHE_MANAGER.register("hierarchical")
def create_hierarchical_cache(
    device: torch.device,
    page_size: int = 1,
    spill_backend: BaseSpillBackend | None = None,
):
    return HierarchicalPrefixCache(
        device=device, page_size=page_size, spill_backend=spill_backend
    )


def create_prefix_cache(device: torch.device, type: str, page_size: int = 1) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device, page_size)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
    "MHAKVCache",
    "HierarchicalPrefixCache",
    "HierarchicalCacheHandle",
    "BaseSpillBackend",
    "MoonCakeKVBackend",
    "NoopSpillBackend",
    "AsyncTransferQueue",
    "mooncake_available",
    "build_model_fingerprint",
]
