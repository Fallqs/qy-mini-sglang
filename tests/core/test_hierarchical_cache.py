"""Tests for hierarchical KV cache with MoonCake-style spill backend."""

from __future__ import annotations

import pytest
import torch

import minisgl.core as core
from minisgl.distributed import set_tp_info
from minisgl.engine.kv_pool import GlobalFineAllocator
from minisgl.kvcache import HierarchicalCacheHandle, HierarchicalPrefixCache
from minisgl.kvcache.async_transfer import AsyncTransferQueue
from minisgl.kvcache.mooncake_backend import BaseSpillBackend, build_model_fingerprint
from minisgl.scheduler.cache import CacheManager, HierarchicalCacheManager
from minisgl.scheduler.utils import PendingReq


@pytest.fixture(scope="session", autouse=True)
def setup_tp_info():
    from minisgl.distributed.info import get_tp_info
    try:
        get_tp_info()
    except RuntimeError:
        set_tp_info(rank=0, size=1)


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctxs = getattr(core._GLOBAL_CTX_STACK, "ctxs", [])
    core._GLOBAL_CTX_STACK.ctxs = []
    yield
    core._GLOBAL_CTX_STACK.ctxs = old_ctxs


class _MockModelConfig:
    def __init__(
        self,
        num_kv_heads: int = 2,
        head_dim: int = 4,
        num_layers: int = 2,
        model_type: str = "llama",
    ):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.model_type = model_type


class _MockSpillBackend(BaseSpillBackend):
    """In-memory spill backend for testing."""

    def __init__(self):
        self._store: dict[str, bytes] = {}
        self.match_calls: list[torch.Tensor] = []
        self.load_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.store_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def match_prefix(self, token_ids: torch.Tensor) -> int:
        self.match_calls.append(token_ids)
        # Simulate storing every other chunk of 4 tokens
        aligned = (len(token_ids) // 4) * 4
        # Only "match" if we've previously stored something
        key = self._key(token_ids[:4])
        if key in self._store:
            return aligned
        return 0

    def load_prefix(self, token_ids, gpu_indices, kv_pool) -> None:
        self.load_calls.append((token_ids, gpu_indices))

    def store_prefix(self, token_ids, gpu_indices, kv_pool) -> None:
        self.store_calls.append((token_ids, gpu_indices))
        # Store in chunks of 4
        for i in range(0, len(token_ids), 4):
            chunk = token_ids[i : i + 4]
            self._store[self._key(chunk)] = b"dummy"

    def reset(self) -> None:
        self._store.clear()

    @staticmethod
    def _key(token_ids: torch.Tensor) -> str:
        return "test:" + ",".join(str(int(x)) for x in token_ids.tolist())


class _MockKVPool:
    def __init__(self, num_layers: int = 2, num_tokens: int = 64):
        self._num_layers = num_layers
        self._k = [torch.zeros((num_tokens, 1, 2, 4)) for _ in range(num_layers)]
        self._v = [torch.zeros((num_tokens, 1, 2, 4)) for _ in range(num_layers)]

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v[index]

    def store_kv(self, k, v, out_loc, layer_id) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def num_layers(self) -> int:
        return self._num_layers


def _make_hierarchical_cache_manager(
    num_pages: int = 8,
    page_size: int = 1,
    spill: BaseSpillBackend | None = None,
) -> HierarchicalCacheManager:
    page_table = torch.zeros((4, 64), dtype=torch.int32, device="cpu")
    model_cfg = _MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=2)
    m_i = model_cfg.num_kv_heads * model_cfg.head_dim
    total_fine_units = num_pages * page_size * m_i
    allocator = GlobalFineAllocator(total_fine_units, device="cpu")
    allocator.register_tenant("test", model_cfg, page_size, num_pages)
    kv_pool = _MockKVPool(num_layers=2, num_tokens=64)
    return HierarchicalCacheManager(
        tenant_id="test",
        num_pages=num_pages,
        page_size=page_size,
        page_table=page_table,
        allocator=allocator,
        kv_pool=kv_pool,
        spill_backend=spill,
        async_queue=AsyncTransferQueue() if spill is not None else None,
    )


def _make_pending_req(input_ids: list[int]) -> PendingReq:
    return PendingReq(
        uid=0,
        input_ids=torch.tensor(input_ids, dtype=torch.int32),
        sampling_params=core.SamplingParams(max_tokens=1),
    )


# --------------------------------------------------------------------------- #
# HierarchicalPrefixCache
# --------------------------------------------------------------------------- #


class TestHierarchicalPrefixCache:
    def test_gpu_hit_no_spill_query(self):
        spill = _MockSpillBackend()
        cache = HierarchicalPrefixCache(device=torch.device("cpu"), page_size=1, spill_backend=spill)
        # Insert into GPU cache
        cache.gpu_cache.insert_prefix(
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([10, 11, 12], dtype=torch.int32),
        )
        result = cache.match_prefix(torch.tensor([1, 2, 3, 4], dtype=torch.int32))
        assert result.cuda_handle.cached_len == 3
        assert result.spilled_handle is None
        # Spill backend is queried for the remaining token [4]; that's expected.
        assert len(spill.match_calls) == 1

    def test_gpu_miss_spill_hit(self):
        spill = _MockSpillBackend()
        cache = HierarchicalPrefixCache(device=torch.device("cpu"), page_size=4, spill_backend=spill)
        # Simulate that spill backend has 8 tokens
        spill._store[spill._key(torch.tensor([5, 6, 7, 8], dtype=torch.int32))] = b"x"
        result = cache.match_prefix(torch.tensor([5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.int32))
        assert result.cuda_handle.cached_len == 0
        assert result.spilled_handle is not None
        assert result.spilled_handle.cached_len == 8

    def test_gpu_partial_spill_partial(self):
        spill = _MockSpillBackend()
        cache = HierarchicalPrefixCache(device=torch.device("cpu"), page_size=4, spill_backend=spill)
        # GPU has first 4 tokens
        cache.gpu_cache.insert_prefix(
            torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        )
        # Spill has next 4 tokens (chunk size = 4)
        spill._store[spill._key(torch.tensor([5, 6, 7, 8], dtype=torch.int32))] = b"x"
        result = cache.match_prefix(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32))
        assert result.cuda_handle.cached_len == 4
        assert result.spilled_handle is not None
        # Only one chunk of 4 tokens is matched in spill
        assert result.spilled_handle.cached_len == 4


# --------------------------------------------------------------------------- #
# HierarchicalCacheManager
# --------------------------------------------------------------------------- #


class TestHierarchicalCacheManager:
    def test_match_req_loads_spilled_prefix(self):
        spill = _MockSpillBackend()
        cm = _make_hierarchical_cache_manager(num_pages=8, page_size=1, spill=spill)
        # Pre-populate GPU radix cache with 2 tokens
        cm.prefix_cache.gpu_cache.insert_prefix(
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([10, 11], dtype=torch.int32),
        )
        # Pre-populate spill backend with next 4 tokens
        spill._store[spill._key(torch.tensor([3, 4, 5, 6], dtype=torch.int32))] = b"x"

        req = _make_pending_req([1, 2, 3, 4, 5, 6, 7])
        result = cm.match_req(req)
        assert result.cuda_handle.cached_len == 6  # 2 GPU + 4 loaded from spill
        assert len(spill.load_calls) == 1

    def test_try_expand_from_spilled(self):
        spill = _MockSpillBackend()
        cm = _make_hierarchical_cache_manager(num_pages=8, page_size=1, spill=spill)
        # GPU has 2 tokens
        cm.prefix_cache.gpu_cache.insert_prefix(
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([10, 11], dtype=torch.int32),
        )
        spill._store[spill._key(torch.tensor([3, 4, 5, 6], dtype=torch.int32))] = b"x"

        gpu_handle = cm.prefix_cache.gpu_cache.match_prefix(
            torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
        ).cuda_handle
        req = _make_pending_req([1, 2, 3, 4, 5, 6, 7])
        expanded = cm.try_expand_from_spilled(req, gpu_handle, table_idx=0)
        assert expanded is not None
        new_handle, new_len = expanded
        assert new_len == 6
        # Page table should be updated for the loaded portion.
        # The loaded indices must be contiguous with the GPU-resident ones.
        loaded_indices = new_handle.get_matched_indices()[gpu_handle.cached_len : new_len]
        assert len(loaded_indices) == 4

    def test_maybe_offload_enqueues_async_task(self):
        spill = _MockSpillBackend()
        cm = _make_hierarchical_cache_manager(num_pages=8, page_size=1, spill=spill)
        # Allocate pages for 3 tokens
        pages = cm._allocate(3)
        # Simulate: token 1 was already cached, token 2 is newly computed.
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        # Manually build a handle with cached_len=1
        handle = HierarchicalCacheHandle(
            cached_len=1,
            indices=pages[:1],
        )
        req = core.Req(
            input_ids=input_ids,
            table_idx=0,
            cached_len=2,
            output_len=2,
            uid=0,
            sampling_params=core.SamplingParams(),
            cache_handle=handle,
        )
        cm.lock(req.cache_handle)
        cm.maybe_offload(req)
        # Give the background worker time to pick up the task.
        import time
        time.sleep(0.5)
        cm._async_queue.shutdown(timeout=2.0)
        # The async worker should have called store_prefix with [2]
        assert len(spill.store_calls) == 1
        stored_ids, stored_indices = spill.store_calls[0]
        assert torch.equal(stored_ids, torch.tensor([2], dtype=torch.int32))

    def test_fallback_to_base_cache_manager_when_disabled(self):
        # When spill_backend is None we should behave exactly like CacheManager
        cm = _make_hierarchical_cache_manager(num_pages=4, page_size=1, spill=None)
        assert type(cm) is HierarchicalCacheManager
        req = _make_pending_req([1, 2, 3])
        result = cm.match_req(req)
        assert result.cuda_handle.cached_len == 0
        assert result.spilled_handle is None


# --------------------------------------------------------------------------- #
# Serialization / fingerprint
# --------------------------------------------------------------------------- #


class TestSerialization:
    def test_build_model_fingerprint_unique(self):
        cfg = _MockModelConfig(num_kv_heads=4, head_dim=64, num_layers=32)
        fp1 = build_model_fingerprint(cfg, tp_size=1, tp_rank=0, dtype=torch.bfloat16)
        fp2 = build_model_fingerprint(cfg, tp_size=2, tp_rank=0, dtype=torch.bfloat16)
        fp3 = build_model_fingerprint(cfg, tp_size=1, tp_rank=0, dtype=torch.float16)
        assert fp1 != fp2
        assert fp1 != fp3


if __name__ == "__main__":
    pytest.main([__file__])
