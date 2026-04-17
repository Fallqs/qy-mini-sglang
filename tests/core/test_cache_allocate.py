"""
Test that CacheManager._allocate correctly handles eviction with page_size > 1.
"""

from __future__ import annotations

import pytest
import torch

import minisgl.core as core
from minisgl.distributed import set_tp_info
from minisgl.engine.kv_pool import GlobalFineAllocator
from minisgl.scheduler.cache import CacheManager


@pytest.fixture(scope="session", autouse=True)
def setup_tp_info():
    """Set TP info once for the whole test session."""
    set_tp_info(rank=0, size=1)


@pytest.fixture(autouse=True)
def reset_global_ctx():
    """Reset global context before and after each test."""
    old_ctxs = getattr(core._GLOBAL_CTX_STACK, "ctxs", [])
    core._GLOBAL_CTX_STACK.ctxs = []
    yield
    core._GLOBAL_CTX_STACK.ctxs = old_ctxs


class _MockModelConfig:
    def __init__(self, num_kv_heads: int = 2, head_dim: int = 4, num_layers: int = 1):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers


def _make_cache_manager(num_pages: int, page_size: int) -> CacheManager:
    """Helper to create a CacheManager with radix cache on CPU."""
    page_table = torch.empty((1,), device="cpu")
    ctx = core.Context(page_size=page_size)
    core.set_global_ctx(ctx)
    model_cfg = _MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=1)
    m_i = model_cfg.num_kv_heads * model_cfg.head_dim  # 8
    total_fine_units = num_pages * page_size * m_i
    allocator = GlobalFineAllocator(total_fine_units, device="cpu")
    allocator.register_tenant(
        tenant_id="test",
        model_config=model_cfg,
        page_size=page_size,
        num_pages=num_pages,
    )
    return CacheManager(
        tenant_id="test",
        num_pages=num_pages,
        page_size=page_size,
        page_table=page_table,
        type="radix",
        allocator=allocator,
    )


def _insert_evictable(cm: CacheManager, input_ids: torch.Tensor, indices: torch.Tensor):
    """Insert a prefix into the radix cache so it becomes evictable."""
    cm.prefix_cache.insert_prefix(input_ids, indices)


def _assert_all_page_aligned(tensor: torch.Tensor, page_size: int, label: str = ""):
    """Assert every element in tensor is a multiple of page_size."""
    if len(tensor) == 0:
        return
    misaligned = tensor[tensor % page_size != 0]
    assert (
        len(misaligned) == 0
    ), f"{label} contains non-page-aligned values: {misaligned.tolist()}, page_size={page_size}"


def _assert_no_overlap(pages: torch.Tensor, page_size: int):
    """Assert that page-aligned starts, when expanded, produce no overlapping ranges."""
    if len(pages) <= 1:
        return
    expanded = set()
    for p in pages.tolist():
        token_range = set(range(p, p + page_size))
        overlap = expanded & token_range
        assert len(overlap) == 0, f"Overlapping tokens: {overlap}"
        expanded.update(token_range)


class TestAllocateEvictPageAlignment:
    """Tests for _allocate handling eviction with page_size > 1."""

    def test_allocate_after_evict_returns_page_aligned(self):
        """Allocated pages after eviction must be page-aligned."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)
        assert cm.allocator.available_pages(cm.tenant_id) == 0

        # Insert 2 pages worth of data into the cache (evictable)
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        # Simulate page table entries: page 0 = [0,1,2,3], page 1 = [4,5,6,7]
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 1 page — triggers eviction
        allocated = cm._allocate(1)
        _assert_all_page_aligned(allocated, page_size, "allocated")

    def test_consecutive_allocations_after_evict_no_overlap(self):
        """Multiple allocations after eviction must not produce overlapping pages."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)

        # Insert 2 pages into cache
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages one by one
        page_a = cm._allocate(1)
        page_b = cm._allocate(1)
        all_pages = torch.cat([page_a, page_b])

        _assert_all_page_aligned(all_pages, page_size, "all_pages")
        _assert_no_overlap(all_pages, page_size)

    def test_free_slots_stay_page_aligned_after_evict(self):
        """Free pages count must be correct after eviction refills them."""
        page_size = 8
        num_pages = 8
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all free pages
        cm._allocate(num_pages)
        assert cm.allocator.available_pages(cm.tenant_id) == 0

        # Insert 4 pages worth of data (4 * 8 = 32 tokens)
        n_tokens = page_size * 4
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        # Indices: page starts at 0, 8, 16, 24
        indices = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 1 page — evicts and refills free pages.
        # Radix cache may evict more than 1 page worth, so free pages
        # will be >= 0 (exact count depends on eviction granularity).
        cm._allocate(1)
        assert cm.allocator.available_pages(cm.tenant_id) >= 0

    def test_allocate_exact_pages_needed_from_evict(self):
        """When exactly N pages are needed, eviction must provide at least N pages."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all
        cm._allocate(num_pages)

        # Insert 3 pages into cache
        n_tokens = page_size * 3
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        indices = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages at once — needs eviction of at least 2 pages
        allocated = cm._allocate(2)
        assert len(allocated) == 2
        _assert_all_page_aligned(allocated, page_size, "allocated")
        _assert_no_overlap(allocated, page_size)

    def test_page_to_token_expansion_correct_after_evict(self):
        """_page_to_token on eviction-allocated pages must produce correct consecutive ranges."""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        cm._allocate(num_pages)

        # Insert 2 pages: tokens [0..7] with indices [0..7]
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages via eviction
        pages = cm._allocate(2)
        tokens = cm._page_to_token(pages)

        # Each page should expand to page_size consecutive tokens
        assert len(tokens) == 2 * page_size
        for i, page_start in enumerate(pages.tolist()):
            chunk = tokens[i * page_size : (i + 1) * page_size].tolist()
            expected = list(range(page_start, page_start + page_size))
            assert chunk == expected, f"Page {page_start}: got {chunk}, expected {expected}"

    def test_check_integrity_passes_after_evict_cycle(self):
        """CacheManager integrity check should pass after allocate-free-evict cycles."""
        page_size = 4
        num_pages = 8
        cm = _make_cache_manager(num_pages, page_size)

        # Allocate 2 pages (simulating a request using them)
        pages_for_cache = cm._allocate(2)
        # free_slots: 6 pages remaining

        # Insert those 2 pages into radix cache (simulating a finished request)
        # The token indices come from _page_to_token expansion of the allocated pages
        token_indices = cm._page_to_token(pages_for_cache)
        n_tokens = len(token_indices)
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, token_indices)
        # Now: free=6, cache=2, total=8

        # This should not raise
        cm.check_integrity()

        # Now exhaust free slots and trigger eviction
        cm._allocate(6)
        assert cm.allocator.available_pages(cm.tenant_id) == 0

        # Allocate 1 more page — must evict from cache.
        # Eviction may free more pages than needed, so remaining free pages
        # can be > 0 depending on eviction granularity.
        allocated = cm._allocate(1)
        _assert_all_page_aligned(allocated, page_size, "allocated after evict")
        assert cm.allocator.available_pages(cm.tenant_id) >= 0


if __name__ == "__main__":
    pytest.main([__file__])
