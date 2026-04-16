"""Test partial overlap free correctness with heterogeneous tenants."""
from __future__ import annotations

from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class MockModelConfig:
    num_kv_heads: int
    head_dim: int
    num_layers: int


@dataclass(frozen=True)
class MockTPInfo:
    rank: int = 0
    size: int = 1


def _inject_minisgl():
    if "minisgl" in sys.modules:
        return
    import types
    import importlib.util

    dist = types.ModuleType("minisgl.distributed")
    dist.get_tp_info = lambda: MockTPInfo()
    sys.modules["minisgl.distributed"] = dist

    utils = types.ModuleType("minisgl.utils")
    utils.div_even = lambda n, d, allow_replicate=False: n // d
    utils.init_logger = lambda name: type("Logger", (), {"info": lambda *a, **k: None, "warning": lambda *a, **k: None})()
    sys.modules["minisgl.utils"] = utils

    spec = importlib.util.spec_from_file_location("minisgl.kvcache.base", "python/minisgl/kvcache/base.py")
    base_mod = importlib.util.module_from_spec(spec)
    sys.modules["minisgl.kvcache.base"] = base_mod
    spec.loader.exec_module(base_mod)

    spec = importlib.util.spec_from_file_location("minisgl.kvcache.mha_pool", "python/minisgl/kvcache/mha_pool.py")
    mha_mod = importlib.util.module_from_spec(spec)
    sys.modules["minisgl.kvcache.mha_pool"] = mha_mod
    spec.loader.exec_module(mha_mod)

    kvcache = types.ModuleType("minisgl.kvcache")
    kvcache.BaseKVCachePool = base_mod.BaseKVCachePool
    kvcache.MHAKVCache = mha_mod.MHAKVCache
    sys.modules["minisgl.kvcache"] = kvcache

    spec = importlib.util.spec_from_file_location("minisgl.engine.segment_list", "python/minisgl/engine/segment_list.py")
    seg_mod = importlib.util.module_from_spec(spec)
    sys.modules["minisgl.engine.segment_list"] = seg_mod
    spec.loader.exec_module(seg_mod)

    engine_pkg = types.ModuleType("minisgl.engine")
    engine_pkg.segment_list = seg_mod
    sys.modules["minisgl.engine"] = engine_pkg

    minisgl = types.ModuleType("minisgl")
    minisgl.distributed = dist
    minisgl.utils = utils
    minisgl.kvcache = kvcache
    minisgl.engine = engine_pkg
    sys.modules["minisgl"] = minisgl


def _load_kv_pool():
    _inject_minisgl()
    import importlib.util
    spec = importlib.util.spec_from_file_location("minisgl.engine.kv_pool", "python/minisgl/engine/kv_pool.py")
    kv_mod = importlib.util.module_from_spec(spec)
    sys.modules["minisgl.engine.kv_pool"] = kv_mod
    spec.loader.exec_module(kv_mod)
    return kv_mod


def test_partial_overlap_free():
    """
    Tenant A has page_fu=32, Tenant B has page_fu=48.
    A allocates page 0 ([0,32)) and page 1 ([32,64)).
    This makes B page 0 ([0,48)) dirty and B page 1 ([48,96)) dirty.
    Then A frees page 1 ([32,64)).
    B page 0 should STAY dirty because A page 0 ([0,32)) still overlaps it.
    B page 1 should become clean because [48,96) is no longer overlapped by A.
    """
    kv_mod = _load_kv_pool()
    GlobalFineAllocator = kv_mod.GlobalFineAllocator

    allocator = GlobalFineAllocator(96, "cpu")
    # A: E=8, page_size=4 -> page_fu=32
    allocator.register_tenant("A", MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=1), page_size=4, num_pages=3)
    # B: E=12, page_size=4 -> page_fu=48  (but capped to 96//48=2 pages)
    allocator.register_tenant("B", MockModelConfig(num_kv_heads=3, head_dim=4, num_layers=1), page_size=4, num_pages=3)

    t0 = allocator.allocate("A", 1)  # [0, 32)
    assert allocator.available_pages("A") == 2
    # B page 0 ([0,48)) should be dirty
    assert allocator.available_pages("B") == 1

    t1 = allocator.allocate("A", 1)  # [32, 64)
    assert allocator.available_pages("A") == 1
    # B page 1 ([48,96)) should also be dirty now
    assert allocator.available_pages("B") == 0

    allocator.free("A", t1)
    # A should have 2 free pages
    assert allocator.available_pages("A") == 2
    # B page 0 should still be dirty (A page 0 still occupies [0,32))
    # B page 1 should be clean now
    assert allocator.available_pages("B") == 1

    allocator.free("A", t0)
    assert allocator.available_pages("A") == 3
    assert allocator.available_pages("B") == 2


if __name__ == "__main__":
    test_partial_overlap_free()
    print("test_partial_overlap_free passed")
