"""Integration test for shared KV pool cross-tenant synchronization."""
from __future__ import annotations

from dataclasses import dataclass
import sys
import torch


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
    import types
    import importlib.util

    # Always inject/override get_tp_info in minisgl.distributed
    if "minisgl.distributed" in sys.modules:
        sys.modules["minisgl.distributed"].get_tp_info = lambda: MockTPInfo()
    else:
        dist = types.ModuleType("minisgl.distributed")
        dist.get_tp_info = lambda: MockTPInfo()
        sys.modules["minisgl.distributed"] = dist

    # minisgl.utils
    if "minisgl.utils" in sys.modules:
        utils = sys.modules["minisgl.utils"]
    else:
        utils = types.ModuleType("minisgl.utils")
        sys.modules["minisgl.utils"] = utils
    if not hasattr(utils, "div_even"):
        utils.div_even = lambda n, d, allow_replicate=False: n // d
    if not hasattr(utils, "init_logger"):
        utils.init_logger = lambda name: type("Logger", (), {"info": lambda *a, **k: None, "warning": lambda *a, **k: None})()

    # minisgl.kvcache.base
    if "minisgl.kvcache.base" in sys.modules:
        base_mod = sys.modules["minisgl.kvcache.base"]
    else:
        spec = importlib.util.spec_from_file_location("minisgl.kvcache.base", "python/minisgl/kvcache/base.py")
        base_mod = importlib.util.module_from_spec(spec)
        sys.modules["minisgl.kvcache.base"] = base_mod
        spec.loader.exec_module(base_mod)

    # minisgl.kvcache.mha_pool
    if "minisgl.kvcache.mha_pool" in sys.modules:
        mha_mod = sys.modules["minisgl.kvcache.mha_pool"]
    else:
        spec = importlib.util.spec_from_file_location("minisgl.kvcache.mha_pool", "python/minisgl/kvcache/mha_pool.py")
        mha_mod = importlib.util.module_from_spec(spec)
        sys.modules["minisgl.kvcache.mha_pool"] = mha_mod
        spec.loader.exec_module(mha_mod)

    # minisgl.kvcache package
    if "minisgl.kvcache" in sys.modules:
        kvcache = sys.modules["minisgl.kvcache"]
    else:
        kvcache = types.ModuleType("minisgl.kvcache")
        sys.modules["minisgl.kvcache"] = kvcache
    kvcache.BaseKVCachePool = base_mod.BaseKVCachePool
    kvcache.MHAKVCache = mha_mod.MHAKVCache

    # minisgl.engine.segment_list
    if "minisgl.engine.segment_list" in sys.modules:
        seg_mod = sys.modules["minisgl.engine.segment_list"]
    else:
        spec = importlib.util.spec_from_file_location("minisgl.engine.segment_list", "python/minisgl/engine/segment_list.py")
        seg_mod = importlib.util.module_from_spec(spec)
        sys.modules["minisgl.engine.segment_list"] = seg_mod
        spec.loader.exec_module(seg_mod)

    # minisgl.engine package
    if "minisgl.engine" in sys.modules:
        engine_pkg = sys.modules["minisgl.engine"]
    else:
        engine_pkg = types.ModuleType("minisgl.engine")
        sys.modules["minisgl.engine"] = engine_pkg
    engine_pkg.segment_list = seg_mod

    # minisgl root
    if "minisgl" in sys.modules:
        minisgl = sys.modules["minisgl"]
    else:
        minisgl = types.ModuleType("minisgl")
        sys.modules["minisgl"] = minisgl
    minisgl.distributed = sys.modules["minisgl.distributed"]
    minisgl.utils = utils
    minisgl.kvcache = kvcache
    minisgl.engine = engine_pkg


def _load_kv_pool():
    _inject_minisgl()
    import importlib.util
    spec = importlib.util.spec_from_file_location("minisgl.engine.kv_pool", "python/minisgl/engine/kv_pool.py")
    kv_mod = importlib.util.module_from_spec(spec)
    sys.modules["minisgl.engine.kv_pool"] = kv_mod
    spec.loader.exec_module(kv_mod)
    return kv_mod


def test_cross_tenant_sync():
    kv_mod = _load_kv_pool()
    GlobalFineAllocator = kv_mod.GlobalFineAllocator

    device = "cpu"
    # total fine units per layer = 100 pages * 4 tokens/page * E_base
    # base model: E=8 (2 heads * 4 dim) => 100*4*8 = 3200 fine units
    allocator = GlobalFineAllocator(3200, device)

    # Tenant A: E=8, page_size=4, num_pages=10
    allocator.register_tenant("A", MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=2), page_size=4, num_pages=10)
    # Tenant B: E=16, page_size=4, num_pages=10
    allocator.register_tenant("B", MockModelConfig(num_kv_heads=4, head_dim=4, num_layers=2), page_size=4, num_pages=10)

    # Initially both tenants have all pages free
    assert allocator.available_pages("A") == 10
    assert allocator.available_pages("B") == 10

    # A allocates 2 pages -> needs 2*4*8 = 64 fine units
    tokens_a = allocator.allocate("A", 2)
    # A now has 8 free pages (2 allocated)
    assert allocator.available_pages("A") == 8
    # B's pages that overlap the allocated f.u. range become dirty.
    # 64 f.u. / (4*16) = 1 page for B. So B should have 9 free pages.
    assert allocator.available_pages("B") == 9

    # B allocates 1 page -> needs 1*4*16 = 64 fine units
    tokens_b = allocator.allocate("B", 1)
    # B now has 8 free pages (9 - 1)
    assert allocator.available_pages("B") == 8
    # A loses 2 more pages because B's 64 f.u. overlaps 2 A-pages (64/32=2)
    assert allocator.available_pages("A") == 6

    # A frees its 2 pages
    allocator.free("A", tokens_a)
    assert allocator.available_pages("A") == 8  # 6 + 2
    # B gets 1 page back
    assert allocator.available_pages("B") == 9  # 8 + 1

    # B frees its 1 page
    allocator.free("B", tokens_b)
    assert allocator.available_pages("B") == 10
    assert allocator.available_pages("A") == 10


def test_mhakvcache_views():
    kv_mod = _load_kv_pool()
    KVPoolManager = kv_mod.KVPoolManager

    # Base model: E=8 (2*4), page_size=4, total_num_pages=10
    base_cfg = MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=2)
    mgr = KVPoolManager(
        device=torch.device("cpu"),
        total_num_pages=10,
        page_size=4,
        dtype=torch.float32,
        base_model_config=base_cfg,
    )

    # Tenant with E=16 (4*4), 3 layers
    tenant_cfg = MockModelConfig(num_kv_heads=4, head_dim=4, num_layers=3)
    pool = mgr.register_pool("T1", tenant_cfg)
    pool.allocate(num_pages=5)

    cache = pool.pool
    assert cache.num_layers == 3
    assert cache.k_cache(0).shape == (mgr.fine_units_per_layer // 16, 1, 4, 4)
    assert cache.v_cache(0).shape == (mgr.fine_units_per_layer // 16, 1, 4, 4)

    # Verify that the view is into the same physical memory
    cache.k_cache(0)[0, 0, 0] = 99.0
    # The global tensor should reflect this change
    assert mgr.global_kv[0][0, 0, 0].item() == 99.0


def test_non_contiguous_allocate():
    kv_mod = _load_kv_pool()
    GlobalFineAllocator = kv_mod.GlobalFineAllocator

    allocator = GlobalFineAllocator(3200, "cpu")
    allocator.register_tenant("A", MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=2), page_size=4, num_pages=10)
    allocator.register_tenant("B", MockModelConfig(num_kv_heads=4, head_dim=4, num_layers=2), page_size=4, num_pages=10)

    # Fragment the pool: A allocates pages 0 and 2 (non-contiguous in its own space,
    # but the global allocator returns whatever segments it has)
    t1 = allocator.allocate("A", 2)  # takes [0, 64)
    t2 = allocator.allocate("A", 2)  # takes [64, 128)
    allocator.free("A", t1)          # frees [0, 64)

    # Now allocate 3 pages - should take [0, 64) and [128, 192)
    t3 = allocator.allocate("A", 3)
    # Should be two segments merged into one token tensor
    assert len(t3) == 3 * 4  # 3 pages * 4 tokens/page = 12 tokens


if __name__ == "__main__":
    test_cross_tenant_sync()
    print("test_cross_tenant_sync passed")
    test_mhakvcache_views()
    print("test_mhakvcache_views passed")
    test_non_contiguous_allocate()
    print("test_non_contiguous_allocate passed")
    print("All shared KV pool tests passed!")
