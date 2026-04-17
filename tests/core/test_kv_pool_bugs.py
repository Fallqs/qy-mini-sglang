"""Regression tests for KV pool bugs in multi-tenant heterogeneous model support.

1. page_size mismatch: register_pool used to hard-code base page_size, causing
   TenantContext's num_tokens to exceed the physical view capacity when
   tenant's actual page_size differs from base.

2. ceil view bug: When total_fine_units is not divisible by m_i, the old ceil
   in num_tokens_global caused .view() to request more elements than the
   underlying tensor slice provides, triggering RuntimeError.
"""
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

    if "minisgl.distributed" in sys.modules:
        sys.modules["minisgl.distributed"].get_tp_info = lambda: MockTPInfo()
    else:
        dist = types.ModuleType("minisgl.distributed")
        dist.get_tp_info = lambda: MockTPInfo()
        sys.modules["minisgl.distributed"] = dist

    if "minisgl.utils" in sys.modules:
        utils = sys.modules["minisgl.utils"]
    else:
        utils = types.ModuleType("minisgl.utils")
        sys.modules["minisgl.utils"] = utils
    if not hasattr(utils, "div_even"):
        utils.div_even = lambda n, d, allow_replicate=False: n // d
    if not hasattr(utils, "init_logger"):
        utils.init_logger = lambda name: type("Logger", (), {
            "info": lambda *a, **k: None,
            "warning": lambda *a, **k: None,
        })()

    if "minisgl.kvcache.base" in sys.modules:
        base_mod = sys.modules["minisgl.kvcache.base"]
    else:
        spec = importlib.util.spec_from_file_location("minisgl.kvcache.base", "python/minisgl/kvcache/base.py")
        base_mod = importlib.util.module_from_spec(spec)
        sys.modules["minisgl.kvcache.base"] = base_mod
        spec.loader.exec_module(base_mod)

    if "minisgl.kvcache.mha_pool" in sys.modules:
        mha_mod = sys.modules["minisgl.kvcache.mha_pool"]
    else:
        spec = importlib.util.spec_from_file_location("minisgl.kvcache.mha_pool", "python/minisgl/kvcache/mha_pool.py")
        mha_mod = importlib.util.module_from_spec(spec)
        sys.modules["minisgl.kvcache.mha_pool"] = mha_mod
        spec.loader.exec_module(mha_mod)

    if "minisgl.kvcache" in sys.modules:
        kvcache = sys.modules["minisgl.kvcache"]
    else:
        kvcache = types.ModuleType("minisgl.kvcache")
        sys.modules["minisgl.kvcache"] = kvcache
    kvcache.BaseKVCachePool = base_mod.BaseKVCachePool
    kvcache.MHAKVCache = mha_mod.MHAKVCache

    if "minisgl.engine.segment_list" in sys.modules:
        seg_mod = sys.modules["minisgl.engine.segment_list"]
    else:
        spec = importlib.util.spec_from_file_location("minisgl.engine.segment_list", "python/minisgl/engine/segment_list.py")
        seg_mod = importlib.util.module_from_spec(spec)
        sys.modules["minisgl.engine.segment_list"] = seg_mod
        spec.loader.exec_module(seg_mod)

    if "minisgl.engine" in sys.modules:
        engine_pkg = sys.modules["minisgl.engine"]
    else:
        engine_pkg = types.ModuleType("minisgl.engine")
        sys.modules["minisgl.engine"] = engine_pkg
    engine_pkg.segment_list = seg_mod

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


def _make_mgr(base_ps: int, base_E: int, total_num_pages: int = 10):
    """Helper: create KVPoolManager."""
    kv_mod = _load_kv_pool()
    KVPoolManager = kv_mod.KVPoolManager
    base_cfg = MockModelConfig(num_kv_heads=base_E // 4, head_dim=4, num_layers=1)
    return KVPoolManager(
        device=torch.device("cpu"),
        total_num_pages=total_num_pages,
        page_size=base_ps,
        dtype=torch.float32,
        base_model_config=base_cfg,
    )


# =============================================================================
# Bug 1: page_size mismatch between base model and tenant
# =============================================================================

def test_page_size_mismatch_double_still_safe():
    """Tenant page_size=8, base page_size=4. With fix, capped_num_pages adapts
    so that num_tokens never exceeds view capacity."""
    mgr = _make_mgr(base_ps=4, base_E=8)
    tenant_cfg = MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=1)
    pool = mgr.register_pool("T", tenant_cfg, page_size=8)
    pool.allocate(num_pages=10)

    # allocator used tenant page_size=8: page_fu=64, max_pages=320//64=5
    assert pool.num_pages == 5

    num_tokens = pool.num_pages * 8  # 5 * 8 = 40
    m_i = tenant_cfg.num_kv_heads * tenant_cfg.head_dim  # 8
    view_token_capacity = mgr.fine_units_per_layer // m_i  # 320 // 8 = 40

    assert num_tokens <= view_token_capacity, (
        f"num_tokens({num_tokens}) > view_capacity({view_token_capacity})"
    )


def test_page_size_mismatch_quadruple_still_safe():
    """Tenant page_size=16, base page_size=4. capped_num_pages adapts to 2."""
    mgr = _make_mgr(base_ps=4, base_E=8)
    tenant_cfg = MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=1)
    pool = mgr.register_pool("T", tenant_cfg, page_size=16)
    pool.allocate(num_pages=10)

    # page_fu=128, max_pages=320//128=2
    assert pool.num_pages == 2

    num_tokens = pool.num_pages * 16  # 2 * 16 = 32
    m_i = tenant_cfg.num_kv_heads * tenant_cfg.head_dim  # 8
    view_token_capacity = mgr.fine_units_per_layer // m_i  # 40

    assert num_tokens <= view_token_capacity


def test_page_size_mismatch_smaller_still_safe():
    """Tenant page_size=2, base page_size=4. capped_num_pages stays at 10,
    num_tokens=20 fits comfortably."""
    mgr = _make_mgr(base_ps=4, base_E=8)
    tenant_cfg = MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=1)
    pool = mgr.register_pool("T", tenant_cfg, page_size=2)
    pool.allocate(num_pages=10)

    # page_fu=16, max_pages=320//16=20, so capped stays at 10
    assert pool.num_pages == 10

    num_tokens = pool.num_pages * 2  # 20
    m_i = tenant_cfg.num_kv_heads * tenant_cfg.head_dim  # 8
    view_token_capacity = mgr.fine_units_per_layer // m_i  # 40

    assert num_tokens <= view_token_capacity


def test_page_size_match_exact():
    """When page_size matches, num_tokens exactly equals view capacity."""
    mgr = _make_mgr(base_ps=4, base_E=8)
    tenant_cfg = MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=1)
    pool = mgr.register_pool("T", tenant_cfg, page_size=4)
    pool.allocate(num_pages=10)

    assert pool.num_pages == 10
    num_tokens = pool.num_pages * 4  # 40
    m_i = tenant_cfg.num_kv_heads * tenant_cfg.head_dim  # 8
    view_token_capacity = mgr.fine_units_per_layer // m_i  # 40

    assert num_tokens == view_token_capacity


# =============================================================================
# Bug 2: ceil in num_tokens_global used to cause .view() RuntimeError
# =============================================================================

def test_non_divisible_mi_allocates_ok():
    """total_fine_units=320, m_i=12 -> 320%12=8. Exact adaptive view avoids
    the old ceil bug; allocate should succeed."""
    mgr = _make_mgr(base_ps=4, base_E=8, total_num_pages=10)
    tenant_cfg = MockModelConfig(num_kv_heads=3, head_dim=4, num_layers=1)  # E=12
    pool = mgr.register_pool("T", tenant_cfg, page_size=4)

    total_fine = mgr.fine_units_per_layer  # 320
    m_i = tenant_cfg.num_kv_heads * tenant_cfg.head_dim  # 12
    assert total_fine % m_i != 0

    # Should NOT raise RuntimeError after fix
    pool.allocate(num_pages=5)
    assert pool.is_allocated
    assert pool.num_pages == 5  # 5*4*12=240 <= 320

    num_tokens = pool.num_pages * 4  # 20
    assert num_tokens * m_i <= total_fine  # 240 <= 320


def test_large_remainder_allocates_ok():
    """total_fine_units=1600, m_i=6 -> 1600%6=4. Should succeed."""
    mgr = _make_mgr(base_ps=4, base_E=4, total_num_pages=100)
    tenant_cfg = MockModelConfig(num_kv_heads=3, head_dim=2, num_layers=1)  # E=6
    pool = mgr.register_pool("T", tenant_cfg, page_size=4)

    total_fine = mgr.fine_units_per_layer  # 1600
    m_i = tenant_cfg.num_kv_heads * tenant_cfg.head_dim  # 6
    assert total_fine % m_i != 0

    pool.allocate(num_pages=5)
    assert pool.is_allocated
    assert pool.num_pages == 5  # 5*4*6=120 <= 1600
    assert pool.num_pages * 4 * m_i <= total_fine


def test_extreme_small_allocates_ok():
    """total_fine_units=3, m_i=2 -> 3%2=1. Should succeed with capped pages."""
    kv_mod = _load_kv_pool()
    KVPoolManager = kv_mod.KVPoolManager

    base_cfg = MockModelConfig(num_kv_heads=3, head_dim=1, num_layers=1)
    mgr = KVPoolManager(
        device=torch.device("cpu"),
        total_num_pages=1,
        page_size=1,
        dtype=torch.float32,
        base_model_config=base_cfg,
    )
    tenant_cfg = MockModelConfig(num_kv_heads=2, head_dim=1, num_layers=1)
    pool = mgr.register_pool("T", tenant_cfg, page_size=1)

    total_fine = mgr.fine_units_per_layer  # 3
    m_i = tenant_cfg.num_kv_heads * tenant_cfg.head_dim  # 2
    assert total_fine % m_i != 0

    pool.allocate(num_pages=1)
    assert pool.is_allocated
    assert pool.num_pages == 1  # capped: 1*1*2=2 <= 3
    assert pool.num_pages * 1 * m_i <= total_fine


def test_exact_division_still_ok():
    """total_fine_units=320, m_i=8 -> exact division. allocate should succeed."""
    mgr = _make_mgr(base_ps=4, base_E=8, total_num_pages=10)
    tenant_cfg = MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=1)
    pool = mgr.register_pool("T", tenant_cfg, page_size=4)

    total_fine = mgr.fine_units_per_layer  # 320
    m_i = tenant_cfg.num_kv_heads * tenant_cfg.head_dim  # 8
    assert total_fine % m_i == 0

    pool.allocate(num_pages=5)
    assert pool.is_allocated
    assert pool.num_pages == 5
    assert pool.num_pages * 4 * m_i <= total_fine


def test_mi_larger_than_total_capped_to_zero():
    """total_fine_units=320, m_i=321 -> m_i > total_fine.
    max_pages=0, so capped_num_pages=0. View should be empty but safe."""
    kv_mod = _load_kv_pool()
    KVPoolManager = kv_mod.KVPoolManager

    base_cfg = MockModelConfig(num_kv_heads=80, head_dim=1, num_layers=1)
    mgr = KVPoolManager(
        device=torch.device("cpu"),
        total_num_pages=1,
        page_size=4,
        dtype=torch.float32,
        base_model_config=base_cfg,
    )
    tenant_cfg = MockModelConfig(num_kv_heads=321, head_dim=1, num_layers=1)
    pool = mgr.register_pool("T", tenant_cfg, page_size=4)

    total_fine = mgr.fine_units_per_layer  # 320
    m_i = tenant_cfg.num_kv_heads * tenant_cfg.head_dim  # 321
    assert total_fine < m_i

    pool.allocate(num_pages=1)
    assert pool.is_allocated
    assert pool.num_pages == 0  # capped to zero


if __name__ == "__main__":
    tests = [
        test_page_size_mismatch_double_still_safe,
        test_page_size_mismatch_quadruple_still_safe,
        test_page_size_mismatch_smaller_still_safe,
        test_page_size_match_exact,
        test_non_divisible_mi_allocates_ok,
        test_large_remainder_allocates_ok,
        test_extreme_small_allocates_ok,
        test_exact_division_still_ok,
        test_mi_larger_than_total_capped_to_zero,
    ]
    for t in tests:
        print(f"Running {t.__name__}...")
        t()
        print("  PASSED")
    print("\nAll bug regression tests passed!")
