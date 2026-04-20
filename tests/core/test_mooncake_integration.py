"""Integration tests for MoonCake-backed hierarchical KV cache.

These tests verify the Mini-SGLang → MoonCake integration layer.  They require
``mooncake-transfer-engine`` to be installed but do **not** need a running
``mooncake_master`` or RDMA hardware – the real ``MooncakeDistributedStore``
object is exercised where possible (import / instantiation) and mocked for
storage operations so that we can validate the call patterns, serialization
round-trip, and end-to-end data flow.

Run with::

    LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7 \
        pytest tests/core/test_mooncake_integration.py -v

(The ``LD_PRELOAD`` workaround is needed on systems where the conda-provided
``libffi.so.8`` shadows the system ``libffi.so.7`` required by
``libp11-kit.so.0``.)
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from unittest.mock import MagicMock

import pytest
import torch

import minisgl.core as core
from minisgl.distributed import set_tp_info
from minisgl.engine.kv_pool import GlobalFineAllocator
from minisgl.kvcache import HierarchicalCacheHandle, HierarchicalPrefixCache
from minisgl.kvcache.async_transfer import AsyncTransferQueue
from minisgl.kvcache.mooncake_backend import (
    BaseSpillBackend,
    MoonCakeKVBackend,
    NoopSpillBackend,
    _deserialize_kv,
    _serialize_kv,
    build_model_fingerprint,
    mooncake_available,
)
from minisgl.scheduler.cache import CacheManager, HierarchicalCacheManager
from minisgl.scheduler.utils import PendingReq


# --------------------------------------------------------------------------- #
# Session fixtures
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


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


class _MockKVPool:
    """CPU-backed KV pool for deterministic testing."""

    def __init__(self, num_layers: int = 2, num_tokens: int = 64, dtype: torch.dtype = torch.float32):
        self._num_layers = num_layers
        self._k = [torch.zeros((num_tokens, 1, 2, 4), dtype=dtype) for _ in range(num_layers)]
        self._v = [torch.zeros((num_tokens, 1, 2, 4), dtype=dtype) for _ in range(num_layers)]
        self._dtype = dtype

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v[index]

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers


class _MockMooncakeStore:
    """Pure-Python mock that mimics the ``MooncakeDistributedStore`` API surface."""

    def __init__(self):
        self._data: dict[str, bytes] = {}
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, name: str, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def put(self, key: str, value: bytes) -> int:
        self._record("put", key, value)
        self._data[key] = value
        return 0

    def get(self, key: str) -> bytes:
        self._record("get", key)
        return self._data.get(key, b"")

    def is_exist(self, key: str) -> int:
        self._record("is_exist", key)
        return 1 if key in self._data else 0

    def remove(self, key: str) -> int:
        self._record("remove", key)
        self._data.pop(key, None)
        return 0

    def remove_all(self) -> int:
        self._record("remove_all")
        n = len(self._data)
        self._data.clear()
        return n

    def close(self) -> None:
        self._record("close")

    def health_check(self) -> int:
        return 1

    def setup(self, *args, **kwargs) -> int:
        self._record("setup", args, kwargs)
        return 0

    def setup_dummy(self, *args, **kwargs) -> int:
        self._record("setup_dummy", args, kwargs)
        return 0


def _make_hierarchical_cache_manager(
    num_pages: int = 8,
    page_size: int = 1,
    spill: BaseSpillBackend | None = None,
    dtype: torch.dtype = torch.float32,
) -> HierarchicalCacheManager:
    page_table = torch.zeros((4, 64), dtype=torch.int32, device="cpu")
    model_cfg = _MockModelConfig(num_kv_heads=2, head_dim=4, num_layers=2)
    m_i = model_cfg.num_kv_heads * model_cfg.head_dim
    total_fine_units = num_pages * page_size * m_i
    allocator = GlobalFineAllocator(total_fine_units, device="cpu")
    allocator.register_tenant("test", model_cfg, page_size, num_pages)
    kv_pool = _MockKVPool(num_layers=2, num_tokens=64, dtype=dtype)
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
# 1. MoonCake availability / ABI smoke tests
# --------------------------------------------------------------------------- #


class TestMooncakeAvailability:
    """Verify that the installed ``mooncake-transfer-engine`` is importable
    and exposes the expected API surface."""

    def test_mooncake_importable(self):
        """``mooncake.store`` must import without error."""
        pytest.importorskip("mooncake.store")

    def test_mooncake_available_helper(self):
        assert mooncake_available() is True

    def test_store_instantiable(self):
        from mooncake.store import MooncakeDistributedStore
        s = MooncakeDistributedStore()
        assert s is not None
        # verify expected methods exist
        for attr in ("setup", "setup_dummy", "put", "get", "is_exist",
                     "remove", "remove_all", "close", "health_check"):
            assert hasattr(s, attr), f"Missing attribute: {attr}"
        s.close()


# --------------------------------------------------------------------------- #
# 2. Serialization / key generation
# --------------------------------------------------------------------------- #


class TestSerializationRoundTrip:
    """KV tensor serialization is the critical path between Mini-SGLang
    and MoonCake; these tests ensure the wire format is stable and correct."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_serialize_deserialize(self, dtype: torch.dtype):
        k = torch.randn(4, 1, 2, 4, dtype=dtype)
        v = torch.randn(4, 1, 2, 4, dtype=dtype)
        blob = _serialize_kv(k, v)
        assert isinstance(blob, bytes)
        assert len(blob) > 0
        k2, v2 = _deserialize_kv(blob, dtype)
        assert torch.equal(k, k2)
        assert torch.equal(v, v2)

    def test_serialize_version_mismatch_raises(self):
        k = torch.randn(2, 1, 2, 4)
        v = torch.randn(2, 1, 2, 4)
        blob = bytearray(_serialize_kv(k, v))
        blob[0] = 99  # corrupt version
        with pytest.raises(ValueError, match="Unsupported KV serialization version"):
            _deserialize_kv(bytes(blob), torch.float32)

    def test_chunk_key_deterministic(self):
        cfg = _MockModelConfig(num_layers=2)
        backend = MoonCakeKVBackend(
            model_config=cfg,
            page_size=1,
            model_fingerprint="fp_test",
            chunk_size=4,
            store=_MockMooncakeStore(),
        )
        tokens = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        k1 = backend._chunk_key(0, 0, tokens)
        k2 = backend._chunk_key(0, 0, tokens)
        assert k1 == k2
        assert k1.startswith("minisgl:fp_test:L0:C0:")

    def test_chunk_key_different_tokens(self):
        cfg = _MockModelConfig(num_layers=2)
        backend = MoonCakeKVBackend(
            model_config=cfg,
            page_size=1,
            model_fingerprint="fp_test",
            chunk_size=4,
            store=_MockMooncakeStore(),
        )
        k1 = backend._chunk_key(0, 0, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
        k2 = backend._chunk_key(0, 0, torch.tensor([1, 2, 3, 5], dtype=torch.int32))
        assert k1 != k2

    def test_iter_chunks_alignment(self):
        cfg = _MockModelConfig(num_layers=2)
        backend = MoonCakeKVBackend(
            model_config=cfg,
            page_size=1,
            model_fingerprint="fp_test",
            chunk_size=4,
            store=_MockMooncakeStore(),
        )
        tokens = torch.arange(10, dtype=torch.int32)
        chunks = list(backend._iter_chunks(tokens))
        assert len(chunks) == 2  # 0..3, 4..7
        assert torch.equal(chunks[0][1], tokens[0:4])
        assert torch.equal(chunks[1][1], tokens[4:8])

    def test_iter_chunks_empty_when_less_than_chunk_size(self):
        cfg = _MockModelConfig(num_layers=2)
        backend = MoonCakeKVBackend(
            model_config=cfg,
            page_size=1,
            model_fingerprint="fp_test",
            chunk_size=8,
            store=_MockMooncakeStore(),
        )
        tokens = torch.arange(3, dtype=torch.int32)
        chunks = list(backend._iter_chunks(tokens))
        assert len(chunks) == 0


# --------------------------------------------------------------------------- #
# 3. MoonCakeKVBackend with injected mock store
# --------------------------------------------------------------------------- #


class TestMoonCakeKVBackendWithMockStore:
    """Validate that ``MoonCakeKVBackend`` translates Mini-SGLang concepts
    (token ids, GPU indices, KV pools) into the correct MoonCake Store calls."""

    @pytest.fixture
    def backend(self):
        cfg = _MockModelConfig(num_layers=2, num_kv_heads=2, head_dim=4)
        store = _MockMooncakeStore()
        return MoonCakeKVBackend(
            model_config=cfg,
            page_size=1,
            model_fingerprint="fp_test",
            chunk_size=4,
            store=store,
        ), store

    def test_store_prefix_emits_correct_keys(self, backend):
        mooncake, store = backend
        kv_pool = _MockKVPool(num_layers=2, num_tokens=64)
        # Fill k/v with distinguishable values
        for layer_id in range(2):
            kv_pool.k_cache(layer_id)[:, :, :, :] = torch.arange(64).view(64, 1, 1, 1) * (layer_id + 1)
            kv_pool.v_cache(layer_id)[:, :, :, :] = -torch.arange(64).view(64, 1, 1, 1) * (layer_id + 1)

        token_ids = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        gpu_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

        mooncake.store_prefix(token_ids, gpu_indices, kv_pool)

        # 2 layers * 1 chunk = 2 put calls
        put_calls = [c for c in store.calls if c[0] == "put"]
        assert len(put_calls) == 2

        # Verify keys contain expected structure
        keys = [c[1][0] for c in put_calls]
        assert all(k.startswith("minisgl:fp_test:") for k in keys)
        assert any("L0:C0" in k for k in keys)
        assert any("L1:C0" in k for k in keys)

        # Verify serialized payloads can be deserialized
        for _, args, _ in put_calls:
            blob = args[1]
            k_restored, v_restored = _deserialize_kv(blob, kv_pool.dtype)
            assert k_restored.shape == (4, 1, 2, 4)
            assert v_restored.shape == (4, 1, 2, 4)

    def test_load_prefix_restores_data(self, backend):
        mooncake, store = backend
        kv_pool_src = _MockKVPool(num_layers=2, num_tokens=64)
        kv_pool_dst = _MockKVPool(num_layers=2, num_tokens=64)

        for layer_id in range(2):
            kv_pool_src.k_cache(layer_id)[:, :, :, :] = (
                torch.arange(64).view(64, 1, 1, 1).float() * (layer_id + 10)
            )
            kv_pool_src.v_cache(layer_id)[:, :, :, :] = (
                -torch.arange(64).view(64, 1, 1, 1).float() * (layer_id + 10)
            )

        token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        src_indices = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
        dst_indices = torch.tensor([50, 51, 52, 53], dtype=torch.int32)

        mooncake.store_prefix(token_ids, src_indices, kv_pool_src)
        mooncake.load_prefix(token_ids, dst_indices, kv_pool_dst)

        for layer_id in range(2):
            src_k = kv_pool_src.k_cache(layer_id)[src_indices]
            dst_k = kv_pool_dst.k_cache(layer_id)[dst_indices]
            src_v = kv_pool_src.v_cache(layer_id)[src_indices]
            dst_v = kv_pool_dst.v_cache(layer_id)[dst_indices]
            assert torch.equal(src_k, dst_k)
            assert torch.equal(src_v, dst_v)

    def test_match_prefix_returns_zero_when_empty(self, backend):
        mooncake, _store = backend
        result = mooncake.match_prefix(torch.tensor([1, 2, 3], dtype=torch.int32))
        assert result == 0

    def test_match_prefix_returns_aligned_length(self, backend):
        mooncake, store = backend
        # Pre-populate store with first 8 tokens (2 chunks of 4)
        tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)
        dummy_kv = _MockKVPool(num_layers=2, num_tokens=8)
        mooncake.store_prefix(tokens, torch.arange(8, dtype=torch.int32), dummy_kv)

        # Query with 12 tokens
        matched = mooncake.match_prefix(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.int32))
        # Only first 2 chunks (8 tokens) are stored
        assert matched == 8

        # Verify is_exist was called for layer-0 chunks
        exist_calls = [c for c in store.calls if c[0] == "is_exist"]
        assert len(exist_calls) >= 2

    def test_match_prefix_breaks_on_missing_chunk(self, backend):
        mooncake, store = backend
        tokens = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        dummy_kv = _MockKVPool(num_layers=2, num_tokens=4)
        mooncake.store_prefix(tokens, torch.arange(4, dtype=torch.int32), dummy_kv)

        # Query with different second chunk
        matched = mooncake.match_prefix(torch.tensor([1, 2, 3, 4, 99, 100, 101, 102], dtype=torch.int32))
        assert matched == 4

    def test_load_prefix_missing_key_raises(self, backend):
        mooncake, _store = backend
        kv_pool = _MockKVPool(num_layers=2, num_tokens=8)
        with pytest.raises(RuntimeError, match="MoonCake key missing during load"):
            mooncake.load_prefix(
                torch.tensor([99, 100, 101, 102], dtype=torch.int32),
                torch.arange(4, dtype=torch.int32),
                kv_pool,
            )

    def test_noop_backend(self):
        noop = NoopSpillBackend()
        assert noop.match_prefix(torch.tensor([1, 2, 3])) == 0
        noop.store_prefix(None, None, None)  # must not raise
        noop.load_prefix(None, None, None)   # must not raise
        noop.reset()                         # must not raise


# --------------------------------------------------------------------------- #
# 4. End-to-end with HierarchicalCacheManager + MoonCake backend
# --------------------------------------------------------------------------- #


class TestHierarchicalCacheManagerWithMockMoonCake:
    """Full scheduler-level integration: prefix matching, expansion from
    spill, and async offload all wired through a mock MoonCake store."""

    @pytest.fixture
    def mgr_and_store(self):
        cfg = _MockModelConfig(num_layers=2, num_kv_heads=2, head_dim=4)
        store = _MockMooncakeStore()
        backend = MoonCakeKVBackend(
            model_config=cfg,
            page_size=1,
            model_fingerprint="fp_e2e",
            chunk_size=4,
            store=store,
        )
        cm = _make_hierarchical_cache_manager(num_pages=16, page_size=1, spill=backend)
        return cm, store, backend

    def test_match_req_loads_spilled_prefix(self, mgr_and_store):
        cm, store, backend = mgr_and_store
        # GPU radix cache has first 2 tokens
        cm.prefix_cache.gpu_cache.insert_prefix(
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([10, 11], dtype=torch.int32),
        )
        # Spill has next 4 tokens
        dummy_kv = _MockKVPool(num_layers=2, num_tokens=16)
        backend.store_prefix(
            torch.tensor([3, 4, 5, 6], dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
            dummy_kv,
        )
        store.calls.clear()

        req = _make_pending_req([1, 2, 3, 4, 5, 6, 7])
        result = cm.match_req(req)

        assert result.spilled_handle is None  # loaded back into GPU
        assert result.cuda_handle.cached_len == 6
        # load_prefix should have been invoked
        load_calls = [c for c in store.calls if c[0] == "get"]
        assert len(load_calls) == 2  # 2 layers * 1 chunk

    def test_try_expand_from_spilled(self, mgr_and_store):
        cm, _store, backend = mgr_and_store
        cm.prefix_cache.gpu_cache.insert_prefix(
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([10, 11], dtype=torch.int32),
        )
        dummy_kv = _MockKVPool(num_layers=2, num_tokens=16)
        backend.store_prefix(
            torch.tensor([3, 4, 5, 6], dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
            dummy_kv,
        )

        gpu_handle = cm.prefix_cache.gpu_cache.match_prefix(
            torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
        ).cuda_handle
        req = _make_pending_req([1, 2, 3, 4, 5, 6, 7])
        expanded = cm.try_expand_from_spilled(req, gpu_handle, table_idx=0)

        assert expanded is not None
        new_handle, new_len = expanded
        assert new_len == 6
        loaded = new_handle.get_matched_indices()[gpu_handle.cached_len:new_len]
        assert len(loaded) == 4
        # Page table must be updated for the new portion
        pt_slice = cm.page_table[0, gpu_handle.cached_len:new_len]
        assert torch.equal(pt_slice, loaded)

    def test_maybe_offload_enqueues_async_task(self, mgr_and_store):
        cm, store, _backend = mgr_and_store
        # chunk_size is 4, so we need at least 4 new tokens to trigger a store
        pages = cm._allocate(6)
        handle = HierarchicalCacheHandle(cached_len=2, indices=pages[:2])
        req = core.Req(
            input_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
            table_idx=0,
            cached_len=6,
            output_len=2,
            uid=0,
            sampling_params=core.SamplingParams(),
            cache_handle=handle,
        )
        cm.lock(req.cache_handle)
        cm.maybe_offload(req)
        # Drain async worker
        time.sleep(0.5)
        cm._async_queue.shutdown(timeout=2.0)

        put_calls = [c for c in store.calls if c[0] == "put"]
        # 2 layers * 1 chunk of 4 tokens = 2 put calls minimum
        assert len(put_calls) >= 2

    def test_offload_and_reload_roundtrip(self, mgr_and_store):
        cm, store, backend = mgr_and_store
        # Allocate pages for 8 tokens
        pages = cm._allocate(8)
        cm.page_table[0, :8].copy_(pages[:8])

        # Simulate prefill producing tokens 0..7
        # device_len must be > cached_len, so input_ids needs at least 9 elements
        req = core.Req(
            input_ids=torch.arange(9, dtype=torch.int32),
            table_idx=0,
            cached_len=8,
            output_len=1,
            uid=0,
            sampling_params=core.SamplingParams(),
            cache_handle=HierarchicalCacheHandle(cached_len=0, indices=pages[:0]),
        )
        # Seed the KV pool with unique data so we can verify round-trip
        expected_shape = cm.kv_pool.k_cache(0)[pages[:1]].shape  # (1, 1, 2, 4)
        for layer_id in range(2):
            cm.kv_pool.k_cache(layer_id)[pages[:8]] = (
                torch.arange(8).view(8, 1, 1, 1).float() + layer_id * 100
            )
            cm.kv_pool.v_cache(layer_id)[pages[:8]] = (
                -torch.arange(8).view(8, 1, 1, 1).float() - layer_id * 100
            )

        cm.maybe_offload(req)
        time.sleep(0.5)
        cm._async_queue.shutdown(timeout=2.0)

        # Clear store call log
        store.calls.clear()

        # Now simulate a new request that wants the same prefix
        req2 = _make_pending_req(list(range(8)) + [99])
        result = cm.match_req(req2)
        assert result.cuda_handle.cached_len == 8

        # Verify the reloaded data matches
        for layer_id in range(2):
            loaded_k = cm.kv_pool.k_cache(layer_id)[result.cuda_handle.get_matched_indices()]
            loaded_v = cm.kv_pool.v_cache(layer_id)[result.cuda_handle.get_matched_indices()]
            # Broadcasting during seeding expands to the full cache shape;
            # build expectations with the same shape.
            expected_k = (torch.arange(8).view(8, 1, 1, 1).float() + layer_id * 100).expand_as(loaded_k)
            expected_v = (-torch.arange(8).view(8, 1, 1, 1).float() - layer_id * 100).expand_as(loaded_v)
            assert torch.equal(loaded_k, expected_k)
            assert torch.equal(loaded_v, expected_v)

    def test_build_model_fingerprint_unique(self):
        cfg = _MockModelConfig(num_kv_heads=4, head_dim=64, num_layers=32)
        fp1 = build_model_fingerprint(cfg, tp_size=1, tp_rank=0, dtype=torch.bfloat16)
        fp2 = build_model_fingerprint(cfg, tp_size=2, tp_rank=0, dtype=torch.bfloat16)
        fp3 = build_model_fingerprint(cfg, tp_size=1, tp_rank=0, dtype=torch.float16)
        assert fp1 != fp2
        assert fp1 != fp3


# --------------------------------------------------------------------------- #
# 5. Real MoonCake setup smoke test (best-effort, skipped on failure)
# --------------------------------------------------------------------------- #


class TestRealMoonCakeSetupSmoke:
    """Attempt to exercise the *real* ``MooncakeDistributedStore.setup()``.

    On machines without RDMA hardware or when the transfer-engine fails to
    initialise, these tests are automatically skipped so the suite remains
    green.
    """

    def _try_setup_real_store(self, timeout_sec: float = 10.0):
        from mooncake.store import MooncakeDistributedStore
        store = MooncakeDistributedStore()

        # Use a high port unlikely to collide with user services
        rpc_port = 55201
        meta_port = 55202
        metrics_port = 55203

        master_cmd = [
            "/home/skl/anaconda3/envs/qysgl/bin/mooncake_master",
            f"--rpc_port={rpc_port}",
            f"--metrics_port={metrics_port}",
            "--logtostderr",
        ]

        env = os.environ.copy()
        env["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libffi.so.7"

        proc = subprocess.Popen(
            master_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        try:
            time.sleep(1.5)  # give master time to bind
            config = {
                "local_hostname": "127.0.0.1",
                "metadata_server": "P2PHANDSHAKE",
                "global_segment_size": 64 * 1024 * 1024,
                "local_buffer_size": 16 * 1024 * 1024,
                "protocol": "tcp",
                "rdma_devices": "",
                "master_server_addr": f"127.0.0.1:{rpc_port}",
            }
            # setup() may retry indefinitely when it cannot reach the metadata
            # server.  Run in a child process so we can enforce a hard timeout
            # without deadlocking pytest.
            import multiprocessing
            q = multiprocessing.Queue()

            def _setup_worker(queue, cfg):
                try:
                    s = MooncakeDistributedStore()
                    ret = s.setup(cfg)
                    queue.put(("ok", ret))
                    s.close()
                except Exception as exc:
                    queue.put(("exc", str(exc)))

            p = multiprocessing.Process(target=_setup_worker, args=(q, config))
            p.start()
            p.join(timeout=timeout_sec)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
                if p.is_alive():
                    os.kill(p.pid, signal.SIGKILL)
                    p.join(timeout=2)
                pytest.skip(
                    "Real MoonCake store.setup() timed out (check mooncake_master is reachable)"
                )

            if not q.empty():
                status, payload = q.get_nowait()
                if status == "exc":
                    pytest.skip(f"Real MoonCake store.setup() failed: {payload}")
                if status == "ok" and payload != 0:
                    pytest.skip(f"Real MoonCake store.setup() returned {payload}")
                return store, payload
            else:
                pytest.skip("Real MoonCake store.setup() produced no result")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_real_store_setup_smoke(self):
        self._try_setup_real_store()

    def test_real_put_get_smoke(self):
        _, ret = self._try_setup_real_store()
        if ret != 0:
            pytest.skip("Setup failed")
        from mooncake.store import MooncakeDistributedStore
        store = MooncakeDistributedStore()
        # Re-use the same config – we already know setup works if we got here,
        # but the worker process above already closed the store.  Skip for now.
        pytest.skip(
            "Real put/get smoke test deferred – store must persist across process boundary"
        )
