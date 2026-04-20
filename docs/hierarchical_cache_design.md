# Hierarchical KV Cache Design for Mini-SGLang

## 1. Goals & Motivation

Mini-SGLang already supports **multi-tenant universal shared cache** across heterogeneous models on GPU memory. However, GPU HBM is expensive and limited. The goal is to introduce a **hierarchical cache tier** that transparently extends GPU-resident KV cache into CPU DRAM and SSD via a single external backend — **MoonCake Store**.

This enables:
- **Larger effective cache capacity** → higher prefix hit ratios, especially for long-context / RAG workloads
- **Reduced GPU memory pressure** → more concurrent requests or larger batch sizes
- **Cross-instance cache reuse** → eliminate redundant prefill computation across restarts or nodes
- **Better multi-turn / agentic workload performance** → historical KV caches survive tenant switching

---

## 2. Design Philosophy

The design follows the existing Mini-SGLang principles:
- **Minimal intrusion**: Extend existing abstractions (`BasePrefixCache`, `MatchResult`, `CacheManager`) rather than replacing them
- **Async-by-default**: Data movement must not block the GPU scheduling loop
- **Tenant-aware**: Hierarchical cache is shared across tenants (universal cache), but eviction policies respect per-tenant isolation
- **Optional & pluggable**: Hierarchical caching can be enabled/disabled per deployment; MoonCake Store is an optional dependency
- **One package for all tiers**: MoonCake Store internally manages DRAM → SSD → Remote, so we don't need LMCache or native tier implementations

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Scheduler / Prefill                          │
│                     (unchanged scheduling logic)                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   HierarchicalCacheManager                           │
│  ┌─────────────────┐  ┌─────────────────────────────────────────┐   │
│  │  GPU Prefix     │  │  MoonCake Store (single dependency)     │   │
│  │  (RadixCache)   │◄─┤  - DRAM tier (hot spilled KV)           │   │
│  │  hot working set│  │  - SSD tier (cold spilled KV)           │   │
│  └─────────────────┘  │  - Remote tier (cross-node via RDMA)    │   │
│                       │  - LRU eviction, replication, persistence │   │
│                       └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Async Transfer Queue                            │
│         Background thread for D2MC serialization & put()             │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight**: MoonCake Store itself is a hierarchical storage engine (DRAM/SSD/Remote). We do NOT need LMCache as a middleman, nor do we need to build native CPU/disk tiers from scratch. Our code provides:
1. **HiRadixTree** — token-to-cache-location indexing on top of the existing radix cache
2. **MoonCakeKVBackend** — serializes KV pages and calls MoonCake Store's `put`/`get`
3. **AsyncTransferQueue** — non-blocking offload so the scheduler loop never stalls

---

## 4. Key Abstraction Extensions

### 4.1 Extended `MatchResult`

The current `MatchResult` was extended with a `spilled_handle` field:

```python
class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
    spilled_handle: BaseCacheHandle | None = None
```

This is backward-compatible: existing code that only reads `cuda_handle` continues to work.

### 4.2 `HierarchicalPrefixCache`

A new prefix cache implementation (registered as `"hierarchical"`) wraps the existing `RadixPrefixCache` and adds spill-backend queries:

```python
class HierarchicalPrefixCache(BasePrefixCache):
    def __init__(self, device, page_size=1, spill_backend=None):
        self.gpu_cache = RadixPrefixCache(device=device, page_size=page_size)
        self.spill = spill_backend

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        gpu_result = self.gpu_cache.match_prefix(input_ids)
        gpu_len = gpu_result.cuda_handle.cached_len
        if self.spill is None or gpu_len >= len(input_ids):
            return MatchResult(cuda_handle=gpu_result.cuda_handle)
        # Query spill backend for remaining suffix
        remaining = input_ids[gpu_len:]
        spill_len = self.spill.match_prefix(remaining)
        if spill_len == 0:
            return MatchResult(cuda_handle=gpu_result.cuda_handle)
        spilled = _SpilledCacheHandle(cached_len=spill_len, token_ids=remaining[:spill_len])
        return MatchResult(cuda_handle=gpu_result.cuda_handle, spilled_handle=spilled)
```

### 4.3 `HierarchicalCacheManager`

Replaces `CacheManager` when hierarchical caching is enabled. It orchestrates:
- **Tiered reads**: `match_req()` loads spilled prefixes back into GPU synchronously
- **Tiered writes**: `maybe_offload()` enqueues async D2MC copies after prefill
- **Expansion**: `try_expand_from_spilled()` lets `PrefillAdder` increase `cached_len` after table allocation

---

## 5. MoonCake Store Integration

### 5.1 Why MoonCake Store Alone?

MoonCake Store **already implements** what we would otherwise build:
- **DRAM + SSD tiering** with LRU eviction
- **Distributed pooling** across cluster nodes
- **RDMA / TCP transfer** via MoonCake Transfer Engine
- **Persistence and replication**
- **Python API**: `pip install mooncake-transfer-engine`

### 5.2 Our Thin Adapter: `MoonCakeKVBackend`

Since MoonCake Store exposes a generic blob API (`put(key, bytes)`, `get(key)`), we add a thin LLM-aware layer:

- **Chunking**: KV cache is stored in page-aligned chunks (default 256 tokens)
- **Key format**: `minisgl:{model_fp}:L{layer_id}:C{chunk_idx}:{hash}`
- **Serialization**: Compact binary format using `torch.Tensor` + `struct` headers
- **Model fingerprint**: Derived from `(model_type, num_layers, num_kv_heads, head_dim, tp_size, tp_rank, dtype)` to prevent cross-model cache poisoning

### 5.3 Optional Dependency

MoonCake Store is **optional**. If not installed, the system falls back to `NoopSpillBackend` and behaves exactly like the original radix cache.

---

## 6. Async Transfer Pipeline

To avoid blocking the scheduler loop, all offloads are asynchronous:

```
Scheduler Loop:
  1. match_req() → returns MatchResult (GPU + possibly spilled)
  2. PrefillAdder tries expand_from_spilled() after table allocation
  3. Forward pass computes missing tokens
  4. _process_last_data() calls cache_req() then maybe_offload()
  5. maybe_offload() enqueues D2MC copy to AsyncTransferQueue
  6. Background thread serializes and writes to MoonCake Store
```

The read path is currently **synchronous** (simpler and correct). Async prefetch can be added later without architectural changes.

---

## 7. Multi-Tenant Considerations

The existing `GlobalFineAllocator` and `VirtualKVPool` enable **shared GPU memory** across heterogeneous tenants. For hierarchical cache:

1. **Per-tenant spill backend**: Each tenant gets its own `MoonCakeKVBackend` instance with a unique model fingerprint
2. **No cross-tenant prefix sharing** in the spill backend (enforced by model fingerprint in keys)
3. **Async queue is per-tenant**: Each `TenantUnit` has its own `AsyncTransferQueue`

---

## 8. Configuration

### CLI Flags

```bash
python -m minisgl \
  --model "Qwen/Qwen3-0.6B" \
  --enable-hierarchical-cache \
  --hicache-backend mooncake \
  --hicache-chunk-size 256 \
  --hicache-max-inflight 4 \
  --hicache-config-json '{"local_hostname":"localhost","metadata_server":"http://localhost:8080/metadata","master_server_address":"localhost:50051","protocol":"tcp","global_segment_size":"4gb"}'
```

### Config Fields (SchedulerConfig)

```python
enable_hierarchical_cache: bool = False
hicache_backend: str = "noop"  # "noop" | "mooncake"
hicache_backend_config: dict | None = None
hicache_chunk_size: int = 256
hicache_max_inflight: int = 4
```

---

## 9. File Changes

| File | Change |
|------|--------|
| `python/minisgl/kvcache/base.py` | Add `spilled_handle` to `MatchResult` |
| `python/minisgl/kvcache/hierarchical.py` | New: `HierarchicalPrefixCache`, `HierarchicalCacheHandle` |
| `python/minisgl/kvcache/mooncake_backend.py` | New: `MoonCakeKVBackend`, `BaseSpillBackend`, `NoopSpillBackend`, serialization helpers |
| `python/minisgl/kvcache/async_transfer.py` | New: `AsyncTransferQueue` |
| `python/minisgl/kvcache/__init__.py` | Register `"hierarchical"` cache type; export new classes |
| `python/minisgl/scheduler/cache.py` | New: `HierarchicalCacheManager` with tiered read/write |
| `python/minisgl/scheduler/prefill.py` | Hook `try_expand_from_spilled()` in `PrefillAdder` |
| `python/minisgl/scheduler/scheduler.py` | Wire up `_build_spill_backend()` in `TenantUnit`; call `maybe_offload()` after prefill |
| `python/minisgl/scheduler/config.py` | Add hierarchical cache config fields |
| `python/minisgl/server/args.py` | Add `--enable-hierarchical-cache`, `--hicache-backend`, etc. |
| `tests/core/test_hierarchical_cache.py` | New: comprehensive unit tests |

---

## 10. Performance Considerations

| Operation | Bandwidth | Latency | Strategy |
|-----------|-----------|---------|----------|
| GPU↔CPU (PCIe) | ~64 GB/s | ~10 μs | Synchronous load for small prefixes |
| CPU↔NVMe | ~7 GB/s | ~100 μs | Async offload only |
| RDMA (400 GbE) | ~50 GB/s | ~2 μs | MoonCake Store handles transparently |

**Critical optimizations implemented:**
1. **Async offload**: Scheduler never waits on I/O
2. **Chunking**: 256-token chunks amortize per-key overhead
3. **Write-through**: KV is offloaded immediately after prefill, not on eviction
4. **Layer-wise serialization**: Per-layer keys allow partial load/store

---

## 11. Compatibility

| Feature | Status |
|---------|--------|
| Radix Cache | ✅ Extended naturally — GPU tree unchanged |
| Chunked Prefill | ✅ `TieredMatchResult` guides chunking |
| Tensor Parallelism | ✅ Per-rank spill backend with unique fingerprint |
| CUDA Graphs | ✅ Unaffected — transfers happen outside graph region |
| Overlap Scheduling | ✅ Async queue uses separate thread |
| Multi-Tenant | ✅ Per-tenant backend instances |
| No MoonCake installed | ✅ Falls back to `NoopSpillBackend` |

---

## 12. Summary

This design introduces a **MoonCake-centric hierarchical cache** for Mini-SGLang that:

1. **Uses MoonCake Store as the single backend** for DRAM/SSD/Remote tiers — no native tier reimplementation needed
2. **Extends existing abstractions** with minimal intrusion (~500 lines of new Python code)
3. **Supports async offloading** via `AsyncTransferQueue` to keep the scheduler loop non-blocking
4. **Maintains multi-tenant compatibility** with per-tenant model fingerprinting
5. **Is fully optional** — installs and runs without MoonCake when not needed

The incremental path is: **HierarchicalPrefixCache → MoonCakeKVBackend → AsyncQueue → Scheduler integration**, each independently testable.
