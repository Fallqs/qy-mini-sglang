from __future__ import annotations

import hashlib
import io
import struct
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from minisgl.utils import div_ceil, init_logger

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

logger = init_logger(__name__)

# --------------------------------------------------------------------------- #
# Optional mooncake import
# --------------------------------------------------------------------------- #

_MooncakeStore = None


def _get_mooncake_store_cls():
    global _MooncakeStore
    if _MooncakeStore is not None:
        return _MooncakeStore
    try:
        from mooncake.store import MooncakeDistributedStore as _MooncakeStore
    except Exception as exc:
        logger.debug(f"mooncake.store not available: {exc}")
        _MooncakeStore = False  # type: ignore[assignment]
    return _MooncakeStore


def mooncake_available() -> bool:
    return _get_mooncake_store_cls() not in (False, None)


# --------------------------------------------------------------------------- #
# Abstract spill backend
# --------------------------------------------------------------------------- #


class BaseSpillBackend(ABC):
    """Interface for a tier-2/3 KV cache store (CPU/SSD/Remote)."""

    @abstractmethod
    def match_prefix(self, token_ids: torch.Tensor) -> int:
        """Return the length (in tokens) of the longest cached prefix of *token_ids*."""

    @abstractmethod
    def load_prefix(
        self, token_ids: torch.Tensor, gpu_indices: torch.Tensor, kv_pool
    ) -> None:
        """Load KV cache for *token_ids* into the given *gpu_indices*."""

    @abstractmethod
    def store_prefix(
        self, token_ids: torch.Tensor, gpu_indices: torch.Tensor, kv_pool
    ) -> None:
        """Store KV cache for *token_ids* from the given *gpu_indices*."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored entries."""


# --------------------------------------------------------------------------- #
# No-op fallback
# --------------------------------------------------------------------------- #


class NoopSpillBackend(BaseSpillBackend):
    """Used when hierarchical caching is disabled or mooncake is not installed."""

    def match_prefix(self, token_ids: torch.Tensor) -> int:
        return 0

    def load_prefix(
        self, token_ids: torch.Tensor, gpu_indices: torch.Tensor, kv_pool
    ) -> None:
        pass

    def store_prefix(
        self, token_ids: torch.Tensor, gpu_indices: torch.Tensor, kv_pool
    ) -> None:
        pass

    def reset(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# MoonCake-backed spill backend
# --------------------------------------------------------------------------- #


class MoonCakeKVBackend(BaseSpillBackend):
    """
    Chunk-based KV cache spill backend backed by MoonCake Store.

    KV data is stored at page-aligned chunk granularity.  Each chunk key is:
        minisgl:{model_fp}:L{layer_id}:C{chunk_idx}:{hash}

    This keeps individual objects small enough for efficient random access
    while amortising per-key overhead.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        page_size: int,
        model_fingerprint: str,
        chunk_size: int = 256,
        store=None,
    ):
        self.model_config = model_config
        self.page_size = page_size
        self.model_fp = model_fingerprint
        self.chunk_size = chunk_size
        self._num_layers = model_config.num_layers
        self._store = store

        if self._store is None:
            cls = _get_mooncake_store_cls()
            if cls in (False, None):
                raise RuntimeError(
                    "mooncake-transfer-engine is not installed; "
                    "install it with: pip install mooncake-transfer-engine"
                )
            self._store = cls()
            logger.info("MoonCakeKVBackend initialized (no store.setup() called yet).")

    def setup(self, **kwargs) -> int:
        """Forward setup() to the underlying MoonCake store."""
        return self._store.setup(**kwargs)

    # ------------------------------------------------------------------ #
    # Key helpers
    # ------------------------------------------------------------------ #

    def _chunk_key(self, layer_id: int, chunk_idx: int, token_ids: torch.Tensor) -> str:
        """Deterministic key for a single chunk."""
        data = token_ids.cpu().numpy().tobytes()
        h = hashlib.blake2b(data, digest_size=8).hexdigest()
        return f"minisgl:{self.model_fp}:L{layer_id}:C{chunk_idx}:{h}"

    def _iter_chunks(self, token_ids: torch.Tensor):
        """Yield (chunk_idx, chunk_tokens) for *token_ids* aligned to chunk_size."""
        n = len(token_ids)
        aligned = (n // self.chunk_size) * self.chunk_size
        for i in range(0, aligned, self.chunk_size):
            yield i // self.chunk_size, token_ids[i : i + self.chunk_size]

    # ------------------------------------------------------------------ #
    # Metadata-only prefix match
    # ------------------------------------------------------------------ #

    def match_prefix(self, token_ids: torch.Tensor) -> int:
        """Return the number of leading tokens that are stored in MoonCake."""
        matched = 0
        for chunk_idx, chunk_tokens in self._iter_chunks(token_ids):
            layer_key = self._chunk_key(0, chunk_idx, chunk_tokens)
            if self._store.is_exist(layer_key) <= 0:
                break
            # Verify all layers exist for this chunk
            all_exist = all(
                self._store.is_exist(self._chunk_key(layer_id, chunk_idx, chunk_tokens)) > 0
                for layer_id in range(self._num_layers)
            )
            if not all_exist:
                break
            matched += len(chunk_tokens)
        return matched

    # ------------------------------------------------------------------ #
    # Load / Store
    # ------------------------------------------------------------------ #

    def load_prefix(
        self, token_ids: torch.Tensor, gpu_indices: torch.Tensor, kv_pool
    ) -> None:
        """Synchronous load from MoonCake into GPU pages."""
        assert len(token_ids) == len(gpu_indices)
        device = kv_pool.device
        dtype = kv_pool.dtype

        for layer_id in range(self._num_layers):
            k_dst = kv_pool.k_cache(layer_id)
            v_dst = kv_pool.v_cache(layer_id)
            offset = 0
            for chunk_idx, chunk_tokens in self._iter_chunks(token_ids):
                key = self._chunk_key(layer_id, chunk_idx, chunk_tokens)
                raw = self._store.get(key)
                if not raw:
                    raise RuntimeError(f"MoonCake key missing during load: {key}")
                k_cpu, v_cpu = _deserialize_kv(raw, dtype)
                chunk_len = len(chunk_tokens)
                idx = gpu_indices[offset : offset + chunk_len].long().to(device)
                k_dst.index_copy_(0, idx, k_cpu.to(device))
                v_dst.index_copy_(0, idx, v_cpu.to(device))
                offset += chunk_len

    def store_prefix(
        self, token_ids: torch.Tensor, gpu_indices: torch.Tensor, kv_pool
    ) -> None:
        """Synchronous store from GPU pages into MoonCake."""
        assert len(token_ids) == len(gpu_indices)
        device = kv_pool.device

        for layer_id in range(self._num_layers):
            k_src = kv_pool.k_cache(layer_id)
            v_src = kv_pool.v_cache(layer_id)
            offset = 0
            for chunk_idx, chunk_tokens in self._iter_chunks(token_ids):
                chunk_len = len(chunk_tokens)
                idx = gpu_indices[offset : offset + chunk_len].long().to(device)
                k_cpu = k_src.index_select(0, idx).cpu()
                v_cpu = v_src.index_select(0, idx).cpu()
                key = self._chunk_key(layer_id, chunk_idx, chunk_tokens)
                self._store.put(key, _serialize_kv(k_cpu, v_cpu))
                offset += chunk_len

    def reset(self) -> None:
        # MoonCake Store does not expose a global clear API; eviction is
        # handled internally by the store backend.  For explicit cleanup
        # one would track keys and delete them individually.
        pass


# --------------------------------------------------------------------------- #
# Serialization helpers (stable, versioned)
# --------------------------------------------------------------------------- #

_FMT_VERSION = 1
_HEADER = struct.Struct("<B Q")  # version, num_bytes


def _serialize_kv(k: torch.Tensor, v: torch.Tensor) -> bytes:
    """Serialize a (K, V) pair into a compact byte blob."""
    buf = io.BytesIO()
    # bfloat16 is not directly convertible to numpy; view as uint16
    if k.dtype == torch.bfloat16:
        k_np = k.cpu().contiguous().view(torch.uint16).numpy()
        v_np = v.cpu().contiguous().view(torch.uint16).numpy()
    else:
        k_np = k.cpu().contiguous().numpy()
        v_np = v.cpu().contiguous().numpy()
    k_bytes = k_np.tobytes()
    v_bytes = v_np.tobytes()
    buf.write(struct.pack("<B", _FMT_VERSION))
    buf.write(struct.pack("<QQ", len(k_bytes), len(v_bytes)))
    buf.write(k_bytes)
    buf.write(v_bytes)
    # Store full shape for faithful reconstruction
    buf.write(struct.pack("<Q", len(k_np.shape)))
    for dim in k_np.shape:
        buf.write(struct.pack("<Q", dim))
    return buf.getvalue()


def _deserialize_kv(data: bytes, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    buf = io.BytesIO(data)
    version = struct.unpack("<B", buf.read(1))[0]
    if version != _FMT_VERSION:
        raise ValueError(f"Unsupported KV serialization version: {version}")
    k_len, v_len = struct.unpack("<QQ", buf.read(16))
    k_bytes = buf.read(k_len)
    v_bytes = buf.read(v_len)
    # Reconstruct shape
    ndim = struct.unpack("<Q", buf.read(8))[0]
    shape = tuple(struct.unpack("<Q", buf.read(8))[0] for _ in range(ndim))
    # Reconstruct tensors
    if dtype == torch.bfloat16:
        k = torch.frombuffer(bytearray(k_bytes), dtype=torch.uint16).view(torch.bfloat16).clone().reshape(shape)
        v = torch.frombuffer(bytearray(v_bytes), dtype=torch.uint16).view(torch.bfloat16).clone().reshape(shape)
    else:
        k = torch.frombuffer(bytearray(k_bytes), dtype=dtype).clone().reshape(shape)
        v = torch.frombuffer(bytearray(v_bytes), dtype=dtype).clone().reshape(shape)
    return k, v


def build_model_fingerprint(
    model_config: ModelConfig,
    tp_size: int,
    tp_rank: int,
    dtype: torch.dtype,
) -> str:
    """Unique identifier for a model+TP configuration."""
    parts = [
        model_config.model_type,
        str(model_config.num_layers),
        str(model_config.num_kv_heads),
        str(model_config.head_dim),
        str(tp_size),
        str(tp_rank),
        str(dtype),
    ]
    raw = "|".join(parts).encode()
    return hashlib.blake2b(raw, digest_size=8).hexdigest()
