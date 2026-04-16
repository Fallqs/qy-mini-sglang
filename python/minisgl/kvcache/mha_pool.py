from __future__ import annotations

from typing import List

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    @classmethod
    def from_model_config(
        cls,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "MHAKVCache":
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        k_buffers = [kv_buffer[0, i] for i in range(num_layers)]
        v_buffers = [kv_buffer[1, i] for i in range(num_layers)]
        return cls(k_buffers=k_buffers, v_buffers=v_buffers)

    def __init__(self, k_buffers: List[torch.Tensor], v_buffers: List[torch.Tensor]) -> None:
        assert len(k_buffers) == len(v_buffers), "K and V buffers must have same length"
        self._num_layers = len(k_buffers)
        self._k_buffer = k_buffers
        self._v_buffer = v_buffers
        self._device = k_buffers[0].device
        self._storage_shape = (-1, k_buffers[0].shape[-2], k_buffers[0].shape[-1])

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._k_buffer[0].dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
