from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from minisgl.utils import init_logger

if TYPE_CHECKING:
    from minisgl.kvcache.mooncake_backend import BaseSpillBackend

logger = init_logger(__name__)


@dataclass
class _OffloadTask:
    token_ids: torch.Tensor
    gpu_indices: torch.Tensor
    backend: BaseSpillBackend
    kv_pool: object  # BaseKVCachePool


class AsyncTransferQueue:
    """
    Background queue for asynchronous KV cache offloading.

    The scheduler thread enqueues offload tasks; a dedicated worker thread
    performs the (potentially slow) D2H serialization and store.put().
    """

    def __init__(self, max_inflight: int = 4):
        self._queue: deque[_OffloadTask] = deque()
        self._inflight = 0
        self._max_inflight = max_inflight
        self._shutdown = False
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._cv = threading.Condition()
        self._worker.start()

    def enqueue(
        self,
        token_ids: torch.Tensor,
        gpu_indices: torch.Tensor,
        backend: BaseSpillBackend,
        kv_pool: object,
    ) -> None:
        """Schedule an async offload of the given KV pages."""
        if len(gpu_indices) == 0:
            return
        with self._cv:
            self._queue.append(
                _OffloadTask(
                    token_ids=token_ids.cpu().clone(),
                    gpu_indices=gpu_indices.cpu().clone(),
                    backend=backend,
                    kv_pool=kv_pool,
                )
            )
            self._cv.notify()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._shutdown = True
        with self._cv:
            self._cv.notify_all()
        self._worker.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._shutdown:
            with self._cv:
                while not self._queue and not self._shutdown:
                    self._cv.wait(timeout=0.1)
                if self._shutdown:
                    break
                task = self._queue.popleft()
                self._inflight += 1

            try:
                start = time.monotonic()
                task.backend.store_prefix(
                    task.token_ids, task.gpu_indices, task.kv_pool
                )
                elapsed = time.monotonic() - start
                logger.debug(
                    f"Async offload finished: {len(task.token_ids)} tokens "
                    f"in {elapsed:.3f}s"
                )
            except Exception:
                logger.error("Async offload failed:\n" + traceback.format_exc())
            finally:
                with self._cv:
                    self._inflight -= 1
                    self._cv.notify()
