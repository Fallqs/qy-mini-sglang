from __future__ import annotations

import functools
import pathlib
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


def _get_nccl_lib_path() -> str | None:
    """Find the NCCL library path from nvidia-nccl-cu12 package."""
    try:
        import nvidia.nccl

        # nvidia.nccl is a namespace package, use __path__ instead of __file__
        nccl_pkg_path = pathlib.Path(nvidia.nccl.__path__[0])
        nccl_lib = nccl_pkg_path / "lib"
        if (nccl_lib / "libnccl.so.2").exists():
            return str(nccl_lib)
    except (ImportError, IndexError):
        pass

    # Fallback: check common locations
    for path in [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
    ]:
        if pathlib.Path(path).joinpath("libnccl.so").exists() or \
           pathlib.Path(path).joinpath("libnccl.so.2").exists():
            return path

    return None


@functools.cache
def _load_nccl_module() -> Module:
    nccl_lib_path = _get_nccl_lib_path()
    extra_ldflags = []
    if nccl_lib_path:
        # Use full path to the library since libnccl.so may not exist (only libnccl.so.2)
        nccl_lib = pathlib.Path(nccl_lib_path) / "libnccl.so.2"
        if nccl_lib.exists():
            extra_ldflags.append(str(nccl_lib))
        else:
            # Fallback to -L/-l if full path doesn't work
            extra_ldflags.append(f"-L{nccl_lib_path}")
            extra_ldflags.append("-lnccl")
    else:
        extra_ldflags.append("-lnccl")
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=extra_ldflags)


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
