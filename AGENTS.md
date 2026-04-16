# AGENTS.md - Mini-SGLang Development Guide

This document provides essential information for AI coding agents working on the Mini-SGLang project.

## Project Overview

Mini-SGLang is a **lightweight yet high-performance** inference framework for Large Language Models. It is a compact implementation of [SGLang](https://github.com/sgl-project/sglang), designed to demystify the complexities of modern LLM serving systems. With approximately **5,000 lines of Python**, it serves as both a capable inference engine and a transparent reference for researchers and developers.

### Key Features

- **High Performance**: State-of-the-art throughput and latency with advanced optimizations
- **Lightweight & Readable**: Clean, modular, and fully type-annotated codebase
- **Advanced Optimizations**:
  - **Radix Cache**: Reuses KV cache for shared prefixes across requests
  - **Chunked Prefill**: Reduces peak memory usage for long-context serving
  - **Overlap Scheduling**: Hides CPU scheduling overhead with GPU computation
  - **Tensor Parallelism**: Scales inference across multiple GPUs
  - **Optimized Kernels**: FlashAttention and FlashInfer integration

### Platform Support

- **Supported**: Linux only (x86_64 and aarch64)
- **Not Supported**: Windows and macOS (due to Linux-specific CUDA kernels)
- **Workaround**: Use WSL2 on Windows or Docker for cross-platform compatibility

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10+ |
| **ML Framework** | PyTorch (<2.10.0) |
| **Model Loading** | Transformers (4.56.0 - 4.57.3), Accelerate |
| **Attention Kernels** | FlashAttention, FlashInfer, TensorRT-LLM fmha |
| **Communication** | ZeroMQ (ZMQ), NCCL (via torch.distributed) |
| **API Server** | FastAPI, Uvicorn |
| **Custom Kernels** | CUDA, TVM-FFI for Python bindings |
| **Optional** | ModelScope (for model downloading in China) |

### Key Dependencies

```
torch<2.10.0
transformers>=4.56.0,<=4.57.3
flashinfer-python>=0.5.3
apache-tvm-ffi>=0.1.4
sgl_kernel>=0.3.17.post1
quack-kernels
pyzmq, uvicorn, fastapi
```

## Project Structure

```
mini-sglang/
├── python/minisgl/          # Main source code
│   ├── __main__.py          # Entry point: python -m minisgl
│   ├── core.py              # Core dataclasses (Req, Batch, Context, SamplingParams)
│   ├── server/              # Server and API components
│   │   ├── args.py          # CLI argument parsing
│   │   ├── launch.py        # Server process launcher
│   │   └── api_server.py    # FastAPI OpenAI-compatible API
│   ├── scheduler/           # Request scheduling logic
│   │   ├── scheduler.py     # Main scheduler implementation
│   │   ├── config.py        # Scheduler configuration
│   │   ├── prefill.py       # Prefill phase handling
│   │   └── decode.py        # Decode phase handling
│   ├── engine/              # Inference engine
│   │   ├── engine.py        # Core engine (TP worker)
│   │   ├── config.py        # Engine configuration
│   │   └── graph.py         # CUDA graph management
│   ├── models/              # LLM model implementations
│   │   ├── llama.py         # Llama-3 architecture
│   │   ├── qwen3.py         # Qwen-3 architecture
│   │   ├── qwen2.py         # Qwen-2.5 architecture
│   │   └── mistral.py       # Mistral architecture
│   ├── layers/              # Neural network layers
│   │   ├── attention.py     # Attention layers
│   │   ├── linear.py        # Linear layers with TP
│   │   ├── norm.py          # Normalization layers
│   │   └── moe.py           # MoE layers
│   ├── attention/           # Attention backends
│   │   ├── fa.py            # FlashAttention backend
│   │   ├── fi.py            # FlashInfer backend
│   │   └── trtllm.py        # TensorRT-LLM backend
│   ├── kvcache/             # KV cache management
│   │   ├── radix_cache.py   # Radix Cache implementation
│   │   └── naive_cache.py   # Naive cache strategy
│   ├── distributed/         # Tensor parallelism
│   ├── message/             # Inter-process messaging (ZMQ)
│   ├── tokenizer/           # Tokenization workers
│   ├── kernel/              # Custom CUDA kernels
│   │   └── csrc/            # C++/CUDA source files
│   ├── moe/                 # Mixture of Experts
│   └── utils/               # Utility functions
├── tests/                   # Test suite
├── benchmark/               # Benchmarking scripts
├── docs/                    # Documentation
├── pyproject.toml           # Python project configuration
└── Dockerfile               # Docker image definition
```

## System Architecture

Mini-SGLang is designed as a **distributed system** with multiple independent processes:

### Components

1. **API Server** (`api_server.py`): Entry point providing OpenAI-compatible API (`/v1/chat/completions`)
2. **Tokenizer Worker** (`tokenizer/`): Converts text to tokens (can be shared with detokenizer)
3. **Detokenizer Worker** (`tokenizer/`): Converts tokens back to text
4. **Scheduler Workers** (`scheduler/`): One per GPU (TP Rank). Manages computation and resource allocation

### Communication

- **Control Messages**: ZeroMQ (ZMQ)
- **Heavy Tensor Data**: NCCL (via `torch.distributed`)

### Request Lifecycle

1. User sends request to **API Server**
2. **API Server** forwards to **Tokenizer**
3. **Tokenizer** converts text to tokens, sends to **Scheduler (Rank 0)**
4. **Scheduler (Rank 0)** broadcasts to all other Schedulers (if multi-GPU)
5. **All Schedulers** schedule request and trigger local **Engine**
6. **Scheduler (Rank 0)** collects output token, sends to **Detokenizer**
7. **Detokenizer** converts token to text, returns to **API Server**
8. **API Server** streams result to User

## Build and Installation

### Prerequisites

- NVIDIA CUDA Toolkit (matching driver version)
- Python 3.10+
- Linux environment

### Installation

```bash
# Clone repository
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# Create virtual environment with uv (recommended)
uv venv --python=3.12
source .venv/bin/activate

# Install in editable mode
uv pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
uv pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

## Running the Server

### Basic Usage

```bash
# Single GPU
python -m minisgl --model "Qwen/Qwen3-0.6B"

# Tensor Parallelism (4 GPUs)
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000

# Interactive Shell Mode
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

### Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` / `--model-path` | Model path (local or HuggingFace) | Required |
| `--tp` / `--tensor-parallel-size` | Tensor parallelism size | 1 |
| `--dtype` | Data type (auto/float16/bfloat16/float32) | auto |
| `--max-running-requests` | Max concurrent requests | (from config) |
| `--attention-backend` / `--attn` | Attention backend (fa/fi/trtllm) | auto |
| `--cache-type` | KV cache strategy (radix/naive) | radix |
| `--cuda-graph-max-bs` | Max batch size for CUDA graph | auto |
| `--max-prefill-length` | Chunk prefill size | (from config) |
| `--host` / `--port` | Server binding | 127.0.0.1:1919 |

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=minisgl --cov-report=html

# Run specific test
pytest tests/core/test_scheduler.py -v
```

### Test Structure

```
tests/
├── core/               # Core functionality tests
│   ├── test_cache_allocate.py
│   └── test_scheduler.py
├── kernel/             # Kernel tests
│   ├── test_comm.py
│   ├── test_index.py
│   ├── test_store.py
│   └── test_tensor.py
└── misc/               # Miscellaneous tests
    └── test_serialize.py
```

## Code Style Guidelines

### Formatting

- **Black**: Line length 100, target Python 3.10+
- **Ruff**: Import sorting, pycodestyle warnings, flake8-comprehensions

### Type Checking

- **MyPy**: Strict type checking enabled
- All function definitions should have type annotations
- Use `from __future__ import annotations` for forward references

### Pre-commit Hooks

```yaml
# Hooks configured in .pre-commit-config.yaml
- trailing-whitespace
- end-of-file-fixer
- check-yaml, check-toml, check-ast
- check-added-large-files
- detect-private-key
- debug-statements
- black (formatting)
- ruff (linting)
- clang-format (C++/CUDA)
```

### Coding Conventions

1. **Always use type annotations** - The codebase is fully type-annotated
2. **Use dataclasses** for data containers (`@dataclass`)
3. **Use `from __future__ import annotations`** at the top of files
4. **Logger initialization**: Use `init_logger(__name__)` from `minisgl.utils`
5. **Tensor parallelism awareness**: Use `tp_info.is_primary()` for rank-0 only operations
6. **Import style**: Absolute imports preferred, organized by ruff

### Example Code Pattern

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend

from minisgl.utils import init_logger

logger = init_logger(__name__)


@dataclass
class ExampleClass:
    """Docstring with clear description."""
    field1: torch.Tensor
    field2: int
    
    def method(self) -> bool:
        """Return something useful."""
        return True
```

## Docker Deployment

### Build Image

```bash
docker build -t minisgl .
```

### Run Server

```bash
# Basic run
docker run --gpus all -p 1919:1919 \
    minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0

# With persistent caches
docker run --gpus all -p 1919:1919 \
    -v huggingface_cache:/app/.cache/huggingface \
    -v tvm_cache:/app/.cache/tvm-ffi \
    -v flashinfer_cache:/app/.cache/flashinfer \
    minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0

# Interactive shell
docker run -it --gpus all \
    minisgl --model Qwen/Qwen3-0.6B --shell
```

## Benchmarking

### Offline Benchmark

```bash
# See benchmark/offline/bench.py
python benchmark/offline/bench.py

# Disable overlap scheduling for ablation study
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python benchmark/offline/bench.py
```

### Online Benchmark

```bash
# See benchmark/online/bench_qwen.py
# Requires running server first
python benchmark/online/bench_qwen.py
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `LOG_PID` | Include PID in logs (1/true/yes) |
| `MINISGL_DISABLE_OVERLAP_SCHEDULING` | Disable overlap scheduling |
| `HF_HOME` | HuggingFace cache directory |
| `TVM_FFI_CACHE_DIR` | TVM-FFI cache directory |
| `FLASHINFER_WORKSPACE_BASE` | FlashInfer workspace directory |

## Key Modules Reference

### Core Classes

- `minisgl.core.Req`: Represents a single inference request
- `minisgl.core.Batch`: A batch of requests in prefill or decode phase
- `minisgl.core.Context`: Global state holder for inference context
- `minisgl.core.SamplingParams`: Sampling parameters (temperature, top_p, etc.)

### Engine & Scheduling

- `minisgl.engine.Engine`: Core inference engine (one per GPU/TP rank)
- `minisgl.scheduler.Scheduler`: Manages request scheduling and lifecycle
- `minisgl.scheduler.SchedulerConfig`: Configuration for scheduler

### Model Support

- `minisgl.models`: Model architectures (Llama, Qwen2/3, Mistral)
- `minisgl.layers`: Building blocks (attention, linear, norm, embedding)
- `minisgl.attention`: Attention backends (FlashAttention, FlashInfer, TRT-LLM)

### Cache Management

- `minisgl.kvcache.RadixCacheManager`: Radix Cache for prefix sharing
- `minisgl.kvcache.NaiveCacheManager`: Simple cache management
- `minisgl.kvcache.MHAKVCache`: Multi-head attention KV cache pool

### Communication

- `minisgl.message`: Message types for ZMQ communication
- `minisgl.distributed`: Tensor parallelism utilities

## Security Considerations

1. **Model Sources**: Supports HuggingFace and ModelScope for model downloads
2. **Docker**: Runs as non-root user (`minisgl`, UID 1001) in production
3. **Private Keys**: Pre-commit hook detects private keys in commits
4. **Network**: API server binds to localhost by default; use `--host 0.0.0.0` for remote access

## Troubleshooting

### Common Issues

1. **CUDA Kernels**: Ensure CUDA toolkit version matches driver version
2. **Model Loading**: Use `--model-source modelscope` for network issues in China
3. **Memory**: Adjust `--memory-ratio` or `--max-running-requests` for OOM issues
4. **Windows**: Use WSL2 or Docker (native Windows not supported)

### Debug Logging

```bash
LOG_LEVEL=DEBUG python -m minisgl --model "Qwen/Qwen3-0.6B"
```

## Contributing Notes

- All code should be type-annotated
- Run `pre-commit run --all-files` before committing
- Tests should use `pytest` framework
- Follow existing module structure for new features
- Update documentation in `docs/` for significant changes
