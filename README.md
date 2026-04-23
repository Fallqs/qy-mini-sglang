<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang

A **lightweight yet high-performance** inference framework for Large Language Models.

---

Mini-SGLang is a compact implementation of [SGLang](https://github.com/sgl-project/sglang), designed to demystify the complexities of modern LLM serving systems. With a compact codebase of **~5,000 lines of Python**, it serves as both a capable inference engine and a transparent reference for researchers and developers.

## ✨ Key Features

- **High Performance**: Achieves state-of-the-art throughput and latency with advanced optimizations.
- **Lightweight & Readable**: A clean, modular, and fully type-annotated codebase that is easy to understand and modify.
- **Advanced Optimizations**:
  - **Radix Cache**: Reuses KV cache for shared prefixes across requests.
  - **Chunked Prefill**: Reduces peak memory usage for long-context serving.
  - **Overlap Scheduling**: Hides CPU scheduling overhead with GPU computation.
  - **Tensor Parallelism**: Scales inference across multiple GPUs.
  - **Optimized Kernels**: Integrates **FlashAttention** and **FlashInfer** for maximum efficiency.
  - **Multi-Tenant Engine**: Serve multiple independent models simultaneously on the same GPUs with a shared KV memory pool (tensor-parallelism supported).
  - **Parameter Offloading**: Offload inactive tenant models back to CPU pinned memory and reactivate them on demand.
  - **Layer Offloading**: Keep only a small number of decoder blocks resident on GPU while running the rest through a CPU-backed block cache.
  - ...

## 🚀 Quick Start

> **⚠️ Platform Support**: Mini-SGLang currently supports **Linux only** (x86_64 and aarch64). Windows and macOS are not supported due to dependencies on Linux-specific CUDA kernels (`sgl-kernel`, `flashinfer`). We recommend using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows or Docker for cross-platform compatibility.

### 1. Environment Setup

We recommend using `uv` for a fast and reliable installation (note that `uv` does not conflict with `conda`).

```bash
# Create a virtual environment (Python 3.10+ recommended)
uv venv --python=3.12
source .venv/bin/activate
```

**Prerequisites**: Mini-SGLang relies on CUDA kernels that are JIT-compiled. Ensure you have the **NVIDIA CUDA Toolkit** installed and that its version matches your driver's version. You can check your driver's CUDA capability with `nvidia-smi`.

### 2. Installation

Install Mini-SGLang directly from the source:

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

<details>
<summary><b>💡 Installing on Windows (WSL2)</b></summary>

Since Mini-SGLang requires Linux-specific dependencies, Windows users should use WSL2:

1. **Install WSL2** (if not already installed):
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install
   ```

2. **Install CUDA on WSL2**:
   - Follow [NVIDIA's WSL2 CUDA guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - Ensure your Windows GPU drivers support WSL2

3. **Install Mini-SGLang in WSL2**:
   ```bash
   # Inside WSL2 terminal
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **Access from Windows**: The server will be accessible at `http://localhost:8000` from Windows browsers and applications.

</details>

<details>
<summary><b>🐳 Running with Docker</b></summary>

**Prerequisites**:
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **Build the Docker image**:
   ```bash
   docker build -t minisgl .
   ```

2. **Run the server**:
   ```bash
   docker run --gpus all -p 1919:1919 \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **Run in interactive shell mode**:
   ```bash
   docker run -it --gpus all \
       minisgl --model Qwen/Qwen3-0.6B --shell
   ```

4. **Using Docker Volumes for persistent caches** (recommended for faster subsequent startups):
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

</details>

### 3. Online Serving

Launch an OpenAI-compatible API server with a single command.

```bash
# Deploy Qwen/Qwen3-0.6B on a single GPU
python -m minisgl --model "Qwen/Qwen3-0.6B"

# Deploy meta-llama/Llama-3.1-70B-Instruct on 4 GPUs with Tensor Parallelism, on port 30000
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

#### Multi-Tenant Serving

Serve multiple models on the same GPU cluster by registering additional "tenants" with `--extra-model`. Each tenant gets its own independent KV cache while sharing the underlying GPU memory pool.

```bash
# Serve Llama-3.2-1B (default) and Qwen3-8B on 4 GPUs
python -m minisgl \
  --model "meta-llama/Llama-3.2-1B-Instruct" \
  --extra-model qwen="Qwen/Qwen3-8B" \
  --tp 4
```

When multi-tenant mode is active, the OpenAI-compatible API uses the standard `model` field to select the tenant (e.g., `"qwen"` or `"default"`). If the provided value does not match a registered tenant, the default model is used.

Once the server is running, you can send requests using standard tools like `curl` or any OpenAI-compatible client.

#### Multi-Tenant Parameter Offloading

Mini-SGLang can also offload inactive tenant models back to CPU memory. This is useful when one engine hosts multiple tenants but only a small number of them should stay resident on GPU at the same time.

```bash
python -m minisgl \
  --model "/data/vlm/jlk/qyinfra/Qwen3-0.6B" \
  --extra-model qwen="/data/vlm/jlk/qyinfra/Qwen3-0.6B" \
  --enable-parameter-offloading \
  --offload-idle-seconds 0 \
  --max-active-models 1
```

With this configuration:
- inactive tenants are eligible for whole-model offloading;
- at most one tenant model stays active on GPU at a time.

#### Layer Offloading

For decoder-only models with a standard `model.layers` layout, Mini-SGLang can further reduce peak model residency by keeping only a small number of decoder blocks on GPU.

```bash
python -m minisgl \
  --model "/data/vlm/jlk/qyinfra/Qwen3-0.6B" \
  --extra-model qwen="/data/vlm/jlk/qyinfra/Qwen3-0.6B" \
  --enable-parameter-offloading \
  --offload-idle-seconds 0 \
  --max-active-models 1 \
  --enable-layer-offloading \
  --max-resident-blocks 1
```

Notes:
- `--enable-layer-offloading` currently disables CUDA graph capture for that tenant path.
- Shell mode does not support `--dummy-weight`.
- The OpenAI-compatible API is often more convenient than shell mode for repeatable multi-tenant tests.

### 4. Interactive Shell

Chat with your model directly in the terminal by adding the `--shell` flag.

```bash
python -u -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

You can also use `/reset` to clear the chat history, or switch between loaded models with `/model:<name>` when running in multi-tenant mode.

```bash
# Interactive multi-tenant shell
python -u -m minisgl \
  --model "meta-llama/Llama-3.2-1B-Instruct" \
  --extra-model qwen="Qwen/Qwen3-8B" \
  --shell

# Inside the shell:
#   [default] $ Hello!
#   /model:qwen
#   [qwen] $ 你好!
```

For multi-tenant offloading tests, a typical shell command is:

```bash
python -u -m minisgl \
  --model "/data/vlm/jlk/qyinfra/Qwen3-0.6B" \
  --extra-model qwen="/data/vlm/jlk/qyinfra/Qwen3-0.6B" \
  --shell \
  --enable-parameter-offloading \
  --offload-idle-seconds 0 \
  --max-active-models 1 \
  --enable-layer-offloading \
  --max-resident-blocks 1
```

And the equivalent API test flow is:

```bash
curl -N http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32,
    "temperature": 0
  }'

curl -N http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

## Benchmark

### Offline inference

See [bench.py](./benchmark/offline/bench.py) for more details. Set `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` for ablation study on overlap scheduling.

Test Configuration:

- Hardware: 1xH200 GPU.
- Model: Qwen3-0.6B, Qwen3-14B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100-1024 tokens
- Output Length: Randomly sampled between 100-1024 tokens

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### Online inference

See [benchmark_qwen.py](./benchmark/online/bench_qwen.py) for more details.

Test Configuration:

- Hardware: 4xH200 GPU, connected by NVLink.
- Model: Qwen3-32B
- Dataset: [Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl), replaying first 1000 requests.

Launch command:

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **Note**: If you encounter network issues when downloading models from HuggingFace, try using `--model-source modelscope` to download from ModelScope instead:
> ```bash
> python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 Learn More

- **[Detailed Features](./docs/features.md)**: Explore all available features and command-line arguments.
- **[System Architecture](./docs/structures.md)**: Dive deep into the design and data flow of Mini-SGLang.
