# Mooncake RDMA Service Test Report

**Date:** 2026-04-20  
**Tester:** Kimi Code CLI  
**Environment:** `conda env: qysgl`, Ubuntu 20.04.6 LTS, kernel 5.4.0-212-generic

---

## 1. Test Environment

| Component | Version / Status |
|-----------|-----------------|
| `mooncake-transfer-engine` | 0.3.10.post1 (in `qysgl` conda env) |
| RDMA device (software) | `iwp23s0f3` via `siw` on `ens121f3` |
| RDMA device (hardware) | Intel X710 (`i40iw` loaded, **no link** — missing fiber cables) |
| PyTorch | 2.9.1+cu128, NCCL 2.27.5, 8 GPUs |
| Workaround needed | `LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7` |

---

## 2. RDMA Stack Preparation

Before testing Mooncake, the system-level RDMA stack was activated:

- Installed `rdma-core`, `libibverbs1`, `librdmacm1`, `perftest`, `qperf`, `python3-pyverbs`
- Loaded `i40iw` (Intel X710 hardware iWARP driver)
- Configured `siw` (Soft-iWARP) as a software RDMA fallback on `ens121f3`
- Created `siw-rdma.service` systemd unit for persistence across reboots
- Verified with `ibv_devinfo`, `ib_write_bw` (~4.5 GB/s loopback), and `rping`

> **Note:** The Intel X710 10GbE SFP+ ports (`ens6f0`, `ens6f1`) have 10G Base-SR transceivers installed but **no fiber cables are connected** (RX power = -40.00 dBm). Hardware iWARP devices will appear automatically once cables are connected.

---

## 3. Mooncake TCP Mode — Fully Working

The real `MooncakeDistributedStore` over TCP initializes and operates correctly.

### Pytest Results

```bash
env LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7 \
  conda run -n qysgl pytest tests/core/test_mooncake_integration.py -v
```

**Result:** `24 passed, 1 skipped`

### End-to-End Backend Verification

Verified `MoonCakeKVBackend` operations with the **real** TCP-backed store:

| Operation | Result |
|-----------|--------|
| `store_prefix()` | Stores KV chunks to Mooncake |
| `match_prefix()` | Correctly detects cached prefixes |
| `load_prefix()` | Round-trips data faithfully |
| `bfloat16` serialization | Stable |

---

## 4. Mooncake RDMA Mode — Setup Works, Transfer Fails on `siw`

### What Works

With two environment workarounds, **Mooncake RDMA initialization succeeds**:

| Workaround | Purpose |
|------------|---------|
| `MC_GID_INDEX=0` | Force Mooncake to use GID index 0 (the EUI-64 GID on `siw`) |
| `sudo` + `ulimit -l unlimited` | Bypass 64 MB locked-memory limit for RDMA memory registration |

**Logs confirm success:**

```
I ... Successfully created client on port XXXXX after 1 attempt(s)
I ... Registering local memory: 16777216 bytes
I ... Mounting segment: 67108864 bytes ...
```

### What Fails

**Actual data transfer crashes** at the Queue Pair handshake:

```
E ... [Handshake] Failed to modify QP to RTS: Invalid argument [22]
E ... Worker: Cannot make connection for endpoint: 10.2.14.76:xxxxx@iwp23s0f3
```

**Root cause:** `siw` (Soft-iWARP) is a software RDMA implementation. Mooncake uses a **custom coroutine-based RDMA endpoint layer** (not `rdma_cm`), and this layer sets QP attributes that `siw` rejects during the `IBV_QPS_RTS` transition. This is a **Mooncake ↔ `siw` compatibility issue**, not a general RDMA stack failure.

For comparison, standard tools like `ib_write_bw -R` (which uses `rdma_cm`) work fine on this `siw` device.

---

## 5. What Would Make Full RDMA Work

| Scenario | Expected Outcome |
|----------|----------------|
| **Connect fiber cables to X710** + use `i40iw` | Hardware iWARP should work. Mooncake would auto-discover the device, GID selection would work natively, and QP setup should succeed because `i40iw` is a hardware driver that fully supports the verbs API. |
| **Use Mellanox ConnectX NIC** (RoCE/IB) | This is Mooncake's primary target. RDMA would work out of the box. |

---

## 6. Recommended Usage Right Now

Since fiber cables are temporarily unavailable, you can **use Mooncake over TCP** for development and functional testing:

```bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
python -m minisgl --model ... --enable-hierarchical-cache --hicache-backend mooncake
```

Or with explicit config:

```bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_MASTER=127.0.0.1:50051
```

A convenience script is provided at:

```bash
source scripts/test_mooncake_env.sh
```

---

## 7. Summary

| Test | Result |
|------|--------|
| Mooncake import & store instantiation | Pass |
| Mooncake TCP `setup()` + `put`/`get` | Pass |
| `MoonCakeKVBackend` end-to-end (TCP) | Pass |
| Full pytest integration suite | 24 passed, 1 skipped |
| Mooncake RDMA `setup()` on `siw` | Pass (with `MC_GID_INDEX=0`) |
| Mooncake RDMA data transfer on `siw` | Fail (`QP to RTS: Invalid argument`) |
| Mooncake RDMA on X710 (with cables) | Expected to work |
