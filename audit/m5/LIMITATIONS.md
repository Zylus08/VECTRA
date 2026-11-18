# Project VECTRA — Milestone M5: Limitations & Future Engineering Frontiers

**Document Purpose:** Honest, rigorous accounting of architectural boundaries, unoptimized subsystems, and technical debt remaining after Milestone M5.

---

## 1. Identified Limitations in M5

While Milestone M5 successfully recovered prefill throughput ($22-25\text{ tok/s}$) and accelerated decode throughput to $19.3-20.5\text{ tok/s}$, the following engineering boundaries remain:

### 1. Prefill Throughput vs L1-Cache Tiled GEMM
- **Current Architecture:** Batched GEMV-N streams quantized weights from DDR5 RAM once per prompt.
- **Limitation:** For long context windows ($B \ge 128$ tokens), memory bandwidth continues to cap throughput around $25-30\text{ tok/s}$.
- **Future Direction (M6):** Implement L1-cache register-tiled GEMM ($16 \times 16$ micro-tiles) to keep weight blocks resident in the $48\text{ KB}$ L1 data cache across token tiles, which can achieve $60-80\text{ tok/s}$ prefill.

### 2. Lack of Explicit Thread Affinity Pinning on Windows
- **Current Architecture:** Relies on Windows NT OS scheduler across Alder Lake's heterogeneous core topology (6 Golden Cove P-cores + 4 Gracemont E-cores).
- **Limitation:** At thread counts $> 10$, Windows schedules threads across SMT logical cores, causing pipeline conflicts and dropping throughput to $8.3\text{ tok/s}$.
- **Future Direction:** Introduce explicit Win32 `SetThreadAffinityMask` or `SetThreadGroupAffinity` to pin worker threads strictly to high-IPC Golden Cove P-cores without SMT sharing.

### 3. Sampling Capabilities
- **Current Architecture:** Supports greedy argmax (`--temp 0.0`), top-k, and top-p nucleus sampling.
- **Limitation:** Repetition penalties, frequency/presence penalties, and min-p sampling are not yet implemented in `vectra_cli.exe`.

### 4. KV-Cache Quantization
- **Current Architecture:** KV-cache is stored in full 32-bit floating point precision (`FP32`).
- **Limitation:** For sequence lengths up to 2048 tokens, the KV-cache consumes $\approx 45\text{ MB}$. For sequence lengths $\ge 32\text{k}$, FP32 KV-cache would consume $> 700\text{ MB}$.
- **Future Direction:** Implement Q8_0 or Q4_0 quantized KV-cache storage to reduce memory usage by $4\times$.

---

## 2. Purity & Independence Guarantee

None of the above limitations will be resolved by incorporating external inference libraries. Project VECTRA remains **100% committed to original from-scratch C++20 algorithms**.
