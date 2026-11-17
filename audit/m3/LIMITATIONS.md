# Project VECTRA — Milestone M3: Known Limitations & Future Roadmap

**Milestone:** M3 High-Performance Quantized CPU Inference  
**Architecture:** Project VECTRA CPU LLM Inference Engine  

---

## 1. Current Architectural Limitations

### 1. Activation Representation (FP32 Activations)
- **Current Behavior:** VECTRA preserves single-precision 32-bit floating-point format for intermediate activations ($x, q, k, v, \text{gate}, \text{up}, \text{act}$). Quantized weights are multiplied against FP32 activations using AVX2 FP32 FMAs (`_mm256_fmadd_ps`).
- **Limitation:** AVX2 FMA pipelines process 8 single-precision floats per vector instruction. This limits compute density compared to 8-bit integer dot products.

### 2. Prefill GEMV vs Tiled 2D GEMM
- **Current Behavior:** In `forward_prompt_batch`, prompt ingestion uses a 1D batched matrix-vector routine (`matvec_batch_dispatch`), where a single weight row is evaluated against $B$ tokens in registers.
- **Limitation:** For short prompts ($B \le 16$), this achieves 5.2 - 5.4 tok/s. For long prompts ($B \ge 128$), it does not perform 2D spatial cache tiling across weight blocks and activation blocks in L2 cache, capping prefill scaling.

### 3. Asymmetric Core Scheduling on Hybrid Architecture
- **Current Behavior:** The persistent thread pool statically partitions matrix rows into equal slices $\lceil \text{rows} / N \rceil$ across all worker threads.
- **Limitation:** On Intel 12th Gen Alder Lake (6 P-cores + 4 E-cores), Gracemont E-cores run at ~3.5 GHz with narrower instruction decode compared to Golden Cove P-cores at ~4.7 GHz. Static chunking causes P-cores to finish their slices faster and wait at the generation barrier for E-cores to complete.

### 4. Quantization Type Coverage
- **Current Behavior:** VECTRA natively supports `Q4_0`, `Q4_K`, and `Q6_K` super-blocks (which fully covers the target `qwen2.5-1.5b-instruct-q4_k_m.gguf` checkpoint).
- **Limitation:** Models quantized with `Q2_K`, `Q3_K`, `Q5_K`, or `IQ` (importance matrix) quantizations are not yet supported.

### 5. KV Cache Precision
- **Current Behavior:** Contiguous FP32 storage ($56\text{ KB/token}$).
- **Limitation:** For long context windows ($S \ge 4096$), storing FP32 keys and values consumes $224\text{ MB}$ of cache traffic.

---

## 2. Engineering Roadmap (M4 & M5 Milestones)

| Feature / Enhancement | Proposed Architecture | Expected Performance Impact | Target Milestone |
| :--- | :--- | :---: | :---: |
| **Dynamic INT8 Activation Quantization** | Quantize activations to `Q8_0` on-the-fly; use `_mm256_maddubs_epi16` and `_mm256_madd_epi16` for 32 integer MACs/cycle | **3x – 4x decode speedup** (~10 – 15 tok/s) | Milestone M4 |
| **Asymmetric Work Partitioning / Affinity** | Pin threads to Golden Cove cores via `SetThreadAffinityMask`; partition rows proportionally ($1.6\times$ more to P-cores) | **15% – 25% reduction in barrier stall** | Milestone M4 |
| **Tiled 2D GEMM Prefill** | Register-blocked GEMM ($4 \times 8$ or $6 \times 8$ accumulator tiles) in L2 cache for batched prompt ingestion | **5x – 10x prefill speedup** (>30 tok/s on $B \ge 64$) | Milestone M4 |
| **FP16 KV Cache** | Store keys and values in 16-bit half precision with F16C hardware conversion | **50% reduction in KV memory bandwidth** | Milestone M5 |
| **Speculative Decoding** | Draft tokens with a small model or n-gram cache, verified in parallel by VECTRA batched forward pass | **2x – 3x wall-clock decode speedup** | Milestone M5 |
