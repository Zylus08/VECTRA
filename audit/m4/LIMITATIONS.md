# Project VECTRA — Milestone M4: Architectural Limitations & Scope Boundaries

**Document Version:** 1.0.0  
**Scope:** Milestone M4 Implementation

---

## 1. Supported vs Unsupported Quantization Schemes

1. **Fully Supported Production Formats:**
   - **`Q4_K`:** AVX2 FP32 FMA and INT8 DP4A kernels fully implemented for all attention and feed-forward projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`).
   - **`Q6_K`:** AVX2 FP32 FMA and INT8 DP4A kernels fully implemented for FFN down-projection (`down_proj`) and vocabulary head (`output.weight`).
   - **`Q4_0`:** Introductory baseline format preserved for backwards compatibility.
   - **`Q8_K`:** Full dynamic activation quantization format implemented on stack.
2. **Unsupported Formats (Deferred to Future Milestones):**
   - `Q2_K`, `Q3_K`, `Q5_K`, `IQ4_NL`, `IQ3_XXS`. Attempts to load models quantized with these types will gracefully throw descriptive `VectraException` errors.

---

## 2. Microarchitectural & Threading Limitations

1. **Heterogeneous Thread Affinity:**
   VECTRA currently relies on Windows OS thread scheduling across the Alder Lake hybrid topology. It does not yet invoke Win32 `SetThreadAffinityMask` or `SetThreadInformation` to strictly bind worker threads to Golden Cove P-cores.
2. **Prefill Register-Tiling (Outer-Product GEMM):**
   In prefill mode, VECTRA uses batched streaming GEMV-N (`forward_prompt_batch`), which streams weights once per prompt from DDR5 RAM. While this achieves $\sim 3.2-5.4\text{ tok/s}$ ($12\times$ faster than M2), it does not yet implement $L1$-cache-blocked register-tiled GEMM (which achieves $60-80\text{ tok/s}$ in `llama.cpp`).
3. **Cascading INT8 Quantization Drift:**
   In pure `--int8` mode, cascading 112 activation quantizations across 28 layers introduces small cumulative quantization noise that can alter greedy argmax ranking after 2-3 tokens. For mission-critical bitwise reference matching, users should specify `--fp32` (which delivers 100% exact bitwise parity with reference runtimes).

---

## 3. Serving & Context Limitations

1. **Single-Sequence Execution:**
   VECTRA M4 is optimized for single-session local interactive execution. Continuous batching, PagedAttention, speculative decoding, and concurrent multi-request serving are out of scope for M4.
2. **Context Window Limit:**
   Context capacity defaults to 1024 tokens (configurable up to the model's 32,768 training limit). Very large context windows ($> 4096$) require linear memory scaling ($56\text{ MB}$ per 1024 tokens in FP32).
3. **ISA Requirements:**
   Optimal performance requires hardware support for AVX2 and FMA3. While scalar C++20 fallbacks are provided for portability and regression testing, scalar execution is memory-latency bound.
