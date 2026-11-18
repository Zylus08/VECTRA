# Project VECTRA — Milestone M5: Starting Audit & Integrity Assessment

**Auditor:** Principal Inference-Runtime Architect & Performance Certification Engineer  
**Date:** November 18, 2026  
**Repository Branch:** `main`  
**Starting Commit:** `4941654`  
**Target Architecture:** Windows 11 x64, Intel Core i7-12650H (6 P-cores + 4 E-cores, 16 threads), AVX2, FMA3, 16 GB DDR5  
**Target Model:** `Qwen2.5-1.5B-Instruct` (`models/qwen2.5-1.5b-instruct-q4_k_m.gguf`, SHA-256: `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`)

---

## 1. Executive Assessment

At the conclusion of Milestone M4, Project VECTRA established genuine autoregressive inference, dynamic INT8 activation quantization (`BlockQ8_K_256`), AVX2 dot-product acceleration (`vec_dot_q4_k_q8_avx2`, `vec_dot_q6_k_q8_avx2`), and a persistent thread pool.

However, entering Milestone M5, a critical forensic audit of M4 revealed several severe performance and numerical discrepancies:

1. **Catastrophic Prefill Degradation:**
   While decode throughput was accelerated to 12.7 tok/s in M4, prefill throughput severely degraded from 5.39 tok/s (in M3) down to **2.67 – 3.62 tok/s** in M4. Profiling revealed that M4's batched prefill (`forward_prompt_batch`) had not been updated to use the batched INT8 GEMV-N kernels, repeatedly dequantizing weights and activations per token.
2. **LM-Head Bottleneck:**
   The vocabulary projection (`151936 x 1536` weights) consumed **42.0 ms per token** (nearly 50% of the entire decode cycle), bottlenecked by inefficient 16-bit integer horizontal adds (`_mm_hadd_epi32`) in `vec_dot_q6_k_q8_avx2`.
3. **Thread Pool Dispatch Latency & Barrier Overhead:**
   Every decode token performed separate parallel dispatches for Q, K, and V projections, followed by separate parallel dispatches for Gate and Up projections, introducing **196 thread pool synchronization barriers** and **336 dynamic heap allocations** per decode token.
4. **Quantization Contract & Clamping Discrepancy:**
   In M4, `quantize_activation_q8_k_256` clamped activation values to `[-128, 127]`. However, in x86 AVX2, `_mm256_maddubs_epi16` treats the first operand as unsigned and the second as signed 8-bit integers. If an activation value was $-128$, the signed product can overflow 16-bit integers, requiring strict clamping to `[-127, 127]`.

---

## 2. Regression Baseline Verification

Before applying any code modifications, all 40 existing regression tests across 7 historical test suites were executed and verified:

| Test Suite Executable | Test Count | Status | Description |
| :--- | :---: | :---: | :--- |
| `vectra_unit_tests.exe` | 7 | **PASSED** | Memory alignment, FP16 conversion, Tensor basics & views, Q4 block math, GGUF error handling |
| `vectra_numerical_tests.exe` | 6 | **PASSED** | Dot products, Matvec, RMSNorm, Softmax, RoPE, pathological inputs |
| `vectra_integration_tests.exe` | 3 | **PASSED** | Causal masking, tiny Transformer step, quantized matvec |
| `vectra_tokenizer_tests.exe` | 11 | **PASSED** | Exact parity on English, numerals, whitespace, code, Hindi, Japanese, emoji, ChatML |
| `vectra_m2_tests.exe` | 4 | **PASSED** | Memory-mapped GGUF loader, Q4_K/Q6_K correctness, greedy sampler |
| `vectra_m3_tests.exe` | 4 | **PASSED** | KV cache bounds, thread pool determinism, AVX2 fused Q4_K, greedy token parity |
| `vectra_m4_tests.exe` | 5 | **PASSED** | INT8 activation bounds, Q4_K/Q6_K INT8 parity, fused FP32 parity, multi-layer bounds |
| **Total Baseline Tests** | **40** | **40/40 PASSED** | **100% Historical Integrity Maintained** |

---

## 3. Scope of Milestone M5 Deliverables

1. **Prefill Recovery:** Design and integrate `matvec_q4_k_q8_batch` and `matvec_q6_k_q8_batch` into `forward_prompt_batch` to achieve prefill throughput $> 18.0\text{ tok/s}$.
2. **LM-Head Vectorization:** Vectorize `vec_dot_q6_k_q8_avx2` using AVX2 256-bit horizontal adds and interleaved sub-block sums, dropping LM-head latency below $10\text{ ms}$.
3. **Decode Projection Fusion:** Fuse Q, K, V into a single parallel dispatch (`matvec_q4_k_q8_qkv`) and Gate, Up into a single parallel dispatch (`matvec_q4_k_q8_2x`), eliminating 84 barriers and 336 heap allocations per token.
4. **AVX2 Arithmetic Safety Suite:** Implement 7 comprehensive edge-case tests validating strict $[-127, 127]$ clamping, NaN/Inf handling, subnormal floats, and adversarial saturation guards.
5. **Full INT8 Numerical Certification:** Formulate `tests/integration/m5_tests.cpp` measuring cosine similarity ($> 0.9999$), max absolute error, RMS error, and multi-token greedy sequence agreement.
6. **Controlled Multi-Thread Benchmarking:** Evaluate prompts P1, P2, P3 across 1 to 16 threads, documenting exact prefill, decode, and memory footprints.
