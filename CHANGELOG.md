# Changelog

All notable changes to Project VECTRA are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0-rc1] - 2026-11-18

### Milestone RC1 — Release Candidate 1: Final Scientific Audit & Release Certification

This release candidate consolidates all five engineering milestones (M1–M5) into a certified, reproducible, standalone CPU inference engine for Transformer language models.

#### Added
- **Batched INT8 Prefill Recovery Engine:** Multi-token causal GEMV-N (`matvec_q4_k_q8_batch` and `matvec_q6_k_q8_batch`) achieving $17.4 – 25.9\text{ tok/s}$ prefill ($4.9\times$ speedup over M4).
- **Vectorized Vocabulary Projection (LM-Head):** 256-bit AVX2 horizontal additions (`_mm256_hadd_epi32`) and interleaved scale sums in `vec_dot_q6_k_q8_avx2`, cutting LM-head projection from $42.0\text{ ms}$ to $5.7\text{ ms}$ ($33.8\text{ GB/s}$).
- **Fused Multi-Projection Kernels:** Single-dispatch Query-Key-Value projection (`matvec_q4_k_q8_qkv`) and Gate-Up projection (`matvec_q4_k_q8_2x`), eliminating 84 thread pool synchronization barriers per decode token.
- **Persistent Memory Scratchpad:** Pre-allocated static working workspace (`LayerScratch`), dropping dynamic heap allocations during steady-state decode to exactly zero.
- **AVX2 Arithmetic Safety Suite:** 7 rigorous edge-case tests in `vectra_arithmetic_safety_tests` covering strict $[-127, 127]$ clamping, overflow prevention, and NaN/Inf isolation.
- **Active Caller-Thread ThreadPool:** Caller thread immediately computes chunk 0 in `ThreadPool::parallel_for`, eliminating thread dispatch latency.
- **Full-Model M5 Integration Suite:** Comprehensive multi-layer numerical parity and throughput verification suite (`vectra_m5_tests.exe`).
- **Comprehensive 13-Document Release Audit:** Formally documented in `audit/rc1/`.

#### Changed
- Inlined 6-bit scales and minimums bitfield unpacking directly into `vec_dot_q4_k_q8_avx2`.
- Tightened activation quantization clamping from `[-128, 127]` to `[-127, 127]` to mathematically prevent signed 16-bit saturation in `_mm256_maddubs_epi16`.
- Separated measurement populations for P1, P2, and P3 benchmarks, providing clear statistical medians and standard deviations.
- Reduced peak memory footprint to $1,061.4\text{ MB}$ (leaner than `llama.cpp` by $111\text{ MB}$).

#### Fixed
- Fixed timing bug in `test_m5_prefill_throughput_recovery` where timer was placed before dynamic allocation of 23 MB KVCache buffer.
- Fixed command-line parser in `vectra_cli.cpp` to correctly parse short flags `-m`, `-p`, `-n`, `-t`.
- Fixed CTest working directory configuration in `CMakeLists.txt` to execute all tests relative to project root.

---

## [0.1.0-m5] - 2026-11-18

### Milestone M5 — INT8 Numerical Certification, Prefill Recovery & Kernel Optimization
- Added batched multi-token GEMV-N for prefill.
- Added fused QKV and GateUp projections.
- Vectorized Q6_K dot product.
- Expanded test suite to 53 tests.

---

## [0.1.0-m4] - 2026-11-18

### Milestone M4 — Quantized CPU Inference Optimization & Hardware Scheduling
- Added dynamic INT8 activation quantization (`BlockQ8_K_256`).
- Added AVX2 DP4A integer dot-product kernels (`vec_dot_q4_k_q8_avx2`, `vec_dot_q6_k_q8_avx2`).
- Integrated persistent thread pool with condition-variable synchronization.
- Accelerated decode throughput to $12.7\text{ tok/s}$.

---

## [0.1.0-m3] - 2026-11-18

### Milestone M3 — High-Performance Quantized Inference & AVX2 Packed Kernels
- Added packed Q4_K and Q6_K quantized matrix-vector kernels with AVX2/FMA.
- Added multi-threaded persistent thread pool.
- Added full Transformer autoregressive greedy generation loop with KV-cache.
- Accelerated decode throughput from $0.22\text{ tok/s}$ to $3.72\text{ tok/s}$.

---

## [0.1.0-m2] - 2026-11-18

### Milestone M2 — Real Qwen2.5-1.5B Inference, GGUF Loader & Tokenizer
- Added memory-mapped GGUF file loader (`MMapLoader`).
- Added BPE tokenizer runtime for Qwen2.5 151,665-token vocabulary.
- Added greedy, top-k, and top-p sampling.
- Verified initial autoregressive token generation.

---

## [0.1.0-m1] - 2026-11-17

### Milestone M1 — Runtime Foundation, Model Format & Numerical Verification
- Initialized clean-room C++20 repository.
- Built contiguous N-dimensional Tensor runtime with 64-byte alignment.
- Implemented scalar and AVX2 RMSNorm, Softmax, RoPE, and SwiGLU.
- Added introductory Q4_0 block quantization.
- Established numerical oracle verification against Python reference.
