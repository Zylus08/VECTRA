# Milestone M2 — Known Limitations & Architectural Roadmap

**Date:** 2026-11-18  
**Project:** VECTRA Milestone M2  

---

## 1. Single-Threaded CPU Execution

- **Observation:** Matrix-vector multiplications across all 28 layers and the 151,936-dimensional LM head currently execute synchronously on a single CPU thread.
- **Impact:** Generation throughput is approximately 0.22 to 0.27 tokens per second on an Intel Core i7-12650H.
- **Planned Resolution for M3:** Implement a lightweight, zero-dependency C++20 thread pool to parallelize rows across the 6 P-cores and 4 E-cores of the Alder Lake processor, targeting >5 tok/s.

---

## 2. LM-Head Computational Bottleneck

- **Observation:** The final vocabulary projection (`output.weight`) multiplies a 1,536-dimensional activation by a $1536 \times 151936$ matrix (233.4 million parameters, Q6_K).
- **Impact:** Nearly 40% of per-token decode latency is spent on this single matrix-vector multiplication.
- **Planned Optimization:** Vectorized multi-row AVX2 kernels and multi-core work stealing.

---

## 3. Sequential Prompt Prefill

- **Observation:** Prompt tokens are ingested one token at a time through `forward_token(t, pos, ...)`.
- **Impact:** Prefill latency scales linearly ($O(N)$) with prompt length rather than leveraging batched matrix-matrix multiplication (GEMM) for simultaneous prompt token processing.
- **Planned Optimization:** Introduce a batched prefill kernel (`forward_prompt_batch`) that computes attention over prompt tokens in parallel.

---

## 4. Quantization Scope

- **Observation:** M2 focuses on the exact formats required for `qwen2.5-1.5b-instruct-q4_k_m.gguf`, namely `Q4_K`, `Q6_K`, `Q4_0`, and `F32`.
- **Impact:** Other variants (e.g., `Q2_K`, `Q3_K`, `Q5_K`, `Q8_0`) are parsed for byte sizes in the GGUF reader but do not yet have dedicated AVX2 dot-product kernels.
- **Planned Optimization:** Expand kernel registry to support all standard GGUF quantizations as needed.

---

## 5. OS-Specific Memory Mapping

- **Observation:** `MemoryMappedFile` is implemented directly using Win32 APIs (`CreateFileW`, `CreateFileMappingW`, `MapViewOfFile`).
- **Impact:** Builds cleanly on Windows 11 MSVC, but requires a POSIX abstraction layer for Linux/macOS cross-compilation.
