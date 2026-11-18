# Project VECTRA — Milestone M5: Controlled Benchmark Protocol & Methodology

**Document Purpose:** Complete specification of benchmarking methodologies, hardware controls, environment freezing, and timing boundary definitions.

---

## 1. Controlled Hardware & Software Environment

All benchmarks in Milestone M5 were conducted under strictly controlled environment conditions:
- **Host CPU:** Intel Core i7-12650H, 10 Physical Cores (6P + 4E), 16 Logical Threads.
- **Base Frequency:** 2.30 GHz; Boost Frequency: up to 4.70 GHz.
- **Microarchitectural SIMD:** AVX2 + FMA3 + BMI2 active. (AVX-512 disabled/absent).
- **RAM:** 16 GB DDR5-4800 dual-channel.
- **Operating System:** Microsoft Windows 11 Home 64-bit (Build 26100).
- **Toolchain:** MSVC 19.43.34808.0 (Visual Studio 2022 v17.13.5), CMake 3.31.6, Ninja 1.12.1.
- **Compilation Flags:** `/O2 /Ob2 /Oi /Ot /GL -std:c++20 /arch:AVX2 /MD /permissive- /utf-8 /Zc:preprocessor`.
- **Target Model Checkpoint:** `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`  
  - SHA-256: `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`  
  - File Size: `1,117,460,960 bytes` ($1065.69\text{ MB}$).

---

## 2. Standardized Benchmark Prompts

1. **Prompt P1 (General Knowledge Fact Retrieval):**
   - Text: `"The capital of France is"`
   - Input Tokens: 5 tokens (`[785, 6722, 315, 9625, 374]`)
   - Generation Budget: 16 to 32 tokens
   - Sampling: Greedy argmax (`--temp 0.0`)
2. **Prompt P2 (Code Generation):**
   - Text: `"def fibonacci(n):"`
   - Input Tokens: 4 tokens (`[715, 262, 421, 308]`)
   - Generation Budget: 16 to 32 tokens
   - Sampling: Greedy argmax (`--temp 0.0`)
3. **Prompt P3 (Multi-Turn ChatML Conversation):**
   - Text: `"<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n"`
   - Input Tokens: 16 tokens
   - Generation Budget: 16 tokens
   - Sampling: Greedy argmax (`--temp 0.0`)

---

## 3. Metric Definitions & Formulas

1. **Time-To-First-Token / Prefill Latency ($T_{\text{prefill}}$):**
   The elapsed wall-clock time from the start of multi-token prompt evaluation (`forward_prompt_batch`) until the first next-token logits are computed and the first token is selected by the sampler.
   $$\text{Prefill Throughput} = \frac{N_{\text{prompt}}}{T_{\text{prefill}}} \quad (\text{tok/s})$$
2. **Subsequent Decode Latency ($T_{\text{decode}}$):**
   The elapsed wall-clock time spent generating the remaining $N_{\text{gen}} - 1$ tokens via autoregressive single-token forward passes (`forward_single`).
   $$\text{Decode Throughput} = \frac{N_{\text{gen}} - 1}{T_{\text{decode}}} \quad (\text{tok/s})$$
3. **Total Inference Latency ($T_{\text{total}}$):**
   $$T_{\text{total}} = T_{\text{prefill}} + T_{\text{decode}}$$
4. **Resident Set Size (RSS) & Peak Working Set:**
   Queried via Windows Win32 API `K32GetProcessMemoryInfo`:
   - `WorkingSetSize`: Current resident memory mapped into physical RAM.
   - `PeakWorkingSetSize`: Highest resident memory watermark reached during execution.

---

## 4. Cold vs Warm Execution Protocol

- **Cold Benchmark:** Executed on a fresh process invocation with no prior model memory in the OS file cache. Page faults during initial memory mapping are observed and recorded.
- **Warm Benchmark:** Executed immediately following a prior model execution so that all memory-mapped file pages reside hot in the OS page cache.
- Across all formal audit reports, Cold and Warm measurements are strictly segregated.
