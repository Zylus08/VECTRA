# Project VECTRA — Milestone M3: Controlled Benchmark Protocol & Measurement Methodology

**Auditor / Architect:** Saksham Mishra  
**Repository:** `E:\VECTRA`  
**Milestone:** M3 High-Performance Quantized CPU Inference & AVX2 Optimization  

---

## 1. Experimental Environment & System Controls

To ensure reproducible, low-variance measurements without thermal throttling or operating system jitter, the following system controls were maintained throughout all M3 benchmarks:

1. **Power & Thermal Policy:**
   - Platform: Windows 11 Home x64 (Version 10.0.26100).
   - Machine: Lenovo LOQ with Intel Core i7-12650H (10 Cores: 6 Performance Cores + 4 Efficient Cores, 16 Logical Processors).
   - AC wall adapter connected at 170W capacity.
   - Windows Power Plan: High Performance (`SCHEME_MIN`).
   - Thermal profile: Performance mode (fans running, preventing throttling down from base clocks).
2. **Memory Hierarchy State:**
   - 16.0 GB physical RAM.
   - All benchmarks execute on a warm page cache (one unmeasured warm-up pass executed prior to recorded iterations to ensure model file memory-mapping pages are faulted into physical memory).
3. **Compiler Toolchain & Optimizations:**
   - Microsoft Visual C++ Compiler Version 19.43.34808 for x64.
   - Build Mode: `Release` with Whole Program Optimization.
   - Compiler Flags: `/O2 /Oi /Ot /GL /arch:AVX2 /DVECTRA_USE_AVX2=1 /permissive- /utf-8 /Zc:preprocessor`.
   - Linker Flags: `/LTCG`.

---

## 2. Test Prompts & Generation Scenarios

Three distinct prompt regimes were chosen to test varied prefill lengths and autoregressive generation behaviors:

| Scenario | Raw Prompt Text | Formatted Prompt / Tokens | Prefill Length ($B$) | Max New Tokens | Domain |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **P1: Short Factual** | `"The capital of France is"` | `[785, 6722, 315, 9625, 374]` | 5 tokens | 5 tokens | General Knowledge / M2 Parity |
| **P2: Code Prefill** | `"def fibonacci(n):"` | `[750, 75698, 1445, 1648]` | 4 tokens | 16 tokens | Code / Python Indentation & Control Flow |
| **P3: ChatML Dialog** | `"What is 2 + 2?"` | `<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n` (`[151644, ...]`) | 16 tokens | 12 tokens | Instruction Following / Stop-Token Handling |

---

## 3. Metrics & Measurement Formulas

1. **Time to First Token (TTFT / Prefill Latency):**
   Measured from prompt ingest start until the first output token ID is produced by the sampler:
   $$\text{TTFT} = t_{\text{first\_token}} - t_{\text{start}} \quad (\text{ms})$$
   $$\text{Prefill Throughput} = \frac{N_{\text{prompt\_tokens}}}{\text{TTFT}} \quad (\text{tokens/second})$$
2. **Inter-Token Latency & Decode Throughput:**
   Measured across the autoregressive token loop from first token to generation termination:
   $$\text{Decode Latency} = t_{\text{end}} - t_{\text{first\_token}} \quad (\text{ms})$$
   $$\text{Decode Throughput} = \frac{N_{\text{generated\_tokens}} - 1}{\text{Decode Latency}} \quad (\text{tokens/second})$$
3. **End-to-End Latency:**
   $$t_{\text{total}} = t_{\text{end}} - t_{\text{start}} \quad (\text{ms})$$
4. **Resident Set Size (RSS):**
   Queried dynamically via the Win32 `K32GetProcessMemoryInfo` API (`PROCESS_MEMORY_COUNTERS_EX.WorkingSetSize` and `PeakWorkingSetSize`).
5. **Sampling Policy:**
   Greedy argmax ($\text{Temperature} = 0.0$) to guarantee 100% deterministic, bit-for-bit reproducible token generation across runs.

---

## 4. Multi-Run Statistical Treatment

Each test case was run across 3 independent trials. For all reported metrics, we record:
- Mean value ($\mu$)
- Standard deviation ($\sigma$)
- Coefficient of variation ($\text{CV} = \sigma / \mu$)
Any run displaying $\text{CV} > 5\%$ was automatically flagged for thermal throttling or OS background interference and re-evaluated.
