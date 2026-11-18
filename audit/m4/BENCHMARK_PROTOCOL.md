# Project VECTRA — Milestone M4: Scientific Benchmark Protocol & Measurement Methodology

**Document Version:** 1.0.0  
**Authors:** VECTRA Systems & Benchmark Audit Team  
**Standard Compliance:** Controlled Multi-Run Benchmarking Protocol

---

## 1. Experimental Environment & System Controls

To ensure reproducible, scientifically valid benchmark measurements without confounding variables:
1. **Host Environment:** Windows 11 Home x64 (Build 26200), AC power connected, Windows High Performance power profile enabled to prevent dynamic core downclocking.
2. **Thermal & Background State:** Benchmarks are executed after confirming thermal equilibrium ($T_{\text{package}} < 55^\circ\text{C}$). Background applications, indexing services, and update daemons are suspended.
3. **Cache State Discrimination:**
   - **Cold Run:** Executed immediately following cache invalidation or upon first process invocation.
   - **Warm Runs:** Three consecutive runs executed with weights pre-faulted into the Windows OS unified file cache. All reported numbers represent the mean of the 3 warm runs unless explicitly designated as cold.

---

## 2. Mathematical Definition of Timing Boundaries

```
Process Launch
      |
      v
  [Model mmap & Tokenizer Load]  <-- Initialization (Recorded separately)
      |
      v
  T_start  <----------------------- Prompt Tokenization Begins
      |
      |   (Batched Prefill Pass: X_0 through X_{B-1})
      v
  T_first  <----------------------- First Token Generated & Streamed
      |
      |   (Autoregressive Decode: forward_token for tokens 1 through N-1)
      v
  T_end    <----------------------- Final Token Generated or EOS Reached
```

### Precise Metric Equations
1. **Time To First Token (TTFT) / Prefill Latency:**
   $$T_{\text{prefill}} = T_{\text{first}} - T_{\text{start}}$$
2. **Prefill Throughput:**
   $$\text{Throughput}_{\text{prefill}} = \frac{N_{\text{prompt}}}{T_{\text{prefill}} \text{ (seconds)}} \quad [\text{tokens/sec}]$$
3. **Decode Latency:**
   $$T_{\text{decode}} = T_{\text{end}} - T_{\text{first}}$$
4. **Decode Throughput:**
   $$\text{Throughput}_{\text{decode}} = \frac{N_{\text{gen}} - 1}{T_{\text{decode}} \text{ (seconds)}} \quad [\text{tokens/sec}]$$
   *(Note: The first generated token is produced by the prefill pass; subsequent decode steps produce $N_{\text{gen}} - 1$ tokens).*
5. **Total End-to-End Latency:**
   $$T_{\text{total}} = T_{\text{end}} - T_{\text{start}}$$

---

## 3. Canonical Evaluation Prompts

| Identifier | Domain | Input Prompt Text | Prompt Tokens | Max Tokens | Sampler |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **P1** | General Knowledge | `"The capital of France is"` | 5 tokens | 5 tokens | Greedy (`temp=0.0`) |
| **P2** | Python Code | `"def fibonacci(n):"` | 4 tokens | 16 tokens | Greedy (`temp=0.0`) |
| **P3** | ChatML Conversation | `"<\|im_start\|>user\nWhat is 2 + 2?<\|im_end\|>\n<\|im_start\|>assistant\n"` | 16 tokens | 16 tokens (EOS) | Greedy (`temp=0.0`) |

---

## 4. Instrumentation Tools

- **Timing Clock:** `std::chrono::high_resolution_clock` (monotonic hardware timestamp counter, sub-microsecond resolution on x86_64).
- **Memory Footprint:** Win32 API `GetProcessMemoryInfo` reading `PROCESS_MEMORY_COUNTERS::WorkingSetSize` (resident memory) and `PeakWorkingSetSize` (peak resident commit).
- **Reference Oracle:** `llama.cpp` release `b4500` invoked with `-m <model> -p <prompt> -n <tokens> -t 10 --temp 0.0 -no-cnv --simple-io`.
