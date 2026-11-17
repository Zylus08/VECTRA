# Milestone M2 — First Genuine Generation Evidence

**Date:** 2026-11-18  
**Acceptance Gate:** Genuine text generation from trained Qwen2.5-1.5B weights executed entirely on VECTRA CPU engine.  
**Platform:** Windows 11 x64, Intel Core i7-12650H, AVX2 + FMA, MSVC Release (/O2 /arch:AVX2).  

---

## 1. Generation Run 1: General Knowledge Continuation

### Command Line:
```powershell
.\build_release\vectra_cli.exe --prompt "The capital of France is" --max-tokens 10
```

### Complete Verbatim Execution Output:
```text
=========================================================
          PROJECT VECTRA - CPU INFERENCE RUNTIME         
=========================================================
Host Hardware Vector Instructions: AVX2 + FMA DETECTED & ACTIVE

[1/4] Loading Tokenizer: models/qwen_tokenizer.bin ...
      Loaded 151665 tokens in 162.2 ms

[2/4] Memory-Mapping GGUF Model: models/qwen2.5-1.5b-instruct-q4_k_m.gguf ...
      Model Loaded: 28 layers, hidden=1536, heads=12 (kv=2), intermediate=8960, vocab=151936
      Mapped Size: 1065.6 MB in 18.8 ms

[3/4] Tokenizing Prompt...
      Prompt: "The capital of France is"
      Tokens: 5 tokens: [785, 6722, 315, 9625, 374]

[4/4] Generating (max 10 new tokens, temp=0.0, sampler=greedy)...
---------------------------------------------------------
 Paris. The capital of Italy is Rome. The
---------------------------------------------------------

--- Execution Metrics ---
  Prompt Tokens:      5 tokens
  Generated Tokens:   10 tokens
  Prefill Latency:    18353.26 ms (0.27 tok/s)
  Decode Latency:     41808.23 ms (0.22 tok/s)
  Total Inference:    60161.49 ms
  Process Resident:   1004.18 MB
  Peak Working Set:   1060.43 MB
=========================================================
```

---

## 2. Generation Run 2: Code Synthesis

### Command Line:
```powershell
.\build_release\vectra_cli.exe --prompt "def fibonacci(n):" --max-tokens 12
```

### Complete Verbatim Execution Output:
```text
=========================================================
          PROJECT VECTRA - CPU INFERENCE RUNTIME         
=========================================================
Host Hardware Vector Instructions: AVX2 + FMA DETECTED & ACTIVE

[1/4] Loading Tokenizer: models/qwen_tokenizer.bin ...
      Loaded 151665 tokens in 164.9 ms

[2/4] Memory-Mapping GGUF Model: models/qwen2.5-1.5b-instruct-q4_k_m.gguf ...
      Model Loaded: 28 layers, hidden=1536, heads=12 (kv=2), intermediate=8960, vocab=151936
      Mapped Size: 1065.6 MB in 17.1 ms

[3/4] Tokenizing Prompt...
      Prompt: "def fibonacci(n):"
      Tokens: 4 tokens: [750, 75698, 1445, 1648]

[4/4] Generating (max 12 new tokens, temp=0.0, sampler=greedy)...
---------------------------------------------------------
 
    if n <= 1: 
        return n
---------------------------------------------------------

--- Execution Metrics ---
  Prompt Tokens:      4 tokens
  Generated Tokens:   12 tokens
  Prefill Latency:    14643.04 ms (0.27 tok/s)
  Decode Latency:     50512.82 ms (0.22 tok/s)
  Total Inference:    65155.86 ms
  Process Resident:   1004.19 MB
  Peak Working Set:   1060.43 MB
=========================================================
```

---

## 3. Generation Run 3: ChatML Conversational Reasoning

### Command Line:
```powershell
.\build_release\vectra_cli.exe --chat --prompt "What is 2 + 2?" --max-tokens 8
```

### Complete Verbatim Execution Output:
```text
=========================================================
          PROJECT VECTRA - CPU INFERENCE RUNTIME         
=========================================================
Host Hardware Vector Instructions: AVX2 + FMA DETECTED & ACTIVE

[1/4] Loading Tokenizer: models/qwen_tokenizer.bin ...
      Loaded 151665 tokens in 164.8 ms

[2/4] Memory-Mapping GGUF Model: models/qwen2.5-1.5b-instruct-q4_k_m.gguf ...
      Model Loaded: 28 layers, hidden=1536, heads=12 (kv=2), intermediate=8960, vocab=151936
      Mapped Size: 1065.6 MB in 18.4 ms

[3/4] Tokenizing Prompt...
      Prompt: "What is 2 + 2?"
      Tokens: 16 tokens: [151644, 872, 198, 3838, 374, 220, 17, 488, 220, 17, 30, 151645, 198, 151644, 77091, 198]

[4/4] Generating (max 8 new tokens, temp=0.0, sampler=greedy)...
---------------------------------------------------------
2 + 2 equals 4.
---------------------------------------------------------

--- Execution Metrics ---
  Prompt Tokens:      16 tokens
  Generated Tokens:   8 tokens
  Prefill Latency:    64582.82 ms (0.25 tok/s)
  Decode Latency:     33628.88 ms (0.21 tok/s)
  Total Inference:    98211.69 ms
  Process Resident:   1004.14 MB
  Peak Working Set:   1060.38 MB
=========================================================
```

---

## 4. Acceptance Gate Assessment

1. **Originality:** 0% external inference dependencies. Forward graph, quantization decoding, attention, and sampling are 100% native C++20.
2. **Coherence:** All three outputs are completely coherent, mathematically accurate, and grammatically correct.
3. **Safety & Budget:** Peak process working set was ~1060 MB, strictly complying with the host RAM budget.
4. **Verdict:** **PASS (ACCEPTED)**.
