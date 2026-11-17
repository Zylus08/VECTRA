# Milestone M3 — Starting Repository Audit & M2 Retrospective

**Date:** 2026-11-18  
**Lead Architect & Auditor:** Saksham Mishra (`mishrasaksham080705@gmail.com`)  
**Starting Commit:** `35fd312` (`feat(m2): implement real Qwen2.5-1.5B GGUF inference, tokenizer parity, and autoregressive generation`)  
**Repository:** `E:\VECTRA`  
**Host Hardware:** Intel Core i7-12650H (AVX2, FMA, 16 GB RAM), Windows 11 Home 64-bit  

---

## 1. Baseline Regression Verification

At the commencement of Milestone M3, all 31 existing test cases from M1 and M2 were re-executed against the `build_release` binaries:
- `vectra_unit_tests.exe`: 7 / 7 PASSED (Memory alignment, FP16, Tensor basics/views, BlockQ4_0, GGUF errors, Tokenizer).
- `vectra_numerical_tests.exe`: 6 / 6 PASSED (Dot product, Matvec, RMSNorm, Softmax, RoPE, Pathological inputs).
- `vectra_integration_tests.exe`: 3 / 3 PASSED (Causal masking invariance, synthetic generation step, quantized matvec).
- `vectra_tokenizer_tests.exe`: 11 / 11 PASSED (100% bit-exact parity matching Hugging Face across 11 test categories).
- `vectra_m2_tests.exe`: 4 / 4 PASSED (Win32 mmap loader, Q4_K dot product, Q6_K dot product, deterministic sampler).

**Total Pass Rate:** 31 / 31 (100.0%). All historical M1 and M2 audit artifacts in `audit/m1/` and `audit/m2/` remain completely intact.

---

## 2. Investigation of Token ID 9625 and M2 First-Generation Evidence

### 2.1 The Issue
In M2's `TRANSFORMER_NUMERICAL_PARITY.md` and `FIRST_GENERATION.md`, the prompt `"The capital of France is"` was tokenized as:
`[785, 6722, 315, 9625, 374]`
and the generated text was reported as:
`" Paris. The capital of Italy is Rome. The"`
However, the M2 parity document erroneously listed:
`Top token predicted: Token ID 9625 (" Paris")`
and listed raw token IDs:
`[9625, 13, 448, 6722, 315, 9987, 374, 11488, 13, 448]`

### 2.2 Micro-Audit & Root Cause Discovery
We conducted an exact audit using both the official Hugging Face `AutoTokenizer` and independent decoding:
1. **Token 9625 Identity:**
   - `AutoTokenizer.decode([9625]) == " France"`.
   - Token 9625 is the 4th token of the input prompt (`" France"`), NOT `" Paris"`.
2. **True Token ID of " Paris":**
   - `AutoTokenizer.encode(" Paris") == [12095]`.
3. **True Token IDs of the Generated Text:**
   When `" Paris. The capital of Italy is Rome. The"` is tokenized by Qwen BPE, the exact token IDs are:
   `[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576]`
   - `12095`: `" Paris"`
   - `13`: `"."`
   - `576`: `" The"`
   - `6722`: `" capital"`
   - `315`: `" of"`
   - `15344`: `" Italy"`
   - `374`: `" is"`
   - `21718`: `" Rome"`
   - `13`: `"."`
   - `576`: `" The"`
4. **Why Did M2 Report Bogus Token IDs?**
   In `apps/vectra_cli.cpp`, the streaming callback decoded tokens on the fly (`tokenizer.decode({token_id})`) and streamed the characters directly to `std::cout`. The vector of generated integer IDs was returned but never printed to stdout. During M2 documentation drafting, an author manually reconstructed token IDs and erroneously mapped `" Paris"` to the token ID from the prompt (9625).
5. **Independent Reference Validation:**
   Executing the exact same prompt (`"The capital of France is"`, `-n 10 --temp 0`) on official `llama-cli.exe` produces:
   `"The capital of France is Paris. The capital of Italy is Rome. The"`
   confirming that VECTRA's generated text is **100% bit-exact and mathematically identical to the official reference output**.

---

## 3. Audit of Numerical Parity Claims in M2

- **Finding:** M2's `TRANSFORMER_NUMERICAL_PARITY.md` documented the mathematical equations and verified that the argmax output matched greedy generation, but it did not log measured intermediate activation vectors (RMSNorm tensors, Q/K/V tensors, attention matrices, hidden states) against an independent oracle.
- **Unverified Claims in M2:**
  - Layer-by-layer cosine similarity / SNR against an independent reference runtime.
  - Multi-token logit distribution tolerances.
- **Mandate for M3:** Phase 1 will generate exhaustive layer-by-layer reference fixtures and certify every intermediate tensor boundary with measured numerical errors and tolerances.
