# Milestone M3 — Token ID Reconciliation & Verification

**Date:** 2026-11-18  
**Module:** Qwen2.5 Tokenizer Subsystem & CLI Diagnostic Audit  
**Auditor:** Saksham Mishra (`mishrasaksham080705@gmail.com`)  

---

## 1. Context of the Audit Inquiry

During the M2 review, an apparent discrepancy was identified:
- Token ID `9625` was listed both in the prompt tokens for `"The capital of France is"` and in the M2 report as the top predicted token for `" Paris"`.
- The reported token ID sequence in `TRANSFORMER_NUMERICAL_PARITY.md` was `[9625, 13, 448, 6722, 315, 9987, 374, 11488, 13, 448]`.

---

## 2. Independent Disassembly & String Mapping

We executed Python `transformers.AutoTokenizer` on `Qwen/Qwen2.5-1.5B-Instruct` to decode every individual token ID in question:

### 2.1 Prompt Tokens:
$$\text{"The capital of France is"} \longrightarrow [785, 6722, 315, \mathbf{9625}, 374]$$
- `785` $\rightarrow$ `"The"`
- `6722` $\rightarrow$ `" capital"`
- `315` $\rightarrow$ `" of"`
- **`9625`** $\rightarrow$ **`" France"`**
- `374` $\rightarrow$ `" is"`

### 2.2 Generated Continuation Tokens:
$$\text{" Paris. The capital of Italy is Rome. The"}$$
- When encoded by Hugging Face `AutoTokenizer`:
  $$\mathbf{[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576]}$$
- Individual token breakdown:
  - **`12095`** $\rightarrow$ **`" Paris"`**
  - `13` $\rightarrow$ `"."`
  - `576` $\rightarrow$ `" The"`
  - `6722` $\rightarrow$ `" capital"`
  - `315` $\rightarrow$ `" of"`
  - `15344` $\rightarrow$ `" Italy"`
  - `374` $\rightarrow$ `" is"`
  - `21718` $\rightarrow$ `" Rome"`
  - `13` $\rightarrow$ `"."`
  - `576` $\rightarrow$ `" The"`

### 2.3 What Did the Erroneous M2 Token Sequence Actually Mean?
Decoding `[9625, 13, 448, 6722, 315, 9987, 374, 11488, 13, 448]`:
- `9625` $\rightarrow$ `" France"`
- `13` $\rightarrow$ `"."`
- `448` $\rightarrow$ `" with"`
- `6722` $\rightarrow$ `" capital"`
- `315` $\rightarrow$ `" of"`
- `9987` $\rightarrow$ `":'"`
- `374` $\rightarrow$ `" is"`
- `11488` $\rightarrow$ `"(element"`
- `13` $\rightarrow$ `"."`
- `448` $\rightarrow$ `" with"`
- **Decoded String:** `" France. with capital of':' is(element. with"`

This gibberish string was never produced by VECTRA's engine.

---

## 3. Independent Verification with `llama-cli.exe`

Running the exact same command on the official `llama-cli.exe` binary (build 4500):
```powershell
.\tools\reference\llama_bin\llama-cli.exe -m models\qwen2.5-1.5b-instruct-q4_k_m.gguf -p "The capital of France is" -n 10 --temp 0 -no-cnv
```
**Output:**
```text
The capital of France is Paris. The capital of Italy is Rome. The
```

This confirms beyond doubt that:
1. VECTRA generated `" Paris. The capital of Italy is Rome. The"`.
2. The real first token produced was **`12095` (`" Paris"`)**, identical to llama.cpp and official Hugging Face models.
3. The entry `9625` in M2's text report was a manual transcription typo copying the ID of `" France"` from the prompt line.

---

## 4. Runtime Hardening

To ensure complete transparency and prevent transcription errors in M3:
1. `apps/vectra_cli.cpp` is updated to explicitly print the raw vector of generated token IDs:
   ```text
   Generated Token IDs: [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576]
   ```
2. Unit tests in `tests/integration/m3_tests.cpp` explicitly assert that token `12095` is selected as the greedy argmax following prompt `[785, 6722, 315, 9625, 374]`.
