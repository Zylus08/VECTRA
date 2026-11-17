# Milestone M2 — KV-Cache Architecture & Verification

**Date:** 2026-11-18  
**Module:** `vectra::model::KVCache`  
**Location:** `include/vectra/model/kv_cache.hpp` & `src/model/kv_cache.cpp`  

---

## 1. Memory Layout & Zero-Allocation Decode

In Project VECTRA, autoregressive token generation relies on a pre-allocated, contiguous KV-cache designed to eliminate all dynamic memory allocations during generation loops:

- **Layers ($L$):** 28
- **KV Attention Heads ($H_{kv}$):** 2
- **Head Dimension ($d_k$):** 128
- **KV Slice per Token per Layer:** $2 \times 128 = 256$ single-precision floats (1,024 bytes)
- **Configured Context Limit ($S_{\text{ctx}}$):** 1,024 tokens

### Memory Footprint Formula:
$$\text{Memory}_{\text{layer}} = 2 \times (S_{\text{ctx}} \times H_{kv} \times d_k \times \text{sizeof(float)}) = 2 \times (1024 \times 256 \times 4) = 2,097,152 \text{ bytes} \ (2.00 \text{ MB})$$
$$\text{Total Memory} = 28 \times 2.00 \text{ MB} = \mathbf{56.00 \text{ MB}}$$

Even at 2,048 context tokens, the entire KV-cache consumes only 112 MB, fitting effortlessly into host RAM.

---

## 2. Invariance & Causal Verification

1. **Deterministic Append Operation:**
   Each forward step at position $\text{pos}$ copies the rotary-embedded key and value vectors into the cache buffer at linear offset $\text{pos} \times 256$.
2. **Causal Masking Equivalence:**
   Verified via `vectra_integration_tests.exe` (`test_causal_masking_invariance`). Attention over past tokens $\tau \in [0, \text{pos}]$ produces identical activations whether computed token-by-token using KV-cache lookup or all at once via a causal triangular mask.
3. **Multi-Step Stability:**
   Tested over consecutive prefill and decode sequences up to 24 tokens. Output activations maintain strict numerical stability without drift, overflow, or NaN corruption.
