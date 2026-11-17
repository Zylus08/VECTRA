# Project VECTRA — Milestone M3: KV-Cache Scaling, Contiguity & Bounds Verification

**Component:** `vectra::model::KVCache` (`include/vectra/model/kv_cache.hpp`, `src/model/kv_cache.cpp`)  
**Tested Context Lengths:** 16, 64, 128, 256, 512, 1024  
**Layer Count:** 28 layers  
**Key/Value Heads:** 2 KV heads, head dimension $d_k = 128$  
**Precision:** Single-precision FP32 ($4\text{ bytes per element}$)

---

## 1. Architectural Memory Layout & Contiguity

In VECTRA, the KV-cache is allocated as a contiguous, 64-byte aligned flat memory buffer per layer:
- **Key Cache Layout:** Shape $[S_{\max}, n_{kv}, d_k] = [S_{\max}, 2, 128]$ floats per layer.
  Index formula for token $pos$, KV head $h$, element $e$:
  $$\text{Offset}_K(pos, h, e) = pos \cdot (n_{kv} \cdot d_k) + h \cdot d_k + e$$
- **Value Cache Layout:** Shape $[S_{\max}, n_{kv}, d_k] = [S_{\max}, 2, 128]$ floats per layer.
  Index formula for token $pos$, KV head $h$, element $e$:
  $$\text{Offset}_V(pos, h, e) = pos \cdot (n_{kv} \cdot d_k) + h \cdot d_k + e$$

### Memory Footprint Formula:
$$\text{Bytes Per Token (all 28 layers)} = 28 \times 2 \times 128 \times 4\text{ bytes (K)} + 28 \times 2 \times 128 \times 4\text{ bytes (V)} = 57,344\text{ Bytes} \approx \mathbf{56\text{ KB/token}}$$

---

## 2. Context Length Scaling Verification

The scaling test suite (`test_kv_cache_scaling_and_bounds` in `vectra_m3_tests.exe`) executed allocations, population, retrieval, and bounds checks across 6 context scales:

| Context Length ($S_{\max}$) | Key Cache Size (All Layers) | Value Cache Size (All Layers) | Total KV Footprint | Allocation Latency | Invariance & Read Verification |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **16** | 448 KB | 448 KB | **896 KB** | $< 0.01\text{ ms}$ | **PASS** (Bitwise Match) |
| **64** | 1,792 KB | 1,792 KB | **3,584 KB (3.5 MB)** | $< 0.02\text{ ms}$ | **PASS** (Bitwise Match) |
| **128** | 3,584 KB | 3,584 KB | **7,168 KB (7.0 MB)** | $< 0.03\text{ ms}$ | **PASS** (Bitwise Match) |
| **256** | 7,168 KB | 7,168 KB | **14,336 KB (14.0 MB)** | $< 0.05\text{ ms}$ | **PASS** (Bitwise Match) |
| **512** | 14,336 KB | 14,336 KB | **28,672 KB (28.0 MB)** | $< 0.09\text{ ms}$ | **PASS** (Bitwise Match) |
| **1024** | 28,672 KB | 28,672 KB | **57,344 KB (56.0 MB)** | $< 0.16\text{ ms}$ | **PASS** (Bitwise Match) |

---

## 3. Bounds Enforcement & Safety Invariants

1. **Upper Bound Rejection:**
   - Attempting to append at position $pos \ge S_{\max}$ triggers an immediate, safe `VectraException("KVCache: position out of bounds")`.
   - Verified in unit tests: appending at $pos = 16$ on a cache sized for 16 tokens correctly threw and was caught.
2. **Layer Index Bounds:**
   - Accessing layer index $l < 0$ or $l \ge 28$ triggers a checked exception.
3. **No Garbage Collection / Zero Reallocation:**
   - The buffer is pre-allocated upon generation startup; during token generation, no `malloc` or `std::vector::resize` calls occur.
   - Pointers passed to `compute_gqa_attention` remain valid throughout the entire generation lifecycle.
