# Milestone M2 — Quantization Compatibility & Kernel Architecture

**Date:** 2026-11-18  
**Module:** `vectra::quant`  
**Location:** `include/vectra/quant/quant_k.hpp` & `src/quant/quant_k.cpp`  

---

## 1. Quantization Formats Implemented

In milestone M2, VECTRA implemented full support for the modern k-quantization super-block formats utilized by the official Qwen2.5 GGUF checkpoint.

### 1.1 BlockQ4_K (256 Elements, 144 Bytes, 4.5 bits/weight)

The Q4_K super-block groups 256 values into 8 sub-blocks of 32 values each:
- **`d` (2 bytes, FP16):** Global super-block scale factor.
- **`dmin` (2 bytes, FP16):** Global super-block minimum offset.
- **`scales` (12 bytes):** Packed 6-bit scale and minimum offsets for each of the 8 sub-blocks.
  - Sub-block scales $d_j \in [0, 63]$: `(scales[j] & 0x3F) | ((scales[j + 8] & 0x03) << 4)`
  - Sub-block mins $m_j \in [0, 63]$: `(scales[j + 4] & 0x3F) | ((scales[j + 8] & 0x0C) << 2)`
- **`qs` (128 bytes):** 256 4-bit nibbles packed two per byte.

**Reconstruction Formula:**
$$x_{j, k} = d \cdot d_j \cdot q_{j, k} - d_{\text{min}} \cdot m_j$$

### 1.2 BlockQ6_K (256 Elements, 210 Bytes, 6.56 bits/weight)

The Q6_K super-block groups 256 values into 16 sub-blocks of 16 values each:
- **`ql` (128 bytes):** Lower 4 bits for all 256 quants.
- **`qh` (64 bytes):** Upper 2 bits for all 256 quants (packed 4 quants per byte).
- **`scales` (16 bytes, `int8_t`):** Signed 8-bit scale factor for each 16-element sub-block.
- **`d` (2 bytes, FP16):** Global super-block scale factor.

**Reconstruction Formula:**
$$q_i = (ql_i \ \& \ 0x0F) \ | \ ((qh_i \ \& \ 0x03) \ll 4)$$
$$x_i = d \cdot \text{scales}[\lfloor i / 16 \rfloor] \cdot (q_i - 32)$$

---

## 2. AVX2 Vectorized Dot Product Acceleration

Matrix-vector multiplications for Q4_K and Q6_K utilize handwritten AVX2 kernels:
1. **Unpacking:** Nibbles and scale factors are decoded into 256-bit AVX2 registers (`__m256`).
2. **FMA Accumulation:** `_mm256_fmadd_ps` multiplies the dequantized weights directly with the activation vector `x`.
3. **Horizontal Reduction:** Fast horizontal sum across 8 single-precision float lanes via `_mm256_add_ps` and `_mm_add_ss`.

---

## 3. Numerical Verification & Parity

Executed via `tests/integration/m2_tests.cpp`:
- **`test_q4_k_correctness`:**
  - Constructed synthetic block with uniform scales and quants.
  - Dequantized 256 elements: verified max error $< 10^{-4}$.
  - Scalar vs AVX2 dot product against unit vector: **1280.000 vs 1280.000 (0.0000% error)**.
  - **Verdict:** **PASSED**.
- **`test_q6_k_correctness`:**
  - Constructed synthetic block with uniform scales and quants.
  - Dequantized 256 elements: verified max error $< 10^{-4}$.
  - Scalar vs AVX2 dot product against unit vector: **-7680.000 vs -7680.000 (0.0000% error)**.
  - **Verdict:** **PASSED**.
