# Project VECTRA — Milestone M4: Q4_K INT8 Kernel Architecture & AVX2 DP4A Optimization

**Component:** `vec_dot_q4_k_q8_avx2` / `matvec_q4_k_q8`  
**Header:** `include/vectra/quant/quant_k.hpp`  
**Source:** `src/quant/quant_k.cpp`

---

## 1. Mathematical Formulation

A `BlockQ4_K` super-block quantizes $K=256$ weights into 8 sub-blocks of 32 elements:
$$w_j = d \cdot \text{scale}_{j/32} \cdot q_j - d_{\min} \cdot \min_{j/32}$$
where $q_j \in [0, 15]$ is an unsigned 4-bit integer, and $\text{scale}, \min \in [0, 63]$ are 6-bit sub-block parameters.

When multiplying against an activation vector $x$, we quantize $x$ into 8-bit signed integers $a_j \in [-127, 127]$ with sub-block scale $s_{\text{act}, j/32}$:
$$x_j = s_{\text{act}, j/32} \cdot a_j$$

The dot product over each 32-element sub-block is evaluated as:
$$\sum_{j=0}^{31} w_j x_j = \sum_{j=0}^{31} \left( d \cdot \text{scale} \cdot q_j - d_{\min} \cdot \min \right) \cdot \left( s_{\text{act}} \cdot a_j \right)$$
$$= s_{\text{act}} \cdot \left[ d \cdot \text{scale} \cdot \left( \sum_{j=0}^{31} q_j a_j \right) - d_{\min} \cdot \min \cdot \left( \sum_{j=0}^{31} a_j \right) \right]$$

### Key Algorithmic Invariant
The activation sum $\sum_{j=0}^{31} a_j$ is precomputed **once per activation vector** during quantization and stored in `BlockQ8_K_256::sums32[sb]`. It is completely independent of the weight rows, eliminating $N$ repeated summations during matrix-vector multiplication.

---

## 2. AVX2 Instruction Pipeline

To compute $\sum_{j=0}^{31} q_j a_j$ across 32 elements simultaneously:

1. **Weight Nibble Extraction:**
   ```cpp
   __m256i raw32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
   __m256i q0 = _mm256_and_si256(raw32, _mm256_set1_epi8(0x0F));
   __m256i q1 = _mm256_and_si256(_mm256_srli_epi16(raw32, 4), _mm256_set1_epi8(0x0F));
   ```
   A single 32-byte load provides all 64 nibbles (two 32-element sub-blocks).

2. **SIMD Dot-Product with `_mm256_maddubs_epi16`:**
   ```cpp
   __m256i prod0_16 = _mm256_maddubs_epi16(q0, a0);
   __m256i prod1_16 = _mm256_maddubs_epi16(q1, a1);
   ```
   `_mm256_maddubs_epi16` performs 32 unsigned $\times$ signed byte multiplications and horizontally adds adjacent pairs into sixteen 16-bit signed integers in **1 clock cycle**!
   - Max product: $15 \times 127 = 1,905$.
   - Max sum of pair: $1,905 + 1,905 = 3,810$.
   - Bound check: $3,810 < 32,767$ (zero possibility of 16-bit saturation arithmetic overflow).

3. **32-Bit Widening & Horizontal Sum:**
   ```cpp
   __m256i prod0_32 = _mm256_madd_epi16(prod0_16, ones16);
   ```
   Accumulates pairs of 16-bit integers into eight 32-bit signed integers. Lane extraction and `_mm_hadd_epi32` collapse the 8 integers into a scalar `int32_t dot_int0`.

4. **Fused Scale & Min Accumulation:**
   ```cpp
   total_sum += s_act0 * (dl0 * dot_int0 - ml0 * b[i].sums32[sb0]);
   ```

---

## 3. Microbenchmark & Parity Verification

In microbenchmark tests on $1536 \times 1536$ matrices (`Q_proj` / `O_proj` dimensions):
- **FP32 AVX2 FMA Time:** $2,046.6\text{ }\mu\text{s}$ per call.
- **INT8 AVX2 DP4A Time:** $1,149.7\text{ }\mu\text{s}$ per call.
- **Speedup:** **$1.78\times$ faster execution**.
- **Numerical Error:** Max absolute error $< 0.00048$; relative error $< 0.025\%$.
