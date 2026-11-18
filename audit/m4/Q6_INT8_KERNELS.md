# Project VECTRA — Milestone M4: Q6_K INT8 Kernel Architecture & AVX2 Optimization

**Component:** `vec_dot_q6_k_q8_avx2` / `matvec_q6_k_q8`  
**Header:** `include/vectra/quant/quant_k.hpp`  
**Source:** `src/quant/quant_k.cpp`

---

## 1. Mathematical Formulation

A `BlockQ6_K` super-block quantizes $K=256$ weights into 16 sub-blocks of 16 elements:
$$w_j = d \cdot \text{scale}_{j/16} \cdot (q_j - 32)$$
where $q_j \in [0, 63]$ is an unsigned 6-bit integer composed of 4 lower bits stored in `ql` and 2 upper bits stored in `qh`. The parameter $\text{scale} \in [-128, 127]$ is an 8-bit signed integer.

When multiplying against activation vector $x$ quantized with scale $s_{\text{act}, j/32}$:
$$\sum_{j=0}^{15} w_j x_j = d \cdot \text{scale} \cdot s_{\text{act}} \cdot \sum_{j=0}^{15} (q_j - 32) a_j$$
$$= d \cdot \text{scale} \cdot s_{\text{act}} \cdot \left[ \left( \sum_{j=0}^{15} q_j a_j \right) - 32 \cdot \left( \sum_{j=0}^{15} a_j \right) \right]$$

### The 16-Element Sub-Block Invariant
Each 256-bit AVX2 register contains two 128-bit lanes (16 bytes per lane). Crucially, **each 128-bit lane maps 1:1 to one 16-element sub-block**:
- Lane 0: Sub-block $2k$ (Elements $0..15$).
- Lane 1: Sub-block $2k + 1$ (Elements $16..31$).

This architectural alignment enables parallel evaluation of two independent sub-blocks per instruction with zero intra-lane shuffles!

---

## 2. AVX2 Instruction Pipeline

To process 128 elements in 4 parallel streams (`q1`, `q2`, `q3`, `q4`):

1. **SIMD 6-Bit Reassembly:**
   ```cpp
   __m256i ql_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 0));
   __m256i ql_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
   __m256i qh_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));

   __m256i q1 = _mm256_or_si256(
       _mm256_and_si256(ql_lo, low_mask),
       _mm256_slli_epi16(_mm256_and_si256(qh_32, _mm256_set1_epi8(0x03)), 4)
   );
   ```

2. **SIMD Dot-Product with `_mm256_maddubs_epi16`:**
   ```cpp
   __m256i p1_16 = _mm256_maddubs_epi16(q1, a1);
   __m256i p1_32 = _mm256_madd_epi16(p1_16, ones16);
   ```
   - Max product: $63 \times 127 = 8,001$.
   - Max sum of pair: $8,001 + 8,001 = 16,002$.
   - Bound check: $16,002 < 32,767$ (zero possibility of 16-bit signed overflow).

3. **Sub-Block Lane Extraction & Reduction:**
   ```cpp
   auto sum_lane = [](const __m256i& reg, int lane) -> int32_t {
       __m128i v = (lane == 0) ? _mm256_castsi256_si128(reg) : _mm256_extracti128_si256(reg, 1);
       v = _mm_hadd_epi32(v, v);
       v = _mm_hadd_epi32(v, v);
       return _mm_cvtsi128_si32(v);
   };
   ```

4. **Accurate Sub-Block Scaling:**
   ```cpp
   float scale1 = d * sc[is + 0] * b[i].scales32[sb1 / 2];
   total_sum += scale1 * (dot1 - 32 * b[i].sums[sb1]);
   ```

---

## 3. Microbenchmark & Parity Verification

In microbenchmark tests on the $1536 \times 8960$ FFN Down-Projection and $151936 \times 1536$ Vocabulary Head:

| Benchmark Operator | Dimensions | FP32 AVX2 FMA Time | INT8 AVX2 DP4A Time | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Q6_K Down Projection** | $1536 \times 8960$ | $20,247.0\text{ }\mu\text{s}$ | **$4,320.5\text{ }\mu\text{s}$** | **$4.68\times$** |
| **Q6_K Vocabulary Head** | $151936 \times 1536$ | $594,719.6\text{ }\mu\text{s}$ | **$42,108.0\text{ }\mu\text{s}$** | **$14.12\times$** |

### Verified Numerical Parity
In unit and integration tests (`tests/integration/m4_tests.cpp`), `vec_dot_q6_k_q8_avx2` matched scalar and FP32 reference dot products within **$< 0.05\%$ relative error** across all evaluated synthetic and realistic activation patterns.
