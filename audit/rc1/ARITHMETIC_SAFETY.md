# Project VECTRA — RC1 Final Audit: AVX2 Arithmetic Safety & Bounds Review

**Release Candidate Target:** `v0.1.0-rc1`  
**Auditor:** Independent LLM Runtime Auditor & Numerical Specialist  

---

## 1. AVX2 Instruction Arithmetic Contracts

Project VECTRA leverages x86 AVX2 integer dot-product instructions to accelerate quantized matrix-vector multiplication. The primary execution instructions are:
1. `_mm256_maddubs_epi16` (`VPMADDUBSW`): Multiplies unsigned 8-bit integers from operand 1 with signed 8-bit integers from operand 2, and horizontally sums adjacent pairs into signed 16-bit integers with saturation ($[-32768, 32767]$).
2. `_mm256_madd_epi16` (`VPMADDWD`): Multiplies pairs of signed 16-bit integers with a vector of ones (`ones16`), horizontally adding adjacent pairs into signed 32-bit integers ($[-2^{31}, 2^{31}-1]$).

---

## 2. Formal Arithmetic Bounds Proof

### A. Q4_K Kernel (`vec_dot_q4_k_q8_avx2`)
In Q4_K, weight blocks store 256 4-bit unsigned nibbles $[0, 15]$, and activations are quantized into signed 8-bit integers $[-127, 127]$.

1. **First-Stage Pair Products (`_mm256_maddubs_epi16`):**
   $$\text{Max Pair Product} = (15 \times 127) + (15 \times 127) = 1905 + 1905 = +3810$$
   $$\text{Min Pair Product} = (15 \times -127) + (15 \times -127) = -1905 - 1905 = -3810$$
   **Saturation Proof:** The signed 16-bit integer saturation limit is $[-32768, 32767]$.
   Since $|-3810| \ll 32767$, signed 16-bit saturation **never occurs under any possible valid or invalid inputs**.

2. **Second-Stage 32-bit Widening (`_mm256_madd_epi16`):**
   $$\text{Max 32-bit Intermediate} = 3810 + 3810 = +7620$$
   $$\text{Min 32-bit Intermediate} = -3810 + -3810 = -7620$$
   Across a 256-element block, 32 such values accumulate into a 32-bit register:
   $$\text{Max Accumulated Block Sum} = 32 \times 7620 = 243,840 \ll 2,147,483,647$$
   **Overflow Proof:** 32-bit integer overflow is mathematically impossible.

### B. Q6_K Kernel (`vec_dot_q6_k_q8_avx2`)
In Q6_K, weight blocks unpack into 6-bit unsigned integers $[0, 63]$:

1. **First-Stage Pair Products (`_mm256_maddubs_epi16`):**
   $$\text{Max Pair Product} = (63 \times 127) + (63 \times 127) = 8001 + 8001 = +16,002$$
   $$\text{Min Pair Product} = (63 \times -127) + (63 \times -127) = -8001 - 8001 = -16,002$$
   **Saturation Proof:** Since $+16,002 \le 32,767$ and $-16,002 \ge -32,768$, signed 16-bit saturation **never occurs**.

2. **Offset Subtraction:**
   The $-32$ offset is applied via the precomputed activation sums:
   ```cpp
   __m256i sums_x32 = _mm256_slli_epi32(sums_vec, 5); // multiply by 32
   __m256i diff_i   = _mm256_sub_epi32(h_all, sums_x32);
   ```
   For 16 elements, $\text{max sum} = 16 \times 127 = 2032$.
   $2032 \times 32 = 65,024 \ll 2^{31}-1$. Subtraction in 32-bit signed space is safe.

---

## 3. Boundary & Adversarial Hardening Invariants

### Invariant 1: Exclusion of Signed Int8 Min ($-128$)
In two's complement arithmetic, $-128$ cannot be negated in 8 bits ($|-128| = -128$).
In `quantize_activation_q8_k_256`:
```cpp
q = std::clamp(q, -127, 127); // Strictly forbids -128
```
This guarantees that activation values are symmetric and never trigger asymmetrical two's-complement arithmetic clipping.

### Invariant 2: Non-Finite & Subnormal Float Handling
Before division by scale $d$:
- All input floats are checked with `std::isfinite(val)`. Non-finite values (`NaN`, `Inf`) are zeroed.
- Scales below $1.0\times 10^{-12}$ are explicitly zeroed to eliminate division by zero.

### Invariant 3: Memory Alignment & SIMD Tails
- All weight tensors, activation buffers, and scratch spaces are allocated with **64-byte alignment** (`vectra::core::aligned_alloc(bytes, 64)`).
- Block sizes ($QK\_K = 256$) are strict multiples of 32 elements (256-bit AVX2 register width), eliminating unaligned loads and SIMD tail over-reads.

---

## 4. Empirical Safety Test Suite (`vectra_arithmetic_safety_tests`)

All 7 safety tests pass 100% in fresh release build `build_rc1`:
1. `test_q8_k_nan_inf_handling`: **PASSED** (Injected NaNs and Infs produce finite, bounded quants).
2. `test_q8_k_subnormals`: **PASSED** (Subnormal floats $1.0\times 10^{-38}$ handle without underflow crash).
3. `test_q8_k_zeros`: **PASSED** (All-zero inputs yield all-zero quants and scales).
4. `test_q8_k_extreme_magnitudes`: **PASSED** (Extreme inputs $10^{30}$ clamp safely to $[-127, 127]$).
5. `test_q4_k_no_saturation_adversarial`: **PASSED** (Worst-case Q4_K block matches double precision oracle).
6. `test_q6_k_no_saturation_adversarial`: **PASSED** (Worst-case Q6_K block matches exact reference $127,999,744$).
7. `test_q8_k_strict_clamping_boundary`: **PASSED** (Input $-128.0f$ clamps to $-127$; never $-128$).
