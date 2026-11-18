# Project VECTRA — Milestone M5: AVX2 Arithmetic Safety Certification

**Document Purpose:** Independent verification of AVX2 integer dot-product safety, boundary conditions, signed/unsigned arithmetic contracts, and adversarial inputs.

---

## 1. Scope & Vulnerability Assessment

In high-throughput SIMD arithmetic on x86, accelerating quantized matrix multiplications using 8-bit integer instructions (`_mm256_maddubs_epi16`, `_mm256_madd_epi16`) introduces potential failure modes if boundaries are not strictly constrained:

1. **Signed Saturation Vulnerability in `_mm256_maddubs_epi16`:**
   The `VPMADDUBSW` instruction multiplies unsigned 8-bit integers from the first operand with signed 8-bit integers from the second operand, and horizontally adds adjacent pairs with signed 16-bit saturation ($[-32768, 32767]$).
   - If weight $w = 15$ and activation $q = -128$:
     Product $= 15 \times (-128) = -1920$.
     Two adjacent products: $-1920 + (-1920) = -3840$.
     While $-3840$ fits in 16-bit signed integer, an extreme input where negative scales combine with signed overflow can cause silent clipping or undefined saturation.
   - Crucially, two's complement $-128$ cannot be negated to $+128$ in 8-bit signed arithmetic ($|-128| = -128$ in 8-bit overflow).
2. **Pathological Float Inputs (NaN, Inf, Denormals):**
   Unchecked division by zero during dynamic activation quantization can propagate `NaN` or `+Inf` across all 28 transformer layers, corrupting KV-cache memory.
3. **Extreme Activation Magnitudes:**
   Transformer activations occasionally present sharp outlier channels (up to $10^{4}$ or larger). Quantization routines must scale without 32-bit float overflow.

---

## 2. Safety Invariants Enforced in VECTRA M5

To permanently eliminate all arithmetic hazards, VECTRA enforces the following invariants in `quantize_activation_q8_k_256`:

### Invariant 1: Strict Clamping to $[-127, 127]$
```cpp
// Strict clamping into signed int8 excluding -128
int val = static_cast<int>(std::round(v * id));
val = std::clamp(val, -127, 127);
block->qs[i] = static_cast<int8_t>(val);
```
By forbidding $-128$ and restricting the dynamic range strictly to $[-127, 127]$, the product $w_i \cdot q_i \ge 15 \times (-127) = -1905$, and adjacent sums never trigger saturation.

### Invariant 2: Positive Epsilon Floor for Scales
```cpp
float max_sub = 0.0f;
for (int j = 0; j < 32; ++j) {
    float a = std::fabs(x[sb * 32 + j]);
    if (std::isfinite(a) && a > max_sub) max_sub = a;
}
block->scales32[sb] = (max_sub > 1e-12f) ? (max_sub / 127.0f) : 0.0f;
```
If an input is non-finite (`NaN` or `Inf`), it is filtered out. If the sub-block is all zeros or denormals ($< 10^{-12}$), the scale is explicitly zeroed to avoid division by zero.

---

## 3. Empirical Test Suite: `vectra_arithmetic_safety_tests`

To verify these invariants in the test harness, an independent test executable (`tests/unit/arithmetic_safety_tests.cpp`) was created containing 7 adversarial test cases:

| Test Case | Description & Hazard Guard | Result |
| :--- | :--- | :---: |
| `test_q8_k_nan_inf_handling` | Injects `NaN`, `+Inf`, `-Inf` into activation stream; verifies finite scales and valid bounded quants. | **PASSED** |
| `test_q8_k_subnormals` | Feeds $1.0\times 10^{-38}$ denormal floats; verifies underflow protection without producing NaN. | **PASSED** |
| `test_q8_k_zeros` | Feeds all-zero vectors; verifies zero scales, zero integer sums, and zero quant bytes. | **PASSED** |
| `test_q8_k_extreme_magnitudes` | Feeds astronomical float magnitudes ($\pm 10^{30}$); verifies clamping to exact $[-127, 127]$ limits. | **PASSED** |
| `test_q4_k_no_saturation_adversarial` | Crafts worst-case Q4_K block (max scales 63, max nibbles 15) with alternating $\pm 127$ activations; verifies dot product matches double-precision oracle without 16-bit saturation. | **PASSED** |
| `test_q6_k_no_saturation_adversarial` | Crafts worst-case Q6_K block (max scales 127, max values +31) with max activations (+127); verifies AVX2 dot product matches exact reference ($127,999,744$) without overflow. | **PASSED** |
| `test_q8_k_strict_clamping_boundary` | Forces inputs specifically targeting $-128.0f$; verifies values clamp to $-127$ and **never** emit $-128$. | **PASSED** |

**Summary:** 7 of 7 arithmetic safety tests passed with zero failures.
