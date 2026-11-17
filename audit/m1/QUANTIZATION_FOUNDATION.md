# Quantization Foundation Audit: Introductory 4-bit Format

## Specification & Block Layout
Project VECTRA introduces an introductory symmetric 4-bit per-block quantization format (`BlockQ4_0`):

```
+-------------------------------------------------------------+
|                     BlockQ4_0 Structure                     |
+------------------------------+------------------------------+
| uint16_t d (FP16 scale)      | 2 bytes                      |
| uint8_t qs[16] (32 nibbles)  | 16 bytes                     |
+------------------------------+------------------------------+
Total Block Footprint: 18 bytes for 32 scalar values (4.5 bits/weight)
```

### Quantization Formula
For a block of 32 elements $x_0 \dots x_{31}$:
1. Find peak magnitude: $M = \max_{i} |x_i|$.
2. Compute FP16 scale: $d = M / 7.0$ (with $d = 0$ if $M = 0$).
3. Quantize each element to 4-bit integer $q_i \in [-8, 7]$:
   $$q_i = \text{clamp}\left(\text{round}\left(\frac{x_i}{d}\right), -8, 7\right)$$
4. Store packed nibble with zero-point offset 8 ($u_i \in [0, 15]$):
   $$u_i = q_i + 8$$
   Byte $k$ packs $u_{2k}$ in lower 4 bits and $u_{2k+1}$ in upper 4 bits.

### Dequantization Formula
To reconstruct element $i$:
$$x_i \approx (u_i - 8) \times d$$

## Direct Quantized Dot Product
Rather than allocating an intermediate dequantized array, VECTRA computes matrix-vector multiplication directly from packed blocks:
$$\text{dot} = \sum_{\text{block } b} d_b \cdot \left(\sum_{j=0}^{31} (u_{b,j} - 8) \cdot y_{b,j}\right)$$

## Differentiation from GGUF Q4_K_M
It is critical to distinguish this introductory format from GGUF `Q4_K_M`:
- **Block Size**: VECTRA `BlockQ4_0` operates on 32-element blocks with a single uniform FP16 scale.
- **Q4_K Structure**: GGUF `Q4_K` utilizes 256-element super-blocks composed of 8 sub-blocks of 32 elements each, with 6-bit scales, 6-bit mins, and non-linear scale encodings.
- **Interoperability Claim**: Compatibility for GGUF Q4_K_M will be developed in Milestone M2 in a dedicated quantization module without destabilizing the verified reference implementation.

## Empirical Error Metrics
- Evaluated in `tests/unit/unit_tests.cpp` on continuous input signals:
  - Mean Squared Error (MSE): $0.104$
  - Root Mean Squared Error (RMSE): $0.322$
  - Signal-to-Noise Ratio (SNR): **25.09 dB**
- Evaluated on matrix-vector multiplication in `tests/integration/integration_tests.cpp`:
  - Full Matvec SNR: **> 15.0 dB** (exceeding minimum fidelity threshold for 4-bit weights).
- Edge Cases Handled:
  - Exact preservation of all-zero blocks ($d = 0$, packed $u = 8$).
  - Correct zero-padding on partial blocks where tensor dimension is not an exact multiple of 32.
