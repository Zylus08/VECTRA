# Tensor Runtime Audit: Data Structures, Memory & Kernels

## Design Principles
1. **Explicit Ownership**: `Tensor` owns an `AlignedBuffer`, providing deterministic RAII resource deallocation upon destruction. Copy construction is explicitly deleted to prevent accidental memory duplications.
2. **Non-Owning Views**: `TensorView` provides a safe non-owning window onto contiguous or sliced sub-tensors with offset arithmetic and stride validation.
3. **64-Byte Alignment**: All allocations are aligned to 64-byte boundaries, matching cache lines and AVX-512/AVX2 vector alignments.
4. **Checked Arithmetic**: Tensor shape and stride calculations use overflow-checked multiplication, throwing `VectraException` on integer wrap-around.

## Supported Data Types
- `FP32`: Standard single-precision float for intermediate activations.
- `FP16`: IEEE 754 half-precision float conversion utilities (`float32_to_float16` and `float16_to_float32`) supporting subnormals, infinities, and rounding to nearest even.
- `INT32` / `INT64`: Integer types used for token sequence indexing and positional offsets.
- `Q4_0`: 4-bit per-block quantization format (18 bytes per 32 elements).

## Implemented Mathematical Kernels

| Kernel | Scalar Reference | AVX2 Optimized | Verification Status |
|---|---|---|---|
| Vector Addition | Loop unrolled | `_mm256_add_ps` | PASS |
| Vector Multiplication | Elementwise product | `_mm256_mul_ps` | PASS |
| Scalar Scaling | Elementwise scale | `_mm256_mul_ps` with broadcast | PASS |
| Dot Product | Accumulator loop | Dual FMA accumulators + horizontal sum | PASS (7.0x speedup) |
| Matrix-Vector Multiplication | Row-wise dot product | Vectorized row dot with FMA | PASS (10.4x speedup) |
| RMSNorm | Two-pass mean-square + inv-sqrt | Vectorized square sum + vector scale | PASS (2.8x speedup) |
| Softmax | Max-subtraction + exp + sum | Stable scalar exponential | PASS |
| SiLU | $x / (1 + e^{-x})$ | Stable scalar logistic | PASS |
| SwiGLU | $\text{silu}(\text{gate}) \cdot \text{up}$ | Fused activation | PASS |
| RoPE | Frequency half-split rotation | Standard half-split rotation | PASS |
| GQA Attention | Scaled dot-product + causal mask | Multi-head masked attention | PASS |
