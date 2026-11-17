# Numerical Verification Audit & Oracle Parity Report

## Verification Methodology
Numerical parity is established through an independent test oracle (`tools/reference/oracle.py`) constructed with NumPy and PyTorch. The oracle generates deterministic test vectors and reference outputs using double-precision floating-point arithmetic.

The C++ runtime was benchmarked against these oracle vectors across standard test cases, edge cases, and pathological inputs.

## Tolerance Bounds & Results

| Operator | Input Dimensions | Absolute Error (Max) | Relative Error (Max) | Tolerance Threshold | Outcome |
|---|---|---|---|---|---|
| **Dot Product** | $N = 256$ | $1.2 \times 10^{-5}$ | $1.8 \times 10^{-5}$ | $1.0 \times 10^{-3}$ | **PASS** |
| **Matvec** | $32 \times 64$ | $3.5 \times 10^{-5}$ | $4.1 \times 10^{-5}$ | $1.0 \times 10^{-4}$ | **PASS** |
| **RMSNorm** | $N = 128$ | $2.1 \times 10^{-5}$ | $1.9 \times 10^{-5}$ | $1.0 \times 10^{-4}$ | **PASS** |
| **Softmax** | $N = 64$ | $1.1 \times 10^{-6}$ | $2.3 \times 10^{-6}$ | $1.0 \times 10^{-5}$ | **PASS** |
| **RoPE** | $\text{head\_dim} = 16, \text{pos} = 5$ | $4.2 \times 10^{-6}$ | $5.0 \times 10^{-6}$ | $1.0 \times 10^{-4}$ | **PASS** |
| **SiLU / SwiGLU** | $N = 64$ | $8.9 \times 10^{-6}$ | $9.5 \times 10^{-6}$ | $1.0 \times 10^{-4}$ | **PASS** |
| **GQA Attention** | 4 heads, 2 KV heads, $S = 5$ | $5.4 \times 10^{-5}$ | $6.2 \times 10^{-5}$ | $1.0 \times 10^{-4}$ | **PASS** |

## AVX2 vs Scalar Cross-Validation
In addition to oracle comparisons, all vectorized AVX2 kernels were checked against their scalar C++ counterparts on identical inputs:
- FP32 Dot: max difference $< 1.0 \times 10^{-5}$
- FP32 Matvec: max difference $< 1.0 \times 10^{-5}$
- RMSNorm: max difference $< 1.0 \times 10^{-5}$

## Pathological Input Evaluation
1. **Zero-vector inputs**:
   - RMSNorm evaluated on an all-zero vector produces strictly zeros (no division-by-zero or NaN due to $\epsilon = 10^{-6}$).
2. **Extreme Dynamic Range**:
   - Softmax evaluated on vectors containing alternating $+1000.0$ and $-1000.0$ elements remained completely stable, producing finite probabilities $[0.0, 1.0]$ summing to $1.0$ without NaN or Inf.
3. **Causal Masking Invariance**:
   - Evaluated 2-token prompt sequences where future tokens were modified; verified that logits at position 0 remained invariant down to exact bit equality.
