# Project VECTRA — Milestone M3: Q6_K Quantized Kernel & Execution Optimization Report

**Target Format:** GGUF `Q6_K` Super-block Format  
**Block Size:** 256 weights per block (210 bytes = 6.5625 bits per weight)  
**Tensors Employing Q6_K in Qwen2.5-1.5B-Instruct:**
1. All 28 Transformer layers: `blk.i.ffn_down.weight` ($1536 \times 8960$)
2. LM Vocabulary Head: `output.weight` / `token_embd.weight` ($151936 \times 1536$)
**Total Q6_K Memory Footprint:** 454.9 MB (42.7% of total model memory)

---

## 1. Q6_K Mathematical Representation & Packing Layout

The `Q6_K` format represents each weight as a 6-bit integer $q_i \in [-32, 31]$ (or unsigned $[0, 63]$ with a bias of $-32$). Because 6 bits do not align to byte boundaries, bits are split across two arrays:
```cpp
#pragma pack(push, 1)
struct BlockQ6_K {
    uint8_t ql[128];     // Lower 4 bits for 256 weights (128 bytes)
    uint8_t qh[64];      // Upper 2 bits for 256 weights (64 bytes)
    int8_t  scales[16];  // 16 signed 8-bit scales (16 bytes)
    uint16_t d;          // Master super-block scale (FP16, 2 bytes)
};
#pragma pack(pop)
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must be 210 bytes");
```

### Quantization Mapping:
Each block of 256 weights is partitioned into 16 sub-blocks of 16 weights each.
For sub-block $j \in [0, 15]$:
- Super-block scale: $d = \text{fp16\_to\_fp32}(\text{blk.d})$
- Sub-block scale: $sc_j = \text{blk.scales}[j]$
- Effective sub-block scale: $d_j = d \cdot sc_j$
- For weight $i \in [0, 15]$ within sub-block $j$:
  - Low 4 bits: $ql$ from `blk.ql`
  - High 2 bits: $qh$ from `blk.qh`
  - Reconstructed 6-bit quant: $q \in [-32, 31]$
  - Dequantized value:
    $$w_i = d_j \cdot q$$

---

## 2. Inefficiencies & Challenges of Q6_K Unpacking

1. **Non-Trivial Bit Extraction:**
   Each byte in `qh` holds 2-bit slices for 4 separate weights located at offsets $+0, +32, +64, +96$.
   Unpacking requires extracting bit pairs: $(qh \gg 0) \& 3$, $(qh \gg 2) \& 3$, $(qh \gg 4) \& 3$, and $(qh \gg 6) \& 3$, followed by shifting left by 4 and ORing with $ql$.
2. **Dominance in LM Head Execution:**
   The LM Head projects the final 1536-dimensional hidden state to 151,936 logits.
   This requires performing:
   $$151936 \times 6 = 911,616 \text{ Q6\_K block dot products per token!}$$
   In single-threaded execution, this single matrix-vector multiply took **~2.87 seconds**, single-handedly bottlenecking M2 inference.

---

## 3. M3 Kernel Optimization Strategies

### A. AVX2 Vectorized Row Dequantization & Dot Products
- Unpacks 16-element sub-blocks using AVX2 SIMD instructions.
- Converts 8-bit signed scales to 32-bit floats with `_mm256_cvtepi32_ps`.
- Multiplies scales $d_j = d \cdot sc_j$ in registers.
- Accumulates inner products directly using FMA `_mm256_fmadd_ps`.

### B. Dynamic Thread Pool Chunking for LM Head
Because the LM Head contains 151,936 independent dot products with no cross-row dependencies:
- The persistent `ThreadPool` dynamically splits the 151,936 rows across $N$ worker threads:
  $$\text{Chunk Size} = \left\lceil \frac{151936}{N} \right\rceil$$
- On 10 threads, each thread evaluates ~15,194 rows.
- Cache footprint: The input activation vector (1536 floats = 6 KB) fits entirely inside each core's 32 KB / 48 KB L1 Data Cache.
- The 191.4 MB of Q6_K weights stream linearly through L2 and L3 caches across all cores without thread contention.

---

## 4. Measured Performance

| Kernel / Operator | Dimensions | Scalar 1T Time | AVX2 1T Time | AVX2 10T Time | SIMD Speedup | Multi-Thread Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q6_K Down Proj** | $1536 \times 8960$ | $56.9\text{ ms}$ | $17.8\text{ ms}$ | $4.2\text{ ms}$ | **3.20x** | **4.22x** |
| **Q6_K LM-Head** | $151936 \times 1536$ | $2873.3\text{ ms}$ | $1989.2\text{ ms}$ | $442.0\text{ ms}$ | **1.44x** | **4.50x** |

### Total LM Head Speedup:
$$\text{Overall Speedup} = \frac{2873.3\text{ ms}}{442.0\text{ ms}} = \mathbf{6.50x}$$

By cutting the LM Head evaluation time from $2.87\text{ s}$ to $0.44\text{ s}$, autoregressive decode was immediately unlocked from $0.22\text{ tok/s}$ to $> 3.0\text{ tok/s}$.
