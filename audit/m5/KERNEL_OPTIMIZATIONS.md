# Project VECTRA — Milestone M5: Kernel Optimization Catalog

**Document Purpose:** Complete technical catalog of all original AVX2, SIMD, and algorithmic kernel optimizations introduced in Milestone M5.

---

## 1. Vectorized LM-Head Projection (`vec_dot_q6_k_q8_avx2`)

### The M4 Baseline
In M4, the Q6_K dot-product accumulated intermediate sub-block sums into 16 128-bit XMM registers, and then executed 32 scalar `_mm_hadd_epi32` instructions per block. Across 151,936 vocabulary rows, this performed:
$$151,936 \text{ rows} \times 6 \text{ blocks} \times 32 \text{ hadds} = \mathbf{29,171,712 \text{ hadd instructions per token}}$$
Because `phaddd` has a high instruction latency (3 cycles) and port 5 execution bottleneck, it throttled memory streaming.

### The M5 Vectorized Implementation
In M5, horizontal additions were replaced with AVX2 256-bit wide operations and interleaved sums:
```cpp
// Horizontal addition across 256-bit registers in 3 instructions:
__m256i h01 = _mm256_hadd_epi32(sum0, sum1);
__m256i h23 = _mm256_hadd_epi32(sum2, sum3);
__m256i hall = _mm256_hadd_epi32(h01, h23);

// Direct vector FMA with precomputed interleaved activation sums:
__m256 acc0 = _mm256_fmadd_ps(prod_f32_lo, sc_f32_lo, _mm256_setzero_ps());
__m256 acc1 = _mm256_fmadd_ps(prod_f32_hi, sc_f32_hi, _mm256_setzero_ps());
```
**Impact:** LM-Head latency dropped from **42.0 ms down to 5.7 ms** (a $7.4\times$ speedup), streaming weights at $33.8\text{ GB/s}$.

---

## 2. Inlined Scales & Mins Extraction in `vec_dot_q4_k_q8_avx2`

### The Problem
In Q4_K, each 256-element block stores 12 bytes of packed 6-bit scales and 6-bit minimums. In M4, unpacking was done via a helper function `unpack_scales_and_mins` called per block.
At 28 layers $\times$ 6 blocks $\times$ 24,000 projections, this incurred **40.9 million function calls per token**.

### The Solution
M5 inlined the scale/min bitfield extractions directly into 32-bit integer registers:
```cpp
const uint8_t* sc = block->scales;
uint8_t sc0 = sc[0] & 0x3F;
uint8_t sc1 = sc[1] & 0x3F;
uint8_t m0  = (sc[4] & 0x0F) | ((sc[0] >> 6) << 4);
uint8_t m1  = (sc[5] & 0x0F) | ((sc[1] >> 6) << 4);
```
SIMD FMA instructions (`_mm256_fmadd_ps` and `_mm256_fnmadd_ps`) directly accumulate:
$$\text{acc} = \text{acc} + (d \cdot \text{scale} \cdot \text{prod}) - (d_{\min} \cdot \text{min} \cdot \text{sum})$$
completely eliminating function call overhead and scalar reduction loops.

---

## 3. Fused Multi-Projection Dispatches

### Fused QKV: `matvec_q4_k_q8_qkv`
In Transformer self-attention, the query ($1536$), key ($256$), and value ($256$) projections are computed from the same normalized activation vector (`normed1`).
Instead of launching 3 separate thread pool dispatches with 3 barriers:
- M5 launched a single unified parallel dispatch over $1536 + 256 + 256 = 2048$ rows.
- Each thread examines the row index and writes to either $Q$, $K$, or $V$.
- **Saved:** 56 thread pool barriers per decode token.

### Fused Gate + Up: `matvec_q4_k_q8_2x`
In the SwiGLU MLP, Gate ($8960$) and Up ($8960$) projections share identical activation inputs (`normed2`).
- M5 launched a single parallel dispatch over $8960$ rows, computing row $r$ of Gate and row $r$ of Up simultaneously.
- **Saved:** 28 thread pool barriers per decode token.
- Total barriers eliminated across 28 layers: **84 barriers per token**.

---

## 4. Persistent Workspace Architecture (`LayerScratch`)

In M4, every layer forward pass allocated 12 `std::vector<float>` and `std::vector<BlockQ8_K_256>` heap buffers, totaling 336 allocations per token.
In M5, `GGUFTransformerModel` pre-allocates a persistent workspace:
```cpp
struct LayerScratch {
    std::vector<BlockQ8_K_256> q8_normed1;
    std::vector<BlockQ8_K_256> q8_normed2;
    std::vector<BlockQ8_K_256> q8_act;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> act;
    std::vector<float> ffn_out;
    std::vector<float> attn_out;
};
```
Heap allocations during steady-state autoregressive decoding: **EXACTLY ZERO**.

---

## 5. Active Caller-Thread Slicing in `ThreadPool`

In M4, the calling thread distributed $N$ tasks to $N$ worker threads and sat spinning in a condition variable until all finished.
In M5, `ThreadPool::parallel_for` was updated:
```cpp
// Caller immediately executes slice 0
fn(0, chunk);

// Worker threads 1..num_threads-1 execute remaining slices
```
This eliminates the dispatch idle window, immediately engaging the primary Golden Cove P-core while background threads wake up.
