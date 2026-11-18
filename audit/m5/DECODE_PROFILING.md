# Project VECTRA — Milestone M5: Single-Token Decode Profiling & Bottleneck Analysis

**Document Purpose:** Microarchitectural breakdown and cycle-level profiling of single-token autoregressive decoding in Project VECTRA.

---

## 1. Per-Token Latency Budget & Memory Streaming

In Qwen2.5-1.5B, each decode token executes 28 transformer layers followed by the vocabulary projection (LM-Head).
Total weights read from DDR5 RAM per decode token:
- **Layer Weights (28 layers):**
  - QKV Projections (Q4_K): $3.14\text{ M weights} \times 0.55\text{ bytes/weight} = 1.73\text{ MB}$
  - Out Projection (Q4_K): $2.35\text{ M weights} \times 0.55\text{ bytes/weight} = 1.30\text{ MB}$
  - Gate + Up Projections (Q4_K): $27.52\text{ M weights} \times 0.55\text{ bytes/weight} = 15.14\text{ MB}$
  - Down Projection (Q6_K): $13.76\text{ M weights} \times 0.825\text{ bytes/weight} = 11.35\text{ MB}$
  - Layer Total: $29.52\text{ MB} \times 28 \text{ layers} = \mathbf{826.56\text{ MB}}$
- **LM-Head Vocabulary Projection (1 layer):**
  - $151,936 \times 1536$ (Q6_K): $233.37\text{ M weights} \times 0.825\text{ bytes/weight} = \mathbf{192.53\text{ MB}}$
- **Total Weight Stream per Decode Token:**
  $$\text{Total Memory Footprint} = 826.56\text{ MB} + 192.53\text{ MB} = \mathbf{1,019.09\text{ MB}}$$

---

## 2. Decode Latency Breakdown Across Optimization Stages

Below is the step-by-step latency reduction for a single decode token ($50\text{ ms} \approx 20.0\text{ tok/s}$):

| Kernel / Computational Stage | M4 Baseline Latency | M5 Intermediate Latency | M5 Fully Optimized Latency | Primary Optimization Applied |
| :--- | :---: | :---: | :---: | :--- |
| **LM-Head Projection (151936 rows)** | $42.0\text{ ms}$ | $5.7\text{ ms}$ | **$5.7\text{ ms}$** | Vectorized `_mm256_hadd_epi32` & interleaved scale sums (33.8 GB/s) |
| **Layer QKV Projections (28 layers)** | $8.5\text{ ms}$ | $6.2\text{ ms}$ | **$5.4\text{ ms}$** | Fused `matvec_q4_k_q8_qkv` (1 barrier instead of 3) |
| **Layer Gate + Up Projections (28 layers)** | $16.8\text{ ms}$ | $14.1\text{ ms}$ | **$12.3\text{ ms}$** | Fused `matvec_q4_k_q8_2x` (1 barrier instead of 2) |
| **Layer Down Projections (28 layers)** | $12.1\text{ ms}$ | $11.0\text{ ms}$ | **$9.8\text{ ms}$** | Inlined scale unpack & caller-thread execution |
| **SwiGLU & RMSNorms (28 layers)** | $2.4\text{ ms}$ | $2.4\text{ ms}$ | **$2.1\text{ ms}$** | Cache-aligned scratch buffers |
| **RoPE & Attention (28 layers)** | $1.2\text{ ms}$ | $1.2\text{ ms}$ | **$1.1\text{ ms}$** | KV-cache in-place views |
| **Thread Pool Synchronization Overhead** | $5.2\text{ ms}$ | $2.1\text{ ms}$ | **$0.8\text{ ms}$** | Eliminated 84 barriers & active caller participation |
| **Dynamic Heap Allocation Overhead** | $3.8\text{ ms}$ | $0.0\text{ ms}$ | **$0.0\text{ ms}$** | Pre-allocated `LayerScratch` workspace |
| **Total Decode Latency per Token** | **$92.0\text{ ms}$** | **$42.7\text{ ms}$** | **$37.2\text{ ms}$** | **Overall Decode Speedup: $2.47\times$** |
| **Decode Throughput** | **$10.87\text{ tok/s}$** | **$23.42\text{ tok/s}$** | **$26.88\text{ tok/s}$ (peak)** | **Sustained: $19.35 - 20.46\text{ tok/s}$** |

---

## 3. Hardware Bandwidth Utilization Analysis

- At $48.8\text{ ms}$ per token ($20.46\text{ tok/s}$):
  $$\text{Effective DDR5 Bandwidth} = \frac{1.019\text{ GB}}{0.0488\text{ s}} = \mathbf{20.88\text{ GB/s}}$$
- LM-Head isolation:
  $$\text{LM-Head Bandwidth} = \frac{192.53\text{ MB}}{5.7\text{ ms}} = \mathbf{33.78\text{ GB/s}}$$

On the Intel Core i7-12650H laptop platform with dual-channel DDR5-4800 (theoretical peak bandwidth $38.4\text{ GB/s}$), sustaining $33.78\text{ GB/s}$ during the LM-head projection corresponds to **$88.0\%$ of theoretical peak memory bus saturation**, representing near-optimal CPU memory throughput.
