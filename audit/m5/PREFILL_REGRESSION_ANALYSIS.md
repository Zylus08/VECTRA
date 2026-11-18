# Project VECTRA — Milestone M5: Prefill Regression Root Cause Analysis

**Document Purpose:** Detailed technical analysis of the prefill throughput degradation between Milestone M3 and Milestone M4.

---

## 1. The Phenomenon

During Milestone M3, prefill throughput was:
- Prompt P1 ($5$ tokens): **$5.17\text{ tok/s}$** ($966.5\text{ ms}$)
- Prompt P3 ($16$ tokens): **$5.39\text{ tok/s}$** ($2,968.9\text{ ms}$)

During Milestone M4, prefill throughput experienced a severe regression:
- Prompt P1 ($5$ tokens): **$3.62\text{ tok/s}$** ($1,382.8\text{ ms}$)
- Prompt P3 ($16$ tokens): **$4.16\text{ tok/s}$** ($3,842.1\text{ ms}$)

This regression meant that while single-token decoding sped up by $3.8\times$ in M4, Time-To-First-Token (TTFT) degraded by nearly $40\%$.

---

## 2. Technical Root Cause Breakdown

Forensic instrumentation identified three compounding factors:

### Factor 1: The Activation Re-quantization & Dynamic Allocation Tax
In M4, the runtime introduced dynamic INT8 activation quantization for decoding:
```cpp
// In M4: GGUFTransformerLayer::forward called per token inside forward_prompt_batch
std::vector<BlockQ8_K_256> q8_normed(hidden_dim / 256);
quantize_activation_q8_k_256(normed.data(), q8_normed.data(), hidden_dim);
```
In `forward_prompt_batch`, instead of quantizing all $B$ tokens once and performing a batched GEMV-N, M4 executed a per-token loop that reallocated heap vectors for `q8_normed`, `q8_act`, `gate`, `up`, and `act` at every layer for every prompt token:
$$\text{Heap Allocations per Prompt} = B \times 28 \text{ layers} \times 12 \text{ temporary buffers}$$
For a 16-token prompt, this resulted in **5,376 dynamic heap allocations** during prefill alone, causing severe heap lock contention on the Windows memory allocator.

### Factor 2: Loss of Weight Streaming Amortization
The fundamental advantage of batched prefill GEMV-N is that the weights for a layer matrix (e.g., Gate projection: 15.14 MB) are streamed from DDR5 RAM **once**, and reused across all $B$ tokens in CPU L1/L2 cache:
$$\text{DDR5 Bandwidth with Batched GEMV-N} = \frac{\text{Matrix Size}}{T}$$
Without batched INT8 GEMV-N, M4 streamed the 15.14 MB weights from DDR5 $B$ separate times!
For $B = 16$ tokens, DDR5 was forced to transfer:
$$16 \times 15.14\text{ MB} = 242.2\text{ MB}$$
instead of 15.14 MB! The memory bus was completely saturated by redundant weight transfers, throttling throughput.

### Factor 3: Lack of Fused QKV and Fused Gate-Up Kernels
In M4 prefill, each layer triggered separate thread pool dispatches for Q, K, V, O, Gate, Up, and Down projections. Across 28 layers, a single 16-token prompt incurred:
$$16 \text{ tokens} \times 28 \text{ layers} \times 7 \text{ dispatches} = 3,136 \text{ thread pool barriers}$$
The synchronization latency of 3,136 barriers dominated computation time.

---

## 3. Corrective Strategy Formulated for M5

To completely recover and exceed prefill baselines, M5 established four architectural requirements:
1. **True Batched INT8 Kernels:** Implement `matvec_q4_k_q8_batch` and `matvec_q6_k_q8_batch` operating on $[B \times K]$ activation blocks against $[M \times K]$ weight matrices.
2. **One Weight Stream per Layer:** Read each weight block from memory exactly once, and accumulate dot products into $B$ independent registers simultaneously.
3. **Persistent Pre-allocated Working Buffers:** Eliminate all dynamic heap allocations inside `forward_prompt_batch`.
4. **Fused Multi-Projection Dispatch:** Compute Gate and Up projections simultaneously in the batched kernel.
