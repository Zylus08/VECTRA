# Project VECTRA — Milestone M3: Batched Causal Prefill Engine & Memory Bandwidth Optimization

**Operator Implementation:** `vectra::model::GGUFTransformerModel::forward_prompt_batch`  
**Kernel Implementation:** `matvec_q4_k_f32_batch`, `matvec_q6_k_f32_batch`, `matvec_batch_dispatch`  
**Key Optimization:** Weight Streaming Reuse Across Multiple Prompt Tokens (GEMV-N) + Skip Intermediate Vocabulary Projections

---

## 1. Problem Statement: Inefficiency of Sequential Prompt Ingestion

In Milestone M2, prompt ingestion was implemented by looping through the prompt tokens one by one:
```cpp
for (size_t pos = 0; pos < prompt_tokens.size(); ++pos) {
    forward_token(prompt_tokens[pos], pos, kv_cache, logits);
}
```
For a prompt of length $B$:
1. **Redundant Memory Bandwidth Consumption:**
   The entire 1.06 GB weight payload was streamed from memory into CPU caches $B$ times. For $B=5$ tokens, $> 5.3\text{ GB}$ was transferred across the memory bus.
2. **Wasteful Intermediate Logits Computation:**
   At each position $pos \in [0, B-2]$, the 151,936-dimensional LM vocabulary head was evaluated to compute output logits. Because autoregressive sampling only begins at position $B-1$, all intermediate logits were computed and immediately discarded, wasting ~442 ms per intermediate prompt token.
3. **Severe Latency:**
   For a 5-token prompt, M2 prefill required **18,353 ms (0.27 tok/s)**.

---

## 2. Mathematical Formulation of Batched Causal Prefill

Let $X \in \mathbb{R}^{B \times d}$ denote the batch of $B$ prompt token embeddings ($d = 1536$).
For each Transformer layer $l \in [0, 27]$:

1. **Layer-Norm:**
   Each token row $x_b$ ($b \in [0, B-1]$) is normalized independently using AVX2 RMSNorm:
   $$\hat{x}_b = \text{RMSNorm}(x_b, \gamma)$$

2. **Batched Matrix-Vector Multiplication (GEMV-N):**
   Instead of computing $y_b = W \hat{x}_b$ sequentially in separate passes, we iterate over the rows of weight matrix $W$.
   For each row $W_r$:
   $$\text{Load row } W_r \text{ into CPU registers once.}$$
   $$\text{For each } b \in [0, B-1]: \quad y_{b, r} = \vec{\text{dot}}(W_r, \hat{x}_b)$$
   **Memory Bandwidth Reduction Factor:**
   $$\text{Traffic Ratio} = \frac{\text{Sequential Traffic}}{\text{Batched Traffic}} = \frac{B \cdot |W|}{|W| + B \cdot |\hat{x}|} \approx B$$
   For $B=5$, weight memory bandwidth pressure is reduced by **5x**!

3. **Rotary Position Embedding (RoPE) & KV-Cache Append:**
   For each token $b \in [0, B-1]$:
   - RoPE is evaluated with position index $pos = b$:
     $$q_b = \text{RoPE}(q_b, b), \quad k_b = \text{RoPE}(k_b, b)$$
   - The key and value vectors $k_b, v_b$ are appended to the contiguous KV-cache at position $b$.

4. **Causal Attention Enforcement:**
   To maintain strict causal invariance, each token $b$ attends only to previous and current positions $[0, b]$ in the KV-cache:
   $$A_{b, t} = \frac{q_b \cdot k_t^T}{\sqrt{d_k}} \quad (t \le b), \quad A_{b, t} = -\infty \quad (t > b)$$
   Attention weights are normalized with Softmax and multiplied by cached values $v_t$ ($t \le b$).

5. **Skip Intermediate Vocabulary Projections:**
   After layer 27, only the final token's hidden state $X_{B-1}$ undergoes final RMSNorm and is projected through the $151936 \times 1536$ LM Head matrix. Intermediate tokens $0 \dots B-2$ completely bypass the LM Head.

---

## 3. Empirical Results & Latency Comparison

Measured on prompt `"The capital of France is"` ($B = 5$ tokens) with 10 threads:

| Metric | M2 Sequential Prefill | M3 Batched Causal Prefill | Improvement |
| :--- | :--- | :--- | :--- |
| **Prefill Latency** | 18,353 ms | **966.49 ms** | **18.99x faster** |
| **Prefill Throughput** | 0.27 tok/s | **5.17 tok/s** | **19.1x higher** |
| **DRAM Bytes Read** | 5.33 GB | **1.07 GB** | **4.98x reduction** |
| **LM Head Calls** | 5 calls (2,210 ms) | **1 call (442 ms)** | **80% reduction** |
| **Output Token Parity** | 12095 (`" Paris"`) | 12095 (`" Paris"`) | **Exact Match** |

On prompt `"What is 2 + 2?"` ($B = 16$ ChatML tokens):
- **Prefill Latency:** **2,968.89 ms (5.39 tok/s)**
- Under sequential evaluation, 16 tokens would have required $> 58\text{ seconds}$!
- Batched prefill delivered a **>19x real-world speedup**, easily surpassing the 5.0 tok/s prefill milestone goal.
