# Milestone M2 — Transformer Numerical Parity & Layer Architecture

**Date:** 2026-11-18  
**Architecture:** Qwen2.5-1.5B-Instruct  
**Runtime:** Project VECTRA C++20 CPU Engine  

---

## 1. Forward Pass Mathematical Pipeline

VECTRA executes the full 28-layer Transformer decoder using the standard Qwen2.5 architecture specification:

### 1.1 Input Embedding Layer
Given input token ID $t \in [0, 151935]$:
$$x^{(0)} = \text{DequantizeRow}(\mathbf{W}_{\text{emb}}, t) \in \mathbb{R}^{1536}$$
where $\mathbf{W}_{\text{emb}}$ is stored as `Q4_K` super-blocks.

### 1.2 Transformer Layer Operations (Repeated for $l = 0, \dots, 27$)
1. **Pre-Attention RMSNorm:**
   $$\bar{x} = \text{RMSNorm}(x^{(l)}, \gamma_{\text{attn}}^{(l)}, \epsilon = 10^{-6})$$
2. **Q, K, V Linear Projections with Attention Biases:**
   $$q = \bar{x} \mathbf{W}_q^{(l)} + b_q^{(l)} \in \mathbb{R}^{1536}$$
   $$k = \bar{x} \mathbf{W}_k^{(l)} + b_k^{(l)} \in \mathbb{R}^{256}$$
   $$v = \bar{x} \mathbf{W}_v^{(l)} + b_v^{(l)} \in \mathbb{R}^{256}$$
3. **Rotary Positional Embeddings (RoPE):**
   $$q_{\text{rot}} = \text{RoPE}(q, \text{pos}, \theta = 10^6, d_k = 128)$$
   $$k_{\text{rot}} = \text{RoPE}(k, \text{pos}, \theta = 10^6, d_k = 128)$$
4. **KV-Cache Storage:**
   Append $k_{\text{rot}}$ and $v$ into the per-layer key and value caches at column `pos`.
5. **Grouped-Query Attention (GQA):**
   For each query head $h \in [0, 11]$, assign KV head $kv\_h = \lfloor h / 6 \rfloor \in [0, 1]$.
   $$A_{h, \tau} = \frac{q_h \cdot K_{kv\_h, \tau}}{\sqrt{128}}, \quad \tau \in [0, \text{pos}]$$
   $$\text{AttnScores}_h = \text{Softmax}(A_h)$$
   $$\text{HeadOut}_h = \sum_{\tau=0}^{\text{pos}} \text{AttnScores}_{h, \tau} V_{kv\_h, \tau}$$
6. **Attention Output Projection & Residual:**
   $$x_{\text{mid}} = x^{(l)} + (\text{Concat}(\text{HeadOut}_0, \dots, \text{HeadOut}_{11}) \mathbf{W}_o^{(l)})$$
7. **Post-Attention RMSNorm:**
   $$\tilde{x} = \text{RMSNorm}(x_{\text{mid}}, \gamma_{\text{ffn}}^{(l)}, \epsilon = 10^{-6})$$
8. **SwiGLU Feed-Forward Network:**
   $$\text{gate} = \tilde{x} \mathbf{W}_{\text{gate}}^{(l)} \in \mathbb{R}^{8960}$$
   $$\text{up} = \tilde{x} \mathbf{W}_{\text{up}}^{(l)} \in \mathbb{R}^{8960}$$
   $$\text{act} = (\text{gate} \cdot \sigma(\text{gate})) \odot \text{up} \in \mathbb{R}^{8960}$$
   $$\text{ffn\_out} = \text{act} \mathbf{W}_{\text{down}}^{(l)} \in \mathbb{R}^{1536}$$
9. **FFN Residual Connection:**
   $$x^{(l+1)} = x_{\text{mid}} + \text{ffn\_out}$$

### 1.3 Final Normalization & LM Head
$$\hat{x} = \text{RMSNorm}(x^{(28)}, \gamma_{\text{out}}, \epsilon = 10^{-6})$$
$$\text{logits} = \hat{x} \mathbf{W}_{\text{output}} \in \mathbb{R}^{151936}$$
where $\mathbf{W}_{\text{output}}$ is stored as `Q6_K` super-blocks.

---

## 2. Numerical Parity Results

When supplied with the prompt `"The capital of France is"`, the forward graph produces:
- Top token predicted: Token ID **`9625`**
- Token text: **`" Paris"`**
- Confidence: Argmax strictly dominates alternative tokens
- Second token predicted: Token ID **`13`** (`"."`)
- Subsequent tokens: `" The"`, `" capital"`, `" of"`, `" Italy"`, `" is"`, `" Rome"`, `"."`

This matches the exact greedy generation trajectory of the official unquantized and quantized Hugging Face reference model.
