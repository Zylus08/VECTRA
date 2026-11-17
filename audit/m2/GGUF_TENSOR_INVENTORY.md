# Milestone M2 — GGUF Tensor Inventory

**Target File:** `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`  
**Total Tensors:** 339  
**Total Weight Size:** 1,059.89 MB (1,111,370,240 bytes)  

---

## 1. Tensor Type Distribution

| Quantization Type | Count | Representative Tensors | Total Footprint | Role in Architecture |
|---|---|---|---|---|
| **F32** (4 bytes/elem) | 141 | `attn_norm`, `ffn_norm`, `output_norm`, `attn_q.bias`, `attn_k.bias`, `attn_v.bias` | ~1.64 MB | Normalization scale factors and attention biases requiring full numerical precision |
| **Q4_K** (0.5625 bytes/elem) | 141 | `token_embd.weight`, `attn_q.weight`, `attn_k.weight`, `attn_output.weight`, `ffn_gate.weight`, `ffn_up.weight` | ~684.34 MB | High-volume projections and embedding matrix (256-elem super-blocks with 4-bit quants and 6-bit scales) |
| **Q6_K** (0.8203 bytes/elem) | 57 | `output.weight`, `attn_v.weight`, `ffn_down.weight` | ~373.91 MB | Precision-sensitive projections (value projection, down projection, and LM vocabulary head) |
| **Total** | **339** | Full 28-layer transformer graph | **1059.89 MB** | Complete Model Representation |

---

## 2. Per-Layer Architecture Breakdown (Repeated for 28 Blocks: `blk.0` to `blk.27`)

For each layer $i \in [0, 27]$:

| Tensor Suffix | Type | Dimensions `[cols, rows]` | Element Count | Size on Disk | Role |
|---|---|---|---|---|---|
| `attn_norm.weight` | F32 | `[1536]` | 1,536 | 6.0 KB | Pre-attention RMSNorm scale |
| `attn_q.weight` | Q4_K | `[1536, 1536]` | 2,359,296 | 1,296.0 KB | Query linear projection |
| `attn_q.bias` | F32 | `[1536]` | 1,536 | 6.0 KB | Query projection bias |
| `attn_k.weight` | Q4_K | `[1536, 256]` | 393,216 | 216.0 KB | Key linear projection (2 KV heads * 128) |
| `attn_k.bias` | F32 | `[256]` | 256 | 1.0 KB | Key projection bias |
| `attn_v.weight` | Q6_K | `[1536, 256]` | 393,216 | 315.0 KB | Value linear projection (2 KV heads * 128) |
| `attn_v.bias` | F32 | `[256]` | 256 | 1.0 KB | Value projection bias |
| `attn_output.weight` | Q4_K | `[1536, 1536]` | 2,359,296 | 1,296.0 KB | Attention output linear projection |
| `ffn_norm.weight` | F32 | `[1536]` | 1,536 | 6.0 KB | Post-attention RMSNorm scale |
| `ffn_gate.weight` | Q4_K | `[1536, 8960]` | 13,762,560 | 7,560.0 KB | SwiGLU gate projection |
| `ffn_up.weight` | Q4_K | `[1536, 8960]` | 13,762,560 | 7,560.0 KB | SwiGLU up projection |
| `ffn_down.weight` | Q6_K | `[8960, 1536]` | 13,762,560 | 11,025.0 KB | SwiGLU down projection |

## 3. Global Tensors

| Tensor Name | Type | Dimensions `[cols, rows]` | Size on Disk | Role |
|---|---|---|---|---|
| `token_embd.weight` | Q4_K | `[1536, 151936]` | 128,196.0 KB (125.19 MB) | Vocabulary embedding matrix |
| `output_norm.weight` | F32 | `[1536]` | 6.0 KB | Final Transformer output RMSNorm scale |
| `output.weight` | Q6_K | `[1536, 151936]` | 186,952.5 KB (182.57 MB) | Vocabulary LM-head projection |
