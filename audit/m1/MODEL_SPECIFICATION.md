# Target Model Specification: Qwen2.5-1.5B-Instruct

## Verified Architecture Parameters
Extracted directly from the official model configuration (`Qwen/Qwen2.5-1.5B-Instruct`):

| Parameter | Value | Architectural Significance |
|---|---|---|
| `architectures` | `["Qwen2ForCausalLM"]` | Decoder-only autoregressive causal LM |
| `hidden_size` | 1536 | Representation width per token |
| `num_hidden_layers` | 28 | Sequential Transformer decoder blocks |
| `num_attention_heads` | 12 | Query heads in Multi-Head Attention |
| `num_key_value_heads` | 2 | Grouped-Query Attention (GQA) with 6:1 group ratio |
| `head_dim` | 128 | $1536 / 12 = 128$; KV projection width = $2 \times 128 = 256$ |
| `intermediate_size` | 8960 | FFN intermediate expansion ratio $\approx 5.83\times$ |
| `vocab_size` | 151936 | Extended multilingual vocabulary |
| `rope_theta` | 1000000.0 | High-base RoPE for long-context stability |
| `max_position_embeddings`| 32768 | Native context window length |
| `rms_norm_eps` | 1e-06 | Epsilon for variance numerical stability in RMSNorm |
| `tie_word_embeddings` | `true` | `lm_head.weight` is tied to `model.embed_tokens.weight` |
| `hidden_act` | `silu` | SwiGLU activation in feed-forward layers |

## Tensor Layout & Naming Conventions
Comparison between standard HuggingFace Safetensors and GGUF conventions:

| Functional Role | HuggingFace Safetensors Name | GGUF Format Name | Dimensions |
|---|---|---|---|
| Token Embeddings | `model.embed_tokens.weight` | `token_embd.weight` | `[151936, 1536]` |
| Pre-Attn Layernorm | `model.layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` | `[1536]` |
| Query Projection | `model.layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` | `[1536, 1536]` |
| Query Bias | `model.layers.{i}.self_attn.q_proj.bias` | `blk.{i}.attn_q.bias` | `[1536]` |
| Key Projection | `model.layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` | `[256, 1536]` |
| Key Bias | `model.layers.{i}.self_attn.k_proj.bias` | `blk.{i}.attn_k.bias` | `[256]` |
| Value Projection | `model.layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight` | `[256, 1536]` |
| Value Bias | `model.layers.{i}.self_attn.v_proj.bias` | `blk.{i}.attn_v.bias` | `[256]` |
| Output Projection | `model.layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_output.weight` | `[1536, 1536]` |
| Post-Attn Layernorm | `model.layers.{i}.post_attention_layernorm.weight` | `blk.{i}.ffn_norm.weight` | `[1536]` |
| FFN Gate Projection | `model.layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` | `[8960, 1536]` |
| FFN Up Projection | `model.layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` | `[8960, 1536]` |
| FFN Down Projection | `model.layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` | `[1536, 8960]` |
| Final Layernorm | `model.norm.weight` | `output_norm.weight` | `[1536]` |
| Vocab Head | (Tied to embed_tokens) | `output.weight` | `[151936, 1536]` |

## Architectural Nuances
1. **QKV Biases**: Qwen2.5 includes explicit bias vectors on `q_proj`, `k_proj`, and `v_proj`, but **no bias** on `o_proj`, `gate_proj`, `up_proj`, or `down_proj`.
2. **Embedding Tying**: The output classification head shares weights with token embeddings.
3. **Prompt & Chat Template**: Uses ChatML standard delimiters:
   - System / User / Assistant message framing: `<|im_start|>{role}\n{content}<|im_end|>\n`
   - Generation prefix: `<|im_start|>assistant\n`
