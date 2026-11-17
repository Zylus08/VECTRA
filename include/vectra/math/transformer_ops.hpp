#pragma once

#include "vectra/tensor/tensor.hpp"
#include "vectra/tensor/views.hpp"
#include <vector>

namespace vectra::math {

// Rotary Position Embedding
void apply_rope(float* vec, int64_t pos, int64_t head_dim, float rope_theta = 1000000.0f);

// Multi-head RoPE for Q and K tensors
void apply_rope_to_heads(float* q_heads, int64_t num_q_heads,
                         float* k_heads, int64_t num_kv_heads,
                         int64_t head_dim, int64_t pos, float rope_theta = 1000000.0f);

// Grouped Query Attention for a single query token against KV-cache
void compute_gqa_attention(
    const float* q,               // [num_heads, head_dim]
    const float* k_cache,         // [seq_len, num_kv_heads, head_dim]
    const float* v_cache,         // [seq_len, num_kv_heads, head_dim]
    float* attn_out,              // [num_heads, head_dim]
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t seq_len,
    int64_t current_pos
);

// SwiGLU Feed Forward block
void compute_swiglu_ffn(
    const float* x,               // [hidden_dim]
    const float* gate_w,          // [intermediate_dim, hidden_dim]
    const float* up_w,            // [intermediate_dim, hidden_dim]
    const float* down_w,          // [hidden_dim, intermediate_dim]
    float* ffn_out,               // [hidden_dim]
    int64_t hidden_dim,
    int64_t intermediate_dim
);

} // namespace vectra::math
