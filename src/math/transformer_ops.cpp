#include "vectra/math/transformer_ops.hpp"
#include "vectra/math/scalar_math.hpp"
#include "vectra/math/avx2_math.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace vectra::math {

void apply_rope(float* vec, int64_t pos, int64_t head_dim, float rope_theta) {
    int64_t half_dim = head_dim / 2;
    for (int64_t i = 0; i < half_dim; ++i) {
        float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
        float val = static_cast<float>(pos) * freq;
        float cos_val = std::cos(val);
        float sin_val = std::sin(val);

        float v0 = vec[i];
        float v1 = vec[i + half_dim];

        vec[i] = v0 * cos_val - v1 * sin_val;
        vec[i + half_dim] = v0 * sin_val + v1 * cos_val;
    }
}

void apply_rope_to_heads(float* q_heads, int64_t num_q_heads,
                         float* k_heads, int64_t num_kv_heads,
                         int64_t head_dim, int64_t pos, float rope_theta) {
    for (int64_t h = 0; h < num_q_heads; ++h) {
        apply_rope(q_heads + h * head_dim, pos, head_dim, rope_theta);
    }
    for (int64_t h = 0; h < num_kv_heads; ++h) {
        apply_rope(k_heads + h * head_dim, pos, head_dim, rope_theta);
    }
}

void compute_gqa_attention(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* attn_out,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t seq_len,
    int64_t current_pos
) {
    int64_t group_size = num_heads / num_kv_heads;
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> scores(seq_len);

    for (int64_t h = 0; h < num_heads; ++h) {
        int64_t kv_h = h / group_size;
        const float* q_head = q + h * head_dim;

        // Compute attention scores for positions 0..seq_len-1
        for (int64_t t = 0; t < seq_len; ++t) {
            if (t > current_pos) {
                // Causal masking
                scores[t] = -1e9f;
            } else {
                const float* k_vec = k_cache + (t * num_kv_heads + kv_h) * head_dim;
                float dot_val;
                if (avx2::is_supported()) {
                    dot_val = avx2::dot(q_head, k_vec, head_dim);
                } else {
                    dot_val = scalar::dot(q_head, k_vec, head_dim);
                }
                scores[t] = dot_val * scale_factor;
            }
        }

        // Numerically stable softmax over scores
        scalar::softmax(scores.data(), scores.data(), seq_len);

        // Weighted sum over V
        float* out_head = attn_out + h * head_dim;
        for (int64_t d = 0; d < head_dim; ++d) {
            out_head[d] = 0.0f;
        }

        for (int64_t t = 0; t <= current_pos && t < seq_len; ++t) {
            float weight = scores[t];
            if (weight == 0.0f) continue;
            const float* v_vec = v_cache + (t * num_kv_heads + kv_h) * head_dim;
            for (int64_t d = 0; d < head_dim; ++d) {
                out_head[d] += weight * v_vec[d];
            }
        }
    }
}

void compute_swiglu_ffn(
    const float* x,
    const float* gate_w,
    const float* up_w,
    const float* down_w,
    float* ffn_out,
    int64_t hidden_dim,
    int64_t intermediate_dim
) {
    std::vector<float> gate(intermediate_dim);
    std::vector<float> up(intermediate_dim);
    std::vector<float> activated(intermediate_dim);

    // gate = gate_w * x
    if (avx2::is_supported()) {
        avx2::matvec(gate_w, x, nullptr, gate.data(), intermediate_dim, hidden_dim);
        avx2::matvec(up_w, x, nullptr, up.data(), intermediate_dim, hidden_dim);
    } else {
        scalar::matvec(gate_w, x, nullptr, gate.data(), intermediate_dim, hidden_dim);
        scalar::matvec(up_w, x, nullptr, up.data(), intermediate_dim, hidden_dim);
    }

    // activated = silu(gate) * up
    scalar::swiglu(gate.data(), up.data(), activated.data(), intermediate_dim);

    // ffn_out = down_w * activated
    if (avx2::is_supported()) {
        avx2::matvec(down_w, activated.data(), nullptr, ffn_out, hidden_dim, intermediate_dim);
    } else {
        scalar::matvec(down_w, activated.data(), nullptr, ffn_out, hidden_dim, intermediate_dim);
    }
}

} // namespace vectra::math
