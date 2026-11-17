#include "vectra/model/transformer.hpp"
#include "vectra/math/scalar_math.hpp"
#include "vectra/math/avx2_math.hpp"
#include "vectra/math/transformer_ops.hpp"
#include <random>
#include <cmath>

namespace vectra::model {

TransformerLayer::TransformerLayer(const ModelConfig& config)
    : config_(config),
      input_layernorm_w(std::vector<int64_t>{config.hidden_dim}),
      q_proj_w(std::vector<int64_t>{config.num_heads * config.head_dim, config.hidden_dim}),
      q_proj_b(std::vector<int64_t>{config.num_heads * config.head_dim}),
      k_proj_w(std::vector<int64_t>{config.num_kv_heads * config.head_dim, config.hidden_dim}),
      k_proj_b(std::vector<int64_t>{config.num_kv_heads * config.head_dim}),
      v_proj_w(std::vector<int64_t>{config.num_kv_heads * config.head_dim, config.hidden_dim}),
      v_proj_b(std::vector<int64_t>{config.num_kv_heads * config.head_dim}),
      o_proj_w(std::vector<int64_t>{config.hidden_dim, config.num_heads * config.head_dim}),
      post_attention_layernorm_w(std::vector<int64_t>{config.hidden_dim}),
      gate_proj_w(std::vector<int64_t>{config.intermediate_dim, config.hidden_dim}),
      up_proj_w(std::vector<int64_t>{config.intermediate_dim, config.hidden_dim}),
      down_proj_w(std::vector<int64_t>{config.hidden_dim, config.intermediate_dim}) {}

void TransformerLayer::forward(const float* x_in, float* x_out, int64_t pos, int64_t layer_idx, KVCache& kv_cache) {
    int64_t h_dim = config_.hidden_dim;
    int64_t q_dim = config_.num_heads * config_.head_dim;
    int64_t kv_dim = config_.num_kv_heads * config_.head_dim;

    std::vector<float> normed1(h_dim);
    std::vector<float> q(q_dim);
    std::vector<float> k(kv_dim);
    std::vector<float> v(kv_dim);
    std::vector<float> attn_heads(q_dim);
    std::vector<float> attn_out(h_dim);
    std::vector<float> res1(h_dim);
    std::vector<float> normed2(h_dim);
    std::vector<float> ffn_out(h_dim);

    // 1. Pre-attention RMSNorm
    if (math::avx2::is_supported()) {
        math::avx2::rms_norm(x_in, input_layernorm_w.data_ptr<float>(), normed1.data(), h_dim, config_.rms_norm_eps);
    } else {
        math::scalar::rms_norm(x_in, input_layernorm_w.data_ptr<float>(), normed1.data(), h_dim, config_.rms_norm_eps);
    }

    // 2. Q, K, V Projections
    const float* q_b = config_.has_qkv_bias ? q_proj_b.data_ptr<float>() : nullptr;
    const float* k_b = config_.has_qkv_bias ? k_proj_b.data_ptr<float>() : nullptr;
    const float* v_b = config_.has_qkv_bias ? v_proj_b.data_ptr<float>() : nullptr;

    if (math::avx2::is_supported()) {
        math::avx2::matvec(q_proj_w.data_ptr<float>(), normed1.data(), q_b, q.data(), q_dim, h_dim);
        math::avx2::matvec(k_proj_w.data_ptr<float>(), normed1.data(), k_b, k.data(), kv_dim, h_dim);
        math::avx2::matvec(v_proj_w.data_ptr<float>(), normed1.data(), v_b, v.data(), kv_dim, h_dim);
    } else {
        math::scalar::matvec(q_proj_w.data_ptr<float>(), normed1.data(), q_b, q.data(), q_dim, h_dim);
        math::scalar::matvec(k_proj_w.data_ptr<float>(), normed1.data(), k_b, k.data(), kv_dim, h_dim);
        math::scalar::matvec(v_proj_w.data_ptr<float>(), normed1.data(), v_b, v.data(), kv_dim, h_dim);
    }

    // 3. Rotary Position Embedding
    math::apply_rope_to_heads(q.data(), config_.num_heads, k.data(), config_.num_kv_heads, config_.head_dim, pos, config_.rope_theta);

    // 4. Append to KV cache
    kv_cache.append(layer_idx, pos, k.data(), v.data());

    // 5. Grouped-Query Attention
    int64_t current_seq = pos + 1;
    math::compute_gqa_attention(
        q.data(),
        kv_cache.key_cache(layer_idx),
        kv_cache.val_cache(layer_idx),
        attn_heads.data(),
        config_.num_heads,
        config_.num_kv_heads,
        config_.head_dim,
        current_seq,
        pos
    );

    // 6. Attention Output Projection
    if (math::avx2::is_supported()) {
        math::avx2::matvec(o_proj_w.data_ptr<float>(), attn_heads.data(), nullptr, attn_out.data(), h_dim, q_dim);
        math::avx2::add(x_in, attn_out.data(), res1.data(), h_dim);
    } else {
        math::scalar::matvec(o_proj_w.data_ptr<float>(), attn_heads.data(), nullptr, attn_out.data(), h_dim, q_dim);
        math::scalar::add(x_in, attn_out.data(), res1.data(), h_dim);
    }

    // 7. Post-Attention RMSNorm
    if (math::avx2::is_supported()) {
        math::avx2::rms_norm(res1.data(), post_attention_layernorm_w.data_ptr<float>(), normed2.data(), h_dim, config_.rms_norm_eps);
    } else {
        math::scalar::rms_norm(res1.data(), post_attention_layernorm_w.data_ptr<float>(), normed2.data(), h_dim, config_.rms_norm_eps);
    }

    // 8. SwiGLU FFN
    math::compute_swiglu_ffn(
        normed2.data(),
        gate_proj_w.data_ptr<float>(),
        up_proj_w.data_ptr<float>(),
        down_proj_w.data_ptr<float>(),
        ffn_out.data(),
        h_dim,
        config_.intermediate_dim
    );

    // 9. Second Residual Connection
    if (math::avx2::is_supported()) {
        math::avx2::add(res1.data(), ffn_out.data(), x_out, h_dim);
    } else {
        math::scalar::add(res1.data(), ffn_out.data(), x_out, h_dim);
    }
}

TransformerModel::TransformerModel(ModelConfig config)
    : config_(config),
      embed_tokens_w(std::vector<int64_t>{config.vocab_size, config.hidden_dim}),
      final_norm_w(std::vector<int64_t>{config.hidden_dim}),
      lm_head_w(std::vector<int64_t>{config.vocab_size, config.hidden_dim}) {
    layers.reserve(config.num_layers);
    for (int64_t i = 0; i < config.num_layers; ++i) {
        layers.emplace_back(config);
    }
}

void TransformerModel::init_synthetic_weights(uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    auto fill_tensor = [&](Tensor& t, float default_val = 0.0f, bool is_norm = false) {
        float* ptr = t.data_ptr<float>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            if (is_norm) {
                ptr[i] = 1.0f; // Layernorms initialized to 1.0
            } else if (default_val != 0.0f) {
                ptr[i] = default_val;
            } else {
                ptr[i] = dist(rng);
            }
        }
    };

    fill_tensor(embed_tokens_w);
    fill_tensor(final_norm_w, 1.0f, true);
    if (!config_.tie_word_embeddings) {
        fill_tensor(lm_head_w);
    }

    for (auto& layer : layers) {
        fill_tensor(layer.input_layernorm_w, 1.0f, true);
        fill_tensor(layer.q_proj_w);
        fill_tensor(layer.q_proj_b, 0.0f);
        fill_tensor(layer.k_proj_w);
        fill_tensor(layer.k_proj_b, 0.0f);
        fill_tensor(layer.v_proj_w);
        fill_tensor(layer.v_proj_b, 0.0f);
        fill_tensor(layer.o_proj_w);
        fill_tensor(layer.post_attention_layernorm_w, 1.0f, true);
        fill_tensor(layer.gate_proj_w);
        fill_tensor(layer.up_proj_w);
        fill_tensor(layer.down_proj_w);
    }
}

void TransformerModel::forward_token(int32_t token_id, int64_t pos, KVCache& kv_cache, float* logits_out) {
    if (token_id < 0 || token_id >= config_.vocab_size) {
        throw VectraException("Token id out of vocab bounds: " + std::to_string(token_id));
    }

    int64_t h_dim = config_.hidden_dim;
    std::vector<float> x_cur(h_dim);
    std::vector<float> x_next(h_dim);

    // 1. Embed token lookup
    const float* emb_row = embed_tokens_w.data_ptr<float>() + static_cast<size_t>(token_id) * h_dim;
    std::memcpy(x_cur.data(), emb_row, h_dim * sizeof(float));

    // 2. Transformer layers
    for (int64_t l = 0; l < config_.num_layers; ++l) {
        layers[l].forward(x_cur.data(), x_next.data(), pos, l, kv_cache);
        x_cur = x_next;
    }

    // 3. Final RMSNorm
    std::vector<float> final_normed(h_dim);
    if (math::avx2::is_supported()) {
        math::avx2::rms_norm(x_cur.data(), final_norm_w.data_ptr<float>(), final_normed.data(), h_dim, config_.rms_norm_eps);
    } else {
        math::scalar::rms_norm(x_cur.data(), final_norm_w.data_ptr<float>(), final_normed.data(), h_dim, config_.rms_norm_eps);
    }

    // 4. Vocabulary projection to logits
    const float* head_w = config_.tie_word_embeddings ? embed_tokens_w.data_ptr<float>() : lm_head_w.data_ptr<float>();

    if (math::avx2::is_supported()) {
        math::avx2::matvec(head_w, final_normed.data(), nullptr, logits_out, config_.vocab_size, h_dim);
    } else {
        math::scalar::matvec(head_w, final_normed.data(), nullptr, logits_out, config_.vocab_size, h_dim);
    }
}

} // namespace vectra::model
