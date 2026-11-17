#pragma once

#include "vectra/tensor/tensor.hpp"
#include "vectra/model/kv_cache.hpp"
#include <vector>
#include <memory>

namespace vectra::model {

struct ModelConfig {
    int64_t hidden_dim = 1536;
    int64_t num_layers = 28;
    int64_t num_heads = 12;
    int64_t num_kv_heads = 2;
    int64_t head_dim = 128;
    int64_t intermediate_dim = 8960;
    int64_t vocab_size = 151936;
    int64_t max_seq_len = 2048;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;
    bool has_qkv_bias = true;
    bool tie_word_embeddings = true;

    static ModelConfig tiny_test_config() {
        ModelConfig cfg;
        cfg.hidden_dim = 64;
        cfg.num_layers = 2;
        cfg.num_heads = 4;
        cfg.num_kv_heads = 2;
        cfg.head_dim = 16;
        cfg.intermediate_dim = 128;
        cfg.vocab_size = 256;
        cfg.max_seq_len = 64;
        cfg.rope_theta = 10000.0f;
        cfg.rms_norm_eps = 1e-5f;
        cfg.has_qkv_bias = true;
        cfg.tie_word_embeddings = true;
        return cfg;
    }

    static ModelConfig qwen2_5_1_5b_config() {
        return ModelConfig{};
    }
};

class TransformerLayer {
public:
    explicit TransformerLayer(const ModelConfig& config);

    void forward(const float* x_in, float* x_out, int64_t pos, int64_t layer_idx, KVCache& kv_cache);

    // Weights
    Tensor input_layernorm_w;
    Tensor q_proj_w;
    Tensor q_proj_b;
    Tensor k_proj_w;
    Tensor k_proj_b;
    Tensor v_proj_w;
    Tensor v_proj_b;
    Tensor o_proj_w;
    Tensor post_attention_layernorm_w;
    Tensor gate_proj_w;
    Tensor up_proj_w;
    Tensor down_proj_w;

private:
    ModelConfig config_;
};

class TransformerModel {
public:
    explicit TransformerModel(ModelConfig config);

    void init_synthetic_weights(uint32_t seed = 42);

    // Forward single token at position pos, outputs logits of size [vocab_size]
    void forward_token(int32_t token_id, int64_t pos, KVCache& kv_cache, float* logits_out);

    const ModelConfig& config() const { return config_; }

    // Model weights
    Tensor embed_tokens_w;
    std::vector<TransformerLayer> layers;
    Tensor final_norm_w;
    Tensor lm_head_w; // used if tie_word_embeddings is false

private:
    ModelConfig config_;
};

} // namespace vectra::model
