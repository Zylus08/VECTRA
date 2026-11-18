#pragma once

#include "vectra/core/types.hpp"
#include "vectra/format/mmap_loader.hpp"
#include "vectra/format/gguf_parser.hpp"
#include "vectra/model/transformer.hpp"
#include "vectra/model/kv_cache.hpp"
#include "vectra/quant/quant_k.hpp"
#include "vectra/runtime/sampler.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace vectra::model {

struct GGUFTensorView {
    std::string name;
    const void* data = nullptr;
    format::GGMLType type = format::GGMLType::F32;
    int64_t rows = 0;
    int64_t cols = 0;

    bool is_valid() const { return data != nullptr && rows > 0 && cols > 0; }
};

// Unified matrix-vector multiplication supporting Q4_K, Q6_K, Q4_0, and F32
void matvec_dispatch(const GGUFTensorView& W, const float* x, const float* bias, float* y);

// Unified batched matrix-vector multiplication
void matvec_batch_dispatch(
    const GGUFTensorView& W,
    const float* const* X,
    const float* bias,
    float* const* Y,
    size_t batch_size
);

void matvec_batch_dispatch_q8(
    const GGUFTensorView& W,
    const quant::BlockQ8_K_256* const* X_q8,
    const float* bias,
    float* const* Y,
    size_t batch_size
);

// Dequantize a single row from a 2D tensor
void dequantize_row_dispatch(const GGUFTensorView& W, int64_t row_idx, float* out);

struct LayerScratch {
    std::vector<float> normed1;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attn_heads;
    std::vector<float> attn_out;
    std::vector<float> res1;
    std::vector<float> normed2;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> act;
    std::vector<float> ffn_out;
    std::vector<quant::BlockQ8_K_256> q8_normed1;
    std::vector<quant::BlockQ8_K_256> q8_attn;
    std::vector<quant::BlockQ8_K_256> q8_normed2;
    std::vector<quant::BlockQ8_K_256> q8_act;

    void init(int64_t h_dim, int64_t q_dim, int64_t kv_dim, int64_t inter_dim) {
        if (normed1.size() != static_cast<size_t>(h_dim)) {
            normed1.resize(h_dim);
            q.resize(q_dim);
            k.resize(kv_dim);
            v.resize(kv_dim);
            attn_heads.resize(q_dim);
            attn_out.resize(h_dim);
            res1.resize(h_dim);
            normed2.resize(h_dim);
            gate.resize(inter_dim);
            up.resize(inter_dim);
            act.resize(inter_dim);
            ffn_out.resize(h_dim);
            q8_normed1.resize((h_dim + 255) / 256);
            q8_attn.resize((q_dim + 255) / 256);
            q8_normed2.resize((h_dim + 255) / 256);
            q8_act.resize((inter_dim + 255) / 256);
        }
    }
};

class GGUFTransformerLayer {
public:
    GGUFTransformerLayer() = default;

    void forward(
        const float* x_in,
        float* x_out,
        int64_t pos,
        int64_t layer_idx,
        KVCache& kv_cache,
        const ModelConfig& config,
        LayerScratch& scratch,
        bool use_int8 = true
    );

    GGUFTensorView attn_norm_w;
    GGUFTensorView q_proj_w;
    GGUFTensorView q_proj_b;
    GGUFTensorView k_proj_w;
    GGUFTensorView k_proj_b;
    GGUFTensorView v_proj_w;
    GGUFTensorView v_proj_b;
    GGUFTensorView o_proj_w;
    GGUFTensorView ffn_norm_w;
    GGUFTensorView gate_proj_w;
    GGUFTensorView up_proj_w;
    GGUFTensorView down_proj_w;
};

class GGUFTransformerModel {
public:
    GGUFTransformerModel();
    explicit GGUFTransformerModel(const std::string& gguf_path);
    ~GGUFTransformerModel();

    void load(const std::string& gguf_path);

    // Forward single token, returns logits in logits_out [vocab_size]
    void forward_token(
        int32_t token_id,
        int64_t pos,
        KVCache& kv_cache,
        float* logits_out
    );

    // Optimized batched prompt prefill
    void forward_prompt_batch(
        const std::vector<int32_t>& prompt_tokens,
        KVCache& kv_cache,
        float* logits_out
    );

    // Full autoregressive text generation
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_tokens,
        size_t max_new_tokens,
        runtime::Sampler& sampler,
        const std::function<bool(int32_t)>& on_token = nullptr,
        int64_t max_context_len = 1024
    );

    const ModelConfig& config() const { return config_; }
    const format::GGUFParser& parser() const { return parser_; }
    size_t mapped_bytes() const { return mmap_.size(); }

    void set_use_int8_kernels(bool enable) { use_int8_kernels_ = enable; }
    bool use_int8_kernels() const { return use_int8_kernels_; }

private:
    GGUFTensorView resolve_tensor(const std::string& name);

    format::MemoryMappedFile mmap_;
    format::GGUFParser parser_;
    ModelConfig config_;
    bool use_int8_kernels_ = true;

    GGUFTensorView token_embd_w_;
    std::vector<GGUFTransformerLayer> layers_;
    GGUFTensorView output_norm_w_;
    GGUFTensorView output_w_;

    mutable LayerScratch scratch_;
    mutable std::vector<float> x_buf1_;
    mutable std::vector<float> x_buf2_;
};

} // namespace vectra::model
