#pragma once

#include "vectra/core/types.hpp"
#include "vectra/format/mmap_loader.hpp"
#include "vectra/format/gguf_parser.hpp"
#include "vectra/model/transformer.hpp"
#include "vectra/model/kv_cache.hpp"
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

// Dequantize a single row from a 2D tensor
void dequantize_row_dispatch(const GGUFTensorView& W, int64_t row_idx, float* out);

class GGUFTransformerLayer {
public:
    GGUFTransformerLayer() = default;

    void forward(
        const float* x_in,
        float* x_out,
        int64_t pos,
        int64_t layer_idx,
        KVCache& kv_cache,
        const ModelConfig& config
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

private:
    GGUFTensorView resolve_tensor(const std::string& name);

    format::MemoryMappedFile mmap_;
    format::GGUFParser parser_;
    ModelConfig config_;

    GGUFTensorView token_embd_w_;
    std::vector<GGUFTransformerLayer> layers_;
    GGUFTensorView output_norm_w_;
    GGUFTensorView output_w_;
};

} // namespace vectra::model
