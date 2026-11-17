#include "vectra/model/gguf_model.hpp"
#include "vectra/math/scalar_math.hpp"
#include "vectra/math/avx2_math.hpp"
#include "vectra/math/transformer_ops.hpp"
#include "vectra/quant/quant_block.hpp"
#include "vectra/quant/quant_k.hpp"
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>

namespace vectra::model {

void matvec_dispatch(const GGUFTensorView& W, const float* x, const float* bias, float* y) {
    if (!W.is_valid()) {
        throw VectraException("matvec_dispatch called on invalid tensor: " + W.name);
    }

    switch (W.type) {
        case format::GGMLType::Q4_K:
            quant::matvec_q4_k_f32(reinterpret_cast<const quant::BlockQ4_K*>(W.data), x, bias, y, W.rows, W.cols);
            break;
        case format::GGMLType::Q6_K:
            quant::matvec_q6_k_f32(reinterpret_cast<const quant::BlockQ6_K*>(W.data), x, bias, y, W.rows, W.cols);
            break;
        case format::GGMLType::Q4_0:
            quant::matvec_q4_0_f32(reinterpret_cast<const quant::BlockQ4_0*>(W.data), x, bias, y, W.rows, W.cols);
            break;
        case format::GGMLType::F32:
            if (math::avx2::is_supported()) {
                math::avx2::matvec(reinterpret_cast<const float*>(W.data), x, bias, y, W.rows, W.cols);
            } else {
                math::scalar::matvec(reinterpret_cast<const float*>(W.data), x, bias, y, W.rows, W.cols);
            }
            break;
        default:
            throw VectraException("Unsupported tensor type in matvec_dispatch: " + std::to_string(static_cast<int>(W.type)));
    }
}

void dequantize_row_dispatch(const GGUFTensorView& W, int64_t row_idx, float* out) {
    if (!W.is_valid()) {
        throw VectraException("dequantize_row_dispatch called on invalid tensor: " + W.name);
    }
    if (row_idx < 0 || row_idx >= W.rows) {
        throw VectraException("Row index out of bounds: " + std::to_string(row_idx) + " (rows=" + std::to_string(W.rows) + ")");
    }

    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(W.data);

    switch (W.type) {
        case format::GGMLType::Q4_K: {
            size_t blocks_per_row = W.cols / quant::QK_K;
            const auto* block_ptr = reinterpret_cast<const quant::BlockQ4_K*>(byte_ptr + row_idx * blocks_per_row * sizeof(quant::BlockQ4_K));
            quant::dequantize_row_q4_k(block_ptr, out, W.cols);
            break;
        }
        case format::GGMLType::Q6_K: {
            size_t blocks_per_row = W.cols / quant::QK_K;
            const auto* block_ptr = reinterpret_cast<const quant::BlockQ6_K*>(byte_ptr + row_idx * blocks_per_row * sizeof(quant::BlockQ6_K));
            quant::dequantize_row_q6_k(block_ptr, out, W.cols);
            break;
        }
        case format::GGMLType::Q4_0: {
            size_t blocks_per_row = (W.cols + quant::Q4_BLOCK_SIZE - 1) / quant::Q4_BLOCK_SIZE;
            const auto* block_ptr = reinterpret_cast<const quant::BlockQ4_0*>(byte_ptr + row_idx * blocks_per_row * sizeof(quant::BlockQ4_0));
            quant::dequantize_row_q4_0(block_ptr, out, W.cols);
            break;
        }
        case format::GGMLType::F32: {
            const float* f32_ptr = reinterpret_cast<const float*>(byte_ptr + row_idx * W.cols * sizeof(float));
            std::memcpy(out, f32_ptr, W.cols * sizeof(float));
            break;
        }
        default:
            throw VectraException("Unsupported tensor type in dequantize_row_dispatch: " + std::to_string(static_cast<int>(W.type)));
    }
}

void GGUFTransformerLayer::forward(
    const float* x_in,
    float* x_out,
    int64_t pos,
    int64_t layer_idx,
    KVCache& kv_cache,
    const ModelConfig& config
) {
    int64_t h_dim = config.hidden_dim;
    int64_t q_dim = config.num_heads * config.head_dim;
    int64_t kv_dim = config.num_kv_heads * config.head_dim;

    std::vector<float> normed1(h_dim);
    std::vector<float> q(q_dim);
    std::vector<float> k(kv_dim);
    std::vector<float> v(kv_dim);
    std::vector<float> attn_heads(q_dim);
    std::vector<float> attn_out(h_dim);
    std::vector<float> res1(h_dim);
    std::vector<float> normed2(h_dim);
    std::vector<float> gate(config.intermediate_dim);
    std::vector<float> up(config.intermediate_dim);
    std::vector<float> act(config.intermediate_dim);
    std::vector<float> ffn_out(h_dim);

    // 1. Pre-Attention RMSNorm
    const float* attn_norm_ptr = reinterpret_cast<const float*>(attn_norm_w.data);
    if (math::avx2::is_supported()) {
        math::avx2::rms_norm(x_in, attn_norm_ptr, normed1.data(), h_dim, config.rms_norm_eps);
    } else {
        math::scalar::rms_norm(x_in, attn_norm_ptr, normed1.data(), h_dim, config.rms_norm_eps);
    }

    // 2. Q, K, V Projections
    const float* q_bias = q_proj_b.is_valid() ? reinterpret_cast<const float*>(q_proj_b.data) : nullptr;
    const float* k_bias = k_proj_b.is_valid() ? reinterpret_cast<const float*>(k_proj_b.data) : nullptr;
    const float* v_bias = v_proj_b.is_valid() ? reinterpret_cast<const float*>(v_proj_b.data) : nullptr;

    matvec_dispatch(q_proj_w, normed1.data(), q_bias, q.data());
    matvec_dispatch(k_proj_w, normed1.data(), k_bias, k.data());
    matvec_dispatch(v_proj_w, normed1.data(), v_bias, v.data());

    // 3. RoPE
    math::apply_rope_to_heads(q.data(), config.num_heads, k.data(), config.num_kv_heads, config.head_dim, pos, config.rope_theta);

    // 4. Append to KV Cache
    kv_cache.append(layer_idx, pos, k.data(), v.data());

    // 5. GQA Attention
    int64_t current_seq = pos + 1;
    math::compute_gqa_attention(
        q.data(),
        kv_cache.key_cache(layer_idx),
        kv_cache.val_cache(layer_idx),
        attn_heads.data(),
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        current_seq,
        pos
    );

    // 6. Attention Output Projection + Residual
    matvec_dispatch(o_proj_w, attn_heads.data(), nullptr, attn_out.data());
    if (math::avx2::is_supported()) {
        math::avx2::add(x_in, attn_out.data(), res1.data(), h_dim);
    } else {
        math::scalar::add(x_in, attn_out.data(), res1.data(), h_dim);
    }

    // 7. Post-Attention RMSNorm
    const float* ffn_norm_ptr = reinterpret_cast<const float*>(ffn_norm_w.data);
    if (math::avx2::is_supported()) {
        math::avx2::rms_norm(res1.data(), ffn_norm_ptr, normed2.data(), h_dim, config.rms_norm_eps);
    } else {
        math::scalar::rms_norm(res1.data(), ffn_norm_ptr, normed2.data(), h_dim, config.rms_norm_eps);
    }

    // 8. SwiGLU FFN
    matvec_dispatch(gate_proj_w, normed2.data(), nullptr, gate.data());
    matvec_dispatch(up_proj_w, normed2.data(), nullptr, up.data());
    math::scalar::swiglu(gate.data(), up.data(), act.data(), config.intermediate_dim);
    matvec_dispatch(down_proj_w, act.data(), nullptr, ffn_out.data());

    // 9. Second Residual
    if (math::avx2::is_supported()) {
        math::avx2::add(res1.data(), ffn_out.data(), x_out, h_dim);
    } else {
        math::scalar::add(res1.data(), ffn_out.data(), x_out, h_dim);
    }
}

GGUFTransformerModel::GGUFTransformerModel() = default;

GGUFTransformerModel::GGUFTransformerModel(const std::string& gguf_path) {
    load(gguf_path);
}

GGUFTransformerModel::~GGUFTransformerModel() = default;

GGUFTensorView GGUFTransformerModel::resolve_tensor(const std::string& name) {
    const auto* t = parser_.find_tensor(name);
    if (!t) {
        throw VectraException("Tensor not found in GGUF model: " + name);
    }

    GGUFTensorView view;
    view.name = name;
    view.type = t->type;
    view.data = mmap_.get_pointer_at(t->absolute_file_offset, static_cast<size_t>(t->size_bytes));

    if (t->dimensions.size() == 1) {
        view.rows = static_cast<int64_t>(t->dimensions[0]);
        view.cols = 1;
    } else if (t->dimensions.size() == 2) {
        // GGUF stores dimensions as [ne0, ne1] where ne0 is columns and ne1 is rows in C row-major layout
        view.cols = static_cast<int64_t>(t->dimensions[0]);
        view.rows = static_cast<int64_t>(t->dimensions[1]);
    } else {
        throw VectraException("Unsupported tensor rank (" + std::to_string(t->dimensions.size()) + ") for tensor: " + name);
    }

    return view;
}

void GGUFTransformerModel::load(const std::string& gguf_path) {
    mmap_.open(gguf_path);
    parser_.parse_memory(mmap_.data(), mmap_.size());

    // Populate configuration from metadata
    std::string arch = parser_.get_string_metadata("general.architecture", "qwen2");
    config_.hidden_dim = parser_.get_uint32_metadata(arch + ".embedding_length", 1536);
    config_.num_layers = parser_.get_uint32_metadata(arch + ".block_count", 28);
    config_.num_heads = parser_.get_uint32_metadata(arch + ".attention.head_count", 12);
    config_.num_kv_heads = parser_.get_uint32_metadata(arch + ".attention.head_count_kv", 2);
    config_.head_dim = config_.hidden_dim / config_.num_heads;
    config_.intermediate_dim = parser_.get_uint32_metadata(arch + ".feed_forward_length", 8960);
    config_.max_seq_len = parser_.get_uint32_metadata(arch + ".context_length", 32768);
    config_.rope_theta = parser_.get_float_metadata(arch + ".rope.freq_base", 1000000.0f);
    config_.rms_norm_eps = parser_.get_float_metadata(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);

    // Map embedding tensor
    token_embd_w_ = resolve_tensor("token_embd.weight");
    config_.vocab_size = token_embd_w_.rows;

    // Map 28 transformer layers
    layers_.resize(config_.num_layers);
    for (int64_t i = 0; i < config_.num_layers; ++i) {
        std::string pfx = "blk." + std::to_string(i) + ".";
        auto& l = layers_[i];
        l.attn_norm_w = resolve_tensor(pfx + "attn_norm.weight");
        l.q_proj_w    = resolve_tensor(pfx + "attn_q.weight");
        l.q_proj_b    = resolve_tensor(pfx + "attn_q.bias");
        l.k_proj_w    = resolve_tensor(pfx + "attn_k.weight");
        l.k_proj_b    = resolve_tensor(pfx + "attn_k.bias");
        l.v_proj_w    = resolve_tensor(pfx + "attn_v.weight");
        l.v_proj_b    = resolve_tensor(pfx + "attn_v.bias");
        l.o_proj_w    = resolve_tensor(pfx + "attn_output.weight");
        l.ffn_norm_w   = resolve_tensor(pfx + "ffn_norm.weight");
        l.gate_proj_w = resolve_tensor(pfx + "ffn_gate.weight");
        l.up_proj_w   = resolve_tensor(pfx + "ffn_up.weight");
        l.down_proj_w = resolve_tensor(pfx + "ffn_down.weight");
    }

    // Map final norm and output head
    output_norm_w_ = resolve_tensor("output_norm.weight");
    if (parser_.find_tensor("output.weight")) {
        output_w_ = resolve_tensor("output.weight");
    } else {
        output_w_ = token_embd_w_; // Tied weights
    }
}

void GGUFTransformerModel::forward_token(
    int32_t token_id,
    int64_t pos,
    KVCache& kv_cache,
    float* logits_out
) {
    if (token_id < 0 || token_id >= config_.vocab_size) {
        throw VectraException("Token ID out of vocab bounds: " + std::to_string(token_id));
    }

    int64_t h_dim = config_.hidden_dim;
    std::vector<float> x_cur(h_dim);
    std::vector<float> x_next(h_dim);

    // 1. Dequantize token embedding row
    dequantize_row_dispatch(token_embd_w_, token_id, x_cur.data());

    // 2. Forward through all layers
    for (int64_t l = 0; l < config_.num_layers; ++l) {
        layers_[l].forward(x_cur.data(), x_next.data(), pos, l, kv_cache, config_);
        x_cur = x_next;
    }

    // 3. Final RMSNorm
    std::vector<float> final_normed(h_dim);
    const float* norm_ptr = reinterpret_cast<const float*>(output_norm_w_.data);
    if (math::avx2::is_supported()) {
        math::avx2::rms_norm(x_cur.data(), norm_ptr, final_normed.data(), h_dim, config_.rms_norm_eps);
    } else {
        math::scalar::rms_norm(x_cur.data(), norm_ptr, final_normed.data(), h_dim, config_.rms_norm_eps);
    }

    // 4. Output projection to logits
    matvec_dispatch(output_w_, final_normed.data(), nullptr, logits_out);
}

std::vector<int32_t> GGUFTransformerModel::generate(
    const std::vector<int32_t>& prompt_tokens,
    size_t max_new_tokens,
    runtime::Sampler& sampler,
    const std::function<bool(int32_t)>& on_token,
    int64_t max_context_len
) {
    std::vector<int32_t> generated;
    if (prompt_tokens.empty()) return generated;

    int64_t context_limit = std::min(max_context_len, config_.max_seq_len);
    KVCache kv_cache(config_.num_layers, context_limit, config_.num_kv_heads, config_.head_dim);
    std::vector<float> logits(config_.vocab_size);

    // 1. Prefill prompt tokens
    for (size_t pos = 0; pos < prompt_tokens.size(); ++pos) {
        if (static_cast<int64_t>(pos) >= context_limit) break;
        forward_token(prompt_tokens[pos], static_cast<int64_t>(pos), kv_cache, logits.data());
    }

    // 2. Autoregressive decode loop
    int64_t current_pos = static_cast<int64_t>(prompt_tokens.size());

    for (size_t step = 0; step < max_new_tokens; ++step) {
        if (current_pos >= context_limit) {
            break; // Context window filled
        }

        int32_t next_tok = sampler.sample(logits.data(), config_.vocab_size);
        generated.push_back(next_tok);

        if (on_token) {
            bool keep_going = on_token(next_tok);
            if (!keep_going) break;
        }

        // Check for EOS / Stop tokens: <|im_end|> (151645) or <|endoftext|> (151643)
        if (next_tok == 151645 || next_tok == 151643) {
            break;
        }

        forward_token(next_tok, current_pos, kv_cache, logits.data());
        current_pos++;
    }

    return generated;
}

} // namespace vectra::model
