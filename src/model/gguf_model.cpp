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

void matvec_batch_dispatch(
    const GGUFTensorView& W,
    const float* const* X,
    const float* bias,
    float* const* Y,
    size_t batch_size
) {
    if (!W.is_valid()) {
        throw VectraException("matvec_batch_dispatch called on invalid tensor: " + W.name);
    }
    if (batch_size == 0) return;
    if (batch_size == 1) {
        matvec_dispatch(W, X[0], bias, Y[0]);
        return;
    }

    switch (W.type) {
        case format::GGMLType::Q4_K:
            quant::matvec_q4_k_f32_batch(reinterpret_cast<const quant::BlockQ4_K*>(W.data), X, bias, Y, batch_size, W.rows, W.cols);
            break;
        case format::GGMLType::Q6_K:
            quant::matvec_q6_k_f32_batch(reinterpret_cast<const quant::BlockQ6_K*>(W.data), X, bias, Y, batch_size, W.rows, W.cols);
            break;
        default:
            for (size_t b = 0; b < batch_size; ++b) {
                matvec_dispatch(W, X[b], bias, Y[b]);
            }
            break;
    }
}

void matvec_batch_dispatch_q8(
    const GGUFTensorView& W,
    const quant::BlockQ8_K_256* const* X_q8,
    const float* bias,
    float* const* Y,
    size_t batch_size
) {
    if (!W.is_valid()) {
        throw VectraException("matvec_batch_dispatch_q8 called on invalid tensor: " + W.name);
    }
    if (batch_size == 0) return;
    if (batch_size == 1) {
        if (W.type == format::GGMLType::Q4_K) {
            quant::matvec_q4_k_q8(reinterpret_cast<const quant::BlockQ4_K*>(W.data), X_q8[0], bias, Y[0], W.rows, W.cols);
        } else if (W.type == format::GGMLType::Q6_K) {
            quant::matvec_q6_k_q8(reinterpret_cast<const quant::BlockQ6_K*>(W.data), X_q8[0], bias, Y[0], W.rows, W.cols);
        }
        return;
    }

    switch (W.type) {
        case format::GGMLType::Q4_K:
            quant::matvec_q4_k_q8_batch(reinterpret_cast<const quant::BlockQ4_K*>(W.data), X_q8, bias, Y, batch_size, W.rows, W.cols);
            break;
        case format::GGMLType::Q6_K:
            quant::matvec_q6_k_q8_batch(reinterpret_cast<const quant::BlockQ6_K*>(W.data), X_q8, bias, Y, batch_size, W.rows, W.cols);
            break;
        default:
            throw VectraException("Unsupported tensor type for matvec_batch_dispatch_q8: " + std::to_string(static_cast<int>(W.type)));
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
    const ModelConfig& config,
    LayerScratch& scratch,
    bool use_int8
) {
    int64_t h_dim = config.hidden_dim;
    int64_t q_dim = config.num_heads * config.head_dim;
    int64_t inter_dim = config.intermediate_dim;

    float* normed1 = scratch.normed1.data();
    float* q = scratch.q.data();
    float* k = scratch.k.data();
    float* v = scratch.v.data();
    float* attn_heads = scratch.attn_heads.data();
    float* attn_out = scratch.attn_out.data();
    float* res1 = scratch.res1.data();
    float* normed2 = scratch.normed2.data();
    float* gate = scratch.gate.data();
    float* up = scratch.up.data();
    float* act = scratch.act.data();
    float* ffn_out = scratch.ffn_out.data();

    // 1. Pre-Attention RMSNorm
    const float* attn_norm_ptr = reinterpret_cast<const float*>(attn_norm_w.data);
    if (math::avx2::is_supported()) {
        math::avx2::rms_norm(x_in, attn_norm_ptr, normed1, h_dim, config.rms_norm_eps);
    } else {
        math::scalar::rms_norm(x_in, attn_norm_ptr, normed1, h_dim, config.rms_norm_eps);
    }

    // 2. Q, K, V Projections
    const float* q_bias = q_proj_b.is_valid() ? reinterpret_cast<const float*>(q_proj_b.data) : nullptr;
    const float* k_bias = k_proj_b.is_valid() ? reinterpret_cast<const float*>(k_proj_b.data) : nullptr;
    const float* v_bias = v_proj_b.is_valid() ? reinterpret_cast<const float*>(v_proj_b.data) : nullptr;

    if (use_int8 && math::avx2::is_supported() && q_proj_w.type == format::GGMLType::Q4_K) {
        quant::quantize_activation_q8_k_256(normed1, scratch.q8_normed1.data(), h_dim);
        quant::matvec_q4_k_q8_qkv(
            reinterpret_cast<const quant::BlockQ4_K*>(q_proj_w.data), q_bias, q, q_proj_w.rows,
            reinterpret_cast<const quant::BlockQ4_K*>(k_proj_w.data), k_bias, k, k_proj_w.rows,
            reinterpret_cast<const quant::BlockQ4_K*>(v_proj_w.data), v_bias, v, v_proj_w.rows,
            scratch.q8_normed1.data(),
            q_proj_w.cols
        );
    } else {
        matvec_dispatch(q_proj_w, normed1, q_bias, q);
        matvec_dispatch(k_proj_w, normed1, k_bias, k);
        matvec_dispatch(v_proj_w, normed1, v_bias, v);
    }

    // 3. RoPE
    math::apply_rope_to_heads(q, config.num_heads, k, config.num_kv_heads, config.head_dim, pos, config.rope_theta);

    // 4. Append to KV Cache
    kv_cache.append(layer_idx, pos, k, v);

    // 5. GQA Attention
    int64_t current_seq = pos + 1;
    math::compute_gqa_attention(
        q,
        kv_cache.key_cache(layer_idx),
        kv_cache.val_cache(layer_idx),
        attn_heads,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        current_seq,
        pos
    );

    // 6. Attention Output Projection + Residual
    if (use_int8 && math::avx2::is_supported() && o_proj_w.type == format::GGMLType::Q4_K) {
        quant::quantize_activation_q8_k_256(attn_heads, scratch.q8_attn.data(), q_dim);
        quant::matvec_q4_k_q8(reinterpret_cast<const quant::BlockQ4_K*>(o_proj_w.data), scratch.q8_attn.data(), nullptr, attn_out, o_proj_w.rows, o_proj_w.cols);
    } else {
        matvec_dispatch(o_proj_w, attn_heads, nullptr, attn_out);
    }

    if (math::avx2::is_supported()) {
        math::avx2::add(x_in, attn_out, res1, h_dim);
    } else {
        math::scalar::add(x_in, attn_out, res1, h_dim);
    }

    // 7. Post-Attention RMSNorm
    const float* ffn_norm_ptr = reinterpret_cast<const float*>(ffn_norm_w.data);
    if (math::avx2::is_supported()) {
        math::avx2::rms_norm(res1, ffn_norm_ptr, normed2, h_dim, config.rms_norm_eps);
    } else {
        math::scalar::rms_norm(res1, ffn_norm_ptr, normed2, h_dim, config.rms_norm_eps);
    }

    // 8. SwiGLU FFN
    if (use_int8 && math::avx2::is_supported() && gate_proj_w.type == format::GGMLType::Q4_K) {
        quant::quantize_activation_q8_k_256(normed2, scratch.q8_normed2.data(), h_dim);
        quant::matvec_q4_k_q8_2x(
            reinterpret_cast<const quant::BlockQ4_K*>(gate_proj_w.data), gate,
            reinterpret_cast<const quant::BlockQ4_K*>(up_proj_w.data), up,
            scratch.q8_normed2.data(),
            gate_proj_w.rows,
            gate_proj_w.cols
        );
    } else {
        matvec_dispatch(gate_proj_w, normed2, nullptr, gate);
        matvec_dispatch(up_proj_w, normed2, nullptr, up);
    }

    math::scalar::swiglu(gate, up, act, inter_dim);

    if (use_int8 && math::avx2::is_supported() && down_proj_w.type == format::GGMLType::Q6_K) {
        quant::quantize_activation_q8_k_256(act, scratch.q8_act.data(), inter_dim);
        quant::matvec_q6_k_q8(reinterpret_cast<const quant::BlockQ6_K*>(down_proj_w.data), scratch.q8_act.data(), nullptr, ffn_out, down_proj_w.rows, down_proj_w.cols);
    } else {
        matvec_dispatch(down_proj_w, act, nullptr, ffn_out);
    }

    // 9. Second Residual
    if (math::avx2::is_supported()) {
        math::avx2::add(res1, ffn_out, x_out, h_dim);
    } else {
        math::scalar::add(res1, ffn_out, x_out, h_dim);
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
    scratch_.init(h_dim, config_.num_heads * config_.head_dim, config_.num_kv_heads * config_.head_dim, config_.intermediate_dim);

    if (x_buf1_.size() != static_cast<size_t>(h_dim)) {
        x_buf1_.resize(h_dim);
        x_buf2_.resize(h_dim);
    }

    float* cur = x_buf1_.data();
    float* next = x_buf2_.data();

    // 1. Dequantize token embedding row
    dequantize_row_dispatch(token_embd_w_, token_id, cur);

    // 2. Forward through all layers
    auto t_layers_start = std::chrono::high_resolution_clock::now();
    for (int64_t l = 0; l < config_.num_layers; ++l) {
        layers_[l].forward(cur, next, pos, l, kv_cache, config_, scratch_, use_int8_kernels_);
        std::swap(cur, next);
    }
    auto t_layers_end = std::chrono::high_resolution_clock::now();

    // 3. Final RMSNorm
    float* final_normed = scratch_.normed1.data();
    const float* norm_ptr = reinterpret_cast<const float*>(output_norm_w_.data);
    if (math::avx2::is_supported()) {
        math::avx2::rms_norm(cur, norm_ptr, final_normed, h_dim, config_.rms_norm_eps);
    } else {
        math::scalar::rms_norm(cur, norm_ptr, final_normed, h_dim, config_.rms_norm_eps);
    }

    // 4. Output projection to logits
    if (use_int8_kernels_ && math::avx2::is_supported() && output_w_.type == format::GGMLType::Q6_K) {
        quant::quantize_activation_q8_k_256(final_normed, scratch_.q8_normed1.data(), h_dim);
        quant::matvec_q6_k_q8(
            reinterpret_cast<const quant::BlockQ6_K*>(output_w_.data),
            scratch_.q8_normed1.data(),
            nullptr,
            logits_out,
            output_w_.rows,
            output_w_.cols
        );
    } else {
        matvec_dispatch(output_w_, final_normed, nullptr, logits_out);
    }
}

void GGUFTransformerModel::forward_prompt_batch(
    const std::vector<int32_t>& prompt_tokens,
    KVCache& kv_cache,
    float* logits_out
) {
    if (prompt_tokens.empty()) return;
    size_t B = prompt_tokens.size();
    int64_t h_dim = config_.hidden_dim;
    int64_t q_dim = config_.num_heads * config_.head_dim;
    int64_t kv_dim = config_.num_kv_heads * config_.head_dim;
    int64_t inter_dim = config_.intermediate_dim;

    size_t nb_h = (h_dim + 255) / 256;
    size_t nb_q = (q_dim + 255) / 256;
    size_t nb_inter = (inter_dim + 255) / 256;

    std::vector<std::vector<float>> X(B, std::vector<float>(h_dim));
    std::vector<std::vector<float>> normed1(B, std::vector<float>(h_dim));
    std::vector<std::vector<float>> Q(B, std::vector<float>(q_dim));
    std::vector<std::vector<float>> K(B, std::vector<float>(kv_dim));
    std::vector<std::vector<float>> V(B, std::vector<float>(kv_dim));
    std::vector<std::vector<float>> attn_heads(B, std::vector<float>(q_dim));
    std::vector<std::vector<float>> attn_out(B, std::vector<float>(h_dim));
    std::vector<std::vector<float>> res1(B, std::vector<float>(h_dim));
    std::vector<std::vector<float>> normed2(B, std::vector<float>(h_dim));
    std::vector<std::vector<float>> gate(B, std::vector<float>(inter_dim));
    std::vector<std::vector<float>> up(B, std::vector<float>(inter_dim));
    std::vector<std::vector<float>> act(B, std::vector<float>(inter_dim));
    std::vector<std::vector<float>> ffn_out(B, std::vector<float>(h_dim));

    std::vector<const float*> ptr_normed1(B);
    std::vector<float*> ptr_Q(B), ptr_K(B), ptr_V(B);
    std::vector<const float*> ptr_attn_heads(B);
    std::vector<float*> ptr_attn_out(B);
    std::vector<const float*> ptr_normed2(B);
    std::vector<float*> ptr_gate(B), ptr_up(B);
    std::vector<const float*> ptr_act(B);
    std::vector<float*> ptr_ffn_out(B);

    std::vector<std::vector<quant::BlockQ8_K_256>> q8_normed1;
    std::vector<std::vector<quant::BlockQ8_K_256>> q8_attn;
    std::vector<std::vector<quant::BlockQ8_K_256>> q8_normed2;
    std::vector<std::vector<quant::BlockQ8_K_256>> q8_act;
    std::vector<const quant::BlockQ8_K_256*> ptr_q8_normed1(B);
    std::vector<const quant::BlockQ8_K_256*> ptr_q8_attn(B);
    std::vector<const quant::BlockQ8_K_256*> ptr_q8_normed2(B);
    std::vector<const quant::BlockQ8_K_256*> ptr_q8_act(B);

    bool can_use_int8 = use_int8_kernels_ && math::avx2::is_supported();
    if (can_use_int8) {
        q8_normed1.resize(B, std::vector<quant::BlockQ8_K_256>(nb_h));
        q8_attn.resize(B, std::vector<quant::BlockQ8_K_256>(nb_q));
        q8_normed2.resize(B, std::vector<quant::BlockQ8_K_256>(nb_h));
        q8_act.resize(B, std::vector<quant::BlockQ8_K_256>(nb_inter));
        for (size_t b = 0; b < B; ++b) {
            ptr_q8_normed1[b] = q8_normed1[b].data();
            ptr_q8_attn[b] = q8_attn[b].data();
            ptr_q8_normed2[b] = q8_normed2[b].data();
            ptr_q8_act[b] = q8_act[b].data();
        }
    }

    for (size_t b = 0; b < B; ++b) {
        ptr_normed1[b] = normed1[b].data();
        ptr_Q[b] = Q[b].data();
        ptr_K[b] = K[b].data();
        ptr_V[b] = V[b].data();
        ptr_attn_heads[b] = attn_heads[b].data();
        ptr_attn_out[b] = attn_out[b].data();
        ptr_normed2[b] = normed2[b].data();
        ptr_gate[b] = gate[b].data();
        ptr_up[b] = up[b].data();
        ptr_act[b] = act[b].data();
        ptr_ffn_out[b] = ffn_out[b].data();
    }

    // 1. Initial embeddings lookup for all prompt tokens
    for (size_t b = 0; b < B; ++b) {
        dequantize_row_dispatch(token_embd_w_, prompt_tokens[b], X[b].data());
    }

    // 2. Iterate through 28 layers
    for (int64_t l = 0; l < config_.num_layers; ++l) {
        auto& layer = layers_[l];
        const float* attn_norm_ptr = reinterpret_cast<const float*>(layer.attn_norm_w.data);

        // Pre-Attention RMSNorm
        for (size_t b = 0; b < B; ++b) {
            if (math::avx2::is_supported()) {
                math::avx2::rms_norm(X[b].data(), attn_norm_ptr, normed1[b].data(), h_dim, config_.rms_norm_eps);
            } else {
                math::scalar::rms_norm(X[b].data(), attn_norm_ptr, normed1[b].data(), h_dim, config_.rms_norm_eps);
            }
        }

        // Batched Q, K, V Projections
        const float* q_bias = layer.q_proj_b.is_valid() ? reinterpret_cast<const float*>(layer.q_proj_b.data) : nullptr;
        const float* k_bias = layer.k_proj_b.is_valid() ? reinterpret_cast<const float*>(layer.k_proj_b.data) : nullptr;
        const float* v_bias = layer.v_proj_b.is_valid() ? reinterpret_cast<const float*>(layer.v_proj_b.data) : nullptr;

        if (can_use_int8) {
            for (size_t b = 0; b < B; ++b) {
                quant::quantize_activation_q8_k_256(normed1[b].data(), q8_normed1[b].data(), h_dim);
            }
            matvec_batch_dispatch_q8(layer.q_proj_w, ptr_q8_normed1.data(), q_bias, ptr_Q.data(), B);
            matvec_batch_dispatch_q8(layer.k_proj_w, ptr_q8_normed1.data(), k_bias, ptr_K.data(), B);
            matvec_batch_dispatch_q8(layer.v_proj_w, ptr_q8_normed1.data(), v_bias, ptr_V.data(), B);
        } else {
            matvec_batch_dispatch(layer.q_proj_w, ptr_normed1.data(), q_bias, ptr_Q.data(), B);
            matvec_batch_dispatch(layer.k_proj_w, ptr_normed1.data(), k_bias, ptr_K.data(), B);
            matvec_batch_dispatch(layer.v_proj_w, ptr_normed1.data(), v_bias, ptr_V.data(), B);
        }

        // RoPE and KV Cache Append for all tokens in prompt
        for (size_t b = 0; b < B; ++b) {
            int64_t pos = static_cast<int64_t>(b);
            math::apply_rope_to_heads(Q[b].data(), config_.num_heads, K[b].data(), config_.num_kv_heads, config_.head_dim, pos, config_.rope_theta);
            kv_cache.append(l, pos, K[b].data(), V[b].data());
        }

        // Causal GQA Attention: token b attends to tokens [0..b]
        for (size_t b = 0; b < B; ++b) {
            int64_t pos = static_cast<int64_t>(b);
            int64_t current_seq = pos + 1;
            math::compute_gqa_attention(
                Q[b].data(),
                kv_cache.key_cache(l),
                kv_cache.val_cache(l),
                attn_heads[b].data(),
                config_.num_heads,
                config_.num_kv_heads,
                config_.head_dim,
                current_seq,
                pos
            );
        }

        // Attention output projection + residual
        if (can_use_int8) {
            for (size_t b = 0; b < B; ++b) {
                quant::quantize_activation_q8_k_256(attn_heads[b].data(), q8_attn[b].data(), q_dim);
            }
            matvec_batch_dispatch_q8(layer.o_proj_w, ptr_q8_attn.data(), nullptr, ptr_attn_out.data(), B);
        } else {
            matvec_batch_dispatch(layer.o_proj_w, ptr_attn_heads.data(), nullptr, ptr_attn_out.data(), B);
        }

        for (size_t b = 0; b < B; ++b) {
            if (math::avx2::is_supported()) {
                math::avx2::add(X[b].data(), attn_out[b].data(), res1[b].data(), h_dim);
            } else {
                math::scalar::add(X[b].data(), attn_out[b].data(), res1[b].data(), h_dim);
            }
        }

        // Post-Attention RMSNorm
        const float* ffn_norm_ptr = reinterpret_cast<const float*>(layer.ffn_norm_w.data);
        for (size_t b = 0; b < B; ++b) {
            if (math::avx2::is_supported()) {
                math::avx2::rms_norm(res1[b].data(), ffn_norm_ptr, normed2[b].data(), h_dim, config_.rms_norm_eps);
            } else {
                math::scalar::rms_norm(res1[b].data(), ffn_norm_ptr, normed2[b].data(), h_dim, config_.rms_norm_eps);
            }
        }

        // SwiGLU FFN Projections
        if (can_use_int8) {
            for (size_t b = 0; b < B; ++b) {
                quant::quantize_activation_q8_k_256(normed2[b].data(), q8_normed2[b].data(), h_dim);
            }
            matvec_batch_dispatch_q8(layer.gate_proj_w, ptr_q8_normed2.data(), nullptr, ptr_gate.data(), B);
            matvec_batch_dispatch_q8(layer.up_proj_w, ptr_q8_normed2.data(), nullptr, ptr_up.data(), B);
        } else {
            matvec_batch_dispatch(layer.gate_proj_w, ptr_normed2.data(), nullptr, ptr_gate.data(), B);
            matvec_batch_dispatch(layer.up_proj_w, ptr_normed2.data(), nullptr, ptr_up.data(), B);
        }

        for (size_t b = 0; b < B; ++b) {
            math::scalar::swiglu(gate[b].data(), up[b].data(), act[b].data(), inter_dim);
        }

        if (can_use_int8) {
            for (size_t b = 0; b < B; ++b) {
                quant::quantize_activation_q8_k_256(act[b].data(), q8_act[b].data(), inter_dim);
            }
            matvec_batch_dispatch_q8(layer.down_proj_w, ptr_q8_act.data(), nullptr, ptr_ffn_out.data(), B);
        } else {
            matvec_batch_dispatch(layer.down_proj_w, ptr_act.data(), nullptr, ptr_ffn_out.data(), B);
        }

        // Residual connection
        for (size_t b = 0; b < B; ++b) {
            if (math::avx2::is_supported()) {
                math::avx2::add(res1[b].data(), ffn_out[b].data(), X[b].data(), h_dim);
            } else {
                math::scalar::add(res1[b].data(), ffn_out[b].data(), X[b].data(), h_dim);
            }
        }
    }

    // 3. Final RMSNorm and LM Head only for the LAST prompt token (token B-1)
    std::vector<float> final_normed(h_dim);
    const float* norm_ptr = reinterpret_cast<const float*>(output_norm_w_.data);
    if (math::avx2::is_supported()) {
        math::avx2::rms_norm(X[B - 1].data(), norm_ptr, final_normed.data(), h_dim, config_.rms_norm_eps);
    } else {
        math::scalar::rms_norm(X[B - 1].data(), norm_ptr, final_normed.data(), h_dim, config_.rms_norm_eps);
    }

    if (use_int8_kernels_ && math::avx2::is_supported() && output_w_.type == format::GGMLType::Q6_K) {
        alignas(32) quant::BlockQ8_K_256 q8_final[6];
        quant::quantize_activation_q8_k_256(final_normed.data(), q8_final, h_dim);
        quant::matvec_q6_k_q8(
            reinterpret_cast<const quant::BlockQ6_K*>(output_w_.data),
            q8_final,
            nullptr,
            logits_out,
            output_w_.rows,
            output_w_.cols
        );
    } else {
        matvec_dispatch(output_w_, final_normed.data(), nullptr, logits_out);
    }
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

    // 1. Batched prefill prompt tokens
    if (prompt_tokens.size() > 1 && static_cast<int64_t>(prompt_tokens.size()) <= context_limit) {
        forward_prompt_batch(prompt_tokens, kv_cache, logits.data());
    } else {
        for (size_t pos = 0; pos < prompt_tokens.size(); ++pos) {
            if (static_cast<int64_t>(pos) >= context_limit) break;
            forward_token(prompt_tokens[pos], static_cast<int64_t>(pos), kv_cache, logits.data());
        }
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
