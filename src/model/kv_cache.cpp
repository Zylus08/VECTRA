#include "vectra/model/kv_cache.hpp"
#include <cstring>

namespace vectra::model {

KVCache::KVCache(int64_t num_layers, int64_t max_seq_len, int64_t num_kv_heads, int64_t head_dim)
    : num_layers_(num_layers), max_seq_len_(max_seq_len), num_kv_heads_(num_kv_heads),
      head_dim_(head_dim), current_seq_len_(0) {
    k_layers_.reserve(num_layers);
    v_layers_.reserve(num_layers);

    for (int64_t l = 0; l < num_layers; ++l) {
        k_layers_.emplace_back(std::vector<int64_t>{max_seq_len, num_kv_heads, head_dim}, DataType::FP32);
        v_layers_.emplace_back(std::vector<int64_t>{max_seq_len, num_kv_heads, head_dim}, DataType::FP32);
        k_layers_.back().fill_zero();
        v_layers_.back().fill_zero();
    }
}

void KVCache::reset() {
    current_seq_len_ = 0;
    for (int64_t l = 0; l < num_layers_; ++l) {
        k_layers_[l].fill_zero();
        v_layers_[l].fill_zero();
    }
}

void KVCache::append(int64_t layer_idx, int64_t pos, const float* k_data, const float* v_data) {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw VectraException("KVCache layer index out of bounds: " + std::to_string(layer_idx));
    }
    if (pos < 0 || pos >= max_seq_len_) {
        throw VectraException("KVCache position out of bounds: " + std::to_string(pos));
    }

    size_t elements_per_token = static_cast<size_t>(num_kv_heads_ * head_dim_);
    size_t offset = static_cast<size_t>(pos) * elements_per_token;

    float* k_dest = k_layers_[layer_idx].data_ptr<float>() + offset;
    float* v_dest = v_layers_[layer_idx].data_ptr<float>() + offset;

    std::memcpy(k_dest, k_data, elements_per_token * sizeof(float));
    std::memcpy(v_dest, v_data, elements_per_token * sizeof(float));

    if (pos + 1 > current_seq_len_) {
        current_seq_len_ = pos + 1;
    }
}

const float* KVCache::key_cache(int64_t layer_idx) const {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw VectraException("KVCache layer index out of bounds: " + std::to_string(layer_idx));
    }
    return k_layers_[layer_idx].data_ptr<float>();
}

const float* KVCache::val_cache(int64_t layer_idx) const {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw VectraException("KVCache layer index out of bounds: " + std::to_string(layer_idx));
    }
    return v_layers_[layer_idx].data_ptr<float>();
}

size_t KVCache::memory_bytes() const {
    size_t total = 0;
    for (int64_t l = 0; l < num_layers_; ++l) {
        total += k_layers_[l].bytes() + v_layers_[l].bytes();
    }
    return total;
}

} // namespace vectra::model
