#pragma once

#include "vectra/tensor/tensor.hpp"
#include <cstdint>
#include <vector>

namespace vectra::model {

class KVCache {
public:
    KVCache(int64_t num_layers, int64_t max_seq_len, int64_t num_kv_heads, int64_t head_dim);

    void reset();

    // Append key and value vectors for a given layer at current sequence position
    void append(int64_t layer_idx, int64_t pos, const float* k_data, const float* v_data);

    // Get pointer to key cache for a layer: shape [max_seq_len, num_kv_heads, head_dim]
    const float* key_cache(int64_t layer_idx) const;
    const float* val_cache(int64_t layer_idx) const;

    int64_t num_layers() const { return num_layers_; }
    int64_t max_seq_len() const { return max_seq_len_; }
    int64_t num_kv_heads() const { return num_kv_heads_; }
    int64_t head_dim() const { return head_dim_; }
    int64_t current_seq_len() const { return current_seq_len_; }
    void set_current_seq_len(int64_t len) { current_seq_len_ = len; }

    size_t memory_bytes() const;

private:
    int64_t num_layers_;
    int64_t max_seq_len_;
    int64_t num_kv_heads_;
    int64_t head_dim_;
    int64_t current_seq_len_;

    // Storage for each layer: flattened [max_seq_len, num_kv_heads, head_dim]
    std::vector<Tensor> k_layers_;
    std::vector<Tensor> v_layers_;
};

} // namespace vectra::model
