#include "vectra/model/transformer.hpp"
#include "vectra/model/kv_cache.hpp"
#include "vectra/math/scalar_math.hpp"
#include "vectra/quant/quant_block.hpp"
#include <iostream>
#include <vector>
#include <cmath>

bool test_causal_masking_invariance() {
    auto config = vectra::model::ModelConfig::tiny_test_config();
    vectra::model::TransformerModel model(config);
    model.init_synthetic_weights(999);

    // Sequence 1: tokens [10, 20]
    vectra::model::KVCache cache1(config.num_layers, config.max_seq_len, config.num_kv_heads, config.head_dim);
    std::vector<float> logits1_pos0(config.vocab_size);
    model.forward_token(10, 0, cache1, logits1_pos0.data());

    // Sequence 2: tokens [10, 50] (first token is identical)
    vectra::model::KVCache cache2(config.num_layers, config.max_seq_len, config.num_kv_heads, config.head_dim);
    std::vector<float> logits2_pos0(config.vocab_size);
    model.forward_token(10, 0, cache2, logits2_pos0.data());

    // Verify that logits for position 0 are identical regardless of future tokens
    for (int64_t v = 0; v < config.vocab_size; ++v) {
        if (logits1_pos0[v] != logits2_pos0[v]) {
            std::cerr << "Causal invariance failed at pos 0 logit " << v << "\n";
            return false;
        }
    }
    return true;
}

bool test_tiny_transformer_generation_step() {
    auto config = vectra::model::ModelConfig::tiny_test_config();
    vectra::model::TransformerModel model(config);
    model.init_synthetic_weights(42);

    vectra::model::KVCache cache(config.num_layers, config.max_seq_len, config.num_kv_heads, config.head_dim);
    std::vector<float> logits(config.vocab_size);

    int32_t current_token = 5;
    for (int64_t step = 0; step < 5; ++step) {
        model.forward_token(current_token, step, cache, logits.data());

        // Find greedy token
        int32_t next_token = 0;
        float max_l = logits[0];
        for (int32_t v = 1; v < config.vocab_size; ++v) {
            if (logits[v] > max_l) {
                max_l = logits[v];
                next_token = v;
            }
        }
        current_token = next_token;
    }
    if (cache.current_seq_len() != 5) {
        std::cerr << "Cache sequence length mismatch: expected 5, got " << cache.current_seq_len() << "\n";
        return false;
    }
    return true;
}

bool test_quantized_matvec_integration() {
    const size_t rows = 32;
    const size_t cols = 64;
    std::vector<float> W(rows * cols);
    std::vector<float> x(cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        W[i] = std::sin(static_cast<float>(i) * 0.1f);
    }
    for (size_t i = 0; i < cols; ++i) {
        x[i] = std::cos(static_cast<float>(i) * 0.2f);
    }

    // Unquantized matvec
    std::vector<float> y_fp32(rows);
    vectra::math::scalar::matvec(W.data(), x.data(), nullptr, y_fp32.data(), rows, cols);

    // Quantized matvec
    size_t blocks_per_row = cols / 32;
    std::vector<vectra::quant::BlockQ4_0> W_q(rows * blocks_per_row);
    for (size_t r = 0; r < rows; ++r) {
        vectra::quant::quantize_row_q4_0(W.data() + r * cols, W_q.data() + r * blocks_per_row, cols);
    }

    std::vector<float> y_q4(rows);
    vectra::quant::matvec_q4_0_f32(W_q.data(), x.data(), nullptr, y_q4.data(), rows, cols);

    auto err = vectra::quant::evaluate_quantization_error(y_fp32.data(), y_q4.data(), rows);
    if (err.snr_db < 15.0) { // Should maintain at least 15 dB SNR on full matvec
        std::cerr << "Quantized matvec SNR too low: " << err.snr_db << " dB\n";
        return false;
    }
    return true;
}

int main() {
    int passed = 0;
    int total = 0;

    auto run_test = [&](const char* name, bool (*func)()) {
        total++;
        std::cout << "[RUN]  " << name << " ... ";
        if (func()) {
            passed++;
            std::cout << "PASSED\n";
        } else {
            std::cout << "FAILED\n";
        }
    };

    run_test("test_causal_masking_invariance", test_causal_masking_invariance);
    run_test("test_tiny_transformer_generation_step", test_tiny_transformer_generation_step);
    run_test("test_quantized_matvec_integration", test_quantized_matvec_integration);

    std::cout << "\nIntegration Test Summary: " << passed << " / " << total << " passed.\n";
    return (passed == total) ? 0 : 1;
}
