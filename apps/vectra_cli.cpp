#include "vectra/core/types.hpp"
#include "vectra/core/memory.hpp"
#include "vectra/math/avx2_math.hpp"
#include "vectra/quant/quant_block.hpp"
#include "vectra/model/transformer.hpp"
#include "vectra/tokenizer/tokenizer.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    std::cout << "=========================================================\n";
    std::cout << "               PROJECT VECTRA - RUNTIME CLI              \n";
    std::cout << "=========================================================\n";
    std::cout << "Host Hardware Instruction Support:\n";
    std::cout << "  AVX2 / FMA: " << (vectra::math::avx2::is_supported() ? "DETECTED & ENABLED" : "SCALAR FALLBACK") << "\n\n";

    // 1. Quantization verification
    std::cout << "[1] Demonstrating Introductory 4-bit Quantization (BlockQ4_0)...\n";
    const size_t test_k = 64;
    std::vector<float> orig(test_k);
    for (size_t i = 0; i < test_k; ++i) {
        orig[i] = std::sin(static_cast<float>(i) * 0.2f) * 4.5f;
    }

    std::vector<vectra::quant::BlockQ4_0> blocks(test_k / 32);
    vectra::quant::quantize_row_q4_0(orig.data(), blocks.data(), test_k);

    std::vector<float> reconstructed(test_k);
    vectra::quant::dequantize_row_q4_0(blocks.data(), reconstructed.data(), test_k);

    auto err = vectra::quant::evaluate_quantization_error(orig.data(), reconstructed.data(), test_k);
    std::cout << "  Original size:      " << test_k * sizeof(float) << " bytes\n";
    std::cout << "  Quantized size:     " << blocks.size() * sizeof(vectra::quant::BlockQ4_0) << " bytes (Compression: "
              << (float)(test_k * sizeof(float)) / (float)(blocks.size() * sizeof(vectra::quant::BlockQ4_0)) << "x)\n";
    std::cout << "  Reconstruction MSE: " << std::scientific << err.mse << "\n";
    std::cout << "  Reconstruction SNR: " << std::fixed << std::setprecision(2) << err.snr_db << " dB\n\n";

    // 2. Tokenizer demonstration
    std::cout << "[2] Demonstrating Tokenizer Subsystem...\n";
    vectra::tokenizer::Tokenizer tok;
    tok.add_token("Hello", 100);
    tok.add_token(" world", 101);
    tok.add_token("!", 102);
    tok.add_token(" Qwen", 103);

    std::string prompt = "<|im_start|>user\nHello world!<|im_end|>\n<|im_start|>assistant\n";
    auto tokens = tok.encode(prompt);
    std::cout << "  Input prompt: " << prompt;
    std::cout << "  Token count:  " << tokens.size() << " tokens\n";
    std::cout << "  Decoded text: " << tok.decode(tokens) << "\n\n";

    // 3. Tiny Transformer forward pass
    std::cout << "[3] Demonstrating Synthetic Transformer Forward Execution...\n";
    auto config = vectra::model::ModelConfig::tiny_test_config();
    std::cout << "  Tiny Config: hidden=" << config.hidden_dim
              << ", layers=" << config.num_layers
              << ", heads=" << config.num_heads
              << ", kv_heads=" << config.num_kv_heads
              << ", vocab=" << config.vocab_size << "\n";

    vectra::model::TransformerModel model(config);
    model.init_synthetic_weights(12345);

    vectra::model::KVCache kv_cache(config.num_layers, config.max_seq_len, config.num_kv_heads, config.head_dim);
    std::vector<float> logits(config.vocab_size);

    std::cout << "  Simulating 3-token prompt prefill sequence...\n";
    std::vector<int32_t> test_tokens = {10, 25, 42};
    for (size_t pos = 0; pos < test_tokens.size(); ++pos) {
        model.forward_token(test_tokens[pos], pos, kv_cache, logits.data());
        int32_t argmax_tok = 0;
        float max_logit = logits[0];
        for (int32_t v = 1; v < config.vocab_size; ++v) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                argmax_tok = v;
            }
        }
        std::cout << "    Step " << pos << ": Token " << test_tokens[pos]
                  << " -> Argmax Next Token: " << argmax_tok << " (Logit: " << max_logit << ")\n";
    }
    std::cout << "  KV Cache memory allocated: " << kv_cache.memory_bytes() << " bytes\n";
    std::cout << "\nRuntime verification complete: all subsystems operational.\n";
    return 0;
}
