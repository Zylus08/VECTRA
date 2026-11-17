#include "vectra/core/memory.hpp"
#include "vectra/core/types.hpp"
#include "vectra/tensor/tensor.hpp"
#include "vectra/tensor/views.hpp"
#include "vectra/quant/quant_block.hpp"
#include "vectra/format/gguf_parser.hpp"
#include "vectra/tokenizer/tokenizer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while (0)

bool test_memory_alignment() {
    void* ptr = vectra::allocate_aligned(1024, 64);
    TEST_ASSERT(ptr != nullptr, "Aligned allocation failed");
    TEST_ASSERT(reinterpret_cast<uintptr_t>(ptr) % 64 == 0, "Memory is not 64-byte aligned");
    vectra::free_aligned(ptr);

    vectra::AlignedBuffer buf(512, 64);
    TEST_ASSERT(buf.data() != nullptr, "AlignedBuffer failed to allocate");
    TEST_ASSERT(reinterpret_cast<uintptr_t>(buf.data()) % 64 == 0, "AlignedBuffer is not 64-byte aligned");
    TEST_ASSERT(buf.size() == 512, "AlignedBuffer size mismatch");

    vectra::AlignedBuffer moved = std::move(buf);
    TEST_ASSERT(buf.data() == nullptr && buf.size() == 0, "Moved buffer not emptied");
    TEST_ASSERT(moved.data() != nullptr && moved.size() == 512, "Destination buffer invalid after move");
    return true;
}

bool test_fp16_conversion() {
    float vals[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, 0.00006103515625f};
    for (float v : vals) {
        uint16_t h = vectra::float32_to_float16(v);
        float reconstructed = vectra::float16_to_float32(h);
        float err = std::fabs(v - reconstructed);
        TEST_ASSERT(err <= std::fabs(v) * 0.002f + 1e-6f, "FP16 conversion error too large for " + std::to_string(v));
    }
    return true;
}

bool test_tensor_basics() {
    vectra::Tensor t({2, 3, 4}, vectra::DataType::FP32);
    TEST_ASSERT(t.ndim() == 3, "Tensor ndim mismatch");
    TEST_ASSERT(t.numel() == 24, "Tensor numel mismatch");
    TEST_ASSERT(t.bytes() == 24 * sizeof(float), "Tensor bytes mismatch");
    TEST_ASSERT(t.is_contiguous(), "New tensor should be contiguous");

    t.fill_constant(3.14f);
    float* data = t.data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        TEST_ASSERT(std::fabs(data[i] - 3.14f) < 1e-6f, "fill_constant failed");
    }

    vectra::Tensor cloned = t.clone();
    TEST_ASSERT(cloned.numel() == t.numel(), "Clone numel mismatch");
    TEST_ASSERT(cloned.data() != t.data(), "Clone must be a deep copy");

    t.reshape({6, 4});
    TEST_ASSERT(t.ndim() == 2, "Reshaped ndim mismatch");
    TEST_ASSERT(t.shape()[0] == 6 && t.shape()[1] == 4, "Reshaped shape mismatch");

    bool threw = false;
    try {
        t.reshape({5, 5}); // 25 != 24
    } catch (const vectra::VectraException&) {
        threw = true;
    }
    TEST_ASSERT(threw, "Reshape size mismatch must throw");
    return true;
}

bool test_tensor_views() {
    vectra::Tensor t({4, 8}, vectra::DataType::FP32);
    for (int64_t r = 0; r < 4; ++r) {
        for (int64_t c = 0; c < 8; ++c) {
            t.at(r, c) = static_cast<float>(r * 10 + c);
        }
    }

    vectra::TensorView sub = t.slice(0, 1, 2); // rows 1..2 (2 rows)
    TEST_ASSERT(sub.shape()[0] == 2 && sub.shape()[1] == 8, "Slice shape mismatch");
    TEST_ASSERT(sub.at(0, 0) == 10.0f, "Slice element (0,0) mismatch");
    TEST_ASSERT(sub.at(1, 3) == 23.0f, "Slice element (1,3) mismatch");
    return true;
}

bool test_quant_block() {
    const size_t k = 64;
    std::vector<float> orig(k);
    for (size_t i = 0; i < k; ++i) {
        orig[i] = (static_cast<float>(i) - 32.0f) * 0.25f;
    }

    std::vector<vectra::quant::BlockQ4_0> blocks(k / 32);
    vectra::quant::quantize_row_q4_0(orig.data(), blocks.data(), k);

    std::vector<float> recon(k);
    vectra::quant::dequantize_row_q4_0(blocks.data(), recon.data(), k);

    auto err = vectra::quant::evaluate_quantization_error(orig.data(), recon.data(), k);
    TEST_ASSERT(err.rmse < 0.35, "Quantization RMSE too high");
    TEST_ASSERT(err.snr_db > 20.0, "Quantization SNR too low");

    // Zero-valued block handling
    std::vector<float> zeros(32, 0.0f);
    vectra::quant::BlockQ4_0 z_block;
    vectra::quant::quantize_row_q4_0(zeros.data(), &z_block, 32);
    std::vector<float> z_recon(32);
    vectra::quant::dequantize_row_q4_0(&z_block, z_recon.data(), 32);
    for (size_t i = 0; i < 32; ++i) {
        TEST_ASSERT(z_recon[i] == 0.0f, "Zero block failed to reconstruct exact zeros");
    }

    // Partial block handling (k = 40, requires 2 blocks)
    std::vector<float> partial(40, 1.5f);
    std::vector<vectra::quant::BlockQ4_0> part_blocks(2);
    vectra::quant::quantize_row_q4_0(partial.data(), part_blocks.data(), 40);
    std::vector<float> part_recon(40);
    vectra::quant::dequantize_row_q4_0(part_blocks.data(), part_recon.data(), 40);
    for (size_t i = 0; i < 40; ++i) {
        TEST_ASSERT(std::fabs(part_recon[i] - 1.5f) < 0.2f, "Partial block reconstruction error");
    }
    return true;
}

bool test_gguf_negative_cases() {
    vectra::format::GGUFParser parser;

    // Bad magic
    bool threw_magic = false;
    try {
        parser.load_file("tests/fixtures/bad_magic.gguf");
    } catch (const vectra::VectraException&) {
        threw_magic = true;
    }
    TEST_ASSERT(threw_magic, "Invalid magic GGUF did not throw exception");

    // Corrupt offset
    bool threw_offset = false;
    try {
        parser.load_file("tests/fixtures/corrupt_offset.gguf");
    } catch (const vectra::VectraException&) {
        threw_offset = true;
    }
    TEST_ASSERT(threw_offset, "Corrupt offset GGUF did not throw exception");

    // Valid fixture
    parser.load_file("tests/fixtures/valid_test.gguf");
    TEST_ASSERT(parser.version() == 3, "Valid fixture version mismatch");
    TEST_ASSERT(parser.tensor_count() == 2, "Valid fixture tensor count mismatch");
    TEST_ASSERT(parser.get_string_metadata("general.architecture") == "qwen2", "Metadata architecture mismatch");
    TEST_ASSERT(parser.get_uint32_metadata("qwen2.context_length") == 2048, "Metadata context length mismatch");
    return true;
}

bool test_tokenizer() {
    vectra::tokenizer::Tokenizer tok;
    tok.add_token("hello", 1);
    tok.add_token("world", 2);

    std::string text = "<|im_start|>user\nhelloworld<|im_end|>\n";
    auto tokens = tok.encode(text);
    TEST_ASSERT(tokens.size() >= 4, "Token count unexpected");
    TEST_ASSERT(tokens[0] == 151644, "First token must be <|im_start|>");

    std::string round_trip = tok.decode(tokens);
    TEST_ASSERT(round_trip == text, "Tokenizer round trip failed: '" + round_trip + "' vs '" + text + "'");
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

    run_test("test_memory_alignment", test_memory_alignment);
    run_test("test_fp16_conversion", test_fp16_conversion);
    run_test("test_tensor_basics", test_tensor_basics);
    run_test("test_tensor_views", test_tensor_views);
    run_test("test_quant_block", test_quant_block);
    run_test("test_gguf_negative_cases", test_gguf_negative_cases);
    run_test("test_tokenizer", test_tokenizer);

    std::cout << "\nUnit Test Summary: " << passed << " / " << total << " passed.\n";
    return (passed == total) ? 0 : 1;
}
