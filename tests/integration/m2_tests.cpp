#include "vectra/format/mmap_loader.hpp"
#include "vectra/quant/quant_k.hpp"
#include "vectra/runtime/sampler.hpp"
#include "vectra/tokenizer/bpe_tokenizer.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while (0)

bool test_mmap_loader() {
    vectra::format::MemoryMappedFile mmap_file("tests/fixtures/valid_test.gguf");
    TEST_ASSERT(mmap_file.is_open(), "mmap failed to open valid GGUF fixture");
    TEST_ASSERT(mmap_file.size() > 0, "mmap file size must be > 0");
    TEST_ASSERT(mmap_file.data() != nullptr, "mmap pointer is null");

    // Check memory stats
    auto mem = vectra::format::get_process_memory_stats();
    TEST_ASSERT(mem.working_set_bytes > 0, "Process working set should be > 0");
    TEST_ASSERT(mem.peak_working_set_bytes >= mem.working_set_bytes, "Peak working set must be >= current working set");

    // Out of bounds check
    bool threw = false;
    try {
        mmap_file.get_pointer_at(mmap_file.size() + 100, 10);
    } catch (const vectra::VectraException&) {
        threw = true;
    }
    TEST_ASSERT(threw, "Out of bounds mmap access must throw");
    return true;
}

bool test_q4_k_correctness() {
    // Construct a synthetic BlockQ4_K of 256 elements
    vectra::quant::BlockQ4_K block;
    block.d = vectra::float32_to_float16(1.0f);
    block.dmin = vectra::float32_to_float16(0.0f);
    // Set uniform scale 1 for all sub-blocks: scales[j] = 1
    for (int j = 0; j < 4; ++j) {
        block.scales[j] = 1;      // d = 1
        block.scales[j + 4] = 0;  // m = 0
    }
    for (int j = 8; j < 12; ++j) {
        block.scales[j] = 0x01;   // d = 1, m = 0
    }
    // Set quants to 5
    for (size_t i = 0; i < 128; ++i) {
        block.qs[i] = 0x55; // both nibbles = 5
    }

    std::vector<float> dequant(256);
    vectra::quant::dequantize_row_q4_k(&block, dequant.data(), 256);

    for (size_t i = 0; i < 256; ++i) {
        TEST_ASSERT(std::fabs(dequant[i] - 5.0f) < 1e-4f, "Q4_K dequantize value mismatch at index " + std::to_string(i));
    }

    // Dot product with uniform 1.0 vector: 256 * 5.0 = 1280.0
    std::vector<float> ones(256, 1.0f);
    float dot_scalar = vectra::quant::vec_dot_q4_k_f32_scalar(&block, ones.data(), 256);
    TEST_ASSERT(std::fabs(dot_scalar - 1280.0f) < 1e-3f, "Q4_K scalar dot product mismatch");

    float dot_avx2 = vectra::quant::vec_dot_q4_k_f32(&block, ones.data(), 256);
    TEST_ASSERT(std::fabs(dot_avx2 - 1280.0f) < 1e-3f, "Q4_K AVX2 dot product mismatch");
    return true;
}

bool test_q6_k_correctness() {
    vectra::quant::BlockQ6_K block;
    block.d = vectra::float32_to_float16(1.0f);
    // Set 16 scales to 1
    for (int i = 0; i < 16; ++i) {
        block.scales[i] = 1;
    }
    // Set quants: ql = 0x22 (low 4 bits = 2), qh = 0x00 (high 2 bits = 0)
    // Value = 2 - 32 = -30
    for (int i = 0; i < 128; ++i) block.ql[i] = 0x22;
    for (int i = 0; i < 64; ++i) block.qh[i] = 0x00;

    std::vector<float> dequant(256);
    vectra::quant::dequantize_row_q6_k(&block, dequant.data(), 256);

    for (size_t i = 0; i < 256; ++i) {
        TEST_ASSERT(std::fabs(dequant[i] - (-30.0f)) < 1e-4f, "Q6_K dequantize value mismatch at " + std::to_string(i));
    }

    std::vector<float> ones(256, 1.0f);
    float dot_scalar = vectra::quant::vec_dot_q6_k_f32_scalar(&block, ones.data(), 256);
    TEST_ASSERT(std::fabs(dot_scalar - (256 * -30.0f)) < 1e-3f, "Q6_K scalar dot mismatch");

    float dot_avx2 = vectra::quant::vec_dot_q6_k_f32(&block, ones.data(), 256);
    TEST_ASSERT(std::fabs(dot_avx2 - (256 * -30.0f)) < 1e-3f, "Q6_K AVX2 dot mismatch");
    return true;
}

bool test_sampling() {
    vectra::runtime::SamplingParams params;
    params.greedy = true;
    vectra::runtime::Sampler greedy_sampler(params);

    std::vector<float> logits = {0.1f, 2.5f, -1.0f, 5.8f, 3.2f};
    int32_t tok = greedy_sampler.sample(logits.data(), logits.size());
    TEST_ASSERT(tok == 3, "Greedy sampler must choose argmax (index 3)");

    // Temperature sampling repeatability with seed
    params.greedy = false;
    params.temperature = 0.8f;
    params.seed = 12345;
    vectra::runtime::Sampler samp1(params);
    vectra::runtime::Sampler samp2(params);

    logits = {1.0f, 2.0f, 3.0f, 4.0f};
    int32_t t1 = samp1.sample(logits.data(), logits.size());
    logits = {1.0f, 2.0f, 3.0f, 4.0f};
    int32_t t2 = samp2.sample(logits.data(), logits.size());
    TEST_ASSERT(t1 == t2, "Seeded samplers must produce identical deterministic results");
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

    run_test("test_mmap_loader", test_mmap_loader);
    run_test("test_q4_k_correctness", test_q4_k_correctness);
    run_test("test_q6_k_correctness", test_q6_k_correctness);
    run_test("test_sampling", test_sampling);

    std::cout << "\nM2 Integration Test Summary: " << passed << " / " << total << " passed.\n";
    return (passed == total) ? 0 : 1;
}
