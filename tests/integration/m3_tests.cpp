#include "vectra/format/mmap_loader.hpp"
#include "vectra/quant/quant_k.hpp"
#include "vectra/model/kv_cache.hpp"
#include "vectra/model/gguf_model.hpp"
#include "vectra/runtime/sampler.hpp"
#include "vectra/runtime/thread_pool.hpp"
#include "vectra/tokenizer/bpe_tokenizer.hpp"

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while (0)

bool test_kv_cache_scaling_and_bounds() {
    std::vector<int64_t> ctx_lengths = {16, 64, 128, 256, 512, 1024};
    const int64_t num_layers = 28;
    const int64_t num_kv_heads = 2;
    const int64_t head_dim = 128;

    std::vector<float> dummy_k(num_kv_heads * head_dim, 0.42f);
    std::vector<float> dummy_v(num_kv_heads * head_dim, -0.42f);

    for (int64_t ctx : ctx_lengths) {
        auto t0 = std::chrono::high_resolution_clock::now();
        vectra::model::KVCache cache(num_layers, ctx, num_kv_heads, head_dim);
        auto t1 = std::chrono::high_resolution_clock::now();
        double alloc_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        size_t expected_bytes = static_cast<size_t>(num_layers * 2 * ctx * num_kv_heads * head_dim * sizeof(float));
        TEST_ASSERT(cache.memory_bytes() == expected_bytes, "KV cache memory bytes must match exact formula");

        // Benchmark append overhead per token
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int64_t pos = 0; pos < ctx; ++pos) {
            for (int64_t l = 0; l < num_layers; ++l) {
                cache.append(l, pos, dummy_k.data(), dummy_v.data());
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double write_us_per_token = std::chrono::duration<double, std::micro>(t3 - t2).count() / ctx;

        // Verify values
        const float* k_ptr = cache.key_cache(0);
        TEST_ASSERT(std::fabs(k_ptr[0] - 0.42f) < 1e-6f, "Cache value corrupted");

        // Verify reset behavior
        cache.reset();
        TEST_ASSERT(cache.current_seq_len() == 0, "Cache reset must clear current_seq_len");

        // Bounds check
        bool threw = false;
        try {
            cache.append(0, ctx + 5, dummy_k.data(), dummy_v.data());
        } catch (const vectra::VectraException&) {
            threw = true;
        }
        TEST_ASSERT(threw, "Out of bounds cache append must throw");
    }
    return true;
}

bool test_thread_pool_determinism() {
    const size_t rows = 1536;
    const size_t cols = 1536;
    std::vector<vectra::quant::BlockQ4_K> A(rows * (cols / vectra::quant::QK_K));
    std::vector<float> x(cols, 1.0f);
    std::vector<float> y_ref(rows, 0.0f);

    // Initialize deterministic blocks
    for (size_t i = 0; i < A.size(); ++i) {
        A[i].d = vectra::float32_to_float16(0.05f);
        A[i].dmin = vectra::float32_to_float16(0.01f);
        for (int j = 0; j < 12; ++j) A[i].scales[j] = static_cast<uint8_t>(j * 5);
        for (int j = 0; j < 128; ++j) A[i].qs[j] = static_cast<uint8_t>((i + j) & 0xFF);
    }

    // Compute reference with 1 thread
    vectra::runtime::ThreadPool::set_default_threads(1);
    vectra::quant::matvec_q4_k_f32(A.data(), x.data(), nullptr, y_ref.data(), rows, cols);

    // Test with multiple thread counts: 2, 4, 8, 10
    std::vector<size_t> test_threads = {2, 4, 8, 10};
    for (size_t t : test_threads) {
        std::vector<float> y_test(rows, 0.0f);
        vectra::runtime::ThreadPool::set_default_threads(t);
        vectra::quant::matvec_q4_k_f32(A.data(), x.data(), nullptr, y_test.data(), rows, cols);

        for (size_t r = 0; r < rows; ++r) {
            TEST_ASSERT(y_ref[r] == y_test[r], "Multi-thread execution must be 100% bit-exact identical to single thread");
        }
    }
    return true;
}

bool test_q4_k_kernel_parity() {
    const size_t k = 1536;
    const size_t nb = k / vectra::quant::QK_K;
    std::vector<vectra::quant::BlockQ4_K> blocks(nb);
    std::vector<float> activations(k);

    for (size_t i = 0; i < nb; ++i) {
        blocks[i].d = vectra::float32_to_float16(0.025f);
        blocks[i].dmin = vectra::float32_to_float16(0.005f);
        for (int j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 7 + j) & 0x3F);
        for (int j = 0; j < 128; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 13 + j * 3) & 0xFF);
    }
    for (size_t i = 0; i < k; ++i) {
        activations[i] = std::sin(static_cast<float>(i) * 0.1f) * 1.5f;
    }

    float dot_scalar = vectra::quant::vec_dot_q4_k_f32_scalar(blocks.data(), activations.data(), k);
    float dot_avx2 = vectra::quant::vec_dot_q4_k_f32_avx2(blocks.data(), activations.data(), k);

    float diff = std::fabs(dot_scalar - dot_avx2);
    float rel = diff / std::fabs(dot_scalar);

    TEST_ASSERT(rel < 1e-4f, "AVX2 fused Q4_K dot product must match scalar within float tolerance");
    return true;
}

bool test_full_model_reference_parity() {
    // Check if model file exists
    std::string model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    std::ifstream mf(model_path, std::ios::binary);
    if (!mf.good()) {
        std::cout << "[SKIP] Model checkpoint not found for reference parity: " << model_path << "\n";
        return true;
    }
    mf.close();

    // Verify greedy next token on prompt "The capital of France is"
    vectra::model::GGUFTransformerModel model;
    model.load(model_path);

    std::vector<int32_t> prompt = {785, 6722, 315, 9625, 374};
    vectra::runtime::SamplingParams params;
    params.greedy = true;
    vectra::runtime::Sampler sampler(params);

    auto generated = model.generate(prompt, 1, sampler, nullptr, 1024);
    TEST_ASSERT(!generated.empty(), "Model must generate at least 1 token");
    TEST_ASSERT(generated[0] == 12095, "Greedy next token must strictly be 12095 (' Paris')");

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

    std::cout << "=========================================================\n";
    std::cout << "           PROJECT VECTRA - M3 INTEGRATION SUITE         \n";
    std::cout << "=========================================================\n";

    run_test("test_kv_cache_scaling_and_bounds", test_kv_cache_scaling_and_bounds);
    run_test("test_thread_pool_determinism", test_thread_pool_determinism);
    run_test("test_q4_k_kernel_parity", test_q4_k_kernel_parity);
    run_test("test_full_model_reference_parity", test_full_model_reference_parity);

    std::cout << "\nM3 Integration Test Summary: " << passed << " / " << total << " passed.\n";
    return (passed == total) ? 0 : 1;
}
