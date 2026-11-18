#include "vectra/format/mmap_loader.hpp"
#include "vectra/quant/quant_k.hpp"
#include "vectra/model/kv_cache.hpp"
#include "vectra/model/gguf_model.hpp"
#include "vectra/runtime/sampler.hpp"
#include "vectra/runtime/thread_pool.hpp"
#include "vectra/math/scalar_math.hpp"
#include "vectra/math/avx2_math.hpp"

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while (0)

// 1. Activation Quantization Numerical Bounds
bool test_activation_quantization_bounds() {
    const size_t n_elements = 1536;
    const size_t n_blocks = n_elements / vectra::quant::QK_K; // 6 blocks
    std::vector<float> input(n_elements);

    // Populate with realistic Transformer hidden activations
    for (size_t i = 0; i < n_elements; ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.05f) * 2.5f + (i % 128 == 0 ? 8.0f : 0.0f); // with outliers
    }

    std::vector<vectra::quant::BlockQ8_K_256> q8_blocks(n_blocks);
    vectra::quant::quantize_activation_q8_k_256(input.data(), q8_blocks.data(), n_elements);

    // Verify quantization properties for each block
    for (size_t b = 0; b < n_blocks; ++b) {
        // 32-element sub-blocks
        for (int sb = 0; sb < 8; ++sb) {
            TEST_ASSERT(q8_blocks[b].scales32[sb] > 0.0f, "Scale32 must be strictly positive");
            float max_sub = 0.0f;
            for (int j = 0; j < 32; ++j) {
                float a = std::fabs(input[b * 256 + sb * 32 + j]);
                if (a > max_sub) max_sub = a;
            }
            float expected_scale = max_sub / 127.0f;
            TEST_ASSERT(std::fabs(q8_blocks[b].scales32[sb] - expected_scale) < 1e-5f, "Scale32 calculation must match exact formula");
        }

        // 16-element sub-blocks
        for (int sb = 0; sb < 16; ++sb) {
            TEST_ASSERT(q8_blocks[b].scales[sb] > 0.0f, "Scale16 must be strictly positive");
            int32_t expected_sum = 0;
            for (int j = 0; j < 16; ++j) {
                expected_sum += q8_blocks[b].qs[sb * 16 + j];
            }
            TEST_ASSERT(q8_blocks[b].sums[sb] == expected_sum, "Precomputed sum16 must match integer sum of qs");
        }

        // Check value reconstruction error within theoretical quantization step
        for (int i = 0; i < 256; ++i) {
            int sb = i / 32;
            float reconstructed = static_cast<float>(q8_blocks[b].qs[i]) * q8_blocks[b].scales32[sb];
            float original = input[b * 256 + i];
            float err = std::fabs(original - reconstructed);
            float max_step = q8_blocks[b].scales32[sb] + 1e-4f;
            TEST_ASSERT(err <= max_step, "Quantization error must be bounded by sub-block step size");
        }
    }

    return true;
}

// 2. Q4_K INT8 Kernel Parity vs FP32
bool test_q4_k_int8_kernel_parity() {
    const size_t k = 1536;
    const size_t nb = k / vectra::quant::QK_K;
    std::vector<vectra::quant::BlockQ4_K> blocks(nb);
    std::vector<float> activations(k);

    for (size_t i = 0; i < nb; ++i) {
        blocks[i].d = vectra::float32_to_float16(0.04f);
        blocks[i].dmin = vectra::float32_to_float16(0.008f);
        for (int j = 0; j < 12; ++j) blocks[i].scales[j] = static_cast<uint8_t>((i * 5 + j * 3) & 0x3F);
        for (int j = 0; j < 128; ++j) blocks[i].qs[j] = static_cast<uint8_t>((i * 17 + j * 7) & 0xFF);
    }
    for (size_t i = 0; i < k; ++i) {
        activations[i] = std::cos(static_cast<float>(i) * 0.08f) * 1.8f;
    }

    float dot_fp32 = vectra::quant::vec_dot_q4_k_f32_avx2(blocks.data(), activations.data(), k);

    std::vector<vectra::quant::BlockQ8_K_256> q8_act(nb);
    vectra::quant::quantize_activation_q8_k_256(activations.data(), q8_act.data(), k);
    float dot_int8 = vectra::quant::vec_dot_q4_k_q8_avx2(blocks.data(), q8_act.data(), k);

    float diff = std::fabs(dot_fp32 - dot_int8);
    float rel = diff / std::max(std::fabs(dot_fp32), 1e-4f);

    TEST_ASSERT(rel < 0.005f, "Q4_K INT8 DP4A dot product relative error must be under 0.5%");
    return true;
}

// 3. Q6_K INT8 Kernel Parity vs FP32
bool test_q6_k_int8_kernel_parity() {
    const size_t k = 1536;
    const size_t nb = k / vectra::quant::QK_K;
    std::vector<vectra::quant::BlockQ6_K> blocks(nb);
    std::vector<float> activations(k);

    for (size_t i = 0; i < nb; ++i) {
        blocks[i].d = vectra::float32_to_float16(0.02f);
        for (int j = 0; j < 16; ++j) blocks[i].scales[j] = static_cast<int8_t>((i * 3 + j * 2) % 31 - 15);
        for (int j = 0; j < 128; ++j) blocks[i].ql[j] = static_cast<uint8_t>((i * 11 + j * 5) & 0xFF);
        for (int j = 0; j < 64; ++j) blocks[i].qh[j] = static_cast<uint8_t>((i * 7 + j * 13) & 0xFF);
    }
    for (size_t i = 0; i < k; ++i) {
        activations[i] = std::sin(static_cast<float>(i) * 0.04f) * 2.1f;
    }

    float dot_fp32 = vectra::quant::vec_dot_q6_k_f32_avx2(blocks.data(), activations.data(), k);

    std::vector<vectra::quant::BlockQ8_K_256> q8_act(nb);
    vectra::quant::quantize_activation_q8_k_256(activations.data(), q8_act.data(), k);
    float dot_int8 = vectra::quant::vec_dot_q6_k_q8_avx2(blocks.data(), q8_act.data(), k);

    float diff = std::fabs(dot_fp32 - dot_int8);
    float rel = diff / std::max(std::fabs(dot_fp32), 1e-4f);

    TEST_ASSERT(rel < 0.005f, "Q6_K INT8 DP4A dot product relative error must be under 0.5%");
    return true;
}

// 4. Q6_K Fused FP32 Parity vs Scalar Reference
bool test_q6_k_fused_f32_parity() {
    const size_t k = 1536;
    const size_t nb = k / vectra::quant::QK_K;
    std::vector<vectra::quant::BlockQ6_K> blocks(nb);
    std::vector<float> activations(k);

    for (size_t i = 0; i < nb; ++i) {
        blocks[i].d = vectra::float32_to_float16(0.035f);
        for (int j = 0; j < 16; ++j) blocks[i].scales[j] = static_cast<int8_t>((i * 4 + j) % 27 - 13);
        for (int j = 0; j < 128; ++j) blocks[i].ql[j] = static_cast<uint8_t>((i * 19 + j * 2) & 0xFF);
        for (int j = 0; j < 64; ++j) blocks[i].qh[j] = static_cast<uint8_t>((i * 23 + j * 7) & 0xFF);
    }
    for (size_t i = 0; i < k; ++i) {
        activations[i] = std::cos(static_cast<float>(i) * 0.03f) * 1.2f;
    }

    float dot_scalar = vectra::quant::vec_dot_q6_k_f32_scalar(blocks.data(), activations.data(), k);
    float dot_fused = vectra::quant::vec_dot_q6_k_f32_fused_avx2(blocks.data(), activations.data(), k);

    float diff = std::fabs(dot_scalar - dot_fused);
    TEST_ASSERT(diff < 1e-4f, "Fused Q6_K AVX2 dot product must match scalar reference exactly");
    return true;
}

// 5. Multi-Layer Numerical Parity & Greedy Generation
bool test_multi_layer_numerical_bounds() {
    std::string model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    std::ifstream mf(model_path, std::ios::binary);
    if (!mf.good()) {
        std::cout << "[SKIP] Model checkpoint not found: " << model_path << "\n";
        return true;
    }
    mf.close();

    vectra::model::GGUFTransformerModel model;
    model.load(model_path);
    model.set_use_int8_kernels(false); // certified FP32 mode

    std::vector<int32_t> prompt = {785, 6722, 315, 9625, 374}; // "The capital of France is"
    vectra::runtime::SamplingParams params;
    params.greedy = true;
    vectra::runtime::Sampler sampler(params);

    auto generated = model.generate(prompt, 5, sampler, nullptr, 1024);
    TEST_ASSERT(generated.size() == 5, "Model must generate 5 tokens");
    TEST_ASSERT(generated[0] == 12095, "Token 0 must be 12095 (' Paris')");
    TEST_ASSERT(generated[1] == 13,    "Token 1 must be 13 ('.')");
    TEST_ASSERT(generated[2] == 576,   "Token 2 must be 576 (' The')");
    TEST_ASSERT(generated[3] == 6722,  "Token 3 must be 6722 (' capital')");
    TEST_ASSERT(generated[4] == 315,   "Token 4 must be 315 (' of')");

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
    std::cout << "           PROJECT VECTRA - M4 INTEGRATION SUITE         \n";
    std::cout << "=========================================================\n";

    run_test("test_activation_quantization_bounds", test_activation_quantization_bounds);
    run_test("test_q4_k_int8_kernel_parity", test_q4_k_int8_kernel_parity);
    run_test("test_q6_k_int8_kernel_parity", test_q6_k_int8_kernel_parity);
    run_test("test_q6_k_fused_f32_parity", test_q6_k_fused_f32_parity);
    run_test("test_multi_layer_numerical_bounds", test_multi_layer_numerical_bounds);

    std::cout << "\nM4 Integration Test Summary: " << passed << " / " << total << " passed.\n";
    return (passed == total) ? 0 : 1;
}
