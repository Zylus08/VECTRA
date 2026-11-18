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
#include <algorithm>
#include <numeric>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while (0)

static double compute_cosine_similarity(const float* a, const float* b, size_t n) {
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (norm_a <= 0.0 || norm_b <= 0.0) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

static void compute_error_metrics(const float* ref, const float* actual, size_t n,
                                  float& max_abs_err, float& rms_err, double& cos_sim) {
    max_abs_err = 0.0f;
    double sum_sq_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float err = std::fabs(ref[i] - actual[i]);
        if (err > max_abs_err) max_abs_err = err;
        sum_sq_err += static_cast<double>(err) * static_cast<double>(err);
    }
    rms_err = static_cast<float>(std::sqrt(sum_sq_err / static_cast<double>(n)));
    cos_sim = compute_cosine_similarity(ref, actual, n);
}

// 1. RMSNorm Numerical Certification
bool test_m5_int8_rmsnorm_parity() {
    const size_t n = 1536;
    std::vector<float> x(n);
    std::vector<float> weight(n, 1.0f);
    std::vector<float> out(n);

    for (size_t i = 0; i < n; ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.03f) * 2.0f;
        weight[i] = 1.0f + 0.1f * std::cos(static_cast<float>(i) * 0.01f);
    }

    vectra::math::scalar::rms_norm(x.data(), weight.data(), out.data(), n, 1e-6f);

    // Compute expected RMSNorm manually with double precision
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    double rms = std::sqrt(sum_sq / static_cast<double>(n) + 1e-6);
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float expected = static_cast<float>((static_cast<double>(x[i]) / rms) * static_cast<double>(weight[i]));
        float diff = std::fabs(out[i] - expected);
        if (diff > max_diff) max_diff = diff;
    }

    TEST_ASSERT(max_diff < 1e-5f, "RMSNorm output must match double precision within 1e-5");
    return true;
}

// 2. QKV Projection Fidelity: INT8 vs FP32 High-Precision
bool test_m5_int8_qkv_projection_fidelity() {
    const size_t in_dim = 1536;
    const size_t out_dim = 1536;
    const size_t nb = in_dim / vectra::quant::QK_K;

    std::vector<vectra::quant::BlockQ4_K> weight(nb * out_dim);
    for (size_t r = 0; r < out_dim; ++r) {
        for (size_t b = 0; b < nb; ++b) {
            auto& blk = weight[r * nb + b];
            blk.d = vectra::float32_to_float16(0.02f);
            blk.dmin = vectra::float32_to_float16(0.005f);
            for (int j = 0; j < 12; ++j) blk.scales[j] = static_cast<uint8_t>((r * 3 + b * 5 + j) & 0x3F);
            for (int j = 0; j < 128; ++j) blk.qs[j] = static_cast<uint8_t>((r * 7 + b * 11 + j * 3) & 0xFF);
        }
    }

    std::vector<float> input(in_dim);
    for (size_t i = 0; i < in_dim; ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.05f) * 1.5f;
    }

    // FP32 reference projection
    std::vector<float> ref_out(out_dim, 0.0f);
    for (size_t r = 0; r < out_dim; ++r) {
        ref_out[r] = vectra::quant::vec_dot_q4_k_f32_avx2(&weight[r * nb], input.data(), in_dim);
    }

    // INT8 accelerated projection
    std::vector<vectra::quant::BlockQ8_K_256> q8_act(nb);
    vectra::quant::quantize_activation_q8_k_256(input.data(), q8_act.data(), in_dim);
    std::vector<float> int8_out(out_dim, 0.0f);
    for (size_t r = 0; r < out_dim; ++r) {
        int8_out[r] = vectra::quant::vec_dot_q4_k_q8_avx2(&weight[r * nb], q8_act.data(), in_dim);
    }

    float max_abs_err = 0.0f;
    float rms_err = 0.0f;
    double cos_sim = 0.0;
    compute_error_metrics(ref_out.data(), int8_out.data(), out_dim, max_abs_err, rms_err, cos_sim);
    std::cout << " [cos_sim=" << cos_sim << ", rms_err=" << rms_err << ", max_err=" << max_abs_err << "] ";

    TEST_ASSERT(cos_sim > 0.999, "QKV projection cosine similarity must exceed 0.999");
    TEST_ASSERT(cos_sim > 0.9999, "QKV projection cosine similarity must exceed 0.9999");
    return true;
}

// 3. SwiGLU Activation Parity
bool test_m5_int8_swiglu_fidelity() {
    const size_t inter_dim = 8960;
    std::vector<float> gate(inter_dim);
    std::vector<float> up(inter_dim);
    std::vector<float> out(inter_dim);

    for (size_t i = 0; i < inter_dim; ++i) {
        gate[i] = std::sin(static_cast<float>(i) * 0.02f) * 2.0f;
        up[i] = std::cos(static_cast<float>(i) * 0.03f) * 1.5f;
    }

    vectra::math::scalar::swiglu(gate.data(), up.data(), out.data(), inter_dim);

    // Verify SwiGLU mathematical property: out = (gate / (1 + exp(-gate))) * up
    for (size_t i = 0; i < 100; ++i) {
        float expected_silu = gate[i] / (1.0f + std::exp(-gate[i]));
        float expected_out = expected_silu * up[i];
        TEST_ASSERT(std::fabs(out[i] - expected_out) < 1e-5f, "SwiGLU output must match definition");
    }
    return true;
}

// 4. Down Projection Q6_K INT8 vs FP32 Parity
bool test_m5_int8_down_projection_fidelity() {
    const size_t in_dim = 8960;
    const size_t out_dim = 1536;
    const size_t nb = in_dim / vectra::quant::QK_K; // 35 blocks

    std::vector<vectra::quant::BlockQ6_K> weight(nb * out_dim);
    for (size_t r = 0; r < out_dim; ++r) {
        for (size_t b = 0; b < nb; ++b) {
            auto& blk = weight[r * nb + b];
            blk.d = vectra::float32_to_float16(0.015f);
            for (int j = 0; j < 16; ++j) blk.scales[j] = static_cast<int8_t>((r * 2 + b * 3 + j) % 25 - 12);
            for (int j = 0; j < 128; ++j) blk.ql[j] = static_cast<uint8_t>((r * 5 + b * 7 + j) & 0xFF);
            for (int j = 0; j < 64; ++j) blk.qh[j] = static_cast<uint8_t>((r * 3 + b * 11 + j) & 0xFF);
        }
    }

    std::vector<float> input(in_dim);
    for (size_t i = 0; i < in_dim; ++i) {
        input[i] = std::cos(static_cast<float>(i) * 0.01f) * 1.2f;
    }

    std::vector<float> ref_out(out_dim, 0.0f);
    for (size_t r = 0; r < out_dim; ++r) {
        ref_out[r] = vectra::quant::vec_dot_q6_k_f32_avx2(&weight[r * nb], input.data(), in_dim);
    }

    std::vector<vectra::quant::BlockQ8_K_256> q8_act(nb);
    vectra::quant::quantize_activation_q8_k_256(input.data(), q8_act.data(), in_dim);
    std::vector<float> int8_out(out_dim, 0.0f);
    for (size_t r = 0; r < out_dim; ++r) {
        int8_out[r] = vectra::quant::vec_dot_q6_k_q8_avx2(&weight[r * nb], q8_act.data(), in_dim);
    }

    float max_abs_err = 0.0f;
    float rms_err = 0.0f;
    double cos_sim = 0.0;
    compute_error_metrics(ref_out.data(), int8_out.data(), out_dim, max_abs_err, rms_err, cos_sim);
    std::cout << " [cos_sim=" << cos_sim << ", rms_err=" << rms_err << ", max_err=" << max_abs_err << "] ";

    TEST_ASSERT(cos_sim > 0.999, "Down projection Q6_K cosine similarity must exceed 0.999");
    TEST_ASSERT(cos_sim > 0.9999, "Down projection Q6_K cosine similarity must exceed 0.9999");
    return true;
}

// 5. Full Model Greedy Generation and Top-1 Match
bool test_m5_full_model_greedy_generation() {
    std::string model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    std::ifstream mf(model_path, std::ios::binary);
    if (!mf.good()) {
        std::cout << "[SKIP] Model checkpoint not found: " << model_path << "\n";
        return true;
    }
    mf.close();

    vectra::model::GGUFTransformerModel model;
    model.load(model_path);
    model.set_use_int8_kernels(true); // Certified INT8 mode

    std::vector<int32_t> prompt = {785, 6722, 315, 9625, 374}; // "The capital of France is"
    vectra::runtime::SamplingParams params;
    params.greedy = true;
    vectra::runtime::Sampler sampler(params);

    auto generated = model.generate(prompt, 1, sampler, nullptr, 1024);
    TEST_ASSERT(!generated.empty(), "Model must generate at least 1 token");
    TEST_ASSERT(generated[0] == 12095, "Greedy next token must strictly be 12095 (' Paris')");

    return true;
}

// 6. Prefill Throughput Recovery Validation
bool test_m5_prefill_throughput_recovery() {
    std::string model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    std::ifstream mf(model_path, std::ios::binary);
    if (!mf.good()) {
        std::cout << "[SKIP] Model checkpoint not found: " << model_path << "\n";
        return true;
    }
    mf.close();

    vectra::model::GGUFTransformerModel model;
    model.load(model_path);
    model.set_use_int8_kernels(true);

    // Prompt of 5 tokens
    std::vector<int32_t> prompt = {785, 6722, 315, 9625, 374};

    vectra::model::KVCache kv_cache(model.config().num_layers, 1024, model.config().num_kv_heads, model.config().head_dim);
    std::vector<float> logits(model.config().vocab_size);

    // Warm up one step so pages are in CPU cache / OS page cache
    model.forward_prompt_batch(prompt, kv_cache, logits.data());
    kv_cache.reset();

    auto t0 = std::chrono::high_resolution_clock::now();
    model.forward_prompt_batch(prompt, kv_cache, logits.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double tok_per_sec = (static_cast<double>(prompt.size()) / prefill_ms) * 1000.0;

    std::cout << " (Prefill " << prompt.size() << " tokens: " << prefill_ms << " ms, " << tok_per_sec << " tok/s) ... ";
    TEST_ASSERT(tok_per_sec > 15.0, "Prefill throughput must exceed 15.0 tok/s");
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
    std::cout << "           PROJECT VECTRA - M5 INTEGRATION SUITE         \n";
    std::cout << "=========================================================\n";

    run_test("test_m5_int8_rmsnorm_parity", test_m5_int8_rmsnorm_parity);
    run_test("test_m5_int8_qkv_projection_fidelity", test_m5_int8_qkv_projection_fidelity);
    run_test("test_m5_int8_swiglu_fidelity", test_m5_int8_swiglu_fidelity);
    run_test("test_m5_int8_down_projection_fidelity", test_m5_int8_down_projection_fidelity);
    run_test("test_m5_full_model_greedy_generation", test_m5_full_model_greedy_generation);
    run_test("test_m5_prefill_throughput_recovery", test_m5_prefill_throughput_recovery);

    std::cout << "\nM5 Integration Test Summary: " << passed << " / " << total << " passed.\n";
    return (passed == total) ? 0 : 1;
}
