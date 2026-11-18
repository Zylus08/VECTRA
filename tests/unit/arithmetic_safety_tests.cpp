#include "vectra/core/memory.hpp"
#include "vectra/core/types.hpp"
#include "vectra/quant/quant_k.hpp"
#include "vectra/math/avx2_math.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while (0)

// 1. NaN and Inf handling
bool test_q8_k_nan_inf_handling() {
    std::vector<float> input(256, 1.0f);
    input[10] = std::numeric_limits<float>::quiet_NaN();
    input[50] = std::numeric_limits<float>::infinity();
    input[100] = -std::numeric_limits<float>::infinity();

    vectra::quant::BlockQ8_K_256 block;
    vectra::quant::quantize_activation_q8_k_256(input.data(), &block, 256);

    for (int i = 0; i < 8; ++i) {
        TEST_ASSERT(std::isfinite(block.scales32[i]), "Scale32 must be finite even with NaN/Inf inputs");
        TEST_ASSERT(block.scales32[i] >= 0.0f, "Scale32 must be non-negative");
    }
    for (int i = 0; i < 16; ++i) {
        TEST_ASSERT(std::isfinite(block.scales[i]), "Scale must be finite even with NaN/Inf inputs");
        TEST_ASSERT(block.scales[i] >= 0.0f, "Scale must be non-negative");
    }
    for (int i = 0; i < 256; ++i) {
        TEST_ASSERT(block.qs[i] >= -127 && block.qs[i] <= 127, "Quant must strictly be in [-127, 127]");
    }
    return true;
}

// 2. Subnormal floats handling
bool test_q8_k_subnormals() {
    std::vector<float> input(256, 0.0f);
    input[0] = 1.0e-38f;
    input[1] = -1.0e-39f;

    vectra::quant::BlockQ8_K_256 block;
    vectra::quant::quantize_activation_q8_k_256(input.data(), &block, 256);

    for (int i = 0; i < 8; ++i) {
        TEST_ASSERT(std::isfinite(block.scales32[i]), "Subnormals must not produce NaN scales");
    }
    for (int i = 0; i < 256; ++i) {
        TEST_ASSERT(block.qs[i] >= -127 && block.qs[i] <= 127, "Quant must be in [-127, 127]");
    }
    return true;
}

// 3. All zeros handling
bool test_q8_k_zeros() {
    std::vector<float> input(256, 0.0f);
    vectra::quant::BlockQ8_K_256 block;
    vectra::quant::quantize_activation_q8_k_256(input.data(), &block, 256);

    for (int i = 0; i < 8; ++i) {
        TEST_ASSERT(block.scales32[i] == 0.0f, "Zero input must produce zero scale32");
        TEST_ASSERT(block.sums32[i] == 0, "Zero input must produce zero sum32");
    }
    for (int i = 0; i < 16; ++i) {
        TEST_ASSERT(block.scales[i] == 0.0f, "Zero input must produce zero scale");
        TEST_ASSERT(block.sums[i] == 0, "Zero input must produce zero sum");
        TEST_ASSERT(block.sums_interleaved[i] == 0, "Zero input must produce zero interleaved sum");
    }
    for (int i = 0; i < 256; ++i) {
        TEST_ASSERT(block.qs[i] == 0, "Zero input must produce all-zero quants");
    }
    return true;
}

// 4. Extreme magnitude scaling
bool test_q8_k_extreme_magnitudes() {
    std::vector<float> input(256, 1.0e30f);
    input[128] = -1.0e30f;

    vectra::quant::BlockQ8_K_256 block;
    vectra::quant::quantize_activation_q8_k_256(input.data(), &block, 256);

    for (int i = 0; i < 8; ++i) {
        TEST_ASSERT(std::isfinite(block.scales32[i]), "Extreme input must produce finite scale");
    }
    for (int i = 0; i < 256; ++i) {
        TEST_ASSERT(block.qs[i] >= -127 && block.qs[i] <= 127, "Extreme input must stay within [-127, 127]");
    }
    TEST_ASSERT(block.qs[0] == 127, "Max positive input must quantize to +127");
    TEST_ASSERT(block.qs[128] == -127, "Max negative input must quantize to -127");
    return true;
}

// 5. Adversarial Q4_K saturation guard
bool test_q4_k_no_saturation_adversarial() {
    vectra::quant::BlockQ4_K block;
    block.d = vectra::float32_to_float16(1.0f);
    block.dmin = vectra::float32_to_float16(0.0f);
    // All 6-bit scales set to max (63)
    std::fill(std::begin(block.scales), std::end(block.scales), uint8_t(63));
    // All nibbles set to max (15) -> 0xFF has two 15s
    std::fill(std::begin(block.qs), std::end(block.qs), uint8_t(0xFF));

    // Activation block with alternating max values (+127, -127)
    vectra::quant::BlockQ8_K_256 act;
    for (int i = 0; i < 256; ++i) {
        act.qs[i] = (i % 2 == 0) ? 127 : -127;
    }
    for (int sb = 0; sb < 8; ++sb) {
        act.scales32[sb] = 1.0f;
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i) sum += act.qs[sb * 32 + i];
        act.sums32[sb] = sum;
    }

    float dot_avx2 = vectra::quant::vec_dot_q4_k_q8_avx2(&block, &act, 256);

    // Compute exact double-precision reference
    double expected = 0.0;
    for (int sb = 0; sb < 8; ++sb) {
        double dl = 1.0 * 63;
        double sum_dot = 0.0;
        for (int i = 0; i < 32; ++i) {
            sum_dot += 15.0 * static_cast<double>(act.qs[sb * 32 + i]);
        }
        expected += 1.0 * dl * sum_dot;
    }

    double diff = std::fabs(static_cast<double>(dot_avx2) - expected);
    TEST_ASSERT(diff <= 1.0, "Q4_K dot product must match exact reference without 16-bit integer saturation");
    return true;
}

// 6. Adversarial Q6_K saturation guard
bool test_q6_k_no_saturation_adversarial() {
    vectra::quant::BlockQ6_K block;
    block.d = vectra::float32_to_float16(1.0f);
    // Max positive scales (+127)
    std::fill(std::begin(block.scales), std::end(block.scales), int8_t(127));
    // All lower 4 bits = 15, upper 2 bits = 3 -> quant value = 63 (63 - 32 = +31)
    std::fill(std::begin(block.ql), std::end(block.ql), uint8_t(0xFF));
    std::fill(std::begin(block.qh), std::end(block.qh), uint8_t(0xFF));

    // Activation block with max values (+127)
    vectra::quant::BlockQ8_K_256 act;
    for (int i = 0; i < 256; ++i) {
        act.qs[i] = 127;
    }
    for (int sb = 0; sb < 8; ++sb) {
        act.scales32[sb] = 1.0f;
    }
    for (int sb = 0; sb < 16; ++sb) {
        act.scales[sb] = 1.0f;
        act.sums[sb] = 16 * 127;
    }
    for (int i = 0; i < 4; ++i) {
        act.sums_interleaved[i + 0] = act.sums[2 * i + 0];
        act.sums_interleaved[i + 4] = act.sums[2 * i + 1];
        act.sums_interleaved[i + 8] = act.sums[8 + 2 * i + 0];
        act.sums_interleaved[i + 12] = act.sums[8 + 2 * i + 1];
    }

    float dot_avx2 = vectra::quant::vec_dot_q6_k_q8_avx2(&block, &act, 256);

    // Compute exact double-precision reference
    // Each element has weight (63 - 32) = +31, act = 127, scale = 127
    // Per 16-element sub-block: sum = 16 * (31 * 127) * 127 = 16 * 3937 * 127 = 7,999,984
    // Across 16 sub-blocks: 16 * 7,999,984 = 127,999,744
    double expected = 16.0 * (16.0 * 31.0 * 127.0 * 127.0);
    double diff = std::fabs(static_cast<double>(dot_avx2) - expected);
    double rel_err = diff / expected;
    TEST_ASSERT(rel_err < 0.001, "Q6_K dot product must match exact reference without 16-bit integer saturation");
    return true;
}

// 7. Strict range clamping boundary (never -128)
bool test_q8_k_strict_clamping_boundary() {
    std::vector<float> input(256);
    for (int i = 0; i < 256; ++i) {
        input[i] = -128.0f; // Directly request -128
    }

    vectra::quant::BlockQ8_K_256 block;
    vectra::quant::quantize_activation_q8_k_256(input.data(), &block, 256);

    for (int i = 0; i < 256; ++i) {
        TEST_ASSERT(block.qs[i] != -128, "Quant must strictly avoid -128 to prevent maddubs signed saturation");
        TEST_ASSERT(block.qs[i] == -127, "Quant must clamp to -127");
    }
    return true;
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "       PROJECT VECTRA - ARITHMETIC SAFETY TEST SUITE     \n";
    std::cout << "=========================================================\n";

    int passed = 0;
    int total = 0;

    auto run_test = [&](const char* name, bool (*fn)()) {
        total++;
        std::cout << "[RUN]  " << name << " ... ";
        if (fn()) {
            std::cout << "PASSED\n";
            passed++;
        } else {
            std::cout << "FAILED\n";
        }
    };

    run_test("test_q8_k_nan_inf_handling", test_q8_k_nan_inf_handling);
    run_test("test_q8_k_subnormals", test_q8_k_subnormals);
    run_test("test_q8_k_zeros", test_q8_k_zeros);
    run_test("test_q8_k_extreme_magnitudes", test_q8_k_extreme_magnitudes);
    run_test("test_q4_k_no_saturation_adversarial", test_q4_k_no_saturation_adversarial);
    run_test("test_q6_k_no_saturation_adversarial", test_q6_k_no_saturation_adversarial);
    run_test("test_q8_k_strict_clamping_boundary", test_q8_k_strict_clamping_boundary);

    std::cout << "\nArithmetic Safety Test Summary: " << passed << " / " << total << " passed.\n";
    return (passed == total) ? 0 : 1;
}
