#include "vectra/math/scalar_math.hpp"
#include "vectra/math/avx2_math.hpp"
#include "vectra/math/transformer_ops.hpp"
#include "vectra/quant/quant_block.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <random>

namespace {

float max_abs_diff(const float* a, const float* b, size_t n) {
    float max_d = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

float max_rel_diff(const float* a, const float* b, size_t n) {
    float max_r = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float denom = std::max(std::fabs(b[i]), 1e-6f);
        float r = std::fabs(a[i] - b[i]) / denom;
        if (r > max_r) max_r = r;
    }
    return max_r;
}

} // anonymous namespace

bool test_dot_product_numerical() {
    const size_t n = 256;
    std::vector<float> a(n);
    std::vector<float> b(n);
    float oracle_sum = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        a[i] = std::sin(static_cast<float>(i)) * 2.0f;
        b[i] = std::cos(static_cast<float>(i)) * 1.5f;
        oracle_sum += a[i] * b[i];
    }

    float scalar_res = vectra::math::scalar::dot(a.data(), b.data(), n);
    float err_scalar = std::fabs(scalar_res - oracle_sum);
    if (err_scalar > 1e-3f) {
        std::cerr << "Scalar dot failed: err=" << err_scalar << "\n";
        return false;
    }

    if (vectra::math::avx2::is_supported()) {
        float avx2_res = vectra::math::avx2::dot(a.data(), b.data(), n);
        float err_avx2 = std::fabs(avx2_res - oracle_sum);
        if (err_avx2 > 1e-3f) {
            std::cerr << "AVX2 dot failed: err=" << err_avx2 << "\n";
            return false;
        }
        float diff = std::fabs(avx2_res - scalar_res);
        if (diff > 1e-4f) {
            std::cerr << "Scalar vs AVX2 dot diff too large: " << diff << "\n";
            return false;
        }
    }
    return true;
}

bool test_matvec_numerical() {
    const size_t rows = 32;
    const size_t cols = 64;
    std::vector<float> A(rows * cols);
    std::vector<float> x(cols);
    std::vector<float> bias(rows);
    std::vector<float> oracle(rows);

    for (size_t r = 0; r < rows; ++r) {
        bias[r] = static_cast<float>(r) * 0.1f;
        float row_sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            A[r * cols + c] = std::sin(static_cast<float>(r + c)) * 0.5f;
            if (r == 0) x[c] = std::cos(static_cast<float>(c)) * 0.8f;
            row_sum += A[r * cols + c] * (std::cos(static_cast<float>(c)) * 0.8f);
        }
        oracle[r] = row_sum + bias[r];
    }

    std::vector<float> y_scalar(rows);
    vectra::math::scalar::matvec(A.data(), x.data(), bias.data(), y_scalar.data(), rows, cols);
    float max_err = max_abs_diff(y_scalar.data(), oracle.data(), rows);
    if (max_err > 1e-4f) {
        std::cerr << "Scalar matvec max_err: " << max_err << "\n";
        return false;
    }

    if (vectra::math::avx2::is_supported()) {
        std::vector<float> y_avx2(rows);
        vectra::math::avx2::matvec(A.data(), x.data(), bias.data(), y_avx2.data(), rows, cols);
        float avx2_err = max_abs_diff(y_avx2.data(), oracle.data(), rows);
        if (avx2_err > 1e-4f) {
            std::cerr << "AVX2 matvec max_err: " << avx2_err << "\n";
            return false;
        }
    }
    return true;
}

bool test_rmsnorm_numerical() {
    const size_t n = 128;
    std::vector<float> x(n);
    std::vector<float> weight(n, 1.2f);
    std::vector<float> oracle(n);

    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        x[i] = (static_cast<float>(i) - 64.0f) * 0.1f;
        sum_sq += x[i] * x[i];
    }
    float eps = 1e-6f;
    float rms = std::sqrt(sum_sq / static_cast<float>(n) + eps);
    for (size_t i = 0; i < n; ++i) {
        oracle[i] = (x[i] / rms) * weight[i];
    }

    std::vector<float> out_scalar(n);
    vectra::math::scalar::rms_norm(x.data(), weight.data(), out_scalar.data(), n, eps);
    float err = max_abs_diff(out_scalar.data(), oracle.data(), n);
    if (err > 1e-4f) {
        std::cerr << "RMSNorm scalar err: " << err << "\n";
        return false;
    }

    if (vectra::math::avx2::is_supported()) {
        std::vector<float> out_avx2(n);
        vectra::math::avx2::rms_norm(x.data(), weight.data(), out_avx2.data(), n, eps);
        float err_avx = max_abs_diff(out_avx2.data(), oracle.data(), n);
        if (err_avx > 1e-4f) {
            std::cerr << "RMSNorm AVX2 err: " << err_avx << "\n";
            return false;
        }
    }
    return true;
}

bool test_softmax_numerical() {
    const size_t n = 64;
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i % 10) * 1.5f;
    }

    std::vector<float> out(n);
    vectra::math::scalar::softmax(x.data(), out.data(), n);

    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (out[i] < 0.0f || out[i] > 1.0f) {
            std::cerr << "Softmax value out of [0, 1] range: " << out[i] << "\n";
            return false;
        }
        sum += out[i];
    }
    if (std::fabs(sum - 1.0f) > 1e-5f) {
        std::cerr << "Softmax partition sum != 1.0, got " << sum << "\n";
        return false;
    }
    return true;
}

bool test_rope_numerical() {
    const int64_t head_dim = 16;
    const int64_t pos = 5;
    std::vector<float> vec(head_dim);
    for (int64_t i = 0; i < head_dim; ++i) {
        vec[i] = static_cast<float>(i + 1);
    }
    std::vector<float> orig = vec;

    vectra::math::apply_rope(vec.data(), pos, head_dim, 10000.0f);

    // Compute expected manual rotation
    int64_t half = head_dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(head_dim));
        float angle = static_cast<float>(pos) * freq;
        float c = std::cos(angle);
        float s = std::sin(angle);
        float exp0 = orig[i] * c - orig[i + half] * s;
        float exp1 = orig[i] * s + orig[i + half] * c;

        if (std::fabs(vec[i] - exp0) > 1e-4f || std::fabs(vec[i + half] - exp1) > 1e-4f) {
            std::cerr << "RoPE numerical mismatch at index " << i << "\n";
            return false;
        }
    }
    return true;
}

bool test_pathological_inputs() {
    // 1. All zeros
    const size_t n = 64;
    std::vector<float> zeros(n, 0.0f);
    std::vector<float> out(n, 1.0f);
    vectra::math::scalar::rms_norm(zeros.data(), out.data(), out.data(), n, 1e-6f);
    for (float v : out) {
        if (v != 0.0f) {
            std::cerr << "RMSNorm of zeros was non-zero\n";
            return false;
        }
    }

    // 2. Large dynamic range
    std::vector<float> extreme(n);
    for (size_t i = 0; i < n; ++i) {
        extreme[i] = (i % 2 == 0) ? 1000.0f : -1000.0f;
    }
    vectra::math::scalar::softmax(extreme.data(), out.data(), n);
    for (float v : out) {
        if (std::isnan(v) || std::isinf(v)) {
            std::cerr << "Softmax produced NaN/Inf on large inputs\n";
            return false;
        }
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

    run_test("test_dot_product_numerical", test_dot_product_numerical);
    run_test("test_matvec_numerical", test_matvec_numerical);
    run_test("test_rmsnorm_numerical", test_rmsnorm_numerical);
    run_test("test_softmax_numerical", test_softmax_numerical);
    run_test("test_rope_numerical", test_rope_numerical);
    run_test("test_pathological_inputs", test_pathological_inputs);

    std::cout << "\nNumerical Test Summary: " << passed << " / " << total << " passed.\n";
    return (passed == total) ? 0 : 1;
}
