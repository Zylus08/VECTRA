#include "vectra/math/scalar_math.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace vectra::math::scalar {

void add(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void mul(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void scale(const float* a, float s, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * s;
    }
}

float dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void matvec(const float* A, const float* x, const float* bias, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* row_ptr = A + r * cols;
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            sum += row_ptr[c] * x[c];
        }
        if (bias) {
            sum += bias[r];
        }
        y[r] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

void rms_norm(const float* x, const float* weight, float* out, size_t n, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(n) + eps);
    float inv_rms = 1.0f / rms;
    for (size_t i = 0; i < n; ++i) {
        out[i] = (x[i] * inv_rms) * weight[i];
    }
}

void softmax(const float* x, float* out, size_t n) {
    if (n == 0) return;
    float max_val = x[0];
    for (size_t i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum_exp = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float e = std::exp(x[i] - max_val);
        out[i] = e;
        sum_exp += e;
    }
    float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < n; ++i) {
        out[i] *= inv_sum;
    }
}

float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}

void silu(const float* x, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = silu_scalar(x[i]);
    }
}

void swiglu(const float* gate, const float* up, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = silu_scalar(gate[i]) * up[i];
    }
}

} // namespace vectra::math::scalar
