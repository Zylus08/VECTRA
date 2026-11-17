#include "vectra/math/avx2_math.hpp"
#include <cmath>

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <x86intrin.h>
#include <cpuid.h>
#endif

namespace vectra::math::avx2 {

bool is_supported() {
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    int nIds = info[0];
    if (nIds >= 7) {
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 5)) != 0; // AVX2 bit
    }
    return false;
#elif defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

#if defined(VECTRA_USE_AVX2)

static inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v128 = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(v128);
    __m128 sums = _mm_add_ps(v128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void add(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void mul(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void scale(const float* a, float s, float* out, size_t n) {
    __m256 vs = _mm256_set1_ps(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(out + i, vr);
    }
    for (; i < n; ++i) {
        out[i] = a[i] * s;
    }
}

float dot(const float* a, const float* b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }
    float sum = hsum256_ps(acc0);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void matvec(const float* A, const float* x, const float* bias, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const float* row_ptr = A + r * cols;
        float d = dot(row_ptr, x, cols);
        if (bias) {
            d += bias[r];
        }
        y[r] = d;
    }
}

void rms_norm(const float* x, const float* weight, float* out, size_t n, float eps) {
    __m256 sum_sq_vec = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        sum_sq_vec = _mm256_fmadd_ps(vx, vx, sum_sq_vec);
    }
    float sum_sq = hsum256_ps(sum_sq_vec);
    for (; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(n) + eps);
    float inv_rms = 1.0f / rms;
    __m256 vinv = _mm256_set1_ps(inv_rms);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 scaled = _mm256_mul_ps(vx, vinv);
        __m256 res = _mm256_mul_ps(scaled, vw);
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < n; ++i) {
        out[i] = (x[i] * inv_rms) * weight[i];
    }
}

#else

// Fallback to scalar if AVX2 disabled
void add(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
void mul(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}
void scale(const float* a, float s, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * s;
}
float dot(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
void matvec(const float* A, const float* x, const float* bias, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) sum += A[r * cols + c] * x[c];
        if (bias) sum += bias[r];
        y[r] = sum;
    }
}
void rms_norm(const float* x, const float* weight, float* out, size_t n, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
    for (size_t i = 0; i < n; ++i) out[i] = (x[i] * inv_rms) * weight[i];
}

#endif

} // namespace vectra::math::avx2
