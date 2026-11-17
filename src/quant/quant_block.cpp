#include "vectra/quant/quant_block.hpp"
#include "vectra/math/avx2_math.hpp"
#include "vectra/runtime/thread_pool.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

#if defined(VECTRA_USE_AVX2)
#include <immintrin.h>
#endif

namespace vectra::quant {

void quantize_row_q4_0(const float* x, BlockQ4_0* y, size_t k) {
    size_t nb = (k + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;

    for (size_t b = 0; b < nb; ++b) {
        size_t offset = b * Q4_BLOCK_SIZE;
        size_t block_len = std::min(Q4_BLOCK_SIZE, k - offset);

        // Find max absolute value in the block
        float max_abs = 0.0f;
        for (size_t i = 0; i < block_len; ++i) {
            float a = std::fabs(x[offset + i]);
            if (a > max_abs) max_abs = a;
        }

        if (max_abs == 0.0f) {
            y[b].d = float32_to_float16(0.0f);
            // Packed 8 (offset 8 represents 0)
            std::memset(y[b].qs, 0x88, sizeof(y[b].qs));
            continue;
        }

        // Quantization scale: maps max_abs to 7 (range [-8, 7])
        float d = max_abs / 7.0f;
        float id = 1.0f / d;
        y[b].d = float32_to_float16(d);

        for (size_t i = 0; i < Q4_BLOCK_SIZE / 2; ++i) {
            size_t idx0 = offset + 2 * i;
            size_t idx1 = offset + 2 * i + 1;

            float v0 = (2 * i < block_len) ? x[idx0] : 0.0f;
            float v1 = (2 * i + 1 < block_len) ? x[idx1] : 0.0f;

            int q0 = std::clamp(static_cast<int>(std::round(v0 * id)), -8, 7) + 8;
            int q1 = std::clamp(static_cast<int>(std::round(v1 * id)), -8, 7) + 8;

            y[b].qs[i] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        }
    }
}

void dequantize_row_q4_0(const BlockQ4_0* x, float* y, size_t k) {
    size_t nb = (k + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;

    for (size_t b = 0; b < nb; ++b) {
        float d = float16_to_float32(x[b].d);
        size_t offset = b * Q4_BLOCK_SIZE;
        size_t block_len = std::min(Q4_BLOCK_SIZE, k - offset);

        for (size_t i = 0; i < Q4_BLOCK_SIZE / 2; ++i) {
            uint8_t byte_val = x[b].qs[i];
            int q0 = (byte_val & 0x0F) - 8;
            int q1 = ((byte_val >> 4) & 0x0F) - 8;

            if (2 * i < block_len) {
                y[offset + 2 * i] = static_cast<float>(q0) * d;
            }
            if (2 * i + 1 < block_len) {
                y[offset + 2 * i + 1] = static_cast<float>(q1) * d;
            }
        }
    }
}

float vec_dot_q4_0_f32_scalar(const BlockQ4_0* a, const float* b, size_t k) {
    size_t nb = (k + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    float total_sum = 0.0f;

    for (size_t ib = 0; ib < nb; ++ib) {
        float d = float16_to_float32(a[ib].d);
        size_t offset = ib * Q4_BLOCK_SIZE;
        size_t block_len = std::min(Q4_BLOCK_SIZE, k - offset);

        float block_dot = 0.0f;
        for (size_t i = 0; i < Q4_BLOCK_SIZE / 2; ++i) {
            uint8_t byte_val = a[ib].qs[i];
            int q0 = (byte_val & 0x0F) - 8;
            int q1 = ((byte_val >> 4) & 0x0F) - 8;

            if (2 * i < block_len) {
                block_dot += static_cast<float>(q0) * b[offset + 2 * i];
            }
            if (2 * i + 1 < block_len) {
                block_dot += static_cast<float>(q1) * b[offset + 2 * i + 1];
            }
        }
        total_sum += block_dot * d;
    }
    return total_sum;
}

#if defined(VECTRA_USE_AVX2)

static inline float hsum256(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v128 = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(v128);
    __m128 sums = _mm_add_ps(v128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

float vec_dot_q4_0_f32_avx2(const BlockQ4_0* a, const float* b, size_t k) {
    size_t nb = k / Q4_BLOCK_SIZE;
    float total_sum = 0.0f;

    alignas(32) float block_floats[32];

    for (size_t ib = 0; ib < nb; ++ib) {
        float d = float16_to_float32(a[ib].d);
        __m256 vd = _mm256_set1_ps(d);

        // Unpack 16 bytes into 32 float values
        for (size_t i = 0; i < 16; ++i) {
            uint8_t byte_val = a[ib].qs[i];
            int q0 = (byte_val & 0x0F) - 8;
            int q1 = ((byte_val >> 4) & 0x0F) - 8;
            block_floats[2 * i] = static_cast<float>(q0);
            block_floats[2 * i + 1] = static_cast<float>(q1);
        }

        const float* bp = b + ib * Q4_BLOCK_SIZE;

        __m256 f0 = _mm256_load_ps(block_floats + 0);
        __m256 b0 = _mm256_loadu_ps(bp + 0);
        __m256 acc0 = _mm256_mul_ps(f0, b0);

        __m256 f1 = _mm256_load_ps(block_floats + 8);
        __m256 b1 = _mm256_loadu_ps(bp + 8);
        acc0 = _mm256_fmadd_ps(f1, b1, acc0);

        __m256 f2 = _mm256_load_ps(block_floats + 16);
        __m256 b2 = _mm256_loadu_ps(bp + 16);
        acc0 = _mm256_fmadd_ps(f2, b2, acc0);

        __m256 f3 = _mm256_load_ps(block_floats + 24);
        __m256 b3 = _mm256_loadu_ps(bp + 24);
        acc0 = _mm256_fmadd_ps(f3, b3, acc0);

        acc0 = _mm256_mul_ps(acc0, vd);
        total_sum += hsum256(acc0);
    }

    // Handle tail if k is not a multiple of 32
    if (k > nb * Q4_BLOCK_SIZE) {
        size_t tail_k = k - nb * Q4_BLOCK_SIZE;
        total_sum += vec_dot_q4_0_f32_scalar(a + nb, b + nb * Q4_BLOCK_SIZE, tail_k);
    }

    return total_sum;
}
#else
float vec_dot_q4_0_f32_avx2(const BlockQ4_0* a, const float* b, size_t k) {
    return vec_dot_q4_0_f32_scalar(a, b, k);
}
#endif

float vec_dot_q4_0_f32(const BlockQ4_0* a, const float* b, size_t k) {
    if (math::avx2::is_supported()) {
        return vec_dot_q4_0_f32_avx2(a, b, k);
    }
    return vec_dot_q4_0_f32_scalar(a, b, k);
}

void matvec_q4_0_f32(const BlockQ4_0* A, const float* x, const float* bias, float* y, size_t rows, size_t cols) {
    size_t blocks_per_row = (cols + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    auto& pool = runtime::ThreadPool::get_default_pool();
    pool.parallel_for(0, rows, [=](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            const BlockQ4_0* row_blocks = A + r * blocks_per_row;
            float d = vec_dot_q4_0_f32(row_blocks, x, cols);
            if (bias) d += bias[r];
            y[r] = d;
        }
    });
}

QuantErrorMetrics evaluate_quantization_error(const float* original, const float* reconstructed, size_t n) {
    QuantErrorMetrics metrics{};
    if (n == 0) return metrics;

    double sum_sq_err = 0.0;
    double sum_sq_orig = 0.0;
    double max_err = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double err = std::fabs(static_cast<double>(original[i]) - static_cast<double>(reconstructed[i]));
        if (err > max_err) max_err = err;
        sum_sq_err += err * err;
        sum_sq_orig += static_cast<double>(original[i]) * static_cast<double>(original[i]);
    }

    metrics.mse = sum_sq_err / static_cast<double>(n);
    metrics.rmse = std::sqrt(metrics.mse);
    metrics.max_abs_error = max_err;

    if (sum_sq_err > 0.0) {
        metrics.snr_db = 10.0 * std::log10(sum_sq_orig / sum_sq_err);
    } else {
        metrics.snr_db = 999.0; // Infinite SNR
    }

    return metrics;
}

} // namespace vectra::quant
