#include "vectra/quant/quant_k.hpp"
#include "vectra/math/avx2_math.hpp"
#include "vectra/runtime/thread_pool.hpp"
#include <cmath>
#include <algorithm>

#if defined(VECTRA_USE_AVX2)
#include <immintrin.h>
#endif

namespace vectra::quant {

void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}

void dequantize_row_q4_k(const BlockQ4_K* x, float* y, size_t k) {
    const size_t nb = k / QK_K;

    for (size_t i = 0; i < nb; ++i) {
        const float d = float16_to_float32(x[i].d);
        const float min_val = float16_to_float32(x[i].dmin);
        const uint8_t* q = x[i].qs;

        uint8_t sc[8], m[8];
        for (int j = 0; j < 8; ++j) {
            get_scale_min_k4(j, x[i].scales, &sc[j], &m[j]);
        }

        for (int j = 0; j < 256; j += 64) {
            const float dl1 = d * sc[j / 32 + 0];
            const float ml1 = min_val * m[j / 32 + 0];
            const float dl2 = d * sc[j / 32 + 1];
            const float ml2 = min_val * m[j / 32 + 1];

            for (int l = 0; l < 32; ++l) {
                y[j + l + 0]  = dl1 * static_cast<float>(q[l] & 0x0F) - ml1;
                y[j + l + 32] = dl2 * static_cast<float>(q[l] >> 4)   - ml2;
            }
            q += 32;
        }
        y += QK_K;
    }
}

void dequantize_row_q6_k(const BlockQ6_K* x, float* y, size_t k) {
    const size_t nb = k / QK_K;

    for (size_t i = 0; i < nb; ++i) {
        const float d = float16_to_float32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = static_cast<int8_t>((ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = static_cast<int8_t>((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l + 0]  = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

float vec_dot_q4_k_f32_scalar(const BlockQ4_K* a, const float* b, size_t k) {
    const size_t nb = k / QK_K;
    float sum = 0.0f;

    alignas(32) float block_vals[256];

    for (size_t i = 0; i < nb; ++i) {
        dequantize_row_q4_k(a + i, block_vals, QK_K);
        const float* bp = b + i * QK_K;
        float bsum = 0.0f;
        for (size_t j = 0; j < QK_K; ++j) {
            bsum += block_vals[j] * bp[j];
        }
        sum += bsum;
    }
    return sum;
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

float vec_dot_q4_k_f32_avx2(const BlockQ4_K* a, const float* b, size_t k) {
    const size_t nb = k / QK_K;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (size_t i = 0; i < nb; ++i) {
        const float d = float16_to_float32(a[i].d);
        const float min_val = float16_to_float32(a[i].dmin);
        const uint8_t* q = a[i].qs;
        const float* bp = b + i * QK_K;

        uint8_t sc[8], m[8];
        for (int j = 0; j < 8; ++j) {
            get_scale_min_k4(j, a[i].scales, &sc[j], &m[j]);
        }

        for (int j = 0; j < 256; j += 64) {
            const float dl1 = d * sc[j / 32 + 0];
            const float ml1 = min_val * m[j / 32 + 0];
            const float dl2 = d * sc[j / 32 + 1];
            const float ml2 = min_val * m[j / 32 + 1];

            const __m256 v_dl1 = _mm256_set1_ps(dl1);
            const __m256 v_ml1 = _mm256_set1_ps(ml1);
            const __m256 v_dl2 = _mm256_set1_ps(dl2);
            const __m256 v_ml2 = _mm256_set1_ps(ml2);

            for (int l = 0; l < 32; l += 8) {
                __m128i raw8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
                __m128i raw16 = _mm_cvtepu8_epi16(raw8);
                __m128i low16 = _mm_and_si128(raw16, _mm_set1_epi16(0x0F));
                __m128i high16 = _mm_srli_epi16(raw16, 4);

                __m256 low_f = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(low16));
                __m256 high_f = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(high16));

                __m256 w1 = _mm256_fmsub_ps(low_f, v_dl1, v_ml1);
                __m256 w2 = _mm256_fmsub_ps(high_f, v_dl2, v_ml2);

                __m256 b1 = _mm256_loadu_ps(bp + j + l);
                __m256 b2 = _mm256_loadu_ps(bp + j + 32 + l);

                acc0 = _mm256_fmadd_ps(w1, b1, acc0);
                acc1 = _mm256_fmadd_ps(w2, b2, acc1);
            }
            q += 32;
        }
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    return hsum256_ps(acc0);
}

float vec_dot_q6_k_f32_avx2(const BlockQ6_K* a, const float* b, size_t k) {
    const size_t nb = k / QK_K;
    float sum = 0.0f;
    alignas(32) float block_vals[256];

    for (size_t i = 0; i < nb; ++i) {
        dequantize_row_q6_k(a + i, block_vals, QK_K);
        const float* bp = b + i * QK_K;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        for (size_t j = 0; j < 256; j += 16) {
            __m256 v0 = _mm256_load_ps(block_vals + j);
            __m256 b0 = _mm256_loadu_ps(bp + j);
            acc0 = _mm256_fmadd_ps(v0, b0, acc0);

            __m256 v1 = _mm256_load_ps(block_vals + j + 8);
            __m256 b1 = _mm256_loadu_ps(bp + j + 8);
            acc1 = _mm256_fmadd_ps(v1, b1, acc1);
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        sum += hsum256_ps(acc0);
    }
    return sum;
}
#else
float vec_dot_q4_k_f32_avx2(const BlockQ4_K* a, const float* b, size_t k) {
    return vec_dot_q4_k_f32_scalar(a, b, k);
}
float vec_dot_q6_k_f32_avx2(const BlockQ6_K* a, const float* b, size_t k) {
    return vec_dot_q6_k_f32_scalar(a, b, k);
}
#endif

float vec_dot_q4_k_f32(const BlockQ4_K* a, const float* b, size_t k) {
    if (math::avx2::is_supported()) {
        return vec_dot_q4_k_f32_avx2(a, b, k);
    }
    return vec_dot_q4_k_f32_scalar(a, b, k);
}

float vec_dot_q6_k_f32_scalar(const BlockQ6_K* a, const float* b, size_t k) {
    const size_t nb = k / QK_K;
    float sum = 0.0f;
    alignas(32) float block_vals[256];

    for (size_t i = 0; i < nb; ++i) {
        dequantize_row_q6_k(a + i, block_vals, QK_K);
        const float* bp = b + i * QK_K;
        float bsum = 0.0f;
        for (size_t j = 0; j < QK_K; ++j) {
            bsum += block_vals[j] * bp[j];
        }
        sum += bsum;
    }
    return sum;
}

float vec_dot_q6_k_f32(const BlockQ6_K* a, const float* b, size_t k) {
    if (math::avx2::is_supported()) {
        return vec_dot_q6_k_f32_avx2(a, b, k);
    }
    return vec_dot_q6_k_f32_scalar(a, b, k);
}

void matvec_q4_k_f32(const BlockQ4_K* A, const float* x, const float* bias, float* y, size_t rows, size_t cols) {
    size_t blocks_per_row = cols / QK_K;
    auto& pool = runtime::ThreadPool::get_default_pool();
    pool.parallel_for(0, rows, [=](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            const BlockQ4_K* row_blocks = A + r * blocks_per_row;
            float d = vec_dot_q4_k_f32(row_blocks, x, cols);
            if (bias) d += bias[r];
            y[r] = d;
        }
    });
}

void matvec_q6_k_f32(const BlockQ6_K* A, const float* x, const float* bias, float* y, size_t rows, size_t cols) {
    size_t blocks_per_row = cols / QK_K;
    auto& pool = runtime::ThreadPool::get_default_pool();
    pool.parallel_for(0, rows, [=](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            const BlockQ6_K* row_blocks = A + r * blocks_per_row;
            float d = vec_dot_q6_k_f32(row_blocks, x, cols);
            if (bias) d += bias[r];
            y[r] = d;
        }
    });
}

void matvec_q4_k_f32_batch(
    const BlockQ4_K* A,
    const float* const* X,
    const float* bias,
    float* const* Y,
    size_t batch_size,
    size_t rows,
    size_t cols
) {
    if (batch_size == 0) return;
    if (batch_size == 1) {
        matvec_q4_k_f32(A, X[0], bias, Y[0], rows, cols);
        return;
    }

    size_t blocks_per_row = cols / QK_K;
    auto& pool = runtime::ThreadPool::get_default_pool();

    pool.parallel_for(0, rows, [=](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            const BlockQ4_K* row_blocks = A + r * blocks_per_row;
            for (size_t b = 0; b < batch_size; ++b) {
                float d = vec_dot_q4_k_f32(row_blocks, X[b], cols);
                if (bias) d += bias[r];
                Y[b][r] = d;
            }
        }
    });
}

void matvec_q6_k_f32_batch(
    const BlockQ6_K* A,
    const float* const* X,
    const float* bias,
    float* const* Y,
    size_t batch_size,
    size_t rows,
    size_t cols
) {
    if (batch_size == 0) return;
    if (batch_size == 1) {
        matvec_q6_k_f32(A, X[0], bias, Y[0], rows, cols);
        return;
    }

    size_t blocks_per_row = cols / QK_K;
    auto& pool = runtime::ThreadPool::get_default_pool();

    pool.parallel_for(0, rows, [=](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            const BlockQ6_K* row_blocks = A + r * blocks_per_row;
            for (size_t b = 0; b < batch_size; ++b) {
                float d = vec_dot_q6_k_f32(row_blocks, X[b], cols);
                if (bias) d += bias[r];
                Y[b][r] = d;
            }
        }
    });
}

} // namespace vectra::quant
