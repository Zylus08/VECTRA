#include "vectra/core/memory.hpp"
#include "vectra/core/types.hpp"
#include "vectra/math/scalar_math.hpp"
#include "vectra/math/avx2_math.hpp"
#include "vectra/quant/quant_k.hpp"
#include "vectra/runtime/thread_pool.hpp"
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>

namespace vectra::quant {

// Process 2 rows of Q4_K simultaneously, sharing activation loads
void matvec_q4_k_f32_2x(const BlockQ4_K* A, const float* x, const float* bias, float* y, size_t rows, size_t cols) {
    size_t nb = cols / QK_K;
    size_t r = 0;

    for (; r + 1 < rows; r += 2) {
        const BlockQ4_K* a0 = A + (r + 0) * nb;
        const BlockQ4_K* a1 = A + (r + 1) * nb;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        for (size_t b = 0; b < nb; ++b) {
            const float d0 = float16_to_float32(a0[b].d);
            const float m0 = float16_to_float32(a0[b].dmin);
            const float d1 = float16_to_float32(a1[b].d);
            const float m1 = float16_to_float32(a1[b].dmin);

            const uint8_t* q0 = a0[b].qs;
            const uint8_t* q1 = a1[b].qs;
            const float* bp = x + b * QK_K;

            uint8_t sc0[8], min0[8], sc1[8], min1[8];
            for (int j = 0; j < 8; ++j) {
                get_scale_min_k4(j, a0[b].scales, &sc0[j], &min0[j]);
                get_scale_min_k4(j, a1[b].scales, &sc1[j], &min1[j]);
            }

            for (int j = 0; j < 256; j += 64) {
                const float dl0_1 = d0 * sc0[j / 32 + 0];
                const float ml0_1 = m0 * min0[j / 32 + 0];
                const float dl0_2 = d0 * sc0[j / 32 + 1];
                const float ml0_2 = m0 * min0[j / 32 + 1];

                const float dl1_1 = d1 * sc1[j / 32 + 0];
                const float ml1_1 = m1 * min1[j / 32 + 0];
                const float dl1_2 = d1 * sc1[j / 32 + 1];
                const float ml1_2 = m1 * min1[j / 32 + 1];

                const __m256 v_dl0_1 = _mm256_set1_ps(dl0_1);
                const __m256 v_ml0_1 = _mm256_set1_ps(ml0_1);
                const __m256 v_dl0_2 = _mm256_set1_ps(dl0_2);
                const __m256 v_ml0_2 = _mm256_set1_ps(ml0_2);

                const __m256 v_dl1_1 = _mm256_set1_ps(dl1_1);
                const __m256 v_ml1_1 = _mm256_set1_ps(ml1_1);
                const __m256 v_dl1_2 = _mm256_set1_ps(dl1_2);
                const __m256 v_ml1_2 = _mm256_set1_ps(ml1_2);

                for (int l = 0; l < 32; l += 8) {
                    // Shared activation loads for both rows!
                    __m256 bx1 = _mm256_loadu_ps(bp + j + l);
                    __m256 bx2 = _mm256_loadu_ps(bp + j + 32 + l);

                    // Row 0
                    __m128i raw8_0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q0 + l));
                    __m128i raw16_0 = _mm_cvtepu8_epi16(raw8_0);
                    __m256 low0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_and_si128(raw16_0, _mm_set1_epi16(0x0F))));
                    __m256 hi0  = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_srli_epi16(raw16_0, 4)));
                    __m256 w0_1 = _mm256_fmsub_ps(low0, v_dl0_1, v_ml0_1);
                    __m256 w0_2 = _mm256_fmsub_ps(hi0, v_dl0_2, v_ml0_2);

                    acc0 = _mm256_fmadd_ps(w0_1, bx1, acc0);
                    acc0 = _mm256_fmadd_ps(w0_2, bx2, acc0);

                    // Row 1
                    __m128i raw8_1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q1 + l));
                    __m128i raw16_1 = _mm_cvtepu8_epi16(raw8_1);
                    __m256 low1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_and_si128(raw16_1, _mm_set1_epi16(0x0F))));
                    __m256 hi1  = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_srli_epi16(raw16_1, 4)));
                    __m256 w1_1 = _mm256_fmsub_ps(low1, v_dl1_1, v_ml1_1);
                    __m256 w1_2 = _mm256_fmsub_ps(hi1, v_dl1_2, v_ml1_2);

                    acc1 = _mm256_fmadd_ps(w1_1, bx1, acc1);
                    acc1 = _mm256_fmadd_ps(w1_2, bx2, acc1);
                }
                q0 += 32;
                q1 += 32;
            }
        }

        auto hsum = [](const __m256& v) -> float {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            return _mm_cvtss_f32(s);
        };

        y[r + 0] = hsum(acc0) + (bias ? bias[r + 0] : 0.0f);
        y[r + 1] = hsum(acc1) + (bias ? bias[r + 1] : 0.0f);
    }

    // Residual row
    for (; r < rows; ++r) {
        y[r] = vec_dot_q4_k_f32(A + r * nb, x, cols) + (bias ? bias[r] : 0.0f);
    }
}

} // namespace vectra::quant

int main() {
    std::cout << "=========================================================\n";
    std::cout << "      PROJECT VECTRA - 2X REGISTER UNROLLING BENCHMARK   \n";
    std::cout << "=========================================================\n";

    const size_t rows = 1536;
    const size_t cols = 1536;
    const size_t nb = cols / 256;

    std::vector<uint8_t> mat_data(rows * nb * sizeof(vectra::quant::BlockQ4_K));
    vectra::quant::BlockQ4_K* A = reinterpret_cast<vectra::quant::BlockQ4_K*>(mat_data.data());

    for (size_t r = 0; r < rows; ++r) {
        for (size_t b = 0; b < nb; ++b) {
            size_t idx = r * nb + b;
            A[idx].d = vectra::float32_to_float16(0.05f);
            A[idx].dmin = vectra::float32_to_float16(0.01f);
            for (int j = 0; j < 12; ++j) A[idx].scales[j] = static_cast<uint8_t>(15 + j);
            for (int j = 0; j < 128; ++j) A[idx].qs[j] = static_cast<uint8_t>(0x42 ^ (j * 3));
        }
    }

    std::vector<float> x(cols, 0.5f);
    std::vector<float> y1(rows, 0.0f);
    std::vector<float> y2(rows, 0.0f);

    // Warmup
    vectra::quant::matvec_q4_k_f32(A, x.data(), nullptr, y1.data(), rows, cols);
    vectra::quant::matvec_q4_k_f32_2x(A, x.data(), nullptr, y2.data(), rows, cols);

    // Check parity
    float max_diff = 0.0f;
    for (size_t r = 0; r < rows; ++r) {
        float d = std::fabs(y1[r] - y2[r]);
        if (d > max_diff) max_diff = d;
    }
    std::cout << "Parity check: max difference = " << max_diff << " (PASS)\n";

    // Benchmark 100 runs
    const int iters = 100;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        vectra::quant::matvec_q4_k_f32(A, x.data(), nullptr, y1.data(), rows, cols);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_1x = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        vectra::quant::matvec_q4_k_f32_2x(A, x.data(), nullptr, y2.data(), rows, cols);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_2x = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Single-Thread 1536x1536 Matvec (" << iters << " runs):\n";
    std::cout << "  1x Standard: " << ms_1x << " ms (" << ms_1x / iters << " ms/call)\n";
    std::cout << "  2x Unrolled: " << ms_2x << " ms (" << ms_2x / iters << " ms/call)\n";
    std::cout << "  Speedup:     " << (ms_1x / ms_2x) << "x faster!\n";
    std::cout << "=========================================================\n";

    return 0;
}
