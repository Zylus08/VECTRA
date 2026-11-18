#include "vectra/quant/quant_k.hpp"
#include "vectra/math/avx2_math.hpp"
#include "vectra/runtime/thread_pool.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

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

void quantize_activation_q8_k_256(const float* src, BlockQ8_K_256* dst, size_t k) {
    size_t nb = k / QK_K;
    for (size_t b = 0; b < nb; ++b) {
        const float* bp = src + b * 256;

        // 1. 32-element sub-blocks (for Q4_K)
        for (int sb = 0; sb < 8; ++sb) {
            const float* p = bp + sb * 32;
            float max_val = 0.0f;
            for (int i = 0; i < 32; ++i) {
                float a = std::fabs(p[i]);
                if (a > max_val) max_val = a;
            }
            float d = max_val / 127.0f;
            float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
            dst[b].scales32[sb] = d;

            int32_t sum = 0;
            for (int i = 0; i < 32; ++i) {
                int val = static_cast<int>(std::round(p[i] * id));
                val = std::max(-127, std::min(127, val));
                dst[b].qs[sb * 32 + i] = static_cast<int8_t>(val);
                sum += val;
            }
            dst[b].sums32[sb] = sum;
        }

        // 2. 16-element sub-blocks (for Q6_K)
        for (int sb = 0; sb < 16; ++sb) {
            const float* p = bp + sb * 16;
            float max_val = 0.0f;
            for (int i = 0; i < 16; ++i) {
                float a = std::fabs(p[i]);
                if (a > max_val) max_val = a;
            }
            float d = max_val / 127.0f;
            dst[b].scales[sb] = d;

            int32_t sum = 0;
            for (int i = 0; i < 16; ++i) {
                sum += dst[b].qs[sb * 16 + i];
            }
            dst[b].sums[sb] = sum;
        }
    }
}

float vec_dot_q4_k_q8_avx2(const BlockQ4_K* a, const BlockQ8_K_256* b, size_t k) {
    const size_t nb = k / QK_K;
    float total_sum = 0.0f;

    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i ones16   = _mm256_set1_epi16(1);

    for (size_t i = 0; i < nb; ++i) {
        const float d = float16_to_float32(a[i].d);
        const float min_val = float16_to_float32(a[i].dmin);
        const uint8_t* q = a[i].qs;
        const int8_t* act_qs = b[i].qs;

        uint8_t sc[8], m[8];
        for (int j = 0; j < 8; ++j) {
            get_scale_min_k4(j, a[i].scales, &sc[j], &m[j]);
        }

        for (int j = 0; j < 256; j += 64) {
            int sb0 = j / 32;
            int sb1 = sb0 + 1;

            float dl0 = d * sc[sb0];
            float ml0 = min_val * m[sb0];
            float dl1 = d * sc[sb1];
            float ml1 = min_val * m[sb1];

            float s_act0 = b[i].scales32[sb0];
            float s_act1 = b[i].scales32[sb1];

            __m256i raw32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));

            __m256i q0 = _mm256_and_si256(raw32, low_mask);
            __m256i q1 = _mm256_and_si256(_mm256_srli_epi16(raw32, 4), low_mask);

            __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_qs + j));
            __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_qs + j + 32));

            __m256i prod0_16 = _mm256_maddubs_epi16(q0, a0);
            __m256i prod1_16 = _mm256_maddubs_epi16(q1, a1);

            __m256i prod0_32 = _mm256_madd_epi16(prod0_16, ones16);
            __m256i prod1_32 = _mm256_madd_epi16(prod1_16, ones16);

            __m128i lo0 = _mm256_castsi256_si128(prod0_32);
            __m128i hi0 = _mm256_extracti128_si256(prod0_32, 1);
            __m128i sum0_128 = _mm_add_epi32(lo0, hi0);
            sum0_128 = _mm_hadd_epi32(sum0_128, sum0_128);
            sum0_128 = _mm_hadd_epi32(sum0_128, sum0_128);
            int32_t dot_int0 = _mm_cvtsi128_si32(sum0_128);

            __m128i lo1 = _mm256_castsi256_si128(prod1_32);
            __m128i hi1 = _mm256_extracti128_si256(prod1_32, 1);
            __m128i sum1_128 = _mm_add_epi32(lo1, hi1);
            sum1_128 = _mm_hadd_epi32(sum1_128, sum1_128);
            sum1_128 = _mm_hadd_epi32(sum1_128, sum1_128);
            int32_t dot_int1 = _mm_cvtsi128_si32(sum1_128);

            float sub_res0 = s_act0 * (dl0 * dot_int0 - ml0 * b[i].sums32[sb0]);
            float sub_res1 = s_act1 * (dl1 * dot_int1 - ml1 * b[i].sums32[sb1]);

            total_sum += (sub_res0 + sub_res1);
            q += 32;
        }
    }
    return total_sum;
}

float vec_dot_q6_k_f32_fused_avx2(const BlockQ6_K* a, const float* b, size_t k) {
    const size_t nb = k / QK_K;
    float sum = 0.0f;

    for (size_t i = 0; i < nb; ++i) {
        const float d = float16_to_float32(a[i].d);
        const uint8_t* ql = a[i].ql;
        const uint8_t* qh = a[i].qh;
        const int8_t* sc = a[i].scales;
        const float* bp = b + i * QK_K;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l += 8) {
                int is = l / 16;
                const float s0 = d * sc[is + 0];
                const float s1 = d * sc[is + 2];
                const float s2 = d * sc[is + 4];
                const float s3 = d * sc[is + 6];

                const __m256 vs0 = _mm256_set1_ps(s0);
                const __m256 vs1 = _mm256_set1_ps(s1);
                const __m256 vs2 = _mm256_set1_ps(s2);
                const __m256 vs3 = _mm256_set1_ps(s3);

                alignas(32) float w0[8], w1[8], w2[8], w3[8];
                for (int m = 0; m < 8; ++m) {
                    int idx = l + m;
                    w0[m] = static_cast<float>(static_cast<int8_t>((ql[idx + 0]  & 0x0F) | (((qh[idx] >> 0) & 3) << 4)) - 32);
                    w1[m] = static_cast<float>(static_cast<int8_t>((ql[idx + 32] & 0x0F) | (((qh[idx] >> 2) & 3) << 4)) - 32);
                    w2[m] = static_cast<float>(static_cast<int8_t>((ql[idx + 0]  >> 4)   | (((qh[idx] >> 4) & 3) << 4)) - 32);
                    w3[m] = static_cast<float>(static_cast<int8_t>((ql[idx + 32] >> 4)   | (((qh[idx] >> 6) & 3) << 4)) - 32);
                }

                __m256 vw0 = _mm256_mul_ps(_mm256_load_ps(w0), vs0);
                __m256 vw1 = _mm256_mul_ps(_mm256_load_ps(w1), vs1);
                __m256 vw2 = _mm256_mul_ps(_mm256_load_ps(w2), vs2);
                __m256 vw3 = _mm256_mul_ps(_mm256_load_ps(w3), vs3);

                acc0 = _mm256_fmadd_ps(vw0, _mm256_loadu_ps(bp + n + l + 0), acc0);
                acc1 = _mm256_fmadd_ps(vw1, _mm256_loadu_ps(bp + n + l + 32), acc1);
                acc0 = _mm256_fmadd_ps(vw2, _mm256_loadu_ps(bp + n + l + 64), acc0);
                acc1 = _mm256_fmadd_ps(vw3, _mm256_loadu_ps(bp + n + l + 96), acc1);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 v128 = _mm_add_ps(lo, hi);
        v128 = _mm_hadd_ps(v128, v128);
        v128 = _mm_hadd_ps(v128, v128);
        sum += _mm_cvtss_f32(v128);
    }
    return sum;
}

float vec_dot_q6_k_q8_avx2(const BlockQ6_K* a, const BlockQ8_K_256* b, size_t k) {
    const size_t nb = k / QK_K;
    float total_sum = 0.0f;

    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i ones16   = _mm256_set1_epi16(1);

    for (size_t i = 0; i < nb; ++i) {
        const float d = float16_to_float32(a[i].d);
        const uint8_t* ql = a[i].ql;
        const uint8_t* qh = a[i].qh;
        const int8_t* sc = a[i].scales;
        const int8_t* act_qs = b[i].qs;

        for (int n = 0; n < 256; n += 128) {
            int sb_base = (n / 16);

            __m256i ql_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 0));
            __m256i ql_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
            __m256i qh_32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));

            __m256i q1 = _mm256_or_si256(
                _mm256_and_si256(ql_lo, low_mask),
                _mm256_slli_epi16(_mm256_and_si256(qh_32, _mm256_set1_epi8(0x03)), 4)
            );
            __m256i q2 = _mm256_or_si256(
                _mm256_and_si256(ql_hi, low_mask),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_32, 2), _mm256_set1_epi8(0x03)), 4)
            );
            __m256i q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), low_mask),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_32, 4), _mm256_set1_epi8(0x03)), 4)
            );
            __m256i q4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), low_mask),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_32, 6), _mm256_set1_epi8(0x03)), 4)
            );

            __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_qs + n + 0));
            __m256i a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_qs + n + 32));
            __m256i a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_qs + n + 64));
            __m256i a4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_qs + n + 96));

            __m256i p1_16 = _mm256_maddubs_epi16(q1, a1);
            __m256i p2_16 = _mm256_maddubs_epi16(q2, a2);
            __m256i p3_16 = _mm256_maddubs_epi16(q3, a3);
            __m256i p4_16 = _mm256_maddubs_epi16(q4, a4);

            __m256i p1_32 = _mm256_madd_epi16(p1_16, ones16);
            __m256i p2_32 = _mm256_madd_epi16(p2_16, ones16);
            __m256i p3_32 = _mm256_madd_epi16(p3_16, ones16);
            __m256i p4_32 = _mm256_madd_epi16(p4_16, ones16);

            auto sum_lane = [](const __m256i& reg, int lane) -> int32_t {
                __m128i v = (lane == 0) ? _mm256_castsi256_si128(reg) : _mm256_extracti128_si256(reg, 1);
                v = _mm_hadd_epi32(v, v);
                v = _mm_hadd_epi32(v, v);
                return _mm_cvtsi128_si32(v);
            };

            for (int lane = 0; lane < 2; ++lane) {
                int is = lane;
                int sb1 = sb_base + is;
                int32_t dot1 = sum_lane(p1_32, lane);
                float scale1 = d * sc[is + 0] * b[i].scales32[sb1 / 2];
                total_sum += scale1 * (dot1 - 32 * b[i].sums[sb1]);

                int sb2 = sb_base + 2 + is;
                int32_t dot2 = sum_lane(p2_32, lane);
                float scale2 = d * sc[is + 2] * b[i].scales32[sb2 / 2];
                total_sum += scale2 * (dot2 - 32 * b[i].sums[sb2]);

                int sb3 = sb_base + 4 + is;
                int32_t dot3 = sum_lane(p3_32, lane);
                float scale3 = d * sc[is + 4] * b[i].scales32[sb3 / 2];
                total_sum += scale3 * (dot3 - 32 * b[i].sums[sb3]);

                int sb4 = sb_base + 6 + is;
                int32_t dot4 = sum_lane(p4_32, lane);
                float scale4 = d * sc[is + 6] * b[i].scales32[sb4 / 2];
                total_sum += scale4 * (dot4 - 32 * b[i].sums[sb4]);
            }

            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return total_sum;
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
float vec_dot_q6_k_f32_fused_avx2(const BlockQ6_K* a, const float* b, size_t k) {
    return vec_dot_q6_k_f32_scalar(a, b, k);
}
float vec_dot_q4_k_q8_avx2(const BlockQ4_K* a, const BlockQ8_K_256* b, size_t k) {
    return 0.0f;
}
float vec_dot_q6_k_q8_avx2(const BlockQ6_K* a, const BlockQ8_K_256* b, size_t k) {
    return 0.0f;
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

void matvec_q4_k_q8(const BlockQ4_K* A, const BlockQ8_K_256* x, const float* bias, float* y, size_t rows, size_t cols) {
    size_t blocks_per_row = cols / QK_K;
    auto& pool = runtime::ThreadPool::get_default_pool();
    pool.parallel_for(0, rows, [=](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            const BlockQ4_K* row_blocks = A + r * blocks_per_row;
            float d = vec_dot_q4_k_q8_avx2(row_blocks, x, cols);
            if (bias) d += bias[r];
            y[r] = d;
        }
    });
}

void matvec_q6_k_q8(const BlockQ6_K* A, const BlockQ8_K_256* x, const float* bias, float* y, size_t rows, size_t cols) {
    size_t blocks_per_row = cols / QK_K;
    auto& pool = runtime::ThreadPool::get_default_pool();
    pool.parallel_for(0, rows, [=](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            const BlockQ6_K* row_blocks = A + r * blocks_per_row;
            float d = vec_dot_q6_k_q8_avx2(row_blocks, x, cols);
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
