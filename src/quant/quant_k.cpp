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

        const uint8_t* sc_raw = a[i].scales;
        uint8_t sc[8], m[8];
        sc[0] = sc_raw[0] & 63; sc[1] = sc_raw[1] & 63; sc[2] = sc_raw[2] & 63; sc[3] = sc_raw[3] & 63;
        m[0]  = sc_raw[4] & 63; m[1]  = sc_raw[5] & 63; m[2]  = sc_raw[6] & 63; m[3]  = sc_raw[7] & 63;
        sc[4] = (sc_raw[8]  & 0x0F) | ((sc_raw[0] >> 6) << 4);
        sc[5] = (sc_raw[9]  & 0x0F) | ((sc_raw[1] >> 6) << 4);
        sc[6] = (sc_raw[10] & 0x0F) | ((sc_raw[2] >> 6) << 4);
        sc[7] = (sc_raw[11] & 0x0F) | ((sc_raw[3] >> 6) << 4);
        m[4]  = (sc_raw[8]  >> 4)   | ((sc_raw[4] >> 6) << 4);
        m[5]  = (sc_raw[9]  >> 4)   | ((sc_raw[5] >> 6) << 4);
        m[6]  = (sc_raw[10] >> 4)   | ((sc_raw[6] >> 6) << 4);
        m[7]  = (sc_raw[11] >> 4)   | ((sc_raw[7] >> 6) << 4);

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
    if (!src || !dst || k == 0) return;
    size_t nb = k / QK_K;
    for (size_t b = 0; b < nb; ++b) {
        const float* bp = src + b * 256;

        // 1. 32-element sub-blocks (for Q4_K)
        for (int sb = 0; sb < 8; ++sb) {
            const float* p = bp + sb * 32;
            float max_val = 0.0f;
            for (int i = 0; i < 32; ++i) {
                float val = p[i];
                if (!std::isfinite(val)) val = 0.0f;
                float a = std::fabs(val);
                if (a < 1e-30f) a = 0.0f;
                if (a > max_val) max_val = a;
            }
            float d = max_val / 127.0f;
            float id = (d > 1e-30f) ? (1.0f / d) : 0.0f;
            dst[b].scales32[sb] = d;

            int32_t sum = 0;
            for (int i = 0; i < 32; ++i) {
                float val = p[i];
                if (!std::isfinite(val)) val = 0.0f;
                int q = (id > 0.0f) ? static_cast<int>(std::round(val * id)) : 0;
                q = std::max(-127, std::min(127, q));
                dst[b].qs[sb * 32 + i] = static_cast<int8_t>(q);
                sum += q;
            }
            dst[b].sums32[sb] = sum;
        }

        // 2. 16-element sub-blocks (for Q6_K)
        for (int sb = 0; sb < 16; ++sb) {
            const float* p = bp + sb * 16;
            float max_val = 0.0f;
            for (int i = 0; i < 16; ++i) {
                float val = p[i];
                if (!std::isfinite(val)) val = 0.0f;
                float a = std::fabs(val);
                if (a < 1e-30f) a = 0.0f;
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

        // 3. Populate interleaved sums for vectorized Q6_K kernel:
        // Lower lane has even offsets (0, 2, 4, 6), upper lane has odd offsets (1, 3, 5, 7)
        dst[b].sums_interleaved[0] = dst[b].sums[0];
        dst[b].sums_interleaved[1] = dst[b].sums[2];
        dst[b].sums_interleaved[2] = dst[b].sums[4];
        dst[b].sums_interleaved[3] = dst[b].sums[6];
        dst[b].sums_interleaved[4] = dst[b].sums[1];
        dst[b].sums_interleaved[5] = dst[b].sums[3];
        dst[b].sums_interleaved[6] = dst[b].sums[5];
        dst[b].sums_interleaved[7] = dst[b].sums[7];

        dst[b].sums_interleaved[8]  = dst[b].sums[8];
        dst[b].sums_interleaved[9]  = dst[b].sums[10];
        dst[b].sums_interleaved[10] = dst[b].sums[12];
        dst[b].sums_interleaved[11] = dst[b].sums[14];
        dst[b].sums_interleaved[12] = dst[b].sums[9];
        dst[b].sums_interleaved[13] = dst[b].sums[11];
        dst[b].sums_interleaved[14] = dst[b].sums[13];
        dst[b].sums_interleaved[15] = dst[b].sums[15];
    }
}

float vec_dot_q4_k_q8_avx2(const BlockQ4_K* a, const BlockQ8_K_256* b, size_t k) {
    if (!a || !b || k == 0) return 0.0f;
    const size_t nb = k / QK_K;

    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i ones16   = _mm256_set1_epi16(1);

    __m256 acc_sum = _mm256_setzero_ps();

    for (size_t i = 0; i < nb; ++i) {
        const float d = float16_to_float32(a[i].d);
        const float min_val = float16_to_float32(a[i].dmin);
        const uint8_t* q = a[i].qs;
        const int8_t* act_qs = b[i].qs;

        const uint8_t* sc_raw = a[i].scales;
        uint8_t sc[8], m[8];
        sc[0] = sc_raw[0] & 63; sc[1] = sc_raw[1] & 63; sc[2] = sc_raw[2] & 63; sc[3] = sc_raw[3] & 63;
        m[0]  = sc_raw[4] & 63; m[1]  = sc_raw[5] & 63; m[2]  = sc_raw[6] & 63; m[3]  = sc_raw[7] & 63;
        sc[4] = (sc_raw[8]  & 0x0F) | ((sc_raw[0] >> 6) << 4);
        sc[5] = (sc_raw[9]  & 0x0F) | ((sc_raw[1] >> 6) << 4);
        sc[6] = (sc_raw[10] & 0x0F) | ((sc_raw[2] >> 6) << 4);
        sc[7] = (sc_raw[11] & 0x0F) | ((sc_raw[3] >> 6) << 4);
        m[4]  = (sc_raw[8]  >> 4)   | ((sc_raw[4] >> 6) << 4);
        m[5]  = (sc_raw[9]  >> 4)   | ((sc_raw[5] >> 6) << 4);
        m[6]  = (sc_raw[10] >> 4)   | ((sc_raw[6] >> 6) << 4);
        m[7]  = (sc_raw[11] >> 4)   | ((sc_raw[7] >> 6) << 4);

        __m128i sum_pair[4];

        for (int p = 0; p < 4; ++p) {
            int j = p * 64;
            __m256i raw32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + p * 32));

            __m256i q0 = _mm256_and_si256(raw32, low_mask);
            __m256i q1 = _mm256_and_si256(_mm256_srli_epi16(raw32, 4), low_mask);

            __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_qs + j));
            __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act_qs + j + 32));

            __m256i prod0_16 = _mm256_maddubs_epi16(q0, a0);
            __m256i prod1_16 = _mm256_maddubs_epi16(q1, a1);

            __m256i prod0_32 = _mm256_madd_epi16(prod0_16, ones16);
            __m256i prod1_32 = _mm256_madd_epi16(prod1_16, ones16);

            __m256i h01 = _mm256_hadd_epi32(prod0_32, prod1_32);
            __m128i lo = _mm256_castsi256_si128(h01);
            __m128i hi = _mm256_extracti128_si256(h01, 1);
            sum_pair[p] = _mm_add_epi32(lo, hi);
        }

        __m128i d01 = _mm_hadd_epi32(sum_pair[0], sum_pair[1]);
        __m128i d23 = _mm_hadd_epi32(sum_pair[2], sum_pair[3]);
        __m256i all_dots_i = _mm256_set_m128i(d23, d01);
        __m256  all_dots_f = _mm256_cvtepi32_ps(all_dots_i);

        __m256 v_sums32_f = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].sums32)));
        __m256 v_s_act    = _mm256_loadu_ps(b[i].scales32);

        uint64_t sc_u64;
        std::memcpy(&sc_u64, sc, sizeof(sc_u64));
        __m128i sc_vec = _mm_cvtsi64_si128(sc_u64);
        __m256 sc_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(sc_vec));
        __m256 v_dl = _mm256_mul_ps(_mm256_mul_ps(sc_f, _mm256_set1_ps(d)), v_s_act);

        uint64_t m_u64;
        std::memcpy(&m_u64, m, sizeof(m_u64));
        __m128i m_vec = _mm_cvtsi64_si128(m_u64);
        __m256 m_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(m_vec));
        __m256 v_ml = _mm256_mul_ps(_mm256_mul_ps(m_f, _mm256_set1_ps(min_val)), v_s_act);

        acc_sum = _mm256_fmadd_ps(all_dots_f, v_dl, acc_sum);
        acc_sum = _mm256_fnmadd_ps(v_sums32_f, v_ml, acc_sum);
    }
    return hsum256_ps(acc_sum);
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
    if (!a || !b || k == 0) return 0.0f;
    const size_t nb = k / QK_K;

    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i ones16   = _mm256_set1_epi16(1);

    // Shuffle mask to permute bytes [0, 1, 2, 3, 4, 5, 6, 7] into [0, 2, 4, 6, 1, 3, 5, 7]
    const __m128i sc_shuf_mask = _mm_setr_epi8(
        0, 2, 4, 6, 1, 3, 5, 7,
        -1, -1, -1, -1, -1, -1, -1, -1
    );

    __m256 acc_sum = _mm256_setzero_ps();

    for (size_t i = 0; i < nb; ++i) {
        const float d = float16_to_float32(a[i].d);
        const __m256 v_d = _mm256_set1_ps(d);
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

            // Three 256-bit hadd instructions reduce all 8 dot products into a single YMM register
            __m256i h12 = _mm256_hadd_epi32(p1_32, p2_32);
            __m256i h34 = _mm256_hadd_epi32(p3_32, p4_32);
            __m256i h_all = _mm256_hadd_epi32(h12, h34);

            // Load interleaved sums: [0, 2, 4, 6, 1, 3, 5, 7]
            __m256i sums_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].sums_interleaved + (n == 0 ? 0 : 8)));
            __m256i sums_x32 = _mm256_slli_epi32(sums_vec, 5);
            __m256i diff_i   = _mm256_sub_epi32(h_all, sums_x32);
            __m256  diff_f   = _mm256_cvtepi32_ps(diff_i);

            // Load and shuffle 8 8-bit scales into [0, 2, 4, 6, 1, 3, 5, 7]
            int64_t sc_raw;
            std::memcpy(&sc_raw, sc, sizeof(sc_raw));
            __m128i sc_64 = _mm_cvtsi64_si128(sc_raw);
            __m128i sc_shuf = _mm_shuffle_epi8(sc_64, sc_shuf_mask);
            __m256i sc_32 = _mm256_cvtepi8_epi32(sc_shuf);
            __m256  sc_f  = _mm256_cvtepi32_ps(sc_32);

            __m256 d_sc = _mm256_mul_ps(v_d, sc_f);

            // Load activation scales: 4 floats duplicated across lower and upper 128-bit lanes
            __m128 act_s4 = _mm_loadu_ps(b[i].scales32 + sb_base / 2);
            __m256 act_scales = _mm256_insertf128_ps(_mm256_castps128_ps256(act_s4), act_s4, 1);

            __m256 scales_all = _mm256_mul_ps(d_sc, act_scales);

            acc_sum = _mm256_fmadd_ps(diff_f, scales_all, acc_sum);

            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return hsum256_ps(acc_sum);
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

void matvec_q4_k_q8_2x(
    const BlockQ4_K* A1, float* y1,
    const BlockQ4_K* A2, float* y2,
    const BlockQ8_K_256* x,
    size_t rows,
    size_t cols
) {
    size_t blocks_per_row = cols / QK_K;
    auto& pool = runtime::ThreadPool::get_default_pool();
    pool.parallel_for(0, rows, [=](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            y1[r] = vec_dot_q4_k_q8_avx2(A1 + r * blocks_per_row, x, cols);
            y2[r] = vec_dot_q4_k_q8_avx2(A2 + r * blocks_per_row, x, cols);
        }
    });
}

void matvec_q4_k_q8_qkv(
    const BlockQ4_K* Q_W, const float* q_bias, float* q_out, size_t q_rows,
    const BlockQ4_K* K_W, const float* k_bias, float* k_out, size_t k_rows,
    const BlockQ4_K* V_W, const float* v_bias, float* v_out, size_t v_rows,
    const BlockQ8_K_256* x,
    size_t cols
) {
    size_t blocks_per_row = cols / QK_K;
    size_t total_rows = q_rows + k_rows + v_rows;
    auto& pool = runtime::ThreadPool::get_default_pool();
    pool.parallel_for(0, total_rows, [=](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            if (r < q_rows) {
                float d = vec_dot_q4_k_q8_avx2(Q_W + r * blocks_per_row, x, cols);
                if (q_bias) d += q_bias[r];
                q_out[r] = d;
            } else if (r < q_rows + k_rows) {
                size_t kr = r - q_rows;
                float d = vec_dot_q4_k_q8_avx2(K_W + kr * blocks_per_row, x, cols);
                if (k_bias) d += k_bias[kr];
                k_out[kr] = d;
            } else {
                size_t vr = r - q_rows - k_rows;
                float d = vec_dot_q4_k_q8_avx2(V_W + vr * blocks_per_row, x, cols);
                if (v_bias) d += v_bias[vr];
                v_out[vr] = d;
            }
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

void matvec_q4_k_q8_batch(
    const BlockQ4_K* A,
    const BlockQ8_K_256* const* X,
    const float* bias,
    float* const* Y,
    size_t batch_size,
    size_t rows,
    size_t cols
) {
    if (batch_size == 0) return;
    if (batch_size == 1) {
        matvec_q4_k_q8(A, X[0], bias, Y[0], rows, cols);
        return;
    }

    size_t blocks_per_row = cols / QK_K;
    auto& pool = runtime::ThreadPool::get_default_pool();

    pool.parallel_for(0, rows, [=](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            const BlockQ4_K* row_blocks = A + r * blocks_per_row;
            for (size_t b = 0; b < batch_size; ++b) {
                float d = vec_dot_q4_k_q8_avx2(row_blocks, X[b], cols);
                if (bias) d += bias[r];
                Y[b][r] = d;
            }
        }
    });
}

void matvec_q6_k_q8_batch(
    const BlockQ6_K* A,
    const BlockQ8_K_256* const* X,
    const float* bias,
    float* const* Y,
    size_t batch_size,
    size_t rows,
    size_t cols
) {
    if (batch_size == 0) return;
    if (batch_size == 1) {
        matvec_q6_k_q8(A, X[0], bias, Y[0], rows, cols);
        return;
    }

    size_t blocks_per_row = cols / QK_K;
    auto& pool = runtime::ThreadPool::get_default_pool();

    pool.parallel_for(0, rows, [=](size_t r_start, size_t r_end) {
        for (size_t r = r_start; r < r_end; ++r) {
            const BlockQ6_K* row_blocks = A + r * blocks_per_row;
            for (size_t b = 0; b < batch_size; ++b) {
                float d = vec_dot_q6_k_q8_avx2(row_blocks, X[b], cols);
                if (bias) d += bias[r];
                Y[b][r] = d;
            }
        }
    });
}

} // namespace vectra::quant
