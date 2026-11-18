#pragma once

#include "vectra/core/types.hpp"
#include <cstdint>
#include <cstddef>

namespace vectra::quant {

constexpr size_t QK_K = 256;

#pragma pack(push, 1)
struct BlockQ4_K {
    uint16_t d;          // super-block scale (FP16)
    uint16_t dmin;       // super-block min (FP16)
    uint8_t scales[12];  // 6-bit scales and mins
    uint8_t qs[128];     // 4-bit quants (256 values)
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K must be 144 bytes");

#pragma pack(push, 1)
struct BlockQ6_K {
    uint8_t ql[128];     // quants, lower 4 bits
    uint8_t qh[64];      // quants, upper 2 bits
    int8_t  scales[16];  // 16 8-bit scales
    uint16_t d;          // super-block scale (FP16)
};
#pragma pack(pop)
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must be 210 bytes");

// Helper to decode 6-bit scale and min for sub-block j in [0, 7]
void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m);

// Dequantize row of Q4_K
void dequantize_row_q4_k(const BlockQ4_K* x, float* y, size_t k);

// Dequantize row of Q6_K
void dequantize_row_q6_k(const BlockQ6_K* x, float* y, size_t k);

// Block-quantized activation for 256-element super-blocks (Q8_K format)
struct BlockQ8_K_256 {
    int8_t qs[256];       // Quantized activation values in [-127, 127]
    float scales[16];     // FP32 scale per 16-element sub-block (for Q6_K)
    int32_t sums[16];     // Precomputed sum of qs in each 16-element sub-block
    float scales32[8];    // FP32 scale per 32-element sub-block (for Q4_K)
    int32_t sums32[8];    // Precomputed sum of qs in each 32-element sub-block
    int32_t sums_interleaved[16]; // Interleaved sums for vectorized AVX2 dot product
};

// Quantize an FP32 activation array of length k into BlockQ8_K_256
void quantize_activation_q8_k_256(const float* src, BlockQ8_K_256* dst, size_t k);

// Direct dot product: Q4_K with FP32 vector
float vec_dot_q4_k_f32_scalar(const BlockQ4_K* a, const float* b, size_t k);
float vec_dot_q4_k_f32_avx2(const BlockQ4_K* a, const float* b, size_t k);
float vec_dot_q4_k_f32(const BlockQ4_K* a, const float* b, size_t k);

// Direct dot product: Q4_K with Q8_K quantized activation
float vec_dot_q4_k_q8_avx2(const BlockQ4_K* a, const BlockQ8_K_256* b, size_t k);

// Direct dot product: Q6_K with FP32 vector
float vec_dot_q6_k_f32_scalar(const BlockQ6_K* a, const float* b, size_t k);
float vec_dot_q6_k_f32_avx2(const BlockQ6_K* a, const float* b, size_t k);
float vec_dot_q6_k_f32_fused_avx2(const BlockQ6_K* a, const float* b, size_t k);
float vec_dot_q6_k_f32(const BlockQ6_K* a, const float* b, size_t k);

// Direct dot product: Q6_K with Q8_K quantized activation
float vec_dot_q6_k_q8_avx2(const BlockQ6_K* a, const BlockQ8_K_256* b, size_t k);

// Quantized matvec (FP32 activations)
void matvec_q4_k_f32(const BlockQ4_K* A, const float* x, const float* bias, float* y, size_t rows, size_t cols);
void matvec_q6_k_f32(const BlockQ6_K* A, const float* x, const float* bias, float* y, size_t rows, size_t cols);

// Quantized matvec (INT8 / Q8_K activations)
void matvec_q4_k_q8(const BlockQ4_K* A, const BlockQ8_K_256* x, const float* bias, float* y, size_t rows, size_t cols);
void matvec_q6_k_q8(const BlockQ6_K* A, const BlockQ8_K_256* x, const float* bias, float* y, size_t rows, size_t cols);

// Fused 2x projection (e.g., Gate and Up projections sharing activation)
void matvec_q4_k_q8_2x(
    const BlockQ4_K* A1, float* y1,
    const BlockQ4_K* A2, float* y2,
    const BlockQ8_K_256* x,
    size_t rows,
    size_t cols
);

// Fused QKV projection sharing activation across all 3 heads
void matvec_q4_k_q8_qkv(
    const BlockQ4_K* Q_W, const float* q_bias, float* q_out, size_t q_rows,
    const BlockQ4_K* K_W, const float* k_bias, float* k_out, size_t k_rows,
    const BlockQ4_K* V_W, const float* v_bias, float* v_out, size_t v_rows,
    const BlockQ8_K_256* x,
    size_t cols
);

// Batched Quantized matvec: evaluates N independent vectors against same weights
void matvec_q4_k_f32_batch(
    const BlockQ4_K* A,
    const float* const* X,
    const float* bias,
    float* const* Y,
    size_t batch_size,
    size_t rows,
    size_t cols
);

void matvec_q6_k_f32_batch(
    const BlockQ6_K* A,
    const float* const* X,
    const float* bias,
    float* const* Y,
    size_t batch_size,
    size_t rows,
    size_t cols
);

void matvec_q4_k_q8_batch(
    const BlockQ4_K* A,
    const BlockQ8_K_256* const* X,
    const float* bias,
    float* const* Y,
    size_t batch_size,
    size_t rows,
    size_t cols
);

void matvec_q6_k_q8_batch(
    const BlockQ6_K* A,
    const BlockQ8_K_256* const* X,
    const float* bias,
    float* const* Y,
    size_t batch_size,
    size_t rows,
    size_t cols
);

} // namespace vectra::quant
