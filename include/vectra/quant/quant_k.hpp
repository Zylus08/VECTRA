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

// Direct dot product: Q4_K with FP32 vector
float vec_dot_q4_k_f32_scalar(const BlockQ4_K* a, const float* b, size_t k);
float vec_dot_q4_k_f32_avx2(const BlockQ4_K* a, const float* b, size_t k);
float vec_dot_q4_k_f32(const BlockQ4_K* a, const float* b, size_t k);

// Direct dot product: Q6_K with FP32 vector
float vec_dot_q6_k_f32_scalar(const BlockQ6_K* a, const float* b, size_t k);
float vec_dot_q6_k_f32_avx2(const BlockQ6_K* a, const float* b, size_t k);
float vec_dot_q6_k_f32(const BlockQ6_K* a, const float* b, size_t k);

// Quantized matvec
void matvec_q4_k_f32(const BlockQ4_K* A, const float* x, const float* bias, float* y, size_t rows, size_t cols);
void matvec_q6_k_f32(const BlockQ6_K* A, const float* x, const float* bias, float* y, size_t rows, size_t cols);

} // namespace vectra::quant
