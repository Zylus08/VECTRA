#pragma once

#include "vectra/core/types.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace vectra::quant {

constexpr size_t Q4_BLOCK_SIZE = 32;

#pragma pack(push, 1)
struct BlockQ4_0 {
    uint16_t d;                      // FP16 scale factor
    uint8_t qs[Q4_BLOCK_SIZE / 2];   // 16 bytes = 32 nibbles (4-bit signed representation with offset 8)
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be exactly 18 bytes");

struct QuantErrorMetrics {
    double mse;
    double rmse;
    double max_abs_error;
    double snr_db;
};

// Quantize FP32 array to BlockQ4_0
void quantize_row_q4_0(const float* x, BlockQ4_0* y, size_t k);

// Dequantize BlockQ4_0 array back to FP32
void dequantize_row_q4_0(const BlockQ4_0* x, float* y, size_t k);

// Direct dot product between BlockQ4_0 and FP32 vector (scalar reference)
float vec_dot_q4_0_f32_scalar(const BlockQ4_0* a, const float* b, size_t k);

// Direct dot product between BlockQ4_0 and FP32 vector (AVX2 accelerated)
float vec_dot_q4_0_f32_avx2(const BlockQ4_0* a, const float* b, size_t k);

// Dispatched dot product
float vec_dot_q4_0_f32(const BlockQ4_0* a, const float* b, size_t k);

// Quantized matrix-vector multiplication
void matvec_q4_0_f32(const BlockQ4_0* A, const float* x, const float* bias, float* y, size_t rows, size_t cols);

// Error analysis
QuantErrorMetrics evaluate_quantization_error(const float* original, const float* reconstructed, size_t n);

} // namespace vectra::quant
