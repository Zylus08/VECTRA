#pragma once

#include <cstddef>
#include <cstdint>

namespace vectra::math::avx2 {

bool is_supported();

void add(const float* a, const float* b, float* out, size_t n);
void mul(const float* a, const float* b, float* out, size_t n);
void scale(const float* a, float s, float* out, size_t n);
float dot(const float* a, const float* b, size_t n);
void matvec(const float* A, const float* x, const float* bias, float* y, size_t rows, size_t cols);
void rms_norm(const float* x, const float* weight, float* out, size_t n, float eps);

} // namespace vectra::math::avx2
