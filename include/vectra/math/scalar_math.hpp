#pragma once

#include <cstddef>
#include <cstdint>

namespace vectra::math::scalar {

void add(const float* a, const float* b, float* out, size_t n);
void mul(const float* a, const float* b, float* out, size_t n);
void scale(const float* a, float s, float* out, size_t n);
float dot(const float* a, const float* b, size_t n);
void matvec(const float* A, const float* x, const float* bias, float* y, size_t rows, size_t cols);
void matmul(const float* A, const float* B, float* C, size_t M, size_t K, size_t N);

void rms_norm(const float* x, const float* weight, float* out, size_t n, float eps);
void softmax(const float* x, float* out, size_t n);
float silu_scalar(float x);
void silu(const float* x, float* out, size_t n);
void swiglu(const float* gate, const float* up, float* out, size_t n);

} // namespace vectra::math::scalar
