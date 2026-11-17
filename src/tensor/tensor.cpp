#include "vectra/tensor/tensor.hpp"
#include "vectra/tensor/views.hpp"
#include <numeric>
#include <cstring>

namespace vectra {

Tensor::Tensor() : dtype_(DataType::FP32), numel_(0), is_contiguous_(true) {}

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype)
    : shape_(std::move(shape)), dtype_(dtype), numel_(1), is_contiguous_(true) {
    if (shape_.empty()) {
        numel_ = 0;
        return;
    }
    for (int64_t dim : shape_) {
        if (dim <= 0) {
            throw VectraException("Tensor dimension must be positive, got " + std::to_string(dim));
        }
        // Checked arithmetic for overflow prevention
        int64_t next_numel = numel_ * dim;
        if (next_numel / dim != numel_) {
            throw VectraException("Tensor dimension calculation overflowed integer limits");
        }
        numel_ = next_numel;
    }
    calculate_strides();

    size_t element_bytes = dtype_size(dtype_);
    size_t total_bytes = static_cast<size_t>(numel_) * element_bytes;
    buffer_.resize(total_bytes, VECTRA_DEFAULT_ALIGNMENT);
}

void Tensor::calculate_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;

    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape_.size()) - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
    is_contiguous_ = true;
}

Tensor Tensor::clone() const {
    Tensor copy(shape_, dtype_);
    if (buffer_.size() > 0 && buffer_.data()) {
        std::memcpy(copy.data(), data(), buffer_.size());
    }
    return copy;
}

TensorView Tensor::view() const {
    return TensorView(const_cast<void*>(data()), shape_, strides_, dtype_);
}

TensorView Tensor::slice(int64_t dim, int64_t start, int64_t length) const {
    return view().slice(dim, start, length);
}

void Tensor::reshape(const std::vector<int64_t>& new_shape) {
    if (!is_contiguous_) {
        throw VectraException("Cannot reshape non-contiguous tensor without copying");
    }
    int64_t new_numel = 1;
    for (int64_t dim : new_shape) {
        if (dim <= 0) throw VectraException("New dimension must be positive");
        new_numel *= dim;
    }
    if (new_numel != numel_) {
        throw VectraException("Reshape total element count mismatch: " + std::to_string(numel_) + " vs " + std::to_string(new_numel));
    }
    shape_ = new_shape;
    calculate_strides();
}

void Tensor::fill_zero() {
    if (buffer_.data() && buffer_.size() > 0) {
        std::memset(buffer_.data(), 0, buffer_.size());
    }
}

void Tensor::fill_constant(float value) {
    if (dtype_ != DataType::FP32) {
        throw VectraException("fill_constant currently implemented for FP32");
    }
    float* ptr = data_ptr<float>();
    for (int64_t i = 0; i < numel_; ++i) {
        ptr[i] = value;
    }
}

} // namespace vectra
