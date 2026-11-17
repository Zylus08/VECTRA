#pragma once

#include "vectra/core/types.hpp"
#include "vectra/core/memory.hpp"
#include <vector>
#include <string>
#include <memory>
#include <cassert>

namespace vectra {

class TensorView;

class Tensor {
public:
    Tensor();
    Tensor(std::vector<int64_t> shape, DataType dtype = DataType::FP32);
    
    // Non-copyable by default to prevent accidental massive memory duplicates
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move constructors
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Deep clone
    Tensor clone() const;

    // Accessors
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DataType dtype() const { return dtype_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return numel_; }
    size_t bytes() const { return buffer_.size(); }
    bool is_contiguous() const { return is_contiguous_; }

    void* data() { return buffer_.data(); }
    const void* data() const { return buffer_.data(); }

    template <typename T>
    T* data_ptr() {
        return reinterpret_cast<T*>(buffer_.data());
    }

    template <typename T>
    const T* data_ptr() const {
        return reinterpret_cast<const T*>(buffer_.data());
    }

    // Element access with bounds checking
    float& at(int64_t i) {
        assert(dtype_ == DataType::FP32 && i >= 0 && i < numel_);
        return data_ptr<float>()[i];
    }

    float at(int64_t i) const {
        assert(dtype_ == DataType::FP32 && i >= 0 && i < numel_);
        return data_ptr<float>()[i];
    }

    float& at(int64_t r, int64_t c) {
        assert(dtype_ == DataType::FP32 && shape_.size() == 2 && r >= 0 && r < shape_[0] && c >= 0 && c < shape_[1]);
        return data_ptr<float>()[r * strides_[0] + c * strides_[1]];
    }

    float at(int64_t r, int64_t c) const {
        assert(dtype_ == DataType::FP32 && shape_.size() == 2 && r >= 0 && r < shape_[0] && c >= 0 && c < shape_[1]);
        return data_ptr<float>()[r * strides_[0] + c * strides_[1]];
    }

    // View creation
    TensorView view() const;
    TensorView slice(int64_t dim, int64_t start, int64_t length) const;

    // Reshape (returns a new Tensor or validates contiguity)
    void reshape(const std::vector<int64_t>& new_shape);

    // Fill utilities
    void fill_zero();
    void fill_constant(float value);

private:
    void calculate_strides();

    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DataType dtype_;
    int64_t numel_;
    bool is_contiguous_;
    AlignedBuffer buffer_;
};

} // namespace vectra
