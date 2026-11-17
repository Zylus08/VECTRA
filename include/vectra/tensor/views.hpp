#pragma once

#include "vectra/core/types.hpp"
#include <vector>
#include <cassert>
#include <cstdint>

namespace vectra {

class Tensor;

class TensorView {
public:
    TensorView();
    TensorView(void* data, std::vector<int64_t> shape, std::vector<int64_t> strides, DataType dtype);
    explicit TensorView(const Tensor& tensor);

    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DataType dtype() const { return dtype_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return numel_; }
    bool is_contiguous() const { return is_contiguous_; }

    void* data() { return data_; }
    const void* data() const { return data_; }

    template <typename T>
    T* data_ptr() {
        return reinterpret_cast<T*>(data_);
    }

    template <typename T>
    const T* data_ptr() const {
        return reinterpret_cast<const T*>(data_);
    }

    TensorView slice(int64_t dim, int64_t start, int64_t length) const;

    float at(int64_t i) const {
        assert(dtype_ == DataType::FP32 && i >= 0 && i < numel_);
        return data_ptr<float>()[i];
    }

    float at(int64_t r, int64_t c) const {
        assert(dtype_ == DataType::FP32 && shape_.size() == 2 && r >= 0 && r < shape_[0] && c >= 0 && c < shape_[1]);
        return data_ptr<float>()[r * strides_[0] + c * strides_[1]];
    }

private:
    void* data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DataType dtype_;
    int64_t numel_;
    bool is_contiguous_;
};

} // namespace vectra
