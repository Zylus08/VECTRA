#include "vectra/tensor/views.hpp"
#include "vectra/tensor/tensor.hpp"

namespace vectra {

TensorView::TensorView()
    : data_(nullptr), dtype_(DataType::FP32), numel_(0), is_contiguous_(true) {}

TensorView::TensorView(void* data, std::vector<int64_t> shape, std::vector<int64_t> strides, DataType dtype)
    : data_(data), shape_(std::move(shape)), strides_(std::move(strides)), dtype_(dtype), numel_(1), is_contiguous_(true) {
    if (shape_.empty()) {
        numel_ = 0;
        return;
    }
    for (int64_t dim : shape_) {
        numel_ *= dim;
    }

    // Check contiguity
    int64_t expected_stride = 1;
    for (int64_t i = static_cast<int64_t>(shape_.size()) - 1; i >= 0; --i) {
        if (strides_[i] != expected_stride) {
            is_contiguous_ = false;
            break;
        }
        expected_stride *= shape_[i];
    }
}

TensorView::TensorView(const Tensor& tensor)
    : data_(const_cast<void*>(tensor.data())), shape_(tensor.shape()), strides_(tensor.strides()),
      dtype_(tensor.dtype()), numel_(tensor.numel()), is_contiguous_(tensor.is_contiguous()) {}

TensorView TensorView::slice(int64_t dim, int64_t start, int64_t length) const {
    if (dim < 0 || dim >= static_cast<int64_t>(shape_.size())) {
        throw VectraException("Slice dimension out of bounds");
    }
    if (start < 0 || length <= 0 || (start + length) > shape_[dim]) {
        throw VectraException("Slice index range out of bounds");
    }

    std::vector<int64_t> new_shape = shape_;
    new_shape[dim] = length;

    size_t elem_sz = dtype_size(dtype_);
    char* offset_data = reinterpret_cast<char*>(data_) + (start * strides_[dim] * elem_sz);

    return TensorView(offset_data, new_shape, strides_, dtype_);
}

} // namespace vectra
