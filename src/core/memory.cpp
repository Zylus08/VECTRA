#include "vectra/core/memory.hpp"
#include "vectra/core/types.hpp"
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace vectra {

void* allocate_aligned(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;
#if defined(_MSC_VER)
    void* ptr = _aligned_malloc(bytes, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        ptr = nullptr;
    }
#endif
    if (!ptr) {
        throw VectraException("Out of memory: failed to allocate " + std::to_string(bytes) + " bytes aligned to " + std::to_string(alignment));
    }
    std::memset(ptr, 0, bytes);
    return ptr;
}

void free_aligned(void* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

AlignedBuffer::AlignedBuffer() : data_(nullptr), size_(0), alignment_(VECTRA_DEFAULT_ALIGNMENT) {}

AlignedBuffer::AlignedBuffer(size_t bytes, size_t alignment)
    : data_(nullptr), size_(0), alignment_(alignment) {
    if (bytes > 0) {
        data_ = allocate_aligned(bytes, alignment);
        size_ = bytes;
    }
}

AlignedBuffer::~AlignedBuffer() {
    release();
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
        release();
        data_ = other.data_;
        size_ = other.size_;
        alignment_ = other.alignment_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void AlignedBuffer::resize(size_t bytes, size_t alignment) {
    if (size_ == bytes && alignment_ == alignment && data_ != nullptr) {
        return;
    }
    release();
    if (bytes > 0) {
        data_ = allocate_aligned(bytes, alignment);
        size_ = bytes;
        alignment_ = alignment;
    }
}

void AlignedBuffer::release() {
    if (data_) {
        free_aligned(data_);
        data_ = nullptr;
    }
    size_ = 0;
}

} // namespace vectra
