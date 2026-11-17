#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vectra {

constexpr size_t VECTRA_DEFAULT_ALIGNMENT = 64;

void* allocate_aligned(size_t bytes, size_t alignment = VECTRA_DEFAULT_ALIGNMENT);
void free_aligned(void* ptr);

class AlignedBuffer {
public:
    AlignedBuffer();
    explicit AlignedBuffer(size_t bytes, size_t alignment = VECTRA_DEFAULT_ALIGNMENT);
    ~AlignedBuffer();

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    void resize(size_t bytes, size_t alignment = VECTRA_DEFAULT_ALIGNMENT);
    void release();

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    size_t alignment() const { return alignment_; }
    bool empty() const { return data_ == nullptr || size_ == 0; }

private:
    void* data_;
    size_t size_;
    size_t alignment_;
};

} // namespace vectra
