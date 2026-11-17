#pragma once

#include "vectra/core/types.hpp"
#include <string>
#include <cstdint>
#include <cstddef>
#include <memory>

namespace vectra::format {

struct ProcessMemoryStats {
    size_t working_set_bytes;
    size_t peak_working_set_bytes;
    size_t private_bytes;
};

ProcessMemoryStats get_process_memory_stats();

class MemoryMappedFile {
public:
    MemoryMappedFile();
    explicit MemoryMappedFile(const std::string& filepath);
    ~MemoryMappedFile();

    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    MemoryMappedFile(MemoryMappedFile&& other) noexcept;
    MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept;

    void open(const std::string& filepath);
    void close();

    bool is_open() const { return data_ != nullptr; }
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

    // Direct access to a memory region within the mapped file
    const void* get_pointer_at(uint64_t offset, size_t bytes) const;

    template <typename T>
    const T* get_typed_pointer(uint64_t offset, size_t element_count = 1) const {
        return reinterpret_cast<const T*>(get_pointer_at(offset, element_count * sizeof(T)));
    }

private:
    void* file_handle_;
    void* mapping_handle_;
    const uint8_t* data_;
    size_t size_;
};

} // namespace vectra::format
