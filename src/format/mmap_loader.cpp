#include "vectra/format/mmap_loader.hpp"
#include <iostream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

namespace vectra::format {

ProcessMemoryStats get_process_memory_stats() {
    ProcessMemoryStats stats{0, 0, 0};
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        stats.working_set_bytes = pmc.WorkingSetSize;
        stats.peak_working_set_bytes = pmc.PeakWorkingSetSize;
        stats.private_bytes = pmc.PrivateUsage;
    }
#endif
    return stats;
}

MemoryMappedFile::MemoryMappedFile()
    : file_handle_(nullptr), mapping_handle_(nullptr), data_(nullptr), size_(0) {}

MemoryMappedFile::MemoryMappedFile(const std::string& filepath)
    : file_handle_(nullptr), mapping_handle_(nullptr), data_(nullptr), size_(0) {
    open(filepath);
}

MemoryMappedFile::~MemoryMappedFile() {
    close();
}

MemoryMappedFile::MemoryMappedFile(MemoryMappedFile&& other) noexcept
    : file_handle_(other.file_handle_),
      mapping_handle_(other.mapping_handle_),
      data_(other.data_),
      size_(other.size_) {
    other.file_handle_ = nullptr;
    other.mapping_handle_ = nullptr;
    other.data_ = nullptr;
    other.size_ = 0;
}

MemoryMappedFile& MemoryMappedFile::operator=(MemoryMappedFile&& other) noexcept {
    if (this != &other) {
        close();
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        data_ = other.data_;
        size_ = other.size_;
        other.file_handle_ = nullptr;
        other.mapping_handle_ = nullptr;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MemoryMappedFile::open(const std::string& filepath) {
    close();

#if defined(_WIN32)
    // Convert utf-8 path to wide string
    int wlen = MultiByteToWideChar(CP_UTF8, 0, filepath.c_str(), -1, nullptr, 0);
    if (wlen <= 0) {
        throw VectraException("Invalid file path encoding: " + filepath);
    }
    std::wstring wpath(wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, filepath.c_str(), -1, &wpath[0], wlen);

    HANDLE hFile = CreateFileW(
        wpath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_READONLY | FILE_FLAG_RANDOM_ACCESS,
        nullptr
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        throw VectraException("Failed to open file for mmap (error " + std::to_string(err) + "): " + filepath);
    }
    file_handle_ = hFile;

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        close();
        throw VectraException("Failed to get file size: " + filepath);
    }
    size_ = static_cast<size_t>(fileSize.QuadPart);

    if (size_ == 0) {
        close();
        throw VectraException("Cannot memory map an empty file: " + filepath);
    }

    HANDLE hMapping = CreateFileMappingW(
        hFile,
        nullptr,
        PAGE_READONLY,
        fileSize.HighPart,
        fileSize.LowPart,
        nullptr
    );

    if (!hMapping) {
        DWORD err = GetLastError();
        close();
        throw VectraException("Failed to create file mapping object (error " + std::to_string(err) + "): " + filepath);
    }
    mapping_handle_ = hMapping;

    void* pView = MapViewOfFile(
        hMapping,
        FILE_MAP_READ,
        0,
        0,
        0 // Map entire file
    );

    if (!pView) {
        DWORD err = GetLastError();
        close();
        throw VectraException("Failed to map view of file (error " + std::to_string(err) + "): " + filepath);
    }
    data_ = reinterpret_cast<const uint8_t*>(pView);
#else
    throw VectraException("MemoryMappedFile only implemented for Win32 platform");
#endif
}

void MemoryMappedFile::close() {
#if defined(_WIN32)
    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (mapping_handle_) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = nullptr;
    }
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
    }
#endif
    size_ = 0;
}

const void* MemoryMappedFile::get_pointer_at(uint64_t offset, size_t bytes) const {
    if (!data_) {
        throw VectraException("MemoryMappedFile is not open");
    }
    if (offset + bytes > size_ || (offset + bytes) < offset) {
        throw VectraException("MemoryMappedFile access out of bounds: offset=" +
                              std::to_string(offset) + ", bytes=" + std::to_string(bytes) +
                              ", file_size=" + std::to_string(size_));
    }
    return data_ + offset;
}

} // namespace vectra::format
