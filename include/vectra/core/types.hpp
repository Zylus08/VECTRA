#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace vectra {

enum class DataType : uint32_t {
    FP32 = 0,
    FP16 = 1,
    INT32 = 2,
    INT64 = 3,
    Q4_0 = 4
};

inline const char* dtype_name(DataType dt) {
    switch (dt) {
        case DataType::FP32: return "FP32";
        case DataType::FP16: return "FP16";
        case DataType::INT32: return "INT32";
        case DataType::INT64: return "INT64";
        case DataType::Q4_0: return "Q4_0";
        default: return "UNKNOWN";
    }
}

inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::FP32: return sizeof(float);
        case DataType::FP16: return 2;
        case DataType::INT32: return sizeof(int32_t);
        case DataType::INT64: return sizeof(int64_t);
        case DataType::Q4_0: return 0; // Block-based, handled specifically
        default: return 0;
    }
}

class VectraException : public std::runtime_error {
public:
    explicit VectraException(const std::string& msg) : std::runtime_error(msg) {}
};

// IEEE 754 half-precision float conversion utilities
uint16_t float32_to_float16(float val);
float float16_to_float32(uint16_t val);

} // namespace vectra
