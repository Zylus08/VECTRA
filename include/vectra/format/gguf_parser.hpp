#pragma once

#include "vectra/core/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <cstdint>
#include <memory>

namespace vectra::format {

enum class GGUFType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    BF16 = 29
};

const char* ggml_type_name(GGMLType type);

struct GGUFMetadataValue {
    GGUFType type;
    std::variant<
        uint8_t, int8_t, uint16_t, int16_t,
        uint32_t, int32_t, float, bool,
        std::string,
        uint64_t, int64_t, double,
        std::vector<std::string>,
        std::vector<int32_t>,
        std::vector<float>
    > value;
};

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    GGMLType type;
    uint64_t offset;       // Relative to data section start
    uint64_t absolute_file_offset;
    uint64_t size_bytes;
};

class GGUFParser {
public:
    GGUFParser();
    ~GGUFParser();

    // Parse GGUF file from disk
    void load_file(const std::string& filepath);

    // Parse GGUF from memory buffer
    void parse_memory(const uint8_t* buffer, size_t size);

    uint32_t version() const { return version_; }
    uint64_t tensor_count() const { return tensor_count_; }
    uint64_t metadata_count() const { return metadata_count_; }
    uint32_t alignment() const { return alignment_; }
    uint64_t data_offset() const { return data_offset_; }
    size_t file_size() const { return file_size_; }

    const std::unordered_map<std::string, GGUFMetadataValue>& metadata() const { return metadata_; }
    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }

    const GGUFMetadataValue* get_metadata(const std::string& key) const;
    std::string get_string_metadata(const std::string& key, const std::string& default_val = "") const;
    uint32_t get_uint32_metadata(const std::string& key, uint32_t default_val = 0) const;
    float get_float_metadata(const std::string& key, float default_val = 0.0f) const;

    const GGUFTensorInfo* find_tensor(const std::string& name) const;

private:
    uint32_t version_ = 0;
    uint64_t tensor_count_ = 0;
    uint64_t metadata_count_ = 0;
    uint32_t alignment_ = 32;
    uint64_t data_offset_ = 0;
    size_t file_size_ = 0;

    std::unordered_map<std::string, GGUFMetadataValue> metadata_;
    std::vector<GGUFTensorInfo> tensors_;

    std::vector<uint8_t> owned_buffer_;
};

} // namespace vectra::format
