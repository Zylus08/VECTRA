#include "vectra/format/gguf_parser.hpp"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace vectra::format {

const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::F32: return "F32";
        case GGMLType::F16: return "F16";
        case GGMLType::Q4_0: return "Q4_0";
        case GGMLType::Q4_1: return "Q4_1";
        case GGMLType::Q5_0: return "Q5_0";
        case GGMLType::Q5_1: return "Q5_1";
        case GGMLType::Q8_0: return "Q8_0";
        case GGMLType::Q8_1: return "Q8_1";
        case GGMLType::Q2_K: return "Q2_K";
        case GGMLType::Q3_K: return "Q3_K";
        case GGMLType::Q4_K: return "Q4_K";
        case GGMLType::Q5_K: return "Q5_K";
        case GGMLType::Q6_K: return "Q6_K";
        case GGMLType::Q8_K: return "Q8_K";
        case GGMLType::I8: return "I8";
        case GGMLType::I16: return "I16";
        case GGMLType::I32: return "I32";
        case GGMLType::I64: return "I64";
        case GGMLType::F64: return "F64";
        case GGMLType::BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

namespace {

class BinaryReader {
public:
    BinaryReader(const uint8_t* data, size_t size) : data_(data), size_(size), pos_(0) {
        // Explicit little-endian check
        uint16_t test = 0x0001;
        if (*reinterpret_cast<const uint8_t*>(&test) != 0x01) {
            throw VectraException("GGUF reader requires little-endian architecture");
        }
    }

    size_t pos() const { return pos_; }
    size_t remaining() const { return size_ > pos_ ? size_ - pos_ : 0; }

    template <typename T>
    T read() {
        if (pos_ + sizeof(T) > size_) {
            throw VectraException("Unexpected end of file while reading binary data at offset " + std::to_string(pos_));
        }
        T val;
        std::memcpy(&val, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return val;
    }

    std::string read_string() {
        uint64_t len = read<uint64_t>();
        if (pos_ + len > size_) {
            throw VectraException("String length " + std::to_string(len) + " exceeds buffer boundaries at offset " + std::to_string(pos_));
        }
        std::string s(reinterpret_cast<const char*>(data_ + pos_), static_cast<size_t>(len));
        pos_ += len;
        return s;
    }

    void skip(size_t bytes) {
        if (pos_ + bytes > size_) {
            throw VectraException("Seek exceeds buffer boundaries");
        }
        pos_ += bytes;
    }

    void align_to(size_t alignment) {
        size_t rem = pos_ % alignment;
        if (rem != 0) {
            size_t pad = alignment - rem;
            skip(pad);
        }
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
};

} // anonymous namespace

GGUFParser::GGUFParser() = default;
GGUFParser::~GGUFParser() = default;

void GGUFParser::load_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw VectraException("Failed to open GGUF file: " + filepath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    owned_buffer_.resize(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(owned_buffer_.data()), size)) {
        throw VectraException("Failed to read file contents: " + filepath);
    }

    parse_memory(owned_buffer_.data(), owned_buffer_.size());
}

void GGUFParser::parse_memory(const uint8_t* buffer, size_t size) {
    file_size_ = size;
    metadata_.clear();
    tensors_.clear();

    BinaryReader reader(buffer, size);

    // 1. Magic check: 'G' 'G' 'U' 'F' -> 0x46554747 in little-endian
    uint32_t magic = reader.read<uint32_t>();
    if (magic != 0x46554747) {
        std::stringstream ss;
        ss << "Invalid GGUF magic header: 0x" << std::hex << magic << " (expected 0x46554747 'GGUF')";
        throw VectraException(ss.str());
    }

    // 2. Version check
    version_ = reader.read<uint32_t>();
    if (version_ != 2 && version_ != 3) {
        throw VectraException("Unsupported GGUF version: " + std::to_string(version_) + " (only v2 and v3 supported)");
    }

    // 3. Counts
    tensor_count_ = reader.read<uint64_t>();
    metadata_count_ = reader.read<uint64_t>();

    // 4. Metadata KV pairs
    for (uint64_t i = 0; i < metadata_count_; ++i) {
        std::string key = reader.read_string();
        if (metadata_.find(key) != metadata_.end()) {
            throw VectraException("Duplicate metadata key detected in GGUF: " + key);
        }

        uint32_t type_val = reader.read<uint32_t>();
        GGUFType vtype = static_cast<GGUFType>(type_val);
        GGUFMetadataValue mval;
        mval.type = vtype;

        switch (vtype) {
            case GGUFType::UINT8:   mval.value = reader.read<uint8_t>(); break;
            case GGUFType::INT8:    mval.value = reader.read<int8_t>(); break;
            case GGUFType::UINT16:  mval.value = reader.read<uint16_t>(); break;
            case GGUFType::INT16:   mval.value = reader.read<int16_t>(); break;
            case GGUFType::UINT32:  mval.value = reader.read<uint32_t>(); break;
            case GGUFType::INT32:   mval.value = reader.read<int32_t>(); break;
            case GGUFType::FLOAT32: mval.value = reader.read<float>(); break;
            case GGUFType::BOOL:    mval.value = (reader.read<uint8_t>() != 0); break;
            case GGUFType::STRING:  mval.value = reader.read_string(); break;
            case GGUFType::UINT64:  mval.value = reader.read<uint64_t>(); break;
            case GGUFType::INT64:   mval.value = reader.read<int64_t>(); break;
            case GGUFType::FLOAT64: mval.value = reader.read<double>(); break;
            case GGUFType::ARRAY: {
                uint32_t item_type_raw = reader.read<uint32_t>();
                GGUFType item_type = static_cast<GGUFType>(item_type_raw);
                uint64_t item_count = reader.read<uint64_t>();

                if (item_type == GGUFType::STRING) {
                    std::vector<std::string> str_list;
                    str_list.reserve(static_cast<size_t>(item_count));
                    for (uint64_t j = 0; j < item_count; ++j) {
                        str_list.push_back(reader.read_string());
                    }
                    mval.value = str_list;
                } else if (item_type == GGUFType::INT32) {
                    std::vector<int32_t> i_list;
                    i_list.reserve(static_cast<size_t>(item_count));
                    for (uint64_t j = 0; j < item_count; ++j) {
                        i_list.push_back(reader.read<int32_t>());
                    }
                    mval.value = i_list;
                } else if (item_type == GGUFType::FLOAT32) {
                    std::vector<float> f_list;
                    f_list.reserve(static_cast<size_t>(item_count));
                    for (uint64_t j = 0; j < item_count; ++j) {
                        f_list.push_back(reader.read<float>());
                    }
                    mval.value = f_list;
                } else {
                    // Skip unsupported array types
                    size_t item_elem_sz = 1;
                    if (item_type == GGUFType::UINT16 || item_type == GGUFType::INT16) item_elem_sz = 2;
                    else if (item_type == GGUFType::UINT32 || item_type == GGUFType::INT32 || item_type == GGUFType::FLOAT32) item_elem_sz = 4;
                    else if (item_type == GGUFType::UINT64 || item_type == GGUFType::INT64 || item_type == GGUFType::FLOAT64) item_elem_sz = 8;
                    reader.skip(static_cast<size_t>(item_count * item_elem_sz));
                }
                break;
            }
            default:
                throw VectraException("Unknown GGUF metadata type: " + std::to_string(type_val));
        }

        metadata_[key] = std::move(mval);
    }

    // Read general.alignment if present
    alignment_ = get_uint32_metadata("general.alignment", 32);

    // 5. Tensor info table
    tensors_.reserve(static_cast<size_t>(tensor_count_));
    for (uint64_t i = 0; i < tensor_count_; ++i) {
        GGUFTensorInfo info;
        info.name = reader.read_string();
        uint32_t n_dims = reader.read<uint32_t>();
        if (n_dims > 4) {
            throw VectraException("GGUF tensor has invalid number of dimensions: " + std::to_string(n_dims));
        }

        info.dimensions.resize(n_dims);
        uint64_t total_elements = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            info.dimensions[d] = reader.read<uint64_t>();
            total_elements *= info.dimensions[d];
        }

        info.type = static_cast<GGMLType>(reader.read<uint32_t>());
        info.offset = reader.read<uint64_t>();

        // Estimate size
        if (info.type == GGMLType::F32) {
            info.size_bytes = total_elements * 4;
        } else if (info.type == GGMLType::F16 || info.type == GGMLType::BF16) {
            info.size_bytes = total_elements * 2;
        } else if (info.type == GGMLType::Q4_0) {
            info.size_bytes = ((total_elements + 31) / 32) * 18;
        } else {
            info.size_bytes = total_elements; // estimate for reporting
        }

        tensors_.push_back(std::move(info));
    }

    // Align to data section
    reader.align_to(alignment_);
    data_offset_ = reader.pos();

    // Fill in absolute file offsets and validate boundaries
    for (auto& t : tensors_) {
        t.absolute_file_offset = data_offset_ + t.offset;
        if (t.absolute_file_offset + t.size_bytes > size) {
            // Out of bounds check
            throw VectraException("Tensor " + t.name + " data extends past end of file");
        }
    }
}

const GGUFMetadataValue* GGUFParser::get_metadata(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it != metadata_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::string GGUFParser::get_string_metadata(const std::string& key, const std::string& default_val) const {
    const auto* mv = get_metadata(key);
    if (mv && std::holds_alternative<std::string>(mv->value)) {
        return std::get<std::string>(mv->value);
    }
    return default_val;
}

uint32_t GGUFParser::get_uint32_metadata(const std::string& key, uint32_t default_val) const {
    const auto* mv = get_metadata(key);
    if (mv) {
        if (std::holds_alternative<uint32_t>(mv->value)) return std::get<uint32_t>(mv->value);
        if (std::holds_alternative<int32_t>(mv->value)) return static_cast<uint32_t>(std::get<int32_t>(mv->value));
        if (std::holds_alternative<uint64_t>(mv->value)) return static_cast<uint32_t>(std::get<uint64_t>(mv->value));
    }
    return default_val;
}

float GGUFParser::get_float_metadata(const std::string& key, float default_val) const {
    const auto* mv = get_metadata(key);
    if (mv) {
        if (std::holds_alternative<float>(mv->value)) return std::get<float>(mv->value);
        if (std::holds_alternative<double>(mv->value)) return static_cast<float>(std::get<double>(mv->value));
    }
    return default_val;
}

const GGUFTensorInfo* GGUFParser::find_tensor(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

} // namespace vectra::format
