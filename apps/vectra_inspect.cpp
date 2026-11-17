#include "vectra/format/gguf_parser.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: vectra_inspect <model.gguf>\n";
        return 1;
    }

    std::string filepath = argv[1];
    std::cout << "=========================================================\n";
    std::cout << "           PROJECT VECTRA - GGUF MODEL INSPECTOR         \n";
    std::cout << "=========================================================\n";
    std::cout << "Target File: " << filepath << "\n\n";

    try {
        vectra::format::GGUFParser parser;
        parser.load_file(filepath);

        std::cout << "--- Header Information ---\n";
        std::cout << "GGUF Version:       " << parser.version() << "\n";
        std::cout << "Tensor Count:       " << parser.tensor_count() << "\n";
        std::cout << "Metadata KV Count:  " << parser.metadata_count() << "\n";
        std::cout << "Data Alignment:     " << parser.alignment() << " bytes\n";
        std::cout << "Data Offset:        " << parser.data_offset() << " bytes\n";
        std::cout << "Total File Size:    " << parser.file_size() << " bytes ("
                  << std::fixed << std::setprecision(2) << (double)parser.file_size() / (1024 * 1024) << " MB)\n\n";

        std::cout << "--- Key Metadata Attributes ---\n";
        std::string arch = parser.get_string_metadata("general.architecture", "unknown");
        std::cout << "Architecture:       " << arch << "\n";
        std::cout << "Context Length:     " << parser.get_uint32_metadata(arch + ".context_length", 0) << "\n";
        std::cout << "Embedding Length:   " << parser.get_uint32_metadata(arch + ".embedding_length", 0) << "\n";
        std::cout << "Feed-Forward Dim:   " << parser.get_uint32_metadata(arch + ".feed_forward_length", 0) << "\n";
        std::cout << "Block Count:        " << parser.get_uint32_metadata(arch + ".block_count", 0) << "\n";
        std::cout << "Attention Heads:    " << parser.get_uint32_metadata(arch + ".attention.head_count", 0) << "\n";
        std::cout << "KV Heads:           " << parser.get_uint32_metadata(arch + ".attention.head_count_kv", 0) << "\n";
        std::cout << "RoPE Freq Base:     " << parser.get_float_metadata(arch + ".rope.freq_base", 0.0f) << "\n";
        std::cout << "RMS Norm Epsilon:   " << parser.get_float_metadata(arch + ".attention.layer_norm_rms_epsilon", 0.0f) << "\n\n";

        std::cout << "--- Tensor Table Summary ---\n";
        std::cout << std::left << std::setw(38) << "Tensor Name"
                  << std::setw(10) << "Type"
                  << std::setw(20) << "Dimensions"
                  << std::setw(12) << "Size (KB)"
                  << "Offset\n";
        std::cout << std::string(90, '-') << "\n";

        size_t total_tensor_bytes = 0;
        size_t display_count = std::min(size_t(20), parser.tensors().size());

        for (size_t i = 0; i < display_count; ++i) {
            const auto& t = parser.tensors()[i];
            total_tensor_bytes += t.size_bytes;

            std::stringstream dims_ss;
            dims_ss << "[";
            for (size_t d = 0; d < t.dimensions.size(); ++d) {
                dims_ss << t.dimensions[d] << (d + 1 < t.dimensions.size() ? "," : "");
            }
            dims_ss << "]";

            std::cout << std::left << std::setw(38) << t.name
                      << std::setw(10) << vectra::format::ggml_type_name(t.type)
                      << std::setw(20) << dims_ss.str()
                      << std::setw(12) << std::fixed << std::setprecision(1) << (double)t.size_bytes / 1024.0
                      << t.offset << "\n";
        }

        if (parser.tensors().size() > display_count) {
            std::cout << "... (" << (parser.tensors().size() - display_count) << " more tensors omitted) ...\n";
            for (size_t i = display_count; i < parser.tensors().size(); ++i) {
                total_tensor_bytes += parser.tensors()[i].size_bytes;
            }
        }
        std::cout << std::string(90, '-') << "\n";
        std::cout << "Total Model Weight Footprint: "
                  << std::fixed << std::setprecision(2) << (double)total_tensor_bytes / (1024 * 1024) << " MB\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Inspection Error: " << ex.what() << "\n";
        return 1;
    }
}
