#pragma once

#include "vectra/core/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <map>

namespace vectra::tokenizer {

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    // Load pre-compiled binary tokenizer (vocab + merges)
    void load_binary(const std::string& filepath);

    // Encode text to exact Qwen token IDs
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode token IDs back to UTF-8 text string
    std::string decode(const std::vector<int32_t>& tokens, bool skip_special = false) const;

    // Single token string decode (for streaming generation)
    std::string decode_token(int32_t token_id) const;

    // ChatML template formatting
    std::string format_chatml(const std::string& role, const std::string& content) const;

    int32_t bos_id() const { return 151643; } // <|endoftext|>
    int32_t eos_id() const { return 151645; } // <|im_end|>
    int32_t im_start_id() const { return 151644; } // <|im_start|>
    int32_t im_end_id() const { return 151645; } // <|im_end|>

    size_t vocab_size() const { return vocab_to_id_.size(); }
    bool is_special(int32_t token_id) const;

private:
    void init_byte_mappings();
    std::vector<std::string> bpe_merge(const std::string& piece) const;

    // Vocabulary mapping
    std::unordered_map<std::string, int32_t> vocab_to_id_;
    std::unordered_map<int32_t, std::string> id_to_vocab_;

    // BPE merge rules: pair (a, b) -> rank
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> merge_ranks_;

    // GPT-2 Byte <-> Unicode mapping
    std::unordered_map<uint8_t, std::string> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    // Special tokens map
    std::unordered_map<std::string, int32_t> special_tokens_;
    std::unordered_map<int32_t, std::string> id_to_special_;
};

} // namespace vectra::tokenizer
