#pragma once

#include "vectra/core/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace vectra::tokenizer {

struct SpecialTokens {
    int32_t bos_token_id = 151643;        // <|endoftext|>
    int32_t eos_token_id = 151645;        // <|im_end|>
    int32_t pad_token_id = 151643;        // <|endoftext|>
    int32_t im_start_token_id = 151644;   // <|im_start|>
    int32_t im_end_token_id = 151645;     // <|im_end|>

    std::string bos_token = "<|endoftext|>";
    std::string eos_token = "<|im_end|>";
    std::string im_start_token = "<|im_start|>";
    std::string im_end_token = "<|im_end|>";
};

class Tokenizer {
public:
    Tokenizer();
    explicit Tokenizer(SpecialTokens special_tokens);

    // Initialize with a standard vocabulary map or generated subset
    void add_token(const std::string& token, int32_t id);
    void add_special_token(const std::string& token, int32_t id);

    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text, bool add_special = false) const;

    // Decode token IDs back to text
    std::string decode(const std::vector<int32_t>& tokens, bool skip_special = false) const;

    // Individual token lookups
    int32_t token_to_id(const std::string& token) const;
    std::string id_to_token(int32_t id) const;
    bool is_special(int32_t id) const;

    size_t vocab_size() const { return vocab_to_id_.size(); }
    const SpecialTokens& special_tokens() const { return special_tokens_; }

    // ChatML template formatting helper
    std::string format_chatml(const std::string& role, const std::string& content) const;

private:
    SpecialTokens special_tokens_;
    std::unordered_map<std::string, int32_t> vocab_to_id_;
    std::unordered_map<int32_t, std::string> id_to_vocab_;
    std::unordered_map<std::string, int32_t> special_vocab_to_id_;
    std::unordered_map<int32_t, std::string> id_to_special_vocab_;
};

} // namespace vectra::tokenizer
