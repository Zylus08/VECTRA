#include "vectra/tokenizer/tokenizer.hpp"
#include <iomanip>
#include <sstream>

namespace vectra::tokenizer {

Tokenizer::Tokenizer() : Tokenizer(SpecialTokens{}) {}

Tokenizer::Tokenizer(SpecialTokens special_tokens) : special_tokens_(special_tokens) {
    // Add default special tokens
    add_special_token(special_tokens_.bos_token, special_tokens_.bos_token_id);
    add_special_token(special_tokens_.im_start_token, special_tokens_.im_start_token_id);
    add_special_token(special_tokens_.eos_token, special_tokens_.eos_token_id);

    // Initialize byte-level fallback tokens: <0x00> through <0xFF>
    for (int i = 0; i < 256; ++i) {
        std::stringstream ss;
        ss << "<0x" << std::uppercase << std::hex << std::setw(2) << std::setfill('0') << i << ">";
        add_token(ss.str(), 100000 + i);
    }
}

void Tokenizer::add_token(const std::string& token, int32_t id) {
    vocab_to_id_[token] = id;
    id_to_vocab_[id] = token;
}

void Tokenizer::add_special_token(const std::string& token, int32_t id) {
    special_vocab_to_id_[token] = id;
    id_to_special_vocab_[id] = token;
    vocab_to_id_[token] = id;
    id_to_vocab_[id] = token;
}

int32_t Tokenizer::token_to_id(const std::string& token) const {
    auto it = vocab_to_id_.find(token);
    if (it != vocab_to_id_.end()) {
        return it->second;
    }
    return -1;
}

std::string Tokenizer::id_to_token(int32_t id) const {
    auto it = id_to_vocab_.find(id);
    if (it != id_to_vocab_.end()) {
        return it->second;
    }
    return "";
}

bool Tokenizer::is_special(int32_t id) const {
    return id_to_special_vocab_.find(id) != id_to_special_vocab_.end();
}

std::string Tokenizer::format_chatml(const std::string& role, const std::string& content) const {
    return "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_special) const {
    std::vector<int32_t> tokens;
    if (add_special) {
        tokens.push_back(special_tokens_.bos_token_id);
    }

    size_t i = 0;
    while (i < text.size()) {
        // First, check if current position matches any special token
        bool matched_special = false;
        for (const auto& [spec_str, spec_id] : special_vocab_to_id_) {
            if (text.compare(i, spec_str.size(), spec_str) == 0) {
                tokens.push_back(spec_id);
                i += spec_str.size();
                matched_special = true;
                break;
            }
        }
        if (matched_special) continue;

        // Longest match search in vocabulary
        size_t max_len = std::min(text.size() - i, size_t(32));
        bool found_match = false;
        for (size_t len = max_len; len > 0; --len) {
            std::string sub = text.substr(i, len);
            auto it = vocab_to_id_.find(sub);
            if (it != vocab_to_id_.end()) {
                tokens.push_back(it->second);
                i += len;
                found_match = true;
                break;
            }
        }

        // Byte fallback if no subword matched
        if (!found_match) {
            uint8_t byte_val = static_cast<uint8_t>(text[i]);
            std::stringstream ss;
            ss << "<0x" << std::uppercase << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte_val) << ">";
            int32_t byte_id = token_to_id(ss.str());
            if (byte_id != -1) {
                tokens.push_back(byte_id);
            }
            i += 1;
        }
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens, bool skip_special) const {
    std::string result;
    for (int32_t id : tokens) {
        if (skip_special && is_special(id)) {
            continue;
        }
        auto it = id_to_vocab_.find(id);
        if (it != id_to_vocab_.end()) {
            const std::string& tok = it->second;
            // Check if byte fallback token <0xXX>
            if (tok.size() == 6 && tok.substr(0, 3) == "<0x" && tok[5] == '>') {
                unsigned int byte_val = 0;
                std::stringstream ss;
                ss << std::hex << tok.substr(3, 2);
                if (ss >> byte_val) {
                    result.push_back(static_cast<char>(byte_val));
                    continue;
                }
            }
            result += tok;
        }
    }
    return result;
}

} // namespace vectra::tokenizer
