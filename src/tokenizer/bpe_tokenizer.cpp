#include "vectra/tokenizer/bpe_tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cassert>
#include <algorithm>

namespace vectra::tokenizer {

namespace {

std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp <= 0x7F) {
        s.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        s.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        s.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

uint32_t decode_utf8_char(const std::string& str, size_t& i) {
    uint8_t c0 = static_cast<uint8_t>(str[i]);
    if (c0 < 0x80) {
        i += 1;
        return c0;
    } else if ((c0 & 0xE0) == 0xC0 && i + 1 < str.size()) {
        uint8_t c1 = static_cast<uint8_t>(str[i + 1]);
        i += 2;
        return ((c0 & 0x1F) << 6) | (c1 & 0x3F);
    } else if ((c0 & 0xF0) == 0xE0 && i + 2 < str.size()) {
        uint8_t c1 = static_cast<uint8_t>(str[i + 1]);
        uint8_t c2 = static_cast<uint8_t>(str[i + 2]);
        i += 3;
        return ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    } else if ((c0 & 0xF8) == 0xF0 && i + 3 < str.size()) {
        uint8_t c1 = static_cast<uint8_t>(str[i + 1]);
        uint8_t c2 = static_cast<uint8_t>(str[i + 2]);
        uint8_t c3 = static_cast<uint8_t>(str[i + 3]);
        i += 4;
        return ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
    }
    i += 1;
    return c0;
}

bool is_space_cp(uint32_t cp) {
    return (cp == ' ' || cp == '\t' || cp == '\r' || cp == '\n' || cp == 0x00A0);
}

bool is_digit_cp(uint32_t cp) {
    return (cp >= '0' && cp <= '9');
}

bool is_ascii_punct(uint32_t cp) {
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126);
}

bool is_letter_cp(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp >= 0x80) {
        // Exclude common symbols and emojis from letters
        if (cp >= 0x2000 && cp <= 0x206F) return false; // General punctuation
        if (cp >= 0x2E00 && cp <= 0x2E7F) return false; // Supplemental punctuation
        if (cp >= 0x3000 && cp <= 0x303F) return false; // CJK symbols and punctuation
        if (cp >= 0x1F000) return false;                 // Emojis / pictorial symbols
        return true;
    }
    return false;
}

} // anonymous namespace

BPETokenizer::BPETokenizer() {
    init_byte_mappings();
    // Special tokens
    special_tokens_["<|im_start|>"] = 151644;
    special_tokens_["<|im_end|>"]   = 151645;
    special_tokens_["<|endoftext|>"] = 151643;

    for (const auto& [str, id] : special_tokens_) {
        id_to_special_[id] = str;
    }
}

BPETokenizer::~BPETokenizer() = default;

void BPETokenizer::init_byte_mappings() {
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        uint8_t byte_val = static_cast<uint8_t>(bs[i]);
        std::string utf8_char = codepoint_to_utf8(static_cast<uint32_t>(cs[i]));
        byte_to_unicode_[byte_val] = utf8_char;
        unicode_to_byte_[utf8_char] = byte_val;
    }
}

void BPETokenizer::load_binary(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw VectraException("Failed to open binary tokenizer: " + filepath);
    }

    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x56544F4B) { // 'VTOK'
        throw VectraException("Invalid binary tokenizer magic");
    }

    // Read vocab
    uint32_t vocab_count = 0;
    file.read(reinterpret_cast<char*>(&vocab_count), 4);
    vocab_to_id_.clear();
    id_to_vocab_.clear();

    std::vector<char> str_buf;
    for (uint32_t i = 0; i < vocab_count; ++i) {
        uint16_t len = 0;
        file.read(reinterpret_cast<char*>(&len), 2);
        str_buf.resize(len);
        file.read(str_buf.data(), len);
        std::string token_str(str_buf.data(), len);

        int32_t token_id = 0;
        file.read(reinterpret_cast<char*>(&token_id), 4);

        vocab_to_id_[token_str] = token_id;
        id_to_vocab_[token_id] = token_str;
    }

    // Read merges
    uint32_t merges_count = 0;
    file.read(reinterpret_cast<char*>(&merges_count), 4);
    merge_ranks_.clear();

    for (uint32_t i = 0; i < merges_count; ++i) {
        uint32_t rank = 0;
        file.read(reinterpret_cast<char*>(&rank), 4);

        uint16_t len0 = 0;
        file.read(reinterpret_cast<char*>(&len0), 2);
        str_buf.resize(len0);
        file.read(str_buf.data(), len0);
        std::string p0(str_buf.data(), len0);

        uint16_t len1 = 0;
        file.read(reinterpret_cast<char*>(&len1), 2);
        str_buf.resize(len1);
        file.read(str_buf.data(), len1);
        std::string p1(str_buf.data(), len1);

        merge_ranks_[{p0, p1}] = static_cast<int32_t>(rank);
    }
}

bool BPETokenizer::is_special(int32_t token_id) const {
    return id_to_special_.find(token_id) != id_to_special_.end();
}

std::string BPETokenizer::format_chatml(const std::string& role, const std::string& content) const {
    return "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
}

std::vector<std::string> BPETokenizer::bpe_merge(const std::string& piece) const {
    // Break piece (which is in b2u representation) into individual utf-8 characters
    std::vector<std::string> parts;
    size_t pos = 0;
    while (pos < piece.size()) {
        size_t start = pos;
        decode_utf8_char(piece, pos);
        parts.push_back(piece.substr(start, pos - start));
    }

    while (parts.size() > 1) {
        int32_t min_rank = INT32_MAX;
        int best_idx = -1;

        for (size_t i = 0; i < parts.size() - 1; ++i) {
            auto it = merge_ranks_.find({parts[i], parts[i + 1]});
            if (it != merge_ranks_.end()) {
                if (it->second < min_rank) {
                    min_rank = it->second;
                    best_idx = static_cast<int>(i);
                }
            }
        }

        if (min_rank == INT32_MAX || best_idx == -1) {
            break;
        }

        // Merge parts[best_idx] and parts[best_idx + 1]
        parts[best_idx] += parts[best_idx + 1];
        parts.erase(parts.begin() + best_idx + 1);
    }

    return parts;
}

std::vector<int32_t> BPETokenizer::encode(const std::string& text) const {
    std::vector<int32_t> result;
    if (text.empty()) return result;

    // 1. Split on special tokens first
    size_t i = 0;
    while (i < text.size()) {
        // Check if matching any special token at index i
        bool found_spec = false;
        for (const auto& [spec_str, spec_id] : special_tokens_) {
            if (text.compare(i, spec_str.size(), spec_str) == 0) {
                result.push_back(spec_id);
                i += spec_str.size();
                found_spec = true;
                break;
            }
        }
        if (found_spec) continue;

        // Find end of non-special chunk
        size_t next_spec = text.size();
        for (const auto& [spec_str, _] : special_tokens_) {
            size_t p = text.find(spec_str, i);
            if (p != std::string::npos && p < next_spec) {
                next_spec = p;
            }
        }

        std::string segment = text.substr(i, next_spec - i);
        i = next_spec;

        // 2. Pre-tokenize segment into chunks
        size_t c_idx = 0;
        while (c_idx < segment.size()) {
            size_t chunk_start = c_idx;

            // 1. Check contractions: 's, 't, 're, 've, 'm, 'll, 'd (case insensitive)
            if (segment[c_idx] == '\'') {
                std::string rem = segment.substr(c_idx, 4);
                std::string lower_rem = rem;
                for (auto& ch : lower_rem) ch = static_cast<char>(std::tolower(ch));
                if (lower_rem.substr(0, 2) == "'s" || lower_rem.substr(0, 2) == "'t" ||
                    lower_rem.substr(0, 2) == "'m" || lower_rem.substr(0, 2) == "'d") {
                    c_idx += 2;
                    goto chunk_found;
                } else if (lower_rem.substr(0, 3) == "'re" || lower_rem.substr(0, 3) == "'ve" || lower_rem.substr(0, 3) == "'ll") {
                    c_idx += 3;
                    goto chunk_found;
                }
            }

            {
                size_t p0 = c_idx;
                uint32_t cp0 = decode_utf8_char(segment, p0);

                // 2. Check [^\r\n\p{L}\p{N}]?\p{L}+
                if (is_letter_cp(cp0)) {
                    c_idx = p0;
                    while (c_idx < segment.size()) {
                        size_t p_next = c_idx;
                        uint32_t cp_next = decode_utf8_char(segment, p_next);
                        if (is_letter_cp(cp_next)) {
                            c_idx = p_next;
                        } else {
                            break;
                        }
                    }
                    goto chunk_found;
                } else if (cp0 != '\r' && cp0 != '\n' && !is_digit_cp(cp0)) {
                    // Check if followed immediately by a letter
                    if (p0 < segment.size()) {
                        size_t p1 = p0;
                        uint32_t cp1 = decode_utf8_char(segment, p1);
                        if (is_letter_cp(cp1)) {
                            c_idx = p1;
                            while (c_idx < segment.size()) {
                                size_t p_next = c_idx;
                                uint32_t cp_next = decode_utf8_char(segment, p_next);
                                if (is_letter_cp(cp_next)) {
                                    c_idx = p_next;
                                } else {
                                    break;
                                }
                            }
                            goto chunk_found;
                        }
                    }
                }

                // 3. Single digit (\p{N})
                if (is_digit_cp(cp0)) {
                    c_idx = p0;
                    goto chunk_found;
                }

                // 4. Check  ?[^\s\p{L}\p{N}]+[\r\n]*
                if (cp0 == ' ') {
                    if (p0 < segment.size()) {
                        size_t p1 = p0;
                        uint32_t cp1 = decode_utf8_char(segment, p1);
                        if (!is_space_cp(cp1) && !is_letter_cp(cp1) && !is_digit_cp(cp1)) {
                            c_idx = p1;
                            while (c_idx < segment.size()) {
                                size_t p_next = c_idx;
                                uint32_t cp_next = decode_utf8_char(segment, p_next);
                                if (!is_space_cp(cp_next) && !is_letter_cp(cp_next) && !is_digit_cp(cp_next)) {
                                    c_idx = p_next;
                                } else {
                                    break;
                                }
                            }
                            while (c_idx < segment.size() && (segment[c_idx] == '\r' || segment[c_idx] == '\n')) {
                                c_idx++;
                            }
                            goto chunk_found;
                        }
                    }
                } else if (!is_space_cp(cp0) && !is_letter_cp(cp0) && !is_digit_cp(cp0)) {
                    c_idx = p0;
                    while (c_idx < segment.size()) {
                        size_t p_next = c_idx;
                        uint32_t cp_next = decode_utf8_char(segment, p_next);
                        if (!is_space_cp(cp_next) && !is_letter_cp(cp_next) && !is_digit_cp(cp_next)) {
                            c_idx = p_next;
                        } else {
                            break;
                        }
                    }
                    while (c_idx < segment.size() && (segment[c_idx] == '\r' || segment[c_idx] == '\n')) {
                        c_idx++;
                    }
                    goto chunk_found;
                }

                // 5. Check \s*[\r\n]+
                if (cp0 == '\n' || cp0 == '\r' || (cp0 == ' ' && segment.find_first_not_of(' ', c_idx) != std::string::npos && (segment[segment.find_first_not_of(' ', c_idx)] == '\n' || segment[segment.find_first_not_of(' ', c_idx)] == '\r'))) {
                    c_idx = p0;
                    while (c_idx < segment.size() && (segment[c_idx] == ' ' || segment[c_idx] == '\t')) c_idx++;
                    while (c_idx < segment.size() && (segment[c_idx] == '\r' || segment[c_idx] == '\n')) c_idx++;
                    goto chunk_found;
                }

                // 6. Otherwise: sequence of spaces
                c_idx = p0;
                while (c_idx < segment.size() && segment[c_idx] == ' ') {
                    c_idx++;
                }
                // If followed by non-space non-newline, reserve the last space for the next token
                if (c_idx < segment.size() && segment[c_idx] != '\r' && segment[c_idx] != '\n' && c_idx > chunk_start + 1) {
                    c_idx--;
                }
            }

        chunk_found:
            std::string chunk = segment.substr(chunk_start, c_idx - chunk_start);

            // 3. Map chunk bytes to GPT-2 unicode string
            std::string mapped;
            for (char byte_ch : chunk) {
                uint8_t b = static_cast<uint8_t>(byte_ch);
                auto it = byte_to_unicode_.find(b);
                if (it != byte_to_unicode_.end()) {
                    mapped += it->second;
                }
            }

            // 4. BPE merge
            std::vector<std::string> merged = bpe_merge(mapped);

            // 5. Lookup token IDs
            for (const auto& piece : merged) {
                auto it = vocab_to_id_.find(piece);
                if (it != vocab_to_id_.end()) {
                    result.push_back(it->second);
                } else {
                    // Fallback to byte pieces
                    for (char ch : piece) {
                        std::string s(1, ch);
                        auto b_it = vocab_to_id_.find(s);
                        if (b_it != vocab_to_id_.end()) {
                            result.push_back(b_it->second);
                        }
                    }
                }
            }
        }
    }

    return result;
}

std::string BPETokenizer::decode_token(int32_t token_id) const {
    if (is_special(token_id)) {
        auto it = id_to_special_.find(token_id);
        return (it != id_to_special_.end()) ? it->second : "";
    }
    auto it = id_to_vocab_.find(token_id);
    if (it == id_to_vocab_.end()) return "";

    const std::string& token_str = it->second;
    std::string decoded_bytes;

    size_t i = 0;
    while (i < token_str.size()) {
        size_t start = i;
        decode_utf8_char(token_str, i);
        std::string utf8_char = token_str.substr(start, i - start);
        auto byte_it = unicode_to_byte_.find(utf8_char);
        if (byte_it != unicode_to_byte_.end()) {
            decoded_bytes.push_back(static_cast<char>(byte_it->second));
        } else {
            decoded_bytes += utf8_char;
        }
    }

    return decoded_bytes;
}

std::string BPETokenizer::decode(const std::vector<int32_t>& tokens, bool skip_special) const {
    std::string result;
    for (int32_t tok : tokens) {
        if (skip_special && is_special(tok)) {
            continue;
        }
        result += decode_token(tok);
    }
    return result;
}

} // namespace vectra::tokenizer
