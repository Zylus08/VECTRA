#include "vectra/tokenizer/bpe_tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

namespace {

struct TestCase {
    std::string name;
    std::string text;
    std::vector<int32_t> expected_tokens;
    std::string decoded;
};

std::vector<TestCase> load_parity_fixtures(const std::string& path) {
    std::vector<TestCase> cases;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: Could not open " << path << "\n";
        return cases;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Simple manual parser for the JSON fixtures
    size_t pos = 0;
    while ((pos = content.find("\"name\":", pos)) != std::string::npos) {
        TestCase tc;
        // name
        size_t n_start = content.find('"', pos + 7) + 1;
        size_t n_end = content.find('"', n_start);
        tc.name = content.substr(n_start, n_end - n_start);

        // text
        size_t t_key = content.find("\"text\":", n_end);
        size_t t_start = content.find('"', t_key + 7) + 1;
        size_t t_end = t_start;
        while (t_end < content.size()) {
            if (content[t_end] == '"' && content[t_end - 1] != '\\') break;
            t_end++;
        }
        std::string raw_text = content.substr(t_start, t_end - t_start);
        // unescape \n
        std::string text;
        for (size_t i = 0; i < raw_text.size(); ++i) {
            if (raw_text[i] == '\\' && i + 1 < raw_text.size()) {
                if (raw_text[i + 1] == 'n') { text.push_back('\n'); i++; }
                else if (raw_text[i + 1] == '"') { text.push_back('"'); i++; }
                else if (raw_text[i + 1] == '\\') { text.push_back('\\'); i++; }
                else text.push_back(raw_text[i]);
            } else {
                text.push_back(raw_text[i]);
            }
        }
        tc.text = text;

        // expected_tokens
        size_t tok_key = content.find("\"expected_tokens\":", t_end);
        size_t arr_start = content.find('[', tok_key);
        size_t arr_end = content.find(']', arr_start);
        std::string arr_str = content.substr(arr_start + 1, arr_end - arr_start - 1);
        std::stringstream ss(arr_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            while (!item.empty() && (item.front() == ' ' || item.front() == '\n' || item.front() == '\r')) item.erase(0, 1);
            while (!item.empty() && (item.back() == ' ' || item.back() == '\n' || item.back() == '\r')) item.pop_back();
            if (!item.empty()) {
                tc.expected_tokens.push_back(std::stoi(item));
            }
        }

        cases.push_back(tc);
        pos = arr_end;
    }
    return cases;
}

} // anonymous namespace

int main() {
    std::cout << "=========================================================\n";
    std::cout << "        PROJECT VECTRA - TOKENIZER PARITY TESTS         \n";
    std::cout << "=========================================================\n";

    vectra::tokenizer::BPETokenizer tok;
    try {
        tok.load_binary("models/qwen_tokenizer.bin");
        std::cout << "Loaded Qwen tokenizer: vocab_size=" << tok.vocab_size() << "\n\n";
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load tokenizer binary: " << ex.what() << "\n";
        return 1;
    }

    auto test_cases = load_parity_fixtures("tests/fixtures/tokenizer_parity.json");
    if (test_cases.empty()) {
        std::cerr << "No test cases loaded from tests/fixtures/tokenizer_parity.json!\n";
        return 1;
    }

    int passed = 0;
    int total = static_cast<int>(test_cases.size());

    for (const auto& tc : test_cases) {
        std::cout << "[RUN]  Case: " << tc.name << " ... ";
        auto tokens = tok.encode(tc.text);

        bool match = (tokens == tc.expected_tokens);
        if (match) {
            // Also test round trip decode
            std::string decoded = tok.decode(tokens);
            if (decoded == tc.text) {
                passed++;
                std::cout << "PASSED (Tokens: " << tokens.size() << ", Exact Match)\n";
            } else {
                std::cout << "FAILED (Decode mismatch)\n";
                std::cout << "  Expected: " << tc.text << "\n";
                std::cout << "  Decoded:  " << decoded << "\n";
            }
        } else {
            std::cout << "FAILED (Token ID mismatch)\n";
            std::cout << "  Expected: ";
            for (auto id : tc.expected_tokens) std::cout << id << " ";
            std::cout << "\n  Got:      ";
            for (auto id : tokens) std::cout << id << " ";
            std::cout << "\n";
        }
    }

    std::cout << "\nTokenizer Test Summary: " << passed << " / " << total << " passed.\n";
    return (passed == total) ? 0 : 1;
}
