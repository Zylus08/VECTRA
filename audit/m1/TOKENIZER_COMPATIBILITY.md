# Tokenizer Compatibility Audit

## Target Model Tokenizer Metadata
- **Base Architecture**: Byte-level Byte-Pair Encoding (BPE)
- **Vocabulary Size**: 151,936 tokens
- **Special Tokens**:
  - BOS / Text Boundary: `<|endoftext|>` (Token ID: 151643)
  - ChatML System/User/Assistant Start: `<|im_start|>` (Token ID: 151644)
  - ChatML Turn End: `<|im_end|>` (Token ID: 151645)
  - Vision/Multimodal Delimiters: `<|vision_start|>`, `<|vision_end|>`, etc.

## M1 Implemented Subsystem
In accordance with M1 requirements, an original tokenizer engine was implemented in C++20:
1. **Dictionary & Subword Mapping**:
   - Bi-directional token-to-ID and ID-to-token lookups.
   - Longest-match greedy subword segmentation.
2. **Special Token Preservation**:
   - Delimiters such as `<|im_start|>` and `<|im_end|>` are identified upfront before subword splitting, preventing corruption of structural markers.
3. **Byte-Level Fallback**:
   - Supports 256 byte-level fallback tokens (`<0x00>` through `<0xFF>`).
   - Ensures arbitrary binary or unmapped UTF-8 sequences can be encoded and decoded without data loss.
4. **ChatML Formatting**:
   - Native `format_chatml(role, content)` utility producing `<|im_start|>{role}\n{content}<|im_end|>\n`.

## Verification & Round-Trip Tests
- Round-trip encode/decode verified on ASCII text, whitespace preservation, and multi-turn ChatML prompt structures.
- Tested in `tests/unit/unit_tests.cpp::test_tokenizer`.

## Unimplemented Features & Scope Boundaries (Reserved for M2)
1. **Complex Regex Pre-Tokenization**:
   - HuggingFace Qwen2 tokenizer applies a detailed Python/Rust regex splitting rule:
     `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
   - Full regex pre-tokenization requires a Unicode property regex engine or pre-compiled finite automaton, scheduled for Milestone M2.
2. **Ranked Merge Table Processing**:
   - Processing the full 150,000+ entry `merges.txt` table in priority order will be integrated when full model weights are loaded in M2.
