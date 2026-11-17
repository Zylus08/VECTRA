import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')

cases = [
    {"name": "plain_english", "text": "Hello world"},
    {"name": "numbers", "text": "123 456 7890"},
    {"name": "whitespace", "text": "   leading and trailing   "},
    {"name": "punctuation", "text": "Hello, world! How are you doing today?"},
    {"name": "newlines_code", "text": "def add(a: int, b: int) -> int:\n    return a + b\n"},
    {"name": "hindi_multilingual", "text": "नमस्ते दुनिया"},
    {"name": "japanese_multilingual", "text": "こんにちは世界"},
    {"name": "emojis", "text": "🌟🚀🔥🤖"},
    {"name": "chatml_delimiters", "text": "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"},
    {"name": "empty_string", "text": ""},
    {"name": "multi_turn", "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n"}
]

data = []
for c in cases:
    tokens = tok.encode(c["text"])
    decoded = tok.decode(tokens)
    data.append({
        "name": c["name"],
        "text": c["text"],
        "expected_tokens": tokens,
        "decoded": decoded
    })

out_path = "tests/fixtures/tokenizer_parity.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Exported {len(data)} tokenizer parity cases to {out_path}")
