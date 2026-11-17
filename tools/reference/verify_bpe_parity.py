import json
import regex as re
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
raw = json.loads(tok.backend_tokenizer.to_str())

vocab = raw['model']['vocab']
merges = raw['model']['merges']
merge_ranks = {}
for rank, pair in enumerate(merges):
    if isinstance(pair, list):
        merge_ranks[(pair[0], pair[1])] = rank
    else:
        p = pair.split(' ', 1)
        merge_ranks[(p[0], p[1])] = rank

def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

b2u = bytes_to_unicode()

# Compile pre-tokenizer regex
split_regex = re.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")

special_tokens = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|endoftext|>": 151643
}

def bpe(piece):
    # piece is a list of characters (from b2u)
    parts = list(piece)
    while len(parts) > 1:
        # find pair with lowest rank
        min_rank = float('inf')
        best_pair = None
        best_idx = -1
        for i in range(len(parts) - 1):
            pair = (parts[i], parts[i+1])
            rank = merge_ranks.get(pair, float('inf'))
            if rank < min_rank:
                min_rank = rank
                best_pair = pair
                best_idx = i
        if min_rank == float('inf'):
            break
        # merge
        new_parts = parts[:best_idx] + [parts[best_idx] + parts[best_idx+1]] + parts[best_idx+2:]
        parts = new_parts
    return parts

def encode(text):
    tokens = []
    # 1. Split on special tokens
    pattern = "(<\\|im_start\\|>|<\\|im_end\\|>|<\\|endoftext\\|>)"
    segments = re.split(pattern, text)
    for seg in segments:
        if not seg: continue
        if seg in special_tokens:
            tokens.append(special_tokens[seg])
            continue
        # Pre-tokenize with regex
        chunks = split_regex.findall(seg)
        for chunk in chunks:
            # Map bytes to unicode
            mapped = "".join(b2u[b] for b in chunk.encode('utf-8'))
            merged_parts = bpe(mapped)
            for p in merged_parts:
                tokens.append(vocab[p])
    return tokens

# Verify against fixtures
with open('tests/fixtures/tokenizer_parity.json', 'r', encoding='utf-8') as f:
    fixtures = json.load(f)

all_pass = True
for c in fixtures:
    my_tokens = encode(c['text'])
    ref_tokens = c['expected_tokens']
    match = (my_tokens == ref_tokens)
    print(f"Case {c['name']:25s}: {'PASS' if match else 'FAIL'}")
    if not match:
        all_pass = False
        print(f"  Expected: {ref_tokens}")
        print(f"  Got:      {my_tokens}")

print("All tests passed:", all_pass)
