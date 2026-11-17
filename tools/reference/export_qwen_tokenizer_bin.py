import json
import struct
import os
from transformers import AutoTokenizer

print("Loading Qwen tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
raw = json.loads(tok.backend_tokenizer.to_str())

vocab = raw['model']['vocab']
merges = raw['model']['merges']
added = raw.get('added_tokens', [])

out_dir = "models"
os.makedirs(out_dir, exist_ok=True)
bin_path = os.path.join(out_dir, "qwen_tokenizer.bin")

with open(bin_path, "wb") as f:
    # Magic 'VTOK'
    f.write(struct.pack("<I", 0x56544F4B))
    
    # Vocab items: regular + added
    # Create combined vocab dict
    all_vocab = dict(vocab)
    for at in added:
        all_vocab[at['content']] = at['id']
        
    f.write(struct.pack("<I", len(all_vocab)))
    for token_str, token_id in all_vocab.items():
        encoded = token_str.encode('utf-8')
        f.write(struct.pack("<H", len(encoded)))
        f.write(encoded)
        f.write(struct.pack("<i", token_id))
        
    # Merges
    f.write(struct.pack("<I", len(merges)))
    for rank, pair in enumerate(merges):
        if isinstance(pair, list):
            p0, p1 = pair[0], pair[1]
        else:
            p0, p1 = pair.split(' ', 1)
        enc0 = p0.encode('utf-8')
        enc1 = p1.encode('utf-8')
        f.write(struct.pack("<I", rank))
        f.write(struct.pack("<H", len(enc0)))
        f.write(enc0)
        f.write(struct.pack("<H", len(enc1)))
        f.write(enc1)

sz = os.path.getsize(bin_path)
print(f"Exported binary tokenizer to {bin_path} ({sz / (1024*1024):.2f} MB, {len(all_vocab)} tokens, {len(merges)} merges)")
