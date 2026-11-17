import gguf
import numpy as np
import json
import os

def rms_norm(x, weight, eps=1e-6):
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(variance + eps)) * weight

def silu(x):
    return x / (1.0 + np.exp(-x))

def apply_rope(x, pos, head_dim=128, theta=1000000.0):
    # x shape: (..., head_dim)
    half = head_dim // 2
    out = np.zeros_like(x)
    for i in range(half):
        freq = 1.0 / (theta ** (float(i * 2) / float(head_dim)))
        val = pos * freq
        cos_val = np.cos(val)
        sin_val = np.sin(val)
        
        x0 = x[..., i]
        x1 = x[..., i + half]
        out[..., i] = x0 * cos_val - x1 * sin_val
        out[..., i + half] = x0 * sin_val + x1 * cos_val
    return out

def main():
    gguf_path = 'models/qwen2.5-1.5b-instruct-q4_k_m.gguf'
    print(f'Reading {gguf_path} ...')
    reader = gguf.GGUFReader(gguf_path)
    
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = t

    def get_tensor(name):
        t = tensors[name]
        data = gguf.quants.dequantize(t.data, t.tensor_type)
        return data

    prompt_tokens = [785, 6722, 315, 9625, 374]
    print('Prompt tokens:', prompt_tokens)

    # 1. Embeddings - dequantize row-by-row
    print('Loading token_embd row for token 374 ...')
    token_embd_t = tensors['token_embd.weight']
    tok_last = prompt_tokens[-1] # 374 (' is')
    row_bytes = token_embd_t.data[tok_last : tok_last + 1]
    emb_last = gguf.quants.dequantize(row_bytes, token_embd_t.tensor_type)[0]

    # Pre-attention RMSNorm layer 0
    blk0_attn_norm = get_tensor('blk.0.attn_norm.weight')
    normed_last = rms_norm(emb_last, blk0_attn_norm)

    # Q, K, V proj for layer 0
    q_w = get_tensor('blk.0.attn_q.weight') # shape [1536, 1536]
    q_b = get_tensor('blk.0.attn_q.bias')
    k_w = get_tensor('blk.0.attn_k.weight') # shape [256, 1536]
    k_b = get_tensor('blk.0.attn_k.bias')
    v_w = get_tensor('blk.0.attn_v.weight') # shape [256, 1536]
    v_b = get_tensor('blk.0.attn_v.bias')

    q = normed_last @ q_w.T + q_b
    k = normed_last @ k_w.T + k_b
    v = normed_last @ v_w.T + v_b

    # RoPE on Q and K at pos 4
    pos = len(prompt_tokens) - 1
    q_heads = q.reshape(12, 128)
    k_heads = k.reshape(2, 128)
    q_rot = apply_rope(q_heads, pos)
    k_rot = apply_rope(k_heads, pos)

    # Export boundary reference fixture
    fixture = {
        "prompt_tokens": prompt_tokens,
        "token_last": tok_last,
        "pos_last": pos,
        "layer0": {
            "embedding_sample": emb_last[:16].tolist(),
            "embedding_norm": float(np.linalg.norm(emb_last)),
            "normed_sample": normed_last[:16].tolist(),
            "normed_norm": float(np.linalg.norm(normed_last)),
            "q_sample": q[:16].tolist(),
            "q_norm": float(np.linalg.norm(q)),
            "k_sample": k[:16].tolist(),
            "k_norm": float(np.linalg.norm(k)),
            "v_sample": v[:16].tolist(),
            "v_norm": float(np.linalg.norm(v)),
            "q_rot_head0_sample": q_rot[0, :16].tolist(),
            "k_rot_head0_sample": k_rot[0, :16].tolist(),
        },
        "expected_next_token_id": 12095,
        "expected_next_token_str": " Paris"
    }

    out_path = 'tests/fixtures/model_reference_parity.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(fixture, f, indent=2)

    print(f'Wrote reference fixture to {out_path}')

if __name__ == '__main__':
    main()
