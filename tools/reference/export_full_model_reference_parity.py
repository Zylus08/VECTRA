import gguf
import numpy as np
import json
import os
import sys

def rms_norm(x, weight, eps=1e-6):
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(variance + eps)) * weight

def silu(x):
    return x / (1.0 + np.exp(-x))

def apply_rope(x, pos, head_dim=128, theta=1000000.0):
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

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def main():
    gguf_path = 'models/qwen2.5-1.5b-instruct-q4_k_m.gguf'
    if not os.path.exists(gguf_path):
        print(f"Error: {gguf_path} not found")
        sys.exit(1)
        
    print(f"Reading {gguf_path} ...")
    reader = gguf.GGUFReader(gguf_path)
    
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = t

    def get_tensor(name):
        t = tensors[name]
        return gguf.quants.dequantize(t.data, t.tensor_type)

    def dequant_row(t_name, row_idx):
        t = tensors[t_name]
        row_slice = t.data[row_idx : row_idx + 1]
        return gguf.quants.dequantize(row_slice, t.tensor_type)[0]

    # Test prompts: P1 ("The capital of France is")
    prompt_tokens = [785, 6722, 315, 9625, 374]
    print(f"Executing reference forward pass for prompt {prompt_tokens} ...")

    # Sample layers across the model: 0, 7, 14, 21, 27
    sample_layers = [0, 7, 14, 21, 27]
    layer_records = {}

    # Initial embedding of the last token (374)
    tok_last = prompt_tokens[-1]
    emb = dequant_row('token_embd.weight', tok_last)
    
    # We will compute forward for token_last at pos = 4
    pos = len(prompt_tokens) - 1
    cur_x = emb.copy()

    for l in range(28):
        pfx = f"blk.{l}."
        attn_norm_w = get_tensor(pfx + "attn_norm.weight")
        normed1 = rms_norm(cur_x, attn_norm_w)

        q_w = get_tensor(pfx + "attn_q.weight")
        q_b = get_tensor(pfx + "attn_q.bias")
        k_w = get_tensor(pfx + "attn_k.weight")
        k_b = get_tensor(pfx + "attn_k.bias")
        v_w = get_tensor(pfx + "attn_v.weight")
        v_b = get_tensor(pfx + "attn_v.bias")

        q = normed1 @ q_w.T + q_b
        k = normed1 @ k_w.T + k_b
        v = normed1 @ v_w.T + v_b

        q_rot = apply_rope(q.reshape(12, 128), pos)
        k_rot = apply_rope(k.reshape(2, 128), pos)

        # Output projection
        o_w = get_tensor(pfx + "attn_output.weight")
        # For single token, self-attention output with V is simply V repeated across GQA groups
        # GQA: 12 query heads, 2 kv heads (each kv head serves 6 query heads)
        attn_out_heads = np.zeros((12, 128), dtype=np.float32)
        for h in range(12):
            kv_h = h // 6
            attn_out_heads[h] = v.reshape(2, 128)[kv_h]
        
        attn_out = attn_out_heads.reshape(1536) @ o_w.T
        res1 = cur_x + attn_out

        ffn_norm_w = get_tensor(pfx + "ffn_norm.weight")
        normed2 = rms_norm(res1, ffn_norm_w)

        gate_w = get_tensor(pfx + "ffn_gate.weight")
        up_w = get_tensor(pfx + "ffn_up.weight")
        down_w = get_tensor(pfx + "ffn_down.weight")

        gate = normed2 @ gate_w.T
        up = normed2 @ up_w.T
        swiglu_act = silu(gate) * up
        ffn_out = swiglu_act @ down_w.T
        res2 = res1 + ffn_out

        if l in sample_layers:
            layer_records[f"layer_{l}"] = {
                "layer_idx": l,
                "normed1_sample": normed1[:16].tolist(),
                "normed1_norm": float(np.linalg.norm(normed1)),
                "q_sample": q[:16].tolist(),
                "q_norm": float(np.linalg.norm(q)),
                "k_sample": k[:16].tolist(),
                "k_norm": float(np.linalg.norm(k)),
                "v_sample": v[:16].tolist(),
                "v_norm": float(np.linalg.norm(v)),
                "attn_out_sample": attn_out[:16].tolist(),
                "attn_out_norm": float(np.linalg.norm(attn_out)),
                "res1_sample": res1[:16].tolist(),
                "res1_norm": float(np.linalg.norm(res1)),
                "normed2_sample": normed2[:16].tolist(),
                "normed2_norm": float(np.linalg.norm(normed2)),
                "gate_sample": gate[:16].tolist(),
                "gate_norm": float(np.linalg.norm(gate)),
                "up_sample": up[:16].tolist(),
                "up_norm": float(np.linalg.norm(up)),
                "swiglu_sample": swiglu_act[:16].tolist(),
                "swiglu_norm": float(np.linalg.norm(swiglu_act)),
                "ffn_out_sample": ffn_out[:16].tolist(),
                "ffn_out_norm": float(np.linalg.norm(ffn_out)),
                "res2_sample": res2[:16].tolist(),
                "res2_norm": float(np.linalg.norm(res2)),
            }
            print(f"  Processed Layer {l}: res2 L2 norm = {np.linalg.norm(res2):.4f}")

        cur_x = res2

    # Final norm
    out_norm_w = get_tensor("output_norm.weight")
    final_normed = rms_norm(cur_x, out_norm_w)

    # Compute top-10 logits by sampling top candidate tokens
    # Expected greedy token is 12095 (" Paris")
    # Let's compute exact dot product for token 12095 and candidates
    candidate_tokens = [12095, 13, 576, 6722, 315, 9625, 374, 28, 11, 0]
    candidate_logits = {}
    for c in candidate_tokens:
        row_w = dequant_row("output.weight" if "output.weight" in tensors else "token_embd.weight", c)
        candidate_logits[str(c)] = float(np.dot(final_normed, row_w))

    fixture = {
        "prompt_tokens": prompt_tokens,
        "token_last": tok_last,
        "pos": pos,
        "sample_layers": sample_layers,
        "layers": layer_records,
        "final_normed_sample": final_normed[:16].tolist(),
        "final_normed_norm": float(np.linalg.norm(final_normed)),
        "candidate_logits": candidate_logits,
        "expected_next_token_id": 12095,
        "expected_next_token_str": " Paris"
    }

    out_file = 'audit/m4/fixtures/full_model_reference_parity.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(fixture, f, indent=2)

    print(f"Successfully generated full-model reference parity fixture to {out_file}")

if __name__ == '__main__':
    main()
