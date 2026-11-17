import numpy as np
import json
import struct
import math
import sys

def silu(x):
    return x / (1.0 + np.exp(-x))

def swiglu(gate, up):
    return silu(gate) * up

def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return (x / rms) * weight

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def rope(vec, pos, head_dim, rope_theta=1000000.0):
    half = head_dim // 2
    freqs = 1.0 / (rope_theta ** (2.0 * np.arange(half) / head_dim))
    angles = pos * freqs
    cos_val = np.cos(angles)
    sin_val = np.sin(angles)
    
    out = np.zeros_like(vec)
    out[:half] = vec[:half] * cos_val - vec[half:] * sin_val
    out[half:] = vec[:half] * sin_val + vec[half:] * cos_val
    return out

def gqa_attention(q, k_cache, v_cache, num_heads, num_kv_heads, head_dim, seq_len, current_pos):
    group_size = num_heads // num_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    attn_out = np.zeros((num_heads, head_dim), dtype=np.float32)
    
    for h in range(num_heads):
        kv_h = h // group_size
        q_head = q[h]
        scores = np.zeros(seq_len, dtype=np.float32)
        for t in range(seq_len):
            if t > current_pos:
                scores[t] = -1e9
            else:
                k_vec = k_cache[t, kv_h]
                scores[t] = np.dot(q_head, k_vec) * scale
        weights = softmax(scores)
        head_accum = np.zeros(head_dim, dtype=np.float32)
        for t in range(seq_len):
            if t <= current_pos:
                head_accum += weights[t] * v_cache[t, kv_h]
        attn_out[h] = head_accum
    return attn_out

def generate_test_vectors(output_path):
    np.random.seed(42)
    vectors = {}
    
    # 1. Dot product
    n = 128
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    dot_expected = float(np.dot(a, b))
    vectors["dot"] = {"a": a.tolist(), "b": b.tolist(), "expected": dot_expected}
    
    # 2. Matvec
    rows, cols = 32, 64
    A = np.random.randn(rows, cols).astype(np.float32)
    x = np.random.randn(cols).astype(np.float32)
    bias = np.random.randn(rows).astype(np.float32)
    y_expected = (np.dot(A, x) + bias).astype(np.float32)
    vectors["matvec"] = {
        "A": A.flatten().tolist(), "x": x.tolist(), "bias": bias.tolist(),
        "rows": rows, "cols": cols, "expected": y_expected.tolist()
    }
    
    # 3. RMSNorm
    dim = 64
    x_norm = np.random.randn(dim).astype(np.float32)
    w_norm = np.random.uniform(0.5, 1.5, dim).astype(np.float32)
    eps = 1e-6
    norm_expected = rms_norm(x_norm, w_norm, eps).astype(np.float32)
    vectors["rms_norm"] = {
        "x": x_norm.tolist(), "weight": w_norm.tolist(), "eps": eps,
        "expected": norm_expected.tolist()
    }
    
    # 4. Softmax
    dim = 64
    x_soft = np.random.randn(dim).astype(np.float32)
    soft_expected = softmax(x_soft).astype(np.float32)
    vectors["softmax"] = {"x": x_soft.tolist(), "expected": soft_expected.tolist()}
    
    # 5. SiLU & SwiGLU
    dim = 64
    gate = np.random.randn(dim).astype(np.float32)
    up = np.random.randn(dim).astype(np.float32)
    swiglu_expected = swiglu(gate, up).astype(np.float32)
    vectors["swiglu"] = {"gate": gate.tolist(), "up": up.tolist(), "expected": swiglu_expected.tolist()}
    
    # 6. RoPE
    head_dim = 16
    pos = 7
    rope_vec = np.random.randn(head_dim).astype(np.float32)
    rope_expected = rope(rope_vec, pos, head_dim).astype(np.float32)
    vectors["rope"] = {
        "vec": rope_vec.tolist(), "pos": pos, "head_dim": head_dim,
        "expected": rope_expected.tolist()
    }
    
    # 7. GQA
    num_heads = 4
    num_kv_heads = 2
    seq_len = 5
    current_pos = 3
    q = np.random.randn(num_heads, head_dim).astype(np.float32)
    k_cache = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
    v_cache = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32)
    gqa_expected = gqa_attention(q, k_cache, v_cache, num_heads, num_kv_heads, head_dim, seq_len, current_pos)
    vectors["gqa"] = {
        "q": q.flatten().tolist(),
        "k_cache": k_cache.flatten().tolist(),
        "v_cache": v_cache.flatten().tolist(),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "current_pos": current_pos,
        "expected": gqa_expected.flatten().tolist()
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vectors, f, indent=2)
    print(f"Generated numerical test vectors at {output_path}")

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "tests/numerical/oracle_vectors.json"
    generate_test_vectors(out_file)
