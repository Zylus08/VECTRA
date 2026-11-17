import subprocess
import re
import csv
import json

def run_vectra(threads, prompt="The capital of France is", max_tokens=5):
    cmd = [
        "build_release\\vectra_cli.exe",
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--threads", str(threads),
        "--temp", "0.0"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout
    
    prefill_tok_s = 0.0
    decode_tok_s = 0.0
    prefill_ms = 0.0
    decode_ms = 0.0
    total_ms = 0.0
    peak_mb = 0.0
    
    m_prefill = re.search(r"Prefill Latency:\s+([\d\.]+)\s+ms\s+\(([\d\.]+)\s+tok/s\)", out)
    if m_prefill:
        prefill_ms = float(m_prefill.group(1))
        prefill_tok_s = float(m_prefill.group(2))
        
    m_decode = re.search(r"Decode Latency:\s+([\d\.]+)\s+ms\s+\(([\d\.]+)\s+tok/s\)", out)
    if m_decode:
        decode_ms = float(m_decode.group(1))
        decode_tok_s = float(m_decode.group(2))

    m_total = re.search(r"Total Inference:\s+([\d\.]+)\s+ms", out)
    if m_total:
        total_ms = float(m_total.group(1))

    m_mem = re.search(r"Peak Working Set:\s+([\d\.]+)\s+MB", out)
    if m_mem:
        peak_mb = float(m_mem.group(1))

    m_tokens = re.search(r"Generated Token IDs:\s+\[([\d,\s]+)\]", out)
    token_ids = m_tokens.group(1).strip() if m_tokens else ""

    return {
        "threads": threads,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": prefill_tok_s,
        "decode_ms": decode_ms,
        "decode_tok_s": decode_tok_s,
        "total_ms": total_ms,
        "peak_mb": peak_mb,
        "token_ids": token_ids
    }

def main():
    thread_configs = [1, 2, 4, 6, 8, 10, 12, 16]
    results = []
    print("Benchmarking Thread Scaling on Intel Core i7-12650H (6P + 4E cores)...")
    print(f"{'Threads':<10} {'Prefill ms':<12} {'Prefill tok/s':<15} {'Decode ms':<12} {'Decode tok/s':<15} {'Total ms':<12}")
    print("-" * 76)
    
    for t in thread_configs:
        res = run_vectra(t, max_tokens=5)
        results.append(res)
        print(f"{res['threads']:<10} {res['prefill_ms']:<12.1f} {res['prefill_tok_s']:<15.2f} {res['decode_ms']:<12.1f} {res['decode_tok_s']:<15.2f} {res['total_ms']:<12.1f}")

    with open("audit/m3/thread_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
