# Project VECTRA — RC1 Final Audit: Licensing, Model Provenance & Attribution

**Release Candidate Target:** `v0.1.0-rc1`  
**Auditor:** Independent LLM Runtime Auditor & Release Engineer  

---

## 1. Project VECTRA Licensing & Clean-Room IP Audit

Project VECTRA is an original, clean-room C++20 CPU runtime implementation:
- **Author & Copyright:** Saksham Mishra (`mishrasaksham080705@gmail.com`).
- **Core Codebase:** Contains 100% original C++20 implementations of tensor storage, BPE tokenization, GGUF parsing, quantized mathematical kernels (AVX2), grouped-query attention, rotary positional embeddings, and thread pool scheduling.
- **Dependency Isolation:** No code, headers, or precompiled binaries are vendored or wrapped from `llama.cpp`, `ggml`, `ONNX Runtime`, `PyTorch`, `TensorFlow`, or `OpenVINO`.

---

## 2. Target Model Provenance & Licensing

The reference model evaluated throughout Project VECTRA is:
- **Model Name:** `Qwen2.5-1.5B-Instruct`
- **Developer / Creator:** Qwen Team, Alibaba Cloud
- **Upstream Architecture:** Transformer decoder with RoPE, GQA, and SwiGLU.
- **Model License:** **Apache License 2.0** (Permissive open-weights license permitting commercial use, research, modification, and local inference).
- **Quantization Variant:** `Q4_K_M` GGUF (Medium 4-bit block quantization with selected 6-bit projections).
- **Verified Model File Hash:**
  - File: `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`
  - Size: `1,117,460,960 bytes` ($1065.69\text{ MB}$)
  - SHA-256: `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`

---

## 3. Format Attribution & Test Oracle Dependencies

1. **GGUF Binary Format Attribution:**
   The GGUF binary container format was designed and published by Georgi Gerganov and the `llama.cpp` open-source community. Project VECTRA implements an independent, read-only clean-room parser (`vectra::format::GGUFParser` and `vectra::format::MMapLoader`) adhering to the published open specification.
2. **Development-Time Test Oracles:**
   During development and milestone verification, `llama.cpp` (release `b4500`) and Hugging Face `transformers` / `numpy` were utilized strictly as external, out-of-process numerical oracles to generate validation fixtures (`tests/fixtures/model_reference_parity.json`).
   Neither `llama.cpp` nor Python libraries are linked or required for running the VECTRA binary.

---

## 4. Upstream Model Acquisition Instructions

To ensure Git repository cleanliness and avoid distributing large model weights, model files are excluded via `.gitignore`. Users can obtain and verify the certified model checkpoint using the following commands:

```pwsh
# 1. Create models directory
mkdir -p models

# 2. Download official Qwen2.5-1.5B-Instruct Q4_K_M GGUF checkpoint from Hugging Face
Invoke-WebRequest -Uri "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" -OutFile "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# 3. Verify SHA-256 checksum
$hash = (Get-FileHash -Algorithm SHA256 "models/qwen2.5-1.5b-instruct-q4_k_m.gguf").Hash.ToLower()
if ($hash -eq "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e") {
    Write-Host "Model checksum verified: OK"
} else {
    Write-Error "Checksum mismatch! Expected 6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e, got $hash"
}
```
