# Milestone M2 — Model Provenance & Verification

**Date:** 2026-11-18  
**Model Name:** Qwen2.5-1.5B-Instruct  
**Quantization Target:** Q4_K_M  
**Format:** GGUF v3  

---

## 1. Upstream Artifact Metadata

| Field | Value |
|---|---|
| **Hugging Face Repository** | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` |
| **Model Creator** | Alibaba Cloud / Qwen Team |
| **Official Repository URL** | `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF` |
| **Base Model Architecture** | `qwen2` (Dense Decoder-Only Transformer) |
| **Artifact File** | `qwen2.5-1.5b-instruct-q4_k_m.gguf` |
| **Download URL** | `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf` |
| **License** | Apache 2.0 |
| **Storage Location** | `E:\VECTRA\models\qwen2.5-1.5b-instruct-q4_k_m.gguf` (Git-ignored) |

---

## 2. Cryptographic & File Integrity Verification

| Metric | Measured Value | Verification Status |
|---|---|---|
| **Byte Size** | `1,117,320,736 bytes` (1065.56 MB) | **MATCH** |
| **SHA-256 Digest** | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` | **MATCH (Exact Git-LFS OID)** |
| **GGUF Magic Number** | `0x46554747` (`GGUF`) | **VALID** |
| **GGUF Version** | `3` | **VALID** |
| **Data Alignment** | `32 bytes` | **VALID** |
| **Data Section Offset** | `5,950,496 bytes` | **VALID** |

---

## 3. Architecture Specification

Parsed directly from GGUF metadata by `vectra_inspect.exe`:
- **Embedding Length ($d_{\text{model}}$):** 1536
- **Transformer Block Count ($L$):** 28
- **Attention Query Heads ($H_q$):** 12
- **Key/Value Heads ($H_{kv}$):** 2
- **Head Dimension ($d_k$):** 128 ($1536 / 12$)
- **Grouped-Query Ratio ($G$):** 6:1 ($12 / 2$)
- **Feed-Forward Hidden Dimension ($d_{ffn}$):** 8960
- **Context Length ($S_{\text{max}}$):** 32768
- **RoPE Frequency Base ($\theta$):** 1,000,000.0
- **RMSNorm Epsilon ($\epsilon$):** 1e-6
- **Vocabulary Size ($V$):** 151,936
- **Output Weight Sharing:** Untied (`output.weight` present, [1536, 151936], Q6_K)
- **Special Tokens:**
  - `<|endoftext|>`: `151643`
  - `<|im_start|>`: `151644`
  - `<|im_end|>`: `151645`
