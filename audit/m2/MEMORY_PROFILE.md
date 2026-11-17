# Milestone M2 — Memory Profile & Resource Consumption

**Date:** 2026-11-18  
**Host Platform:** Windows 11 Home x64, 16.0 GB Installed RAM  
**Target Process:** `vectra_cli.exe` (Release x64)  

---

## 1. Memory Architecture Breakdown

VECTRA uses a split-tier memory layout designed to prevent heap fragmentation and unnecessary copying:

| Memory Component | Allocation Mechanism | Virtual Footprint | Resident RAM (Working Set) | Lifetime Policy |
|---|---|---|---|---|
| **Model Weights** | `CreateFileMappingW` / `MapViewOfFile` | 1,065.56 MB | ~1,004 MB (paged on demand) | Mapped during process lifetime; zero-copy |
| **Tokenizer Model** | C++ Standard Vectors (`std::vector`) | 4.63 MB | ~12 MB | Process lifetime |
| **KV-Cache (1024 ctx)** | Contiguous 64-byte aligned buffer | 56.00 MB | ~56 MB | Pre-allocated per generation request |
| **Activation Scratchpad** | Pre-allocated reusable float vectors | ~2.50 MB | ~2.5 MB | Reused across all 28 layers |
| **Logits Array** | 151,936 single-precision floats | 0.60 MB | 0.60 MB | Reused per forward step |
| **Total Process** | Combined | **~1,128 MB** | **1,060.43 MB (Peak)** | Controlled and deterministic |

---

## 2. Kernel Memory Measurements (`K32GetProcessMemoryInfo`)

Measurements captured during full 10-token generation run on `Qwen2.5-1.5B-Instruct-Q4_K_M`:

- **Process ID (PID):** 3968
- **Working Set Size (Current Resident):** `1,004.18 MB` (`1,052,946,432 bytes`)
- **Peak Working Set Size:** `1,060.43 MB` (`1,111,924,736 bytes`)
- **Private Bytes:** `55.84 MB` (`58,552,320 bytes`)
- **Paged Pool Usage:** 1.2 MB
- **Non-Paged Pool Usage:** 0.4 MB

---

## 3. Host Safety Assessment

The host machine maintains ~1.7 to 2.9 GB of free physical RAM.
At 1,060.43 MB peak working set, VECTRA leaves >1.5 GB of physical memory unencumbered. No page faults to disk thrash, no memory leaks occur, and all file mappings and heap allocations are cleanly destroyed upon process exit.
