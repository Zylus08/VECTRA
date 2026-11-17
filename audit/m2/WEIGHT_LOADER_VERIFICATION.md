# Milestone M2 — Memory-Mapped Weight Loader Verification

**Date:** 2026-11-18  
**Module:** `vectra::format::MemoryMappedFile`  
**Location:** `include/vectra/format/mmap_loader.hpp` & `src/format/mmap_loader.cpp`  

---

## 1. Design & Win32 Implementation

To meet the constraint of executing 1.5B parameter models within host RAM (~2.9 GB free) without allocating multi-gigabyte heap copies, VECTRA implements a zero-copy Windows memory-mapped file abstraction:

1. **File Opening:**
   Uses `CreateFileW` with flags `GENERIC_READ`, `FILE_SHARE_READ`, `OPEN_EXISTING`, and `FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS`.
2. **File Mapping Object:**
   Calls `CreateFileMappingW(file_handle, NULL, PAGE_READONLY, 0, 0, NULL)` to back virtual memory addresses directly with the GGUF file on storage drive `E:`.
3. **Virtual Address Space Mapping:**
   Invokes `MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0)` to map the entire 1.06 GB file into contiguous 64-bit user address space.
4. **Deterministic RAII Cleanup:**
   The destructor releases resources in strict reverse order:
   - `UnmapViewOfFile(base_address)`
   - `CloseHandle(mapping_handle)`
   - `CloseHandle(file_handle)`
5. **Pointer Validation & Bounds Checking:**
   `get_pointer_at(offset, required_bytes)` validates that `offset + required_bytes <= file_size_` using 64-bit overflow-safe arithmetic, throwing a `VectraException` on truncation or invalid offsets.

---

## 2. Process Memory Profiling (`GetProcessMemoryInfo`)

VECTRA queries the Windows kernel directly via `K32GetProcessMemoryInfo`:

```cpp
ProcessMemoryStats get_process_memory_stats() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return { pmc.WorkingSetSize, pmc.PeakWorkingSetSize, pmc.PrivateUsage };
    }
    return {0, 0, 0};
}
```

### Measured Runtime Memory Footprint:
- **Mapped File Virtual Size:** 1,117,320,736 bytes (1,065.56 MB)
- **Time to Map File:** **18.8 ms**
- **Process Working Set (Resident Physical RAM):** **1,004.18 MB**
- **Peak Working Set:** **1,060.43 MB**
- **Private Working Set:** < 60 MB (exclusively for activations, KV-cache, and logits buffer)

### Zero-Copy Validation:
Weights are accessed directly from the memory-mapped pointer without intermediary allocations. Individual rows (`token_embd`, `attn_norm`, attention projections) are streamed or dequantized on-the-fly into reusable scratch buffers, keeping heap consumption negligible.
