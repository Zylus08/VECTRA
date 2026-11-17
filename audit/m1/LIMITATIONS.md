# Milestone M1 Scope Limitations & Pending Work

## Explicit Non-Goals for M1
In accordance with the Milestone M1 specification, the primary focus was establishing runtime correctness, numerical verification, and model format parsing. Full autoregressive generation of the 1.5-billion parameter model was explicitly out of scope.

## Documented Architectural Limitations

### 1. Full-Model Weight Ingestion
- **Status**: Scaffolding and synthetic weights fully verified on tiny Transformer configurations.
- **Limitation**: The complete 1.5B model weights (~3.1 GB FP16 or ~1.1 GB Q4) have not yet been streamed into RAM.
- **Reason**: Host reconnaissance established a ~2.9 GB free physical RAM ceiling during normal workloads. Loading weights directly without memory-mapping (mmap) would risk thrashing the page file.
- **Resolution Path (M2)**: Implement memory-mapped GGUF tensor loading (`CreateFileMapping` / `MapViewOfFile` on Windows) to allow lazy demand-paging of model weights.

### 2. Quantization Format Specialization
- **Status**: Introductory symmetric per-block 4-bit (`BlockQ4_0`) format fully implemented with scalar and AVX2 direct dot products.
- **Limitation**: GGUF `Q4_K_M` super-block layout (256 elements, 8 sub-blocks, 6-bit scales) is not yet supported.
- **Resolution Path (M2)**: Author dedicated `quant_q4_k.cpp` kernel using AVX2 byte-shuffle (`_mm256_shuffle_epi8`) once reference weights in Q4_K_M are ready.

### 3. Tokenizer Pre-Tokenizer Regex Engine
- **Status**: Subword dictionary lookup, special token isolation, and 256-byte fallback tokens operational.
- **Limitation**: HuggingFace Qwen2 pre-tokenization regex splitting pattern is not fully emulated.
- **Resolution Path (M2)**: Implement a lightweight deterministic finite state tokenizer or embed a compact regex matcher.

### 4. Multithreading
- **Status**: All M1 benchmarks and kernels executed on a single thread (`threads=1`) to isolate per-core vectorization efficiency.
- **Resolution Path (M2)**: Implement thread-pool work-stealing parallelism across the 6 P-cores and 4 E-cores of the i7-12650H.
