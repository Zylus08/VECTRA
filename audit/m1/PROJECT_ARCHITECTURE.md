# Project VECTRA: Architecture & Modular Design

## Overview
Project VECTRA is an original, clean-room CPU inference runtime implemented in modern C++20. It delivers high-efficiency tensor operations, transformer attention layers, and block quantization specifically optimized for modern x86_64 CPUs with AVX2 instruction sets.

## Repository Layout
```
E:\VECTRA/
├── CMakeLists.txt              # Unified multi-target CMake build script
├── README.md                   # Project description and quickstart
├── LICENSE                     # MIT open-source license
├── .gitignore                  # Exclusion rules for artifacts, models, caches
│
├── include/vectra/             # Public API headers
│   ├── core/                   # Aligned memory allocator, IEEE 754 half precision, types
│   ├── tensor/                 # Contiguous Tensor and non-owning TensorView classes
│   ├── math/                   # Scalar and AVX2 vector math kernels, transformer ops
│   ├── quant/                  # Introductory symmetric 4-bit block quantization
│   ├── model/                  # Model configurations, KVCache, TransformerModel
│   ├── tokenizer/              # Vocabulary lookup, ChatML formatting, byte fallback
│   └── format/                 # Read-only GGUF file parser and metadata extractor
│
├── src/                        # Implementation units
│   ├── core/                   # types.cpp, memory.cpp
│   ├── tensor/                 # tensor.cpp, views.cpp
│   ├── math/                   # scalar_math.cpp, avx2_math.cpp, transformer_ops.cpp
│   ├── quant/                  # quant_block.cpp
│   ├── model/                  # kv_cache.cpp, transformer.cpp
│   ├── tokenizer/              # tokenizer.cpp
│   └── format/                 # gguf_parser.cpp
│
├── apps/                       # User-facing CLI applications
│   ├── vectra_inspect.cpp      # Standalone GGUF model inspector
│   └── vectra_cli.cpp          # Runtime verification and smoke-test CLI
│
├── tests/                      # Verification and regression test suites
│   ├── unit/                   # Memory alignment, types, views, parser negative tests
│   ├── numerical/              # Numerical equivalence against NumPy/PyTorch oracle
│   ├── integration/            # Multi-token execution, KV cache, causal invariance
│   └── fixtures/               # GGUF valid and corrupt test files
│
├── benchmarks/                 # Performance evaluation
│   ├── kernels/                # Repeatable microbenchmark suite
│   └── results/                # Output artifacts and summaries
│
├── tools/                      # Reference scripts
│   └── reference/              # oracle.py, generate_gguf_fixture.py
│
└── audit/m1/                   # Milestone M1 evidence artifacts and audit documentation
```

## Architectural Decoupling
1. **Separation of Generic Compute from Architecture Metadata**:
   Generic kernels (`dot`, `matvec`, `rms_norm`, `softmax`, `swiglu`, `rope`) have no hardcoded model dimensions. All dimensions (`hidden_dim`, `num_heads`, `head_dim`, `seq_len`) are passed as explicit runtime parameters.
2. **Deterministic Memory Ownership**:
   All dynamic buffer allocations use `AlignedBuffer` with 64-byte alignment (`_aligned_malloc` on Windows). Memory is deterministic, RAII-governed, and strictly non-aliased unless explicitly wrapped in a non-owning `TensorView`.
3. **Dual Execution Engine**:
   Every mathematical operation features a clean-room scalar reference implementation alongside an AVX2 vectorized implementation. CPUID runtime inspection selects AVX2 if supported, falling back gracefully to scalar logic if AVX2 is absent.
