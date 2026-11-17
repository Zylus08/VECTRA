# Project VECTRA

High-Performance C++20 CPU Inference Engine for Transformer Language Models.

## Milestone M1: Runtime Foundation, Model Format & Numerical Verification

Project VECTRA is an original, clean-room C++20 CPU runtime designed for efficient local execution of transformer models on modern x86_64 architectures with AVX2 instruction acceleration.

### Features in M1
- **Tensor Runtime**: Contiguous N-dimensional tensor engine with explicit ownership, 64-byte aligned allocation, sub-tensor views, and checked arithmetic.
- **Data Types**: FP32 computation with FP16 storage conversion and signed integer indices.
- **Core Math Kernels**: Scalar and AVX2-accelerated implementations of RMSNorm, Softmax, SiLU, SwiGLU, dot product, and matrix-vector multiplication.
- **Transformer Operators**: Grouped-query attention (GQA), causal masking, rotary positional embeddings (RoPE), residual connections, and persistent KV-cache management.
- **Quantization**: Block-quantized 4-bit representation with symmetric per-block scaling (32 values per block) and direct vector dot products.
- **Model Format**: Read-only GGUF format parser with header parsing, metadata extraction, tensor table verification, and negative bounds checking.
- **Tokenizer**: Byte-level vocabulary mapping with special token handling and UTF-8 decoding.
- **Numerical Verification**: Python oracle comparison framework establishing strict error tolerances across all mathematical kernels.

### System Requirements
- Platform: Windows 11 x64
- Toolchain: MSVC (C++20 compliant) & CMake 3.25+
- Hardware: x86_64 CPU with AVX2 / FMA support
