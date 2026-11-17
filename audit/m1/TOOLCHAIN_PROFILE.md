# Toolchain Profile & Build Environment

## Compiler & Linker
- **Primary C++ Compiler**: Microsoft Visual C++ Compiler (MSVC)
- **Version**: 19.43.34809.0 (Visual Studio 2022 Build Tools v17.13.5)
- **Language Standard**: ISO C++20 (`/std:c++20`)
- **Conformance & Flags**:
  - Standard flags: `/W4 /permissive- /utf-8 /Zc:preprocessor`
  - SIMD vectorization: `/arch:AVX2`
  - Release Optimization: `/O2 /Oi /Ot /GL` with Link-Time Code Generation (`/LTCG`)
  - Debug Configuration: `/Od /Zi /RTC1`

## Build System
- **Build Generator**: CMake 3.30.5-msvc23
- **Build Backend**: Ninja 1.12.1
- **Working Directories**:
  - Release Build: `E:\VECTRA\build_release`
  - Debug Build: `E:\VECTRA\build_debug`

## Python Environment & Testing Oracles
- **Interpreter**: Python 3.12.8 (Anaconda distribution at `C:\Users\mishr\anaconda3\python.exe`)
- **Scientific Computing Libraries**:
  - NumPy: 1.26.4
  - PyTorch: 2.15.0+cpu (located in conda environment `reswork`)
- **Role in VECTRA**:
  - Python scripts (`tools/reference/oracle.py` and `tools/reference/generate_gguf_fixture.py`) serve exclusively as independent numerical oracles, generating deterministic reference vectors and format fixtures.
  - The core C++ runtime contains zero dependencies on Python or PyTorch.

## Version Control
- **Git Version**: 2.47.1.windows.1
- **Repository Identity**:
  - Author / Committer Name: `Saksham Mishra`
  - Author / Committer Email: `mishrasaksham080705@gmail.com`
