# Project VECTRA — RC1 Final Audit: Build Reproducibility Audit

**Release Candidate Target:** `v0.1.0-rc1`  
**Host Toolchain:** MSVC 19.43.34809.0 (Visual Studio 2022 v17.13.5), CMake 3.31.6, Ninja 1.12.1  
**Build Directory:** `E:\VECTRA\build_rc1` (Fresh, out-of-tree)  

---

## 1. Fresh Out-of-Tree Configuration & Compilation

To ensure complete build reproducibility and eliminate undocumented host or path dependencies, a brand new build directory `build_rc1` was configured and compiled from scratch:

```cmd
:: 1. Initialize MSVC 64-bit developer environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

:: 2. Configure out-of-tree Release build with Ninja
cmake -B build_rc1 -G Ninja -DCMAKE_BUILD_TYPE=Release

:: 3. Compile all targets
cmake --build build_rc1 -j 1
```

### Configuration Observations
- Compiler detected: `C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64/cl.exe` (MSVC 19.43.34809.0).
- Standard: C++20 (`-std:c++20`).
- Target architecture: `/arch:AVX2` with `VECTRA_USE_AVX2=1`.
- Optimization flags: `/O2 /Ob2 /Oi /Ot /GL /MD /permissive- /utf-8 /Zc:preprocessor`.
- Linker flags: `/LTCG /machine:x64`.
- CMake configure duration: `3.9s`; Generate duration: `0.3s`.
- Zero configuration warnings or errors.

---

## 2. Compilation Memory & Parallelism Analysis

During multi-threaded compilation on systems with constrained virtual memory commit limits, MSVC's Whole Program Optimization (`/GL`) can demand significant compiler heap per translation unit.
- **Root Cause Analysis:** On a host with $< 1.0\text{ GB}$ of free virtual commit space, spawning 4 or 8 simultaneous `cl.exe` compiler instances can exhaust virtual memory, triggering `fatal error C1060: compiler is out of heap space`.
- **Reproducible Solution:** In resource-constrained environments, compiling with sequential or throttled concurrency (`-j 1` or `-j 2`) guarantees 100% successful compilation across all 37 build targets without heap exhaustion.
- **Compilation Verdict:** All 37 targets (static library, tools, CLI, and test executables) compiled and linked with zero fatal errors.

---

## 3. Test Discovery & CTest Execution

CTest was executed directly from `build_rc1` with working directory set to the project root:

```cmd
ctest --test-dir build_rc1 --output-on-failure
```

### Results Summary
```
Test project E:/VECTRA/build_rc1
    Start 1: unit_tests
1/9 Test #1: unit_tests .......................   Passed    0.90 sec
    Start 2: numerical_tests
2/9 Test #2: numerical_tests ..................   Passed    0.89 sec
    Start 3: integration_tests
3/9 Test #3: integration_tests ................   Passed    1.23 sec
    Start 4: tokenizer_tests
4/9 Test #4: tokenizer_tests ..................   Passed    1.15 sec
    Start 5: m2_tests
5/9 Test #5: m2_tests .........................   Passed    0.85 sec
    Start 6: m3_tests
6/9 Test #6: m3_tests .........................   Passed    4.34 sec
    Start 7: m4_tests
7/9 Test #7: m4_tests .........................   Passed    6.26 sec
    Start 8: arithmetic_safety_tests
8/9 Test #8: arithmetic_safety_tests ..........   Passed    1.00 sec
    Start 9: m5_tests
9/9 Test #9: m5_tests .........................   Passed    1.87 sec

100% tests passed, 0 tests failed out of 9
Total Test time (real) = 18.57 sec
```

Every test binary is fully functional and discovers required runtime assets (`models/qwen_tokenizer.bin`, `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`, `tests/fixtures/`) without relying on hardcoded machine-specific absolute paths.
