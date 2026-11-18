cmake_minimum_required(VERSION 3.20)
project(VECTRA VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VECTRA_ENABLE_AVX2 "Enable AVX2 instruction set optimizations" ON)
option(VECTRA_BUILD_TESTS "Build unit, numerical, and integration tests" ON)
option(VECTRA_BUILD_BENCHMARKS "Build microbenchmarks" ON)
option(VECTRA_BUILD_APPS "Build CLI applications" ON)

if (MSVC)
    add_compile_options(/W4 /permissive- /utf-8 /Zc:preprocessor)
    if (VECTRA_ENABLE_AVX2)
        add_compile_options(/arch:AVX2)
        add_compile_definitions(VECTRA_USE_AVX2=1)
    endif()
    # Optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Ot /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if (VECTRA_ENABLE_AVX2)
        add_compile_options(-mavx2 -mfma -mf16c)
        add_compile_definitions(VECTRA_USE_AVX2=1)
    endif()
endif()

include_directories(include)

# Core library sources
set(VECTRA_SOURCES
    src/core/memory.cpp
    src/core/types.cpp
    src/tensor/tensor.cpp
    src/tensor/views.cpp
    src/math/scalar_math.cpp
    src/math/avx2_math.cpp
    src/math/transformer_ops.cpp
    src/quant/quant_block.cpp
    src/quant/quant_k.cpp
    src/model/kv_cache.cpp
    src/model/transformer.cpp
    src/model/gguf_model.cpp
    src/tokenizer/tokenizer.cpp
    src/tokenizer/bpe_tokenizer.cpp
    src/format/gguf_parser.cpp
    src/format/mmap_loader.cpp
    src/runtime/sampler.cpp
    src/runtime/thread_pool.cpp
)

add_library(vectra_static STATIC ${VECTRA_SOURCES})
target_include_directories(vectra_static PUBLIC include)
if (WIN32)
    target_link_libraries(vectra_static PUBLIC psapi)
endif()

# Applications
if (VECTRA_BUILD_APPS)
    add_executable(vectra_inspect apps/vectra_inspect.cpp)
    target_link_libraries(vectra_inspect PRIVATE vectra_static)

    add_executable(vectra_cli apps/vectra_cli.cpp)
    target_link_libraries(vectra_cli PRIVATE vectra_static)
endif()

# Tests
if (VECTRA_BUILD_TESTS)
    enable_testing()
    
    # M1 Regression tests
    add_executable(vectra_unit_tests tests/unit/unit_tests.cpp)
    target_link_libraries(vectra_unit_tests PRIVATE vectra_static)
    add_test(NAME unit_tests COMMAND vectra_unit_tests)

    add_executable(vectra_numerical_tests tests/numerical/numerical_tests.cpp)
    target_link_libraries(vectra_numerical_tests PRIVATE vectra_static)
    add_test(NAME numerical_tests COMMAND vectra_numerical_tests)

    add_executable(vectra_integration_tests tests/integration/integration_tests.cpp)
    target_link_libraries(vectra_integration_tests PRIVATE vectra_static)
    add_test(NAME integration_tests COMMAND vectra_integration_tests)

    # M2 tests
    add_executable(vectra_tokenizer_tests tests/unit/tokenizer_tests.cpp)
    target_link_libraries(vectra_tokenizer_tests PRIVATE vectra_static)
    add_test(NAME tokenizer_tests COMMAND vectra_tokenizer_tests)

    add_executable(vectra_m2_tests tests/integration/m2_tests.cpp)
    target_link_libraries(vectra_m2_tests PRIVATE vectra_static)
    add_test(NAME m2_tests COMMAND vectra_m2_tests)

    # M3 tests
    add_executable(vectra_m3_tests tests/integration/m3_tests.cpp)
    target_link_libraries(vectra_m3_tests PRIVATE vectra_static)
    add_test(NAME m3_tests COMMAND vectra_m3_tests)

    # M4 tests
    add_executable(vectra_m4_tests tests/integration/m4_tests.cpp)
    target_link_libraries(vectra_m4_tests PRIVATE vectra_static)
    add_test(NAME m4_tests COMMAND vectra_m4_tests)

    # M5 Arithmetic Safety tests
    add_executable(vectra_arithmetic_safety_tests tests/unit/arithmetic_safety_tests.cpp)
    target_link_libraries(vectra_arithmetic_safety_tests PRIVATE vectra_static)
    add_test(NAME arithmetic_safety_tests COMMAND vectra_arithmetic_safety_tests)

    # M5 Integration tests
    add_executable(vectra_m5_tests tests/integration/m5_tests.cpp)
    target_link_libraries(vectra_m5_tests PRIVATE vectra_static)
    add_test(NAME m5_tests COMMAND vectra_m5_tests)

    # Set working directory for all tests to project root
    set_tests_properties(
        unit_tests numerical_tests integration_tests tokenizer_tests
        m2_tests m3_tests m4_tests arithmetic_safety_tests m5_tests
        PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endif()

# Microbenchmarks
if (VECTRA_BUILD_BENCHMARKS)
    add_executable(vectra_benchmarks benchmarks/kernels/benchmark_main.cpp)
    target_link_libraries(vectra_benchmarks PRIVATE vectra_static)

    add_executable(vectra_m3_benchmarks benchmarks/kernels/benchmark_m3.cpp)
    target_link_libraries(vectra_m3_benchmarks PRIVATE vectra_static)

    add_executable(vectra_m4_int8 benchmarks/kernels/benchmark_m4_int8.cpp)
    target_link_libraries(vectra_m4_int8 PRIVATE vectra_static)
endif()
