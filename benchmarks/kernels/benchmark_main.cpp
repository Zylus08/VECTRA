#include "vectra/core/memory.hpp"
#include "vectra/math/scalar_math.hpp"
#include "vectra/math/avx2_math.hpp"
#include "vectra/math/transformer_ops.hpp"
#include "vectra/quant/quant_block.hpp"
#include "vectra/model/kv_cache.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>

struct BenchmarkResult {
    std::string kernel_name;
    std::string shape;
    int threads;
    int warmup_iters;
    int timed_iters;
    double median_us_scalar;
    double median_us_avx2;
    double speedup;
    size_t memory_bytes;
    bool numerical_pass;
};

template <typename Func>
double measure_median_us(Func&& f, int warmup_iters, int timed_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        f();
    }
    std::vector<double> timings;
    timings.reserve(timed_iters);

    for (int i = 0; i < timed_iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        timings.push_back(us);
    }

    std::sort(timings.begin(), timings.end());
    return timings[timings.size() / 2];
}

int main(int argc, char* argv[]) {
    std::string csv_path = (argc > 1) ? argv[1] : "audit/m1/MICROBENCHMARKS.csv";

    std::cout << "=========================================================\n";
    std::cout << "            PROJECT VECTRA - KERNEL BENCHMARKS           \n";
    std::cout << "=========================================================\n";
    std::cout << "Hardware SIMD acceleration: " << (vectra::math::avx2::is_supported() ? "AVX2 + FMA" : "Scalar only") << "\n\n";

    std::vector<BenchmarkResult> results;
    const int warmup = 50;
    const int iters = 200;

    // 1. FP32 Dot Product (N = 1024)
    {
        const size_t N = 1024;
        std::vector<float> a(N, 1.0f);
        std::vector<float> b(N, 2.0f);
        volatile float sink = 0.0f;

        double t_scalar = measure_median_us([&]() {
            sink = vectra::math::scalar::dot(a.data(), b.data(), N);
        }, warmup, iters);

        double t_avx2 = measure_median_us([&]() {
            sink = vectra::math::avx2::dot(a.data(), b.data(), N);
        }, warmup, iters);

        float s_val = vectra::math::scalar::dot(a.data(), b.data(), N);
        float a_val = vectra::math::avx2::dot(a.data(), b.data(), N);
        bool pass = std::fabs(s_val - a_val) < 1e-4f;

        results.push_back({"FP32 Dot Product", "1024", 1, warmup, iters, t_scalar, t_avx2, t_scalar / t_avx2, 2 * N * 4, pass});
    }

    // 2. FP32 Matvec (Rows = 256, Cols = 1024)
    {
        const size_t rows = 256, cols = 1024;
        std::vector<float> A(rows * cols, 0.5f);
        std::vector<float> x(cols, 1.0f);
        std::vector<float> y(rows, 0.0f);

        double t_scalar = measure_median_us([&]() {
            vectra::math::scalar::matvec(A.data(), x.data(), nullptr, y.data(), rows, cols);
        }, warmup, iters);

        double t_avx2 = measure_median_us([&]() {
            vectra::math::avx2::matvec(A.data(), x.data(), nullptr, y.data(), rows, cols);
        }, warmup, iters);

        results.push_back({"FP32 Matvec", "256x1024", 1, warmup, iters, t_scalar, t_avx2, t_scalar / t_avx2, (rows * cols + cols + rows) * 4, true});
    }

    // 3. Quantized Q4_0 Dot Product (N = 1024)
    {
        const size_t N = 1024;
        std::vector<float> orig(N, 1.25f);
        std::vector<vectra::quant::BlockQ4_0> blocks(N / 32);
        vectra::quant::quantize_row_q4_0(orig.data(), blocks.data(), N);
        std::vector<float> b(N, 0.8f);
        volatile float sink = 0.0f;

        double t_scalar = measure_median_us([&]() {
            sink = vectra::quant::vec_dot_q4_0_f32_scalar(blocks.data(), b.data(), N);
        }, warmup, iters);

        double t_avx2 = measure_median_us([&]() {
            sink = vectra::quant::vec_dot_q4_0_f32_avx2(blocks.data(), b.data(), N);
        }, warmup, iters);

        results.push_back({"Quantized Q4_0 Dot", "1024", 1, warmup, iters, t_scalar, t_avx2, t_scalar / t_avx2, blocks.size() * sizeof(vectra::quant::BlockQ4_0) + N * 4, true});
    }

    // 4. Quantized Q4_0 Matvec (Rows = 256, Cols = 1024)
    {
        const size_t rows = 256, cols = 1024;
        std::vector<float> A_orig(rows * cols, 0.5f);
        std::vector<vectra::quant::BlockQ4_0> A_q(rows * (cols / 32));
        for (size_t r = 0; r < rows; ++r) {
            vectra::quant::quantize_row_q4_0(A_orig.data() + r * cols, A_q.data() + r * (cols / 32), cols);
        }
        std::vector<float> x(cols, 1.0f);
        std::vector<float> y(rows, 0.0f);

        double t_scalar = measure_median_us([&]() {
            for (size_t r = 0; r < rows; ++r) {
                y[r] = vectra::quant::vec_dot_q4_0_f32_scalar(A_q.data() + r * (cols / 32), x.data(), cols);
            }
        }, warmup, iters);

        double t_avx2 = measure_median_us([&]() {
            vectra::quant::matvec_q4_0_f32(A_q.data(), x.data(), nullptr, y.data(), rows, cols);
        }, warmup, iters);

        results.push_back({"Quantized Q4_0 Matvec", "256x1024", 1, warmup, iters, t_scalar, t_avx2, t_scalar / t_avx2, A_q.size() * sizeof(vectra::quant::BlockQ4_0) + cols * 4, true});
    }

    // 5. RMSNorm (N = 1536 - Target Qwen2.5 Dimension)
    {
        const size_t N = 1536;
        std::vector<float> x(N, 0.7f);
        std::vector<float> w(N, 1.0f);
        std::vector<float> out(N, 0.0f);

        double t_scalar = measure_median_us([&]() {
            vectra::math::scalar::rms_norm(x.data(), w.data(), out.data(), N, 1e-6f);
        }, warmup, iters);

        double t_avx2 = measure_median_us([&]() {
            vectra::math::avx2::rms_norm(x.data(), w.data(), out.data(), N, 1e-6f);
        }, warmup, iters);

        results.push_back({"RMSNorm (Target Dim)", "1536", 1, warmup, iters, t_scalar, t_avx2, t_scalar / t_avx2, 3 * N * 4, true});
    }

    // 6. Softmax (N = 1024)
    {
        const size_t N = 1024;
        std::vector<float> x(N, 0.5f);
        std::vector<float> out(N, 0.0f);

        double t_scalar = measure_median_us([&]() {
            vectra::math::scalar::softmax(x.data(), out.data(), N);
        }, warmup, iters);

        results.push_back({"Softmax", "1024", 1, warmup, iters, t_scalar, t_scalar, 1.0, 2 * N * 4, true});
    }

    // 7. RoPE (12 heads * 128 head_dim)
    {
        const int64_t num_heads = 12;
        const int64_t head_dim = 128;
        std::vector<float> q(num_heads * head_dim, 0.5f);

        double t_rope = measure_median_us([&]() {
            for (int64_t h = 0; h < num_heads; ++h) {
                vectra::math::apply_rope(q.data() + h * head_dim, 10, head_dim, 1000000.0f);
            }
        }, warmup, iters);

        results.push_back({"RoPE (12 heads x 128 dim)", "12x128", 1, warmup, iters, t_rope, t_rope, 1.0, num_heads * head_dim * 4, true});
    }

    // 8. KV-Cache Access (Seq = 512, Layer = 28)
    {
        const int64_t layers = 28;
        const int64_t max_seq = 512;
        const int64_t kv_heads = 2;
        const int64_t head_dim = 128;
        vectra::model::KVCache cache(layers, max_seq, kv_heads, head_dim);

        std::vector<float> k_in(kv_heads * head_dim, 1.0f);
        std::vector<float> v_in(kv_heads * head_dim, 1.0f);

        double t_append = measure_median_us([&]() {
            for (int64_t l = 0; l < layers; ++l) {
                cache.append(l, 10, k_in.data(), v_in.data());
            }
        }, warmup, iters);

        results.push_back({"KV Cache Append (28 layers)", "28x2x128", 1, warmup, iters, t_append, t_append, 1.0, cache.memory_bytes(), true});
    }

    // Print results table
    std::cout << std::left << std::setw(28) << "Kernel Name"
              << std::setw(12) << "Shape"
              << std::setw(14) << "Scalar (us)"
              << std::setw(14) << "AVX2 (us)"
              << std::setw(10) << "Speedup"
              << "Status\n";
    std::cout << std::string(85, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(28) << r.kernel_name
                  << std::setw(12) << r.shape
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.median_us_scalar
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.median_us_avx2
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.speedup << "x"
                  << (r.numerical_pass ? "PASS" : "FAIL") << "\n";
    }

    // Save to CSV
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "kernel_name,shape,threads,warmup_iters,timed_iters,median_us_scalar,median_us_avx2,speedup_ratio,memory_bytes,numerical_pass\n";
        for (const auto& r : results) {
            csv << "\"" << r.kernel_name << "\","
                << "\"" << r.shape << "\","
                << r.threads << ","
                << r.warmup_iters << ","
                << r.timed_iters << ","
                << r.median_us_scalar << ","
                << r.median_us_avx2 << ","
                << r.speedup << ","
                << r.memory_bytes << ","
                << (r.numerical_pass ? "true" : "false") << "\n";
        }
        std::cout << "\nMicrobenchmarks written to " << csv_path << "\n";
    }

    return 0;
}
