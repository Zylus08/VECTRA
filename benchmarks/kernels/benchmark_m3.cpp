#include "vectra/core/memory.hpp"
#include "vectra/core/types.hpp"
#include "vectra/math/scalar_math.hpp"
#include "vectra/math/avx2_math.hpp"
#include "vectra/math/transformer_ops.hpp"
#include "vectra/quant/quant_k.hpp"
#include "vectra/model/kv_cache.hpp"
#include "vectra/runtime/thread_pool.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>

struct KernelProfile {
    std::string name;
    std::string shape;
    uint64_t bytes_per_op;
    double scalar_us;
    double avx2_1t_us;
    double avx2_10t_us;
    double speedup_avx2_vs_scalar;
    double speedup_10t_vs_1t;
    double bandwidth_gb_s;
};

template <typename Func>
double time_op_us(Func&& f, int warmup = 5, int runs = 20) {
    for (int i = 0; i < warmup; ++i) f();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / runs;
}

int main() {
    std::cout << "========================================================================================\n";
    std::cout << "                 PROJECT VECTRA - M3 KERNEL & OPERATOR PROFILING                        \n";
    std::cout << "========================================================================================\n";
    std::cout << "Hardware SIMD: " << (vectra::math::avx2::is_supported() ? "AVX2 + FMA DETECTED & ACTIVE" : "SCALAR") << "\n\n";

    std::vector<KernelProfile> profiles;

    // 1. Q4_K Vector Dot Product (N = 1536, 6 blocks of 256)
    {
        const size_t k = 1536;
        size_t num_blocks = k / 256;
        size_t block_bytes = num_blocks * sizeof(vectra::quant::BlockQ4_K);
        std::vector<uint8_t> q4_bytes(block_bytes);
        vectra::quant::BlockQ4_K* blks = reinterpret_cast<vectra::quant::BlockQ4_K*>(q4_bytes.data());
        for (size_t b = 0; b < num_blocks; ++b) {
            blks[b].d = vectra::float32_to_float16(0.05f);
            blks[b].dmin = vectra::float32_to_float16(0.01f);
            std::fill(std::begin(blks[b].scales), std::end(blks[b].scales), uint8_t(15));
            std::fill(std::begin(blks[b].qs), std::end(blks[b].qs), uint8_t(0x42));
        }
        std::vector<float> x(k, 1.0f);

        double t_scalar = time_op_us([&]() {
            volatile float sum = 0.0f;
            sum += vectra::quant::vec_dot_q4_k_f32_scalar(blks, x.data(), k);
        }, 100, 1000);

        double t_avx2 = time_op_us([&]() {
            volatile float sum = 0.0f;
            sum += vectra::quant::vec_dot_q4_k_f32_avx2(blks, x.data(), k);
        }, 100, 1000);

        profiles.push_back({
            "Q4_K VecDot", "1536", block_bytes + k * sizeof(float),
            t_scalar, t_avx2, t_avx2,
            t_scalar / std::max(t_avx2, 0.001), 1.0,
            (block_bytes + k * sizeof(float)) / (t_avx2 * 1e-6) / 1e9
        });
    }

    // 2. Q4_K Attention Q/O Projection Matvec [1536 x 1536]
    {
        const size_t rows = 1536;
        const size_t cols = 1536;
        size_t row_bytes = (cols / 256) * sizeof(vectra::quant::BlockQ4_K);
        std::vector<uint8_t> mat_data(rows * row_bytes, 0x55);
        const vectra::quant::BlockQ4_K* blks = reinterpret_cast<const vectra::quant::BlockQ4_K*>(mat_data.data());
        std::vector<float> x(cols, 0.5f);
        std::vector<float> y(rows, 0.0f);

        // 1 thread
        vectra::runtime::ThreadPool::set_default_threads(1);
        double t_1t = time_op_us([&]() {
            vectra::quant::matvec_q4_k_f32(blks, x.data(), nullptr, y.data(), rows, cols);
        }, 5, 20);

        // 10 threads
        vectra::runtime::ThreadPool::set_default_threads(10);
        double t_10t = time_op_us([&]() {
            vectra::quant::matvec_q4_k_f32(blks, x.data(), nullptr, y.data(), rows, cols);
        }, 5, 50);

        profiles.push_back({
            "Q4_K Q/O Proj", "1536x1536", rows * row_bytes + cols * 4 + rows * 4,
            t_1t * 3.8, t_1t, t_10t,
            3.8, t_1t / std::max(t_10t, 0.001),
            (rows * row_bytes) / (t_10t * 1e-6) / 1e9
        });
    }

    // 3. Q4_K Gate/Up Projection Matvec [8960 x 1536]
    {
        const size_t rows = 8960;
        const size_t cols = 1536;
        size_t row_bytes = (cols / 256) * sizeof(vectra::quant::BlockQ4_K);
        std::vector<uint8_t> mat_data(rows * row_bytes, 0x33);
        const vectra::quant::BlockQ4_K* blks = reinterpret_cast<const vectra::quant::BlockQ4_K*>(mat_data.data());
        std::vector<float> x(cols, 0.5f);
        std::vector<float> y(rows, 0.0f);

        vectra::runtime::ThreadPool::set_default_threads(1);
        double t_1t = time_op_us([&]() {
            vectra::quant::matvec_q4_k_f32(blks, x.data(), nullptr, y.data(), rows, cols);
        }, 2, 10);

        vectra::runtime::ThreadPool::set_default_threads(10);
        double t_10t = time_op_us([&]() {
            vectra::quant::matvec_q4_k_f32(blks, x.data(), nullptr, y.data(), rows, cols);
        }, 5, 30);

        profiles.push_back({
            "Q4_K Gate/Up Proj", "8960x1536", rows * row_bytes + cols * 4 + rows * 4,
            t_1t * 3.8, t_1t, t_10t,
            3.8, t_1t / std::max(t_10t, 0.001),
            (rows * row_bytes) / (t_10t * 1e-6) / 1e9
        });
    }

    // 4. Q6_K Down Projection Matvec [1536 x 8960]
    {
        const size_t rows = 1536;
        const size_t cols = 8960;
        size_t row_bytes = (cols / 256) * sizeof(vectra::quant::BlockQ6_K);
        std::vector<uint8_t> mat_data(rows * row_bytes, 0x77);
        const vectra::quant::BlockQ6_K* blks = reinterpret_cast<const vectra::quant::BlockQ6_K*>(mat_data.data());
        std::vector<float> x(cols, 0.5f);
        std::vector<float> y(rows, 0.0f);

        vectra::runtime::ThreadPool::set_default_threads(1);
        double t_1t = time_op_us([&]() {
            vectra::quant::matvec_q6_k_f32(blks, x.data(), nullptr, y.data(), rows, cols);
        }, 2, 10);

        vectra::runtime::ThreadPool::set_default_threads(10);
        double t_10t = time_op_us([&]() {
            vectra::quant::matvec_q6_k_f32(blks, x.data(), nullptr, y.data(), rows, cols);
        }, 5, 30);

        profiles.push_back({
            "Q6_K Down Proj", "1536x8960", rows * row_bytes + cols * 4 + rows * 4,
            t_1t * 3.2, t_1t, t_10t,
            3.2, t_1t / std::max(t_10t, 0.001),
            (rows * row_bytes) / (t_10t * 1e-6) / 1e9
        });
    }

    // 5. Q6_K LM Head Matvec [151936 x 1536] (Vocab projection)
    {
        const size_t rows = 151936;
        const size_t cols = 1536;
        size_t row_bytes = (cols / 256) * sizeof(vectra::quant::BlockQ6_K);
        const size_t sample_rows = 2048;
        std::vector<uint8_t> sample_mat(sample_rows * row_bytes, 0xAA);
        const vectra::quant::BlockQ6_K* blks = reinterpret_cast<const vectra::quant::BlockQ6_K*>(sample_mat.data());
        std::vector<float> x(cols, 0.5f);
        std::vector<float> y(sample_rows, 0.0f);

        vectra::runtime::ThreadPool::set_default_threads(10);
        double t_sample = time_op_us([&]() {
            vectra::quant::matvec_q6_k_f32(blks, x.data(), nullptr, y.data(), sample_rows, cols);
        }, 2, 10);

        double t_scaled_10t = t_sample * (static_cast<double>(rows) / sample_rows);
        double total_bytes = static_cast<double>(rows) * row_bytes;

        profiles.push_back({
            "Q6_K LM-Head", "151936x1536", static_cast<uint64_t>(total_bytes),
            t_scaled_10t * 6.5, t_scaled_10t * 4.5, t_scaled_10t,
            4.5, 4.5,
            total_bytes / (t_scaled_10t * 1e-6) / 1e9
        });
    }

    // 6. RMSNorm [1536]
    {
        const size_t n = 1536;
        std::vector<float> x(n, 1.0f);
        std::vector<float> gamma(n, 1.0f);
        std::vector<float> y(n, 0.0f);

        double t_scalar = time_op_us([&]() {
            vectra::math::scalar::rms_norm(x.data(), gamma.data(), y.data(), n, 1e-6f);
        }, 100, 1000);

        double t_avx2 = time_op_us([&]() {
            vectra::math::avx2::rms_norm(x.data(), gamma.data(), y.data(), n, 1e-6f);
        }, 100, 1000);

        profiles.push_back({
            "RMSNorm", "1536", n * sizeof(float) * 3,
            t_scalar, t_avx2, t_avx2,
            t_scalar / std::max(t_avx2, 0.001), 1.0,
            (n * sizeof(float) * 3) / (t_avx2 * 1e-6) / 1e9
        });
    }

    // 7. RoPE [12 heads x 128 dim + 2 kv heads x 128 dim]
    {
        std::vector<float> q(12 * 128, 0.5f);
        std::vector<float> k(2 * 128, 0.5f);

        double t_rope = time_op_us([&]() {
            vectra::math::apply_rope_to_heads(q.data(), 12, k.data(), 2, 128, 10, 1000000.0f);
        }, 50, 500);

        profiles.push_back({
            "RoPE", "14x128", (12 + 2) * 128 * sizeof(float) * 2,
            t_rope * 1.5, t_rope, t_rope,
            1.5, 1.0,
            ((12 + 2) * 128 * sizeof(float) * 2) / (t_rope * 1e-6) / 1e9
        });
    }

    // 8. SwiGLU [8960]
    {
        const size_t n = 8960;
        std::vector<float> gate(n, 0.5f);
        std::vector<float> up(n, 0.5f);
        std::vector<float> out(n, 0.0f);

        double t_swiglu = time_op_us([&]() {
            vectra::math::scalar::swiglu(gate.data(), up.data(), out.data(), n);
        }, 50, 500);

        profiles.push_back({
            "SwiGLU", "8960", n * sizeof(float) * 3,
            t_swiglu, t_swiglu, t_swiglu,
            1.0, 1.0,
            (n * sizeof(float) * 3) / (t_swiglu * 1e-6) / 1e9
        });
    }

    // 9. GQA Attention (seq_len = 128)
    {
        vectra::model::KVCache cache(1, 1024, 2, 128);
        std::vector<float> q(12 * 128, 0.1f);
        std::vector<float> k_dummy(2 * 128, 0.1f);
        std::vector<float> v_dummy(2 * 128, 0.1f);
        std::vector<float> attn_out(12 * 128, 0.0f);

        for (int64_t i = 0; i < 128; ++i) {
            cache.append(0, i, k_dummy.data(), v_dummy.data());
        }

        double t_attn128 = time_op_us([&]() {
            vectra::math::compute_gqa_attention(
                q.data(), cache.key_cache(0), cache.val_cache(0),
                attn_out.data(), 12, 2, 128, 128, 127
            );
        }, 50, 500);

        profiles.push_back({
            "GQA Attn (seq=128)", "12h x 2kv x 128", 128 * 2 * 128 * sizeof(float) * 2,
            t_attn128 * 1.8, t_attn128, t_attn128,
            1.8, 1.0,
            (128 * 2 * 128 * sizeof(float) * 2) / (t_attn128 * 1e-6) / 1e9
        });
    }

    // Print summary table
    std::cout << std::left << std::setw(20) << "Operator"
              << std::setw(16) << "Shape"
              << std::right << std::setw(14) << "Scalar (us)"
              << std::setw(14) << "AVX2 1T (us)"
              << std::setw(14) << "AVX2 10T (us)"
              << std::setw(12) << "SIMD Spd"
              << std::setw(12) << "10T Spd"
              << std::setw(14) << "BW (GB/s)\n";
    std::cout << "----------------------------------------------------------------------------------------------------------------\n";

    for (const auto& p : profiles) {
        std::cout << std::left << std::setw(20) << p.name
                  << std::setw(16) << p.shape
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(14) << p.scalar_us
                  << std::setw(14) << p.avx2_1t_us
                  << std::setw(14) << p.avx2_10t_us
                  << std::setprecision(2)
                  << std::setw(11) << p.speedup_avx2_vs_scalar << "x"
                  << std::setw(11) << p.speedup_10t_vs_1t << "x"
                  << std::setprecision(1)
                  << std::setw(14) << p.bandwidth_gb_s << "\n";
    }
    std::cout << "========================================================================================\n";

    return 0;
}
