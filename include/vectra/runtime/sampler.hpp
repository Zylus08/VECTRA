#pragma once

#include <cstdint>
#include <vector>
#include <random>

namespace vectra::runtime {

struct SamplingParams {
    float temperature = 0.7f;
    int32_t top_k = 40;
    float top_p = 0.9f;
    uint32_t seed = 42;
    bool greedy = true;
};

class Sampler {
public:
    explicit Sampler(SamplingParams params = SamplingParams{});

    // Sample next token ID from logits array
    int32_t sample(float* logits, size_t vocab_size);

    void set_seed(uint32_t seed);
    void set_greedy(bool greedy) { params_.greedy = greedy; }

private:
    SamplingParams params_;
    std::mt19937 rng_;
};

} // namespace vectra::runtime
