#include "vectra/runtime/sampler.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace vectra::runtime {

Sampler::Sampler(SamplingParams params) : params_(params), rng_(params.seed) {}

void Sampler::set_seed(uint32_t seed) {
    params_.seed = seed;
    rng_.seed(seed);
}

int32_t Sampler::sample(float* logits, size_t vocab_size) {
    if (vocab_size == 0) return 0;

    // 1. Greedy argmax
    if (params_.greedy || params_.temperature <= 0.0f) {
        int32_t best_tok = 0;
        float max_logit = logits[0];
        for (size_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                best_tok = static_cast<int32_t>(i);
            }
        }
        return best_tok;
    }

    // 2. Temperature scaling
    float inv_temp = 1.0f / params_.temperature;
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_temp;
    }

    // 3. Top-k filtering
    struct TokenCandidate {
        int32_t id;
        float logit;
    };
    std::vector<TokenCandidate> candidates(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        candidates[i] = {static_cast<int32_t>(i), logits[i]};
    }

    int k = params_.top_k > 0 ? std::min(params_.top_k, static_cast<int32_t>(vocab_size)) : static_cast<int32_t>(vocab_size);
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                      [](const TokenCandidate& a, const TokenCandidate& b) {
                          return a.logit > b.logit;
                      });
    candidates.resize(k);

    // 4. Stable Softmax over top candidates
    float max_logit = candidates[0].logit;
    float sum_exp = 0.0f;
    for (auto& cand : candidates) {
        cand.logit = std::exp(cand.logit - max_logit);
        sum_exp += cand.logit;
    }
    float inv_sum = 1.0f / sum_exp;
    for (auto& cand : candidates) {
        cand.logit *= inv_sum;
    }

    // 5. Top-p (nucleus) filtering
    if (params_.top_p < 1.0f && params_.top_p > 0.0f) {
        float cumsum = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            cumsum += candidates[i].logit;
            if (cumsum >= params_.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        if (cutoff > 0 && cutoff < candidates.size()) {
            candidates.resize(cutoff);
            // Re-normalize
            float new_sum = 0.0f;
            for (const auto& cand : candidates) new_sum += cand.logit;
            float new_inv = 1.0f / new_sum;
            for (auto& cand : candidates) cand.logit *= new_inv;
        }
    }

    // 6. Stochastic sampling
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float acc = 0.0f;
    for (const auto& cand : candidates) {
        acc += cand.logit;
        if (r <= acc) {
            return cand.id;
        }
    }

    return candidates.back().id;
}

} // namespace vectra::runtime
