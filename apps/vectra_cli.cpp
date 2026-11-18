#include "vectra/core/types.hpp"
#include "vectra/core/memory.hpp"
#include "vectra/math/avx2_math.hpp"
#include "vectra/format/mmap_loader.hpp"
#include "vectra/model/gguf_model.hpp"
#include "vectra/tokenizer/bpe_tokenizer.hpp"
#include "vectra/runtime/sampler.hpp"
#include "vectra/runtime/thread_pool.hpp"
#include "vectra/runtime/profiler.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>

struct CliArgs {
    std::string model_path = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    std::string tokenizer_path = "models/qwen_tokenizer.bin";
    std::string prompt = "The capital of France is";
    size_t max_tokens = 32;
    float temp = 0.0f; // 0.0 = greedy
    float top_p = 0.9f;
    int32_t top_k = 40;
    uint64_t seed = 42;
    size_t threads = 10;
    bool chat_format = false;
    bool use_int8 = true;
};

CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            args.tokenizer_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            args.max_tokens = std::stoul(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            args.temp = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            args.top_p = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            args.top_k = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = std::stoull(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            args.threads = std::stoul(argv[++i]);
        } else if (arg == "--chat") {
            args.chat_format = true;
        } else if (arg == "--fp32") {
            args.use_int8 = false;
        } else if (arg == "--int8") {
            args.use_int8 = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: vectra_cli [options]\n"
                      << "  --model <path>       Path to GGUF model file\n"
                      << "  --tokenizer <path>   Path to binary tokenizer file\n"
                      << "  --prompt <str>       Input prompt text\n"
                      << "  --max-tokens <int>   Maximum new tokens to generate (default: 32)\n"
                      << "  --temp <float>       Temperature (0.0 = greedy argmax, default: 0.0)\n"
                      << "  --top-p <float>      Top-p nucleus sampling (default: 0.9)\n"
                      << "  --top-k <int>        Top-k filtering (default: 40)\n"
                      << "  --threads <int>      Thread count for thread pool (default: 10)\n"
                      << "  --seed <int>         Random seed (default: 42)\n"
                      << "  --chat               Format prompt as ChatML template\n"
                      << "  --int8               Enable INT8 activation quantization (default)\n"
                      << "  --fp32               Disable INT8, use pure FP32 fused AVX2\n";
            std::exit(0);
        }
    }
    return args;
}

int main(int argc, char* argv[]) {
    std::cout << "=========================================================\n";
    std::cout << "          PROJECT VECTRA - CPU INFERENCE RUNTIME         \n";
    std::cout << "=========================================================\n";
    std::cout << "Host Hardware Vector Instructions: "
              << (vectra::math::avx2::is_supported() ? "AVX2 + FMA DETECTED & ACTIVE" : "SCALAR FALLBACK")
              << "\n\n";

    CliArgs args = parse_args(argc, argv);
    vectra::runtime::ThreadPool::set_default_threads(args.threads);

    std::cout << "Active Engine Configuration:\n"
              << "  Worker Threads: " << args.threads << " threads in persistent pool\n"
              << "  Prefill Engine: Batched multi-token causal GEMV-N\n\n";

    // 1. Format input prompt if ChatML requested
    std::string formatted_prompt = args.prompt;
    if (args.chat_format) {
        formatted_prompt = "<|im_start|>user\n" + args.prompt + "<|im_end|>\n<|im_start|>assistant\n";
    }

    // 2. Load tokenizer
    std::cout << "[1/4] Loading Tokenizer: " << args.tokenizer_path << " ...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    vectra::tokenizer::BPETokenizer tokenizer;
    try {
        tokenizer.load_binary(args.tokenizer_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading tokenizer: " << e.what() << "\n";
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double tok_load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "      Loaded " << tokenizer.vocab_size() << " tokens in " << std::fixed << std::setprecision(1) << tok_load_ms << " ms\n\n";

    // 3. Load GGUF model via memory mapping
    std::cout << "[2/4] Memory-Mapping GGUF Model: " << args.model_path << " ...\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    vectra::model::GGUFTransformerModel model;
    try {
        model.load(args.model_path);
        model.set_use_int8_kernels(args.use_int8);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double model_load_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    const auto& cfg = model.config();
    std::cout << "      Model Loaded: " << cfg.num_layers << " layers, hidden=" << cfg.hidden_dim
              << ", heads=" << cfg.num_heads << " (kv=" << cfg.num_kv_heads << ")"
              << ", intermediate=" << cfg.intermediate_dim << ", vocab=" << cfg.vocab_size << "\n";
    std::cout << "      Mapped Size: " << (model.mapped_bytes() / (1024.0 * 1024.0)) << " MB in "
              << std::fixed << std::setprecision(1) << model_load_ms << " ms\n\n";

    // 4. Tokenize prompt
    std::cout << "[3/4] Tokenizing Prompt...\n";
    std::vector<int32_t> prompt_tokens = tokenizer.encode(formatted_prompt);
    std::cout << "      Prompt: \"" << args.prompt << "\"\n";
    std::cout << "      Tokens: " << prompt_tokens.size() << " tokens: [";
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        std::cout << prompt_tokens[i] << (i + 1 < prompt_tokens.size() ? ", " : "");
    }
    std::cout << "]\n\n";

    // 5. Configure sampler
    vectra::runtime::SamplingParams sample_params;
    if (args.temp <= 0.0f) {
        sample_params.greedy = true;
    } else {
        sample_params.greedy = false;
        sample_params.temperature = args.temp;
        sample_params.top_p = args.top_p;
        sample_params.top_k = args.top_k;
        sample_params.seed = args.seed;
    }
    vectra::runtime::Sampler sampler(sample_params);

    // 6. Generate output tokens with streaming callback
    std::cout << "[4/4] Generating (max " << args.max_tokens << " new tokens, temp="
              << args.temp << ", sampler=" << (sample_params.greedy ? "greedy" : "nucleus") << ")...\n";
    std::cout << "---------------------------------------------------------\n";

    size_t generated_count = 0;
    auto gen_start = std::chrono::high_resolution_clock::now();
    auto first_token_time = gen_start;
    bool got_first_token = false;

    auto on_token_cb = [&](int32_t token_id) -> bool {
        if (!got_first_token) {
            first_token_time = std::chrono::high_resolution_clock::now();
            got_first_token = true;
        }
        generated_count++;
        std::string piece = tokenizer.decode({token_id});
        std::cout << piece << std::flush;
        return true;
    };

    std::vector<int32_t> output_tokens = model.generate(
        prompt_tokens,
        args.max_tokens,
        sampler,
        on_token_cb,
        1024
    );

    auto gen_end = std::chrono::high_resolution_clock::now();
    std::cout << "\n---------------------------------------------------------\n\n";

    // 7. Performance & Memory Profile Report
    double prefill_sec = std::chrono::duration<double>(first_token_time - gen_start).count();
    double decode_sec = std::chrono::duration<double>(gen_end - first_token_time).count();
    double total_sec = std::chrono::duration<double>(gen_end - gen_start).count();

    double prefill_tok_per_sec = prompt_tokens.empty() ? 0.0 : (prompt_tokens.size() / std::max(prefill_sec, 0.001));
    double decode_tok_per_sec = (generated_count <= 1) ? 0.0 : ((generated_count - 1) / std::max(decode_sec, 0.001));

    auto mem_stats = vectra::format::get_process_memory_stats();

    std::cout << "--- Execution Metrics ---\n";
    std::cout << "  Prompt Tokens:      " << prompt_tokens.size() << " tokens\n";
    std::cout << "  Generated Tokens:   " << generated_count << " tokens\n";
    std::cout << "  Generated Token IDs: [";
    for (size_t i = 0; i < output_tokens.size(); ++i) {
        std::cout << output_tokens[i] << (i + 1 < output_tokens.size() ? ", " : "");
    }
    std::cout << "]\n";
    std::cout << "  Worker Threads:     " << args.threads << " threads\n";
    std::cout << "  Prefill Latency:    " << std::fixed << std::setprecision(2) << (prefill_sec * 1000.0) << " ms ("
              << prefill_tok_per_sec << " tok/s)\n";
    std::cout << "  Decode Latency:     " << std::fixed << std::setprecision(2) << (decode_sec * 1000.0) << " ms ("
              << decode_tok_per_sec << " tok/s)\n";
    std::cout << "  Total Inference:    " << std::fixed << std::setprecision(2) << (total_sec * 1000.0) << " ms\n";
    std::cout << "  Process Resident:   " << (mem_stats.working_set_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Peak Working Set:   " << (mem_stats.peak_working_set_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "=========================================================\n";

    return 0;
}
