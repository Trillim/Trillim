// CPU-only teacher-forced llama.cpp comparison. Build against its pinned CPU-only libraries.
// Arguments: model.gguf teacher.txt output_prefix threads prompt decode warmups repeats [f16]
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 9 && argc != 10) {
        std::cerr << "model token_file output_prefix threads prompt decode warmups repeats\n";
        return 2;
    }
    const std::string model_path = argv[1], token_path = argv[2], output = argv[3];
    const int threads = std::stoi(argv[4]), prompt = std::stoi(argv[5]);
    const int decode = std::stoi(argv[6]), warmups = std::stoi(argv[7]), repeats = std::stoi(argv[8]);
    const bool cache_f16 = argc == 10 && std::string(argv[9]) == "f16";
    if (threads < 1 || prompt < 1 || decode < 1 || prompt + decode > 512 || warmups < 0 || repeats < 1) { throw std::runtime_error("invalid workload sizes"); }
    std::ifstream token_file(token_path);
    std::vector<llama_token> tokens;
    llama_token token;
    while (token_file >> token) { tokens.push_back(token); }
    if (tokens.size() < static_cast<size_t>(prompt + decode)) { throw std::runtime_error("insufficient teacher tokens"); }
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto *model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { throw std::runtime_error("model loading failed"); }
    const auto *vocab = llama_model_get_vocab(model);
    const int vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<float> copied_logits(static_cast<size_t>(vocab_size));
    std::ofstream logits_file(output + "-first5-logits.bin", std::ios::binary);
    std::ofstream result(output + ".json");
    result << std::setprecision(17) << "{\"engine\":\"llama.cpp\",\"threads\":" << threads
           << ",\"prompt\":" << prompt << ",\"decode\":" << decode << ",\"warmups\":" << warmups
           << ",\"repeats\":" << repeats << ",\"cache_dtype\":\"" << (cache_f16 ? "f16" : "f32") << "\",\"flash_attention\":\"auto\",\"vocab_size\":" << vocab_size << ",\"sessions\":[";
    for (int session = 0; session < warmups + repeats; ++session) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.type_k = cache_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        cp.type_v = cache_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.offload_kqv = false;
        cp.op_offload = false;
        cp.no_perf = true;
        auto *ctx = llama_init_from_model(model, cp);
        if (!ctx) { throw std::runtime_error("context creation failed"); }
        std::vector<double> times;
        std::vector<llama_token> predicted;
        for (int step = -1; step < decode; ++step) {
            const auto start = std::chrono::steady_clock::now();
            auto batch = step < 0 ? llama_batch_get_one(tokens.data(), prompt)
                                 : llama_batch_get_one(tokens.data() + prompt + step, 1);
            if (llama_decode(ctx, batch) != 0) { throw std::runtime_error("decode failed"); }
            llama_synchronize(ctx);
            const float *logits = llama_get_logits_ith(ctx, -1);
            std::copy(logits, logits + vocab_size, copied_logits.begin());
            const auto end = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double>(end - start).count());
            if (!std::all_of(copied_logits.begin(), copied_logits.end(), [](float value) { return std::isfinite(value); })) { throw std::runtime_error("nonfinite logits"); }
            predicted.push_back(static_cast<llama_token>(std::max_element(copied_logits.begin(), copied_logits.end()) - copied_logits.begin()));
            if (session == warmups && step < 4) {
                logits_file.write(reinterpret_cast<const char *>(copied_logits.data()), vocab_size * sizeof(float));
            }
        }
        if (session) { result << ','; }
        result << "{\"warmup\":" << (session < warmups ? "true" : "false") << ",\"prefill_seconds\":" << times[0] << ",\"decode_seconds\":[";
        for (size_t i = 1; i < times.size(); ++i) { if (i > 1) result << ','; result << times[i]; }
        result << "],\"predicted_ids\":[";
        for (size_t i = 0; i < predicted.size(); ++i) { if (i) result << ','; result << predicted[i]; }
        result << "]}";
        result.flush();
        llama_free(ctx);
    }
    result << "]}\n";
    llama_model_free(model);
    llama_backend_free();
}
