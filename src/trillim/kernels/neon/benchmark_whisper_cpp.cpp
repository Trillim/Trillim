// CPU-only whisper.cpp comparison with prepared mel features and teacher tokens.
// Arguments: model.bin mel.f32 teacher.txt output_prefix threads prompt decode warmups repeats
#include "whisper.h"
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
    if (argc != 10) {
        std::cerr << "model.bin mel.f32 teacher.txt output_prefix threads prompt decode warmups repeats\n";
        return 2;
    }
    const std::string model = argv[1], features = argv[2], teacher = argv[3], output = argv[4];
    const int threads = std::stoi(argv[5]), prompt = std::stoi(argv[6]), decode = std::stoi(argv[7]);
    const int warmups = std::stoi(argv[8]), repeats = std::stoi(argv[9]);
    if (threads < 1 || prompt < 1 || decode < 1 || warmups < 0 || repeats < 1) { throw std::runtime_error("invalid workload sizes"); }
    std::ifstream inputs(teacher);
    std::vector<whisper_token> tokens;
    whisper_token token;
    while (inputs >> token) { tokens.push_back(token); }
    if (tokens.size() < static_cast<size_t>(prompt + decode)) { throw std::runtime_error("insufficient teacher tokens"); }
    std::ifstream mel_file(features, std::ios::binary | std::ios::ate);
    if (!mel_file) { throw std::runtime_error("mel file missing"); }
    const size_t mel_bytes = static_cast<size_t>(mel_file.tellg());
    if (mel_bytes % sizeof(float) != 0) { throw std::runtime_error("mel byte size must be divisible by sizeof(float)"); }
    std::vector<float> mel(mel_bytes / sizeof(float));
    mel_file.seekg(0);
    if (!mel_file.read(reinterpret_cast<char *>(mel.data()), static_cast<std::streamsize>(mel_bytes))) { throw std::runtime_error("mel read failed"); }
    auto params = whisper_context_default_params();
    params.use_gpu = false;
    params.flash_attn = false;
    auto *ctx = whisper_init_from_file_with_params_no_state(model.c_str(), params);
    if (!ctx) { throw std::runtime_error("model loading failed"); }
    const int vocab = whisper_n_vocab(ctx), n_mels = whisper_model_n_mels(ctx);
    if (prompt + decode > whisper_model_n_text_ctx(ctx)) { throw std::runtime_error("teacher context exceeds model capacity"); }
    const int frames = static_cast<int>(mel.size() / n_mels);
    if (mel.size() != static_cast<size_t>(frames * n_mels) || frames != 2 * whisper_model_n_audio_ctx(ctx)) { throw std::runtime_error("invalid full-window mel shape"); }
    std::vector<float> copied(static_cast<size_t>(vocab));
    std::ofstream logits_file(output + "-all-logits.bin", std::ios::binary);
    std::ofstream result(output + ".json");
    result << std::setprecision(17) << "{\"engine\":\"whisper.cpp\",\"device\":\"cpu\",\"model_ftype\":" << whisper_model_ftype(ctx) << ",\"cache_dtype\":\"upstream default; see pinned source\",\"threads\":" << threads << ",\"vocab_size\":" << vocab << ",\"sessions\":[";
    for (int session = 0; session < warmups + repeats; ++session) {
        auto *state = whisper_init_state(ctx);
        if (!state) { throw std::runtime_error("state initialization failed"); }
        if (whisper_set_mel_with_state(ctx, state, mel.data(), frames, n_mels)) { throw std::runtime_error("mel binding failed"); }
        auto start = std::chrono::steady_clock::now();
        if (whisper_encode_with_state(ctx, state, 0, threads)) { throw std::runtime_error("encode failed"); }
        const double encoder = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::vector<double> samples;
        int position = 0;
        for (int step = -1; step < decode; ++step) {
            const int count = step < 0 ? prompt : 1;
            const int offset = step < 0 ? 0 : prompt + step;
            start = std::chrono::steady_clock::now();
            if (whisper_decode_with_state(ctx, state, tokens.data() + offset, count, position, threads)) { throw std::runtime_error("decode failed"); }
            const float *logits = whisper_get_logits_from_state(state) + (count - 1) * vocab;
            std::copy(logits, logits + vocab, copied.begin());
            samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            if (!std::all_of(copied.begin(), copied.end(), [](float value) { return std::isfinite(value); })) { throw std::runtime_error("nonfinite logits"); }
            if (session == warmups) { logits_file.write(reinterpret_cast<const char *>(copied.data()), vocab * sizeof(float)); }
            position += count;
        }
        if (session) { result << ','; }
        result << "{\"warmup\":" << (session < warmups ? "true" : "false") << ",\"encoder_and_cross_kv_seconds\":" << encoder << ",\"prefill_seconds\":" << samples[0] << ",\"decode_seconds\":[";
        for (size_t i = 1; i < samples.size(); ++i) { if (i > 1) result << ','; result << samples[i]; }
        result << "]}";
        result.flush();
        whisper_free_state(state);
    }
    result << "]}\n";
    whisper_free(ctx);
}
