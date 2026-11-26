#ifndef TRILLIM_CPU_RUNTIME_H
#define TRILLIM_CPU_RUNTIME_H

#include "native.h"
#include "integer.h"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trillim::cpu {
using Shape = std::vector<int64_t>;

struct Cancelled : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Allocation {
    uint8_t *data = nullptr;
    size_t size = 0;
    Allocation() = default;
    explicit Allocation(size_t bytes, size_t alignment = 64);
    ~Allocation();
    Allocation(Allocation &&other) noexcept;
    Allocation &operator=(Allocation &&other) noexcept;
    Allocation(const Allocation &) = delete;
    Allocation &operator=(const Allocation &) = delete;
};

struct Mapping {
    uint8_t *data = nullptr;
    size_t size = 0;
    Mapping() = default;
    explicit Mapping(const std::filesystem::path &path);
    ~Mapping();
    Mapping(Mapping &&other) noexcept;
    Mapping &operator=(Mapping &&other) noexcept;
    Mapping(const Mapping &) = delete;
    Mapping &operator=(const Mapping &) = delete;
};

struct Input {
    std::string name;
    Shape shape;
    Allocation payload;
};

struct View {
    uint8_t *data = nullptr;
    Shape shape;
    Shape strides; // Elements, as in the executable manifest.
    uint32_t dtype = TRILLIM_FP32;
    size_t span = 0;
};

struct CacheCopyLayout {
    Shape target_shape;
    Shape target_strides;
    Shape source_shape;
    Shape source_strides;
    uint32_t dtype = TRILLIM_FP32;
};

struct CacheCopyProfile {
    uint64_t ns = 0;
    uint64_t count = 0;
    uint64_t bytes = 0;
    CacheCopyLayout first;
    CacheCopyLayout last;
};

struct Kernel {
    TrillimKernel function = nullptr;
    const TrillimParameterSchema *schema = nullptr;
};

struct PreparedCall {
    Json parameters;
    Json varying;
    Allocation packed;
    std::vector<std::vector<uint8_t>> arrays;
    std::vector<const void *> inputs;
    std::vector<void *> outputs;
    std::vector<Integer> key;
    std::unordered_map<std::string, Shape> shapes;
    std::vector<size_t> input_sizes;
    std::vector<size_t> output_sizes;
    size_t scratch_size = 0;
    bool bound = false;
    bool probe = false;
    uint8_t *scratch = nullptr;
    const Kernel *kernel = nullptr;
};

using TokenSink = std::function<void(size_t, const Json &)>;

struct TokenStream {
    TokenSink sink;
    const Json *loop = nullptr;
    std::string output;
    std::string buffer;
    std::string length;
    size_t emitted = 0;
    bool started = false;
};

size_t itemsize(uint32_t dtype);
uint32_t dtype_id(const std::string &name);
size_t elements(const Shape &shape);
Shape contiguous_strides(const Shape &shape);
size_t packed_bytes(uint32_t dtype, size_t rows, size_t columns);
uint64_t now_ns();

class Runtime;

Allocation encode_packed(const float *source, uint32_t dtype, size_t rows, size_t columns,
                         const Runtime &runtime);
Allocation decode_packed(const uint8_t *payload, uint32_t dtype, size_t rows, size_t columns,
                         const Runtime &runtime);

class Runtime {
public:
    Runtime(const std::filesystem::path &directory, size_t threads, uint64_t capabilities);
    ~Runtime();
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    const Json &buffer(const std::string &name) const;
    void activate(const Json &plan);
    void bind_inputs(std::vector<Input> incoming);
    Shape shape(const std::string &name);
    Shape strides(const std::string &name);
    uint8_t *address(const std::string &name);
    size_t span(const std::string &name);
    std::pair<size_t, size_t> storage_shape(const std::string &name);
    View view(const std::string &name);
    View cache_view(const std::string &name, Allocation &decoded, bool capacity = false);
    View logical_view(const std::string &name, Allocation &decoded);
    Integer scalar_integer(const std::string &name, size_t index = 0);
    bool truth(const std::string &name);
    void check_cancelled() const;

    Integer evaluate(const Json &value);
    double scalar_parameter(const Json &value);
    Json parameter(const Json &value);
    void invoke(const Json *identity, const std::string &kernel, const Json &parameters,
                const std::vector<std::pair<uint8_t *, size_t>> &inputs,
                const std::vector<std::pair<uint8_t *, size_t>> &outputs,
                const Json &scratch);
    void call(const Json &instruction);
    void program(const Json &instructions);
    void reduce(const Json &instruction);
    void cache_read(const Json &instruction);
    void cache_write(const Json &instruction);
    void scalar_copy(const std::string &source, const std::string &destination,
                     const Json &transfer, size_t source_offset = 0, size_t destination_offset = 0);
    void copy_view(const View &destination, const View &source);
    void copy_cache(const std::string &name, bool write, const View &destination, const View &source);
    void note_sample();
    void begin_token_stream(TokenSink sink);
    void emit_loop_tokens(const Json &instruction);
    void finish_token_stream();

    std::filesystem::path directory;
    Json base;
    const Json *active_program = nullptr;
    std::unordered_map<std::string, const Json *> buffer_index;
    std::unordered_map<std::string, const Json *> expression_index;
    std::unordered_map<std::string, Shape> shapes;
    std::unordered_map<std::string, int64_t> lengths;
    std::unordered_map<std::string, Integer> expression_values;
    std::unordered_map<std::string, Input> inputs;
    std::unordered_map<std::string, Kernel> kernels;
    std::unordered_map<const Json *, PreparedCall> calls;
    std::vector<void *> libraries;
    Allocation arena;
    Allocation cache;
    Mapping weights;
    Mapping constants;
    size_t weights_start = 0;
    TrillimCpuContext *pool = nullptr;
    std::unordered_set<std::string> probe_outputs;
    std::vector<uint64_t> sample_times;
    uint64_t request_start = 0;
    uint64_t host_copy_bytes = 0;
    bool profile = false;
    uint64_t native_ns = 0;
    uint64_t native_calls = 0;
    uint64_t host_copy_ns = 0;
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> native_by_kernel;
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> instruction_ns;
    uint64_t profiled_instruction_ns = 0;
    std::unordered_map<std::string, CacheCopyProfile> copy_by_cache;

private:
    void emit_tokens(const View &tokens, size_t count);
    TokenStream token_stream;
};
}
#endif
