#include "runtime.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string_view>
#include <system_error>
#include <type_traits>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <malloc.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace trillim::cpu {
namespace {

size_t add(size_t left, size_t right) {
    if (right > SIZE_MAX - left) {
        throw std::overflow_error("byte or element count exceeds size_t");
    }
    return left + right;
}

size_t multiply(size_t left, size_t right) {
    if (right && left > SIZE_MAX / right) {
        throw std::overflow_error("byte or element count exceeds size_t");
    }
    return left * right;
}

template<typename T>
T integer(const Json &value) {
    static_assert(std::is_integral_v<T>);
    if (value.is_boolean()) {
        return static_cast<T>(value.get<bool>());
    }
    if (value.is_number_unsigned()) {
        const auto number = value.get<uint64_t>();
        if (std::in_range<T>(number)) {
            return static_cast<T>(number);
        }
    } else if (value.is_number_integer()) {
        const auto number = value.get<int64_t>();
        if (std::in_range<T>(number)) {
            return static_cast<T>(number);
        }
    }
    throw std::out_of_range("integer out of range: " + value.dump());
}

size_t extent(int64_t value) {
    if (!std::in_range<size_t>(value)) {
        throw std::out_of_range("invalid tensor extent");
    }
    return static_cast<size_t>(value);
}

std::pair<size_t, size_t> matrix_shape(const Shape &shape) {
    if (shape.size() < 2) {
        return {1, elements(shape)};
    }
    return {extent(shape.front()), elements(Shape(shape.begin() + 1, shape.end()))};
}

uint8_t *slice(uint8_t *base, size_t capacity, size_t offset, size_t size) {
    if (offset > capacity || size > capacity - offset) {
        throw std::out_of_range("buffer exceeds its backing storage");
    }
    return offset ? base + offset : base;
}

std::pair<const std::string *, const Json *> backing(const Runtime &runtime, const std::string &name) {
    auto *tensor = &name;
    for (size_t depth = 0; depth < runtime.buffer_index.size(); ++depth) {
        const auto &record = runtime.buffer(*tensor);
        if (record.at("storage") != "alias") {
            return {tensor, &record};
        }
        tensor = &record.at("source").string();
    }
    throw std::invalid_argument(name + ": cyclic alias binding");
}

size_t cache_offset(Runtime &runtime, const Json &record) {
    const auto offset = record.find("byte_offset");
    return offset == nullptr ? 0 : runtime.evaluate(*offset).size();
}

bool overlaps(const void *left, size_t left_bytes, const void *right, size_t right_bytes) {
    if (!left_bytes || !right_bytes) {
        return false;
    }
    const auto start = reinterpret_cast<uintptr_t>(left);
    const auto other = reinterpret_cast<uintptr_t>(right);
    return start <= other ? other - start < left_bytes : start - other < right_bytes;
}

#if defined(_WIN32)
std::system_error windows_error(DWORD code, const std::string &message) {
    return std::system_error(static_cast<int>(code), std::system_category(), message);
}
#endif

void *open_library(const std::filesystem::path &path) {
    const auto absolute = std::filesystem::absolute(path);
#if defined(_WIN32)
    auto library = LoadLibraryExW(
        absolute.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    );
    if (!library) {
        throw windows_error(GetLastError(), "cannot load " + absolute.string());
    }
#else
    auto library = dlopen(absolute.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        throw std::runtime_error("cannot load " + absolute.string() + ": " + dlerror());
    }
#endif
    return library;
}

template<typename T>
T symbol(void *library, const std::string &name) {
#if defined(_WIN32)
    auto entry = GetProcAddress(static_cast<HMODULE>(library), name.c_str());
    if (!entry) {
        throw windows_error(GetLastError(), "cannot load symbol " + name);
    }
#else
    dlerror();
    void *entry = dlsym(library, name.c_str());
    if (const char *error = dlerror()) {
        throw std::runtime_error(error);
    }
#endif
    if (!entry) {
        throw std::runtime_error("invalid native symbol " + name);
    }
    return std::bit_cast<T>(entry);
}

void close_library(void *library) noexcept {
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(library));
#else
    dlclose(library);
#endif
}

void close_native(TrillimCpuContext *pool, const std::vector<void *> &libraries) noexcept {
    if (pool) {
        trillim_cpu_cancel(pool);
        trillim_cpu_destroy(pool);
    }
    for (auto library = libraries.rbegin(); library != libraries.rend(); ++library) {
        close_library(*library);
    }
}

enum class FieldType { Size, Int32, UInt32, Int64, UInt64, Float };

FieldType field_type(std::string_view name) {
    if (name == "size_t") {
        return FieldType::Size;
    }
    if (name == "int32_t") {
        return FieldType::Int32;
    }
    if (name == "uint32_t") {
        return FieldType::UInt32;
    }
    if (name == "int64_t") {
        return FieldType::Int64;
    }
    if (name == "uint64_t") {
        return FieldType::UInt64;
    }
    if (name == "float") {
        return FieldType::Float;
    }
    throw std::invalid_argument("unknown kernel parameter type " + std::string(name));
}

template<typename T>
void store(uint8_t *destination, T value) {
    std::memcpy(destination, &value, sizeof(value));
}

template<typename Function>
decltype(auto) dispatch_field(FieldType type, Function function) {
    switch (type) {
    case FieldType::Size: return function.template operator()<size_t>();
    case FieldType::Int32: return function.template operator()<int32_t>();
    case FieldType::UInt32: return function.template operator()<uint32_t>();
    case FieldType::Int64: return function.template operator()<int64_t>();
    case FieldType::UInt64: return function.template operator()<uint64_t>();
    case FieldType::Float: return function.template operator()<float>();
    }
    throw std::logic_error("unknown kernel parameter type");
}

size_t aligned_size(size_t size, size_t alignment) {
    const auto remainder = size % alignment;
    return remainder ? add(size, alignment - remainder) : size;
}

void verify_schema(
    const TrillimParameterSchema &schema, const Json &metadata, const std::string &name
) {
    if (!schema.requirements || (schema.field_count && !schema.fields) ||
        (schema.input_count && !schema.input_alignments) ||
        (schema.output_count && !schema.output_alignments) ||
        !std::has_single_bit(schema.scratch_alignment)) {
        throw std::invalid_argument(name + ": incomplete parameter schema");
    }
    const auto &record = metadata.at("parameters");
    const auto &fields = record.at("fields");
    if (!fields.is_array() || fields.size() != schema.field_count ||
        integer<size_t>(record.at("size")) != schema.size ||
        integer<size_t>(record.at("alignment")) != schema.alignment ||
        integer<size_t>(metadata.at("inputs")) != schema.input_count ||
        integer<size_t>(metadata.at("outputs")) != schema.output_count ||
        integer<size_t>(metadata.at("scratch_alignment")) != schema.scratch_alignment) {
        throw std::invalid_argument(name + ": manifest disagrees with native parameter schema");
    }
    const auto validate_alignments = [&](const Json &declared, const size_t *actual, size_t count) {
        if (!declared.is_array() || declared.size() != count) {
            throw std::invalid_argument(name + ": manifest disagrees with native alignments");
        }
        for (size_t index = 0; index < count; ++index) {
            if (!std::has_single_bit(actual[index]) || integer<size_t>(declared[index]) != actual[index]) {
                throw std::invalid_argument(name + ": manifest disagrees with native alignments");
            }
        }
    };
    validate_alignments(metadata.at("input_alignments"), schema.input_alignments, schema.input_count);
    validate_alignments(metadata.at("output_alignments"), schema.output_alignments, schema.output_count);

    size_t size = 0;
    size_t alignment = 1;
    std::unordered_set<std::string_view> names;
    for (size_t index = 0; index < schema.field_count; ++index) {
        const auto &field = schema.fields[index];
        const auto &declared = fields.at(index);
        if (!field.name || !field.type || !names.insert(field.name).second ||
            declared.at("name") != field.name || declared.at("type") != field.type ||
            integer<size_t>(declared.at("offset")) != field.offset ||
            declared.at("length") != (field.length ? Json(field.length) : Json(nullptr))) {
            throw std::invalid_argument(name + ": invalid schema field");
        }
        const auto layout = dispatch_field(field_type(field.type), []<typename T> {
            return std::pair{sizeof(T), alignof(T)};
        });
        const auto [width, align] =
            field.length ? std::pair{sizeof(void *), alignof(void *)} : layout;
        size = aligned_size(size, align);
        if (field.offset != size) {
            throw std::invalid_argument(name + ": inconsistent schema field offset");
        }
        size = add(size, width);
        alignment = std::max(alignment, align);
    }
    if (aligned_size(size, alignment) != schema.size || alignment != schema.alignment) {
        throw std::invalid_argument(name + ": inconsistent parameter schema");
    }
    for (size_t index = 0; index < schema.field_count; ++index) {
        const auto &field = schema.fields[index];
        if (!field.length) {
            continue;
        }
        bool valid_length = false;
        for (size_t candidate = 0; candidate < schema.field_count; ++candidate) {
            const auto &other = schema.fields[candidate];
            if (std::string_view(other.name) == field.length) {
                valid_length = !other.length && field_type(other.type) != FieldType::Float;
                break;
            }
        }
        if (!valid_length) {
            throw std::invalid_argument(name + ": invalid schema array length field");
        }
    }
}

void validate_slot(
    const std::pair<uint8_t *, size_t> &buffer, size_t bytes, size_t alignment,
    const std::string &kernel, const char *kind
) {
    if (bytes > buffer.second || (bytes && !buffer.first) ||
        reinterpret_cast<uintptr_t>(buffer.first) % alignment) {
        throw std::invalid_argument(kernel + ": " + kind + " footprint/alignment invalid");
    }
}

TrillimValue read_scalar(const View &tensor, size_t index) {
    if (!itemsize(tensor.dtype)) {
        throw std::invalid_argument("packed tensor cannot be read as a native scalar");
    }
    if (index >= elements(tensor.shape)) {
        throw std::out_of_range("tensor scalar index exceeds its shape");
    }
    size_t offset = 0;
    for (size_t axis = tensor.shape.size(); axis-- > 0;) {
        const auto dimension = static_cast<size_t>(tensor.shape[axis]);
        const auto coordinate = index % dimension;
        if (coordinate) {
            offset += coordinate * static_cast<size_t>(tensor.strides[axis]);
        }
        index /= dimension;
    }
    return trillim_load(
        tensor.data, offset, tensor.dtype == TRILLIM_BF16 ? TRILLIM_UINT16 : tensor.dtype
    );
}

double numeric_value(const Json &value) {
    if (value.is_number_float()) {
        return value.get<double>();
    }
    return json_integer_exact(value).real();
}

float float32_value(const Json &value) {
    if (value.is_number_unsigned()) {
        return static_cast<float>(value.get<uint64_t>());
    }
    if (value.is_number_integer() && !value.is_big_integer()) {
        return static_cast<float>(value.get<int64_t>());
    }
    return static_cast<float>(numeric_value(value));
}

uint16_t cache_fp16(double value) {
    if (!std::isfinite(value)) {
        return trillim_to_fp16(static_cast<float>(value));
    }
    const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
    const double magnitude = std::abs(value);
    if (magnitude >= 65520.0) {
        return sign | 0x7c00;
    }
    const int exponent = magnitude < 0x1.0p-14 ? -14 : std::ilogb(magnitude);
    const double significand = std::ldexp(magnitude, 10 - exponent);
    const auto whole = static_cast<uint16_t>(significand);
    const double fraction = significand - whole;
    const auto rounded = whole + (fraction > 0.5 || (fraction == 0.5 && (whole & 1)));
    return sign | static_cast<uint16_t>((exponent + 14) * 1024 + rounded);
}

uint64_t cache_integer(const Json &value, uint32_t dtype) {
    if (value.is_number_unsigned()) {
        return value.get<uint64_t>();
    }
    if (value.is_number_integer() && !value.is_big_integer()) {
        return static_cast<uint64_t>(value.get<int64_t>());
    }
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    const double number = std::trunc(numeric_value(value));
    if (dtype == TRILLIM_UINT64) {
        if (!(number >= 0)) {
            return 0;
        }
        return number >= 0x1.0p64 ? UINT64_MAX : static_cast<uint64_t>(number);
    }
    if (dtype == TRILLIM_UINT32) {
        if (!(number >= 0)) {
            return 0;
        }
        return number >= 0x1.0p32 ? UINT32_MAX : static_cast<uint32_t>(number);
    }
    if (std::isnan(number)) {
        return 0;
    }
    if (dtype == TRILLIM_INT64) {
        if (number >= 0x1.0p63) {
            return static_cast<uint64_t>(INT64_MAX);
        }
        return static_cast<uint64_t>(static_cast<int64_t>(std::max(number, -0x1.0p63)));
    }
    return static_cast<uint64_t>(static_cast<int32_t>(
        std::clamp(number, static_cast<double>(INT32_MIN), static_cast<double>(INT32_MAX))
    ));
}

bool parameter_equal(const Json &left, const Json &right) {
    if (left.is_array() && right.is_array()) {
        return std::equal(left.begin(), left.end(), right.begin(), right.end(), parameter_equal);
    }
    if (left.is_number_integer() && right.is_number_integer() &&
        !left.is_big_integer() && !right.is_big_integer() &&
        left.is_number_unsigned() != right.is_number_unsigned()) {
        if (left.is_number_unsigned()) {
            return std::cmp_equal(left.get<uint64_t>(), right.get<int64_t>());
        }
        return std::cmp_equal(left.get<int64_t>(), right.get<uint64_t>());
    }
    return left.type() == right.type() && left == right;
}

void prepare(
    PreparedCall &call, const Kernel &kernel, const Json &parameters,
    const std::string &name
) {
    const auto &schema = *kernel.schema;
    if (!parameters.is_object() || parameters.size() != schema.field_count) {
        throw std::invalid_argument(name + ": kernel parameter fields do not match the native schema");
    }
    if (call.kernel == &kernel) {
        bool unchanged = true;
        for (const auto &[field, value] : parameters.items()) {
            const auto previous = call.parameters.find(field);
            if (previous == nullptr || !parameter_equal(value, *previous)) {
                unchanged = false;
                break;
            }
        }
        if (unchanged) {
            return;
        }
    } else {
        call.packed = Allocation(schema.size, schema.alignment);
        std::memset(call.packed.data, 0, call.packed.size);
        call.arrays.resize(schema.field_count);
        call.input_sizes.resize(schema.input_count);
        call.output_sizes.resize(schema.output_count);
        call.inputs.resize(schema.input_count);
        call.outputs.resize(schema.output_count);
        call.parameters = Json::object();
        call.kernel = &kernel;
    }
    for (size_t index = 0; index < schema.field_count; ++index) {
        const auto &field = schema.fields[index];
        const auto &argument = parameters.at(field.name);
        const auto previous = call.parameters.find(field.name);
        if (field.length) {
            const auto &length = parameters.at(field.length);
            if (!argument.is_array() || argument.size() != integer<size_t>(length)) {
                throw std::invalid_argument(name + ": invalid kernel parameter array " + field.name);
            }
        }
        if (previous != nullptr && parameter_equal(argument, *previous)) {
            continue;
        }
        dispatch_field(field_type(field.type), [&]<typename T> {
            auto write = [](uint8_t *destination, const Json &value) {
                if constexpr (std::is_same_v<T, float>) {
                    store(destination, static_cast<float>(numeric_value(value)));
                } else {
                    store(destination, integer<T>(value));
                }
            };
            auto *destination = call.packed.data + field.offset;
            if (field.length) {
                auto &array = call.arrays[index];
                array.resize(std::max(sizeof(T), multiply(argument.size(), sizeof(T))));
                for (size_t element = 0; element < argument.size(); ++element) {
                    write(array.data() + element * sizeof(T), argument[element]);
                }
                store(destination, array.data());
            } else {
                write(destination, argument);
            }
        });
    }
    std::fill(call.input_sizes.begin(), call.input_sizes.end(), 0);
    std::fill(call.output_sizes.begin(), call.output_sizes.end(), 0);
    call.scratch_size = 0;
    const auto status = schema.requirements(
        call.packed.data, call.input_sizes.data(), call.output_sizes.data(), &call.scratch_size
    );
    if (status != TRILLIM_OK) {
        throw std::invalid_argument(
            name + ": kernel rejected bounded parameters (status " + std::to_string(status) + ")"
        );
    }
    call.parameters = parameters;
}

void invoke_bound(Runtime &runtime, const PreparedCall &call, const std::string &name) {
    auto *pool = runtime.pool;
    const uint64_t started = runtime.profile ? now_ns() : 0;
    const auto status = trillim_cpu_invoke(
        pool, call.kernel->function, call.outputs.data(),
        call.inputs.data(), call.scratch, call.packed.data
    );
    if (runtime.profile) {
        const uint64_t elapsed = now_ns() - started;
        runtime.native_ns += elapsed;
        ++runtime.native_calls;
        auto &[kernel_ns, kernel_calls] = runtime.native_by_kernel[name];
        kernel_ns += elapsed;
        ++kernel_calls;
    }
    if (status == TRILLIM_CANCELLED) {
        throw Cancelled(name + ": execution cancelled");
    }
    if (status != TRILLIM_OK) {
        throw std::runtime_error(
            name + " failed with status " + std::to_string(status) +
            ", parameters=" + call.parameters.dump()
        );
    }
}

bool expression_parameter(const Json &value) {
    return value.is_object() && !value.contains("float");
}

} // namespace

Allocation::Allocation(size_t bytes, size_t alignment) : size(bytes) {
    if (!std::has_single_bit(alignment)) {
        throw std::invalid_argument("allocation alignment must be a power of two");
    }
    alignment = std::max(alignment, alignof(void *));
    const auto capacity = std::max<size_t>(1, size);
#if defined(_WIN32)
    void *owner = _aligned_malloc(capacity, alignment);
    if (!owner) {
        throw std::bad_alloc();
    }
#else
    void *owner = nullptr;
    const int status = posix_memalign(&owner, alignment, capacity);
    if (status) {
        throw std::system_error(status, std::generic_category(), "cannot allocate buffer");
    }
#endif
    data = static_cast<uint8_t *>(owner);
}

Allocation::~Allocation() {
#if defined(_WIN32)
    _aligned_free(data);
#else
    std::free(data);
#endif
}

Allocation::Allocation(Allocation &&other) noexcept
    : data(std::exchange(other.data, nullptr)), size(std::exchange(other.size, 0)) {}

Allocation &Allocation::operator=(Allocation &&other) noexcept {
    std::swap(data, other.data);
    std::swap(size, other.size);
    return *this;
}

Mapping::Mapping(const std::filesystem::path &path) {
#if defined(_WIN32)
    const auto file = CreateFileW(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr
    );
    if (file == INVALID_HANDLE_VALUE) {
        throw windows_error(GetLastError(), path.string());
    }
    LARGE_INTEGER length{};
    if (!GetFileSizeEx(file, &length)) {
        const auto error = GetLastError();
        CloseHandle(file);
        throw windows_error(error, path.string());
    }
    if (!std::in_range<size_t>(length.QuadPart)) {
        CloseHandle(file);
        throw std::overflow_error("file size exceeds size_t: " + path.string());
    }
    size = static_cast<size_t>(length.QuadPart);
    if (size == 0) {
        CloseHandle(file);
        return;
    }
    const auto mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    const auto mapping_error = GetLastError();
    CloseHandle(file);
    if (!mapping) {
        throw windows_error(mapping_error, path.string());
    }
    void *address = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, size);
    const auto view_error = GetLastError();
    CloseHandle(mapping);
    if (!address) {
        throw windows_error(view_error, path.string());
    }
#else
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        throw std::system_error(errno, std::generic_category(), path.string());
    }
    struct stat info {};
    if (fstat(descriptor, &info) < 0) {
        const int error = errno;
        close(descriptor);
        throw std::system_error(error, std::generic_category(), path.string());
    }
    if (!std::in_range<size_t>(info.st_size)) {
        close(descriptor);
        throw std::overflow_error("file size exceeds size_t: " + path.string());
    }
    size = static_cast<size_t>(info.st_size);
    if (size == 0) {
        close(descriptor);
        return;
    }
    void *address = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, descriptor, 0);
    const int error = errno;
    close(descriptor);
    if (address == MAP_FAILED) {
        throw std::system_error(error, std::generic_category(), path.string());
    }
#endif
    data = static_cast<uint8_t *>(address);
}

Mapping::~Mapping() {
    if (data) {
#if defined(_WIN32)
        UnmapViewOfFile(data);
#else
        munmap(data, size);
#endif
    }
}

Mapping::Mapping(Mapping &&other) noexcept
    : data(std::exchange(other.data, nullptr)), size(std::exchange(other.size, 0)) {}

Mapping &Mapping::operator=(Mapping &&other) noexcept {
    std::swap(data, other.data);
    std::swap(size, other.size);
    return *this;
}

size_t itemsize(uint32_t dtype) {
    return trillim_itemsize(dtype);
}

uint32_t dtype_id(const std::string &name) {
    static const std::unordered_map<std::string, uint32_t> types = {
        {"bool", TRILLIM_BOOL}, {"int8", TRILLIM_INT8}, {"uint8", TRILLIM_UINT8},
        {"int16", TRILLIM_INT16}, {"uint16", TRILLIM_UINT16}, {"int32", TRILLIM_INT32},
        {"uint32", TRILLIM_UINT32}, {"int64", TRILLIM_INT64}, {"uint64", TRILLIM_UINT64},
        {"fp16", TRILLIM_FP16}, {"bf16", TRILLIM_BF16}, {"fp32", TRILLIM_FP32},
        {"row_int8", TRILLIM_ROW_INT8}, {"mat_ternary", TRILLIM_MAT_TERNARY},
        {"grp_int8", TRILLIM_GRP_INT8}, {"grp_int4", TRILLIM_GRP_INT4},
        {"grp_int5", TRILLIM_GRP_INT5}, {"dubgrp_int6", TRILLIM_DUBGRP_INT6},
        {"grp_binary", TRILLIM_GRP_BINARY}, {"grp_ternary", TRILLIM_GRP_TERNARY},
        {"mxfp4", TRILLIM_MXFP4}, {"fp8_e4m3", TRILLIM_FP8_E4M3},
    };
    return types.at(name);
}

size_t elements(const Shape &shape) {
    size_t product = 1;
    for (auto dimension : shape) {
        product = multiply(product, extent(dimension));
    }
    return product;
}

Shape contiguous_strides(const Shape &shape) {
    Shape result(shape.size());
    size_t stride = 1;
    for (size_t axis = shape.size(); axis-- > 0;) {
        if (!std::in_range<int64_t>(stride)) {
            throw std::overflow_error("tensor stride exceeds int64");
        }
        result[axis] = static_cast<int64_t>(stride);
        stride = multiply(stride, extent(shape[axis]));
    }
    return result;
}

size_t packed_bytes(uint32_t dtype, size_t rows, size_t columns) {
    size_t bytes = 0;
    if ((!trillim_itemsize(dtype) && (!rows || !columns)) ||
        !trillim_packed_payload_size(rows, columns, dtype, &bytes)) {
        throw std::invalid_argument("invalid packed tensor shape or dtype");
    }
    return bytes;
}

uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

void Runtime::note_sample() {
    sample_times.push_back(now_ns() - request_start);
}

Runtime::Runtime(
    const std::filesystem::path &model_directory, size_t threads,
    uint64_t capabilities
) : directory(model_directory) {
    try {
        std::ifstream manifest(directory / "executable.json");
        if (!manifest) {
            throw std::runtime_error("cannot open model executable manifest");
        }
        base = read_manifest(manifest);
        if (std::endian::native != std::endian::little ||
            base.value("format", std::string("trillim.executable")) != "trillim.executable" ||
            base.value("version", Json(1)) != 1 ||
            base.value("abi_version", Json(TRILLIM_KERNEL_ABI_VERSION)) != TRILLIM_KERNEL_ABI_VERSION ||
            base.value("endian", std::string("little")) != "little") {
            throw std::invalid_argument("incompatible executable manifest");
        }
        if (threads == 0) {
            threads = integer<size_t>(base.at("threads"));
        }
        if (threads < 1 || threads > 65536) {
            throw std::invalid_argument("threads must be between 1 and 65536");
        }

        arena = Allocation(integer<size_t>(base.at("arena_size")));
        cache = Allocation(integer<size_t>(base.at("cache_size")));
        std::memset(arena.data, 0, arena.size);
        std::memset(cache.data, 0, cache.size);
        weights = Mapping(directory / base.value("weights", std::string("weights.bin")));
        constants = Mapping(directory / base.value("constants", std::string("constants.bin")));
        uint64_t header_bytes = 0;
        if (weights.size < sizeof(header_bytes)) {
            throw std::invalid_argument("truncated weight bundle header");
        }
        std::memcpy(&header_bytes, weights.data, sizeof(header_bytes));
        if (header_bytes > 100'000'000 || header_bytes > weights.size - sizeof(header_bytes)) {
            throw std::invalid_argument("invalid weight bundle header length");
        }
        weights_start = sizeof(header_bytes) + static_cast<size_t>(header_bytes);
        const Json weight_header = Json::parse(
            weights.data + sizeof(header_bytes), weights.data + weights_start
        );
        if (weights_start % 64 || weight_header.at("data_start") != weights_start ||
            weight_header.value("format", std::string("trillim.weights")) != "trillim.weights" ||
            weight_header.value("version", Json(1)) != 1 ||
            weight_header.value("endian", std::string("little")) != "little") {
            throw std::invalid_argument("weight bundle data_start disagrees with header");
        }

        if (base.at("backend") == "x86") {
            capabilities |= trillim_cpu_features();
        }

        std::unordered_map<std::string, void *> kernel_libraries;
        libraries.reserve(base.at("kernels").size());
        for (const auto &[name, metadata] : base.at("kernels").items()) {
            const auto &filename = metadata.at("library").string();
            auto [entry, fresh] = kernel_libraries.try_emplace(filename);
            if (fresh) {
                entry->second = open_library(directory / filename);
                libraries.push_back(entry->second);
            }
            void *library = entry->second;
            const auto &name_in_library = metadata.at("symbol").string();
            const auto *required = symbol<const TrillimCapabilities *(*)()>(
                library, name_in_library + "_capabilities"
            )();
            if (!required || required->abi_version != TRILLIM_KERNEL_ABI_VERSION) {
                throw std::invalid_argument(name + ": incompatible kernel ABI");
            }
            if (required->required != integer<uint64_t>(metadata.at("capabilities")) ||
                (required->required & capabilities) != required->required) {
                throw std::invalid_argument(name + ": kernel capabilities unavailable on this device");
            }
            const auto *schema = symbol<const TrillimParameterSchema *(*)()>(
                library, name_in_library + "_parameters"
            )();
            if (!schema) {
                throw std::invalid_argument(name + ": missing kernel parameter schema");
            }
            verify_schema(*schema, metadata, name);
            kernels.emplace(name, Kernel{symbol<TrillimKernel>(library, name_in_library), schema});
        }

        pool = trillim_cpu_create(threads);
        if (!pool) {
            throw std::runtime_error("cannot create native pool");
        }
        activate(base);
        for (const auto &[name, metadata] : base.at("caches").items()) {
            lengths.emplace(name, 0);
            const auto dtype = dtype_id(metadata.at("dtype").string());
            Shape dimensions;
            for (const auto &dimension : metadata.at("shape")) {
                dimensions.push_back(integer<int64_t>(dimension));
            }
            const auto [rows, columns] = matrix_shape(dimensions);
            const auto bytes = integer<size_t>(metadata.at("size"));
            auto *destination = slice(cache.data, cache.size, integer<size_t>(metadata.at("offset")), bytes);
            if (packed_bytes(dtype, rows, columns) != bytes) {
                throw std::invalid_argument(name + ": cache size disagrees with its shape");
            }
            const auto initial = parameter(metadata.at("initial"));
            const auto width = itemsize(dtype);
            const bool wide_float = (!width || trillim_float(dtype)) && integer_literal(initial);
            if (!initial.is_number() && !initial.is_boolean() && !wide_float) {
                throw std::invalid_argument(name + ": invalid numeric cache initial value");
            }
            const auto count = multiply(rows, columns);
            if (!width) {
                std::vector<float> values(count, float32_value(initial));
                auto encoded = encode_packed(values.data(), dtype, rows, columns, *this);
                if (encoded.size != bytes) {
                    throw std::runtime_error(name + ": encoded cache size disagrees with its allocation");
                }
                std::memcpy(destination, encoded.data, bytes);
                continue;
            }
            if (numeric_value(initial) == 0) {
                continue;
            }
            if (dtype == TRILLIM_FP16) {
                const auto bits = cache_fp16(numeric_value(initial));
                for (size_t index = 0; index < count; ++index) {
                    store(destination + index * sizeof(bits), bits);
                }
                continue;
            }
            TrillimValue value{};
            if (trillim_float(dtype)) {
                value.f = float32_value(initial);
            } else if (dtype == TRILLIM_BOOL) {
                value.u = 1;
            } else {
                value.u = cache_integer(initial, dtype);
            }
            for (size_t index = 0; index < count; ++index) {
                trillim_store(destination, index, dtype, value);
            }
        }
    } catch (...) {
        close_native(pool, libraries);
        throw;
    }
}

Runtime::~Runtime() {
    close_native(pool, libraries);
}

void Runtime::activate(const Json &plan) {
    const auto *next_program = &plan.at("program");
    if (active_program == next_program) {
        return;
    }
    decltype(buffer_index) next_buffers;
    const auto &buffers = plan.at("buffers");
    next_buffers.reserve(buffers.size());
    for (const auto &[name, metadata] : buffers.items()) {
        next_buffers.emplace(name, &metadata);
    }
    decltype(expression_index) next_expressions;
    auto expressions = plan.find("expressions");
    if (expressions != nullptr) {
        next_expressions.reserve(expressions->size());
        for (const auto &[name, value] : expressions->items()) {
            next_expressions.emplace(name, &value);
        }
    }
    buffer_index.swap(next_buffers);
    expression_index.swap(next_expressions);
    active_program = next_program;
}

const Json &Runtime::buffer(const std::string &name) const {
    return *buffer_index.at(name);
}

void Runtime::bind_inputs(std::vector<Input> incoming) {
    const auto &ports = base.at("inputs");
    if (incoming.size() != ports.size()) {
        throw std::invalid_argument("unexpected number of model inputs");
    }
    const auto &base_buffers = base.at("buffers");
    std::unordered_map<std::string, Input> bound_inputs;
    std::unordered_map<std::string, Shape> bound_shapes;
    bound_inputs.reserve(incoming.size());
    bound_shapes.reserve(incoming.size());
    for (auto &input : incoming) {
        const auto port = ports.find(input.name);
        if (port == nullptr || bound_inputs.contains(input.name)) {
            throw std::invalid_argument(input.name + ": unexpected or duplicate input");
        }
        const auto &tensor = port->string();
        const auto &record = base_buffers.at(tensor);
        const auto &fixed = record.at("shape");
        const auto &maximum = record.at("maximum_shape");
        if (record.at("storage") != "input" || record.at("source") != input.name ||
            input.shape.size() != maximum.size() || fixed.size() != maximum.size()) {
            throw std::invalid_argument(input.name + ": invalid input rank");
        }
        for (size_t axis = 0; axis < input.shape.size(); ++axis) {
            const auto actual = input.shape[axis];
            const auto &dimension = fixed.at(axis);
            if (actual < 0 || Integer{actual} > json_integer_exact(maximum.at(axis)) ||
                (integer_literal(dimension) && Integer{actual} != json_integer_exact(dimension))) {
                throw std::invalid_argument(input.name + ": input shape exceeds or differs from its compiled dimensions");
            }
        }
        const auto [rows, columns] = matrix_shape(input.shape);
        const auto dtype = dtype_id(record.at("dtype").string());
        if (input.payload.size != packed_bytes(dtype, rows, columns) ||
            (input.payload.size && !input.payload.data)) {
            throw std::invalid_argument(input.name + ": invalid input payload size");
        }
        if (!bound_shapes.emplace(tensor, input.shape).second) {
            throw std::invalid_argument(input.name + ": duplicate input");
        }
        const auto name = input.name;
        bound_inputs.emplace(name, std::move(input));
    }

    const Json *plan = &base;
    const auto workload = base.find("workload");
    if (workload != nullptr && !workload->is_null() && !workload->empty()) {
        const auto &tensor = ports.at(workload->at("input").string())
            .string();
        const auto count = bound_shapes.at(tensor).at(integer<size_t>(workload->at("axis")));
        for (const auto &candidate : base.at("plans")) {
            if (integer<int64_t>(candidate.at("minimum_tokens")) <= count &&
                count <= integer<int64_t>(candidate.at("maximum_tokens"))) {
                plan = &candidate;
                break;
            }
        }
    }
    const auto &plan_buffers = plan->at("buffers");
    for (auto &[name, input] : bound_inputs) {
        const auto &tensor = ports.at(name).string();
        const auto &base_record = base_buffers.at(tensor);
        const auto &plan_record = plan_buffers.at(tensor);
        if (plan_record.at("storage") != "input" || plan_record.at("source") != name ||
            plan_record.at("dtype") != base_record.at("dtype")) {
            throw std::invalid_argument(name + ": workload input disagrees with the base input");
        }
        const auto base_alignment = integer<size_t>(base_record.at("alignment"));
        const auto plan_alignment = integer<size_t>(plan_record.at("alignment"));
        if (!std::has_single_bit(base_alignment) || !std::has_single_bit(plan_alignment)) {
            throw std::invalid_argument(name + ": invalid input alignment");
        }
        const auto alignment = std::max(base_alignment, plan_alignment);
        if (!input.payload.data || reinterpret_cast<uintptr_t>(input.payload.data) % alignment) {
            Allocation aligned(input.payload.size, alignment);
            if (aligned.size) {
                std::memcpy(aligned.data, input.payload.data, aligned.size);
            }
            input.payload = std::move(aligned);
        }
    }
    activate(*plan);
    inputs = std::move(bound_inputs);
    shapes = std::move(bound_shapes);
    expression_values.clear();
    for (auto &[identity, call] : calls) {
        call.bound = false;
    }
}

Shape Runtime::shape(const std::string &name) {
    auto recorded = shapes.find(name);
    if (recorded != shapes.end()) {
        return recorded->second;
    }
    const auto &dimensions = buffer(name).at("shape");
    Shape result;
    result.reserve(dimensions.size());
    for (const auto &dimension : dimensions) {
        result.push_back(evaluate(dimension).int64());
    }
    return result;
}

Shape Runtime::strides(const std::string &name) {
    const auto &layout = buffer(name).at("strides");
    Shape result;
    result.reserve(layout.size());
    for (const auto &stride : layout) {
        result.push_back(evaluate(stride).int64());
    }
    return result;
}

uint8_t *Runtime::address(const std::string &name) {
    const auto &record = *backing(*this, name).second;
    const auto &storage = record.at("storage").string();
    if (storage == "input") {
        return inputs.at(record.at("source").string()).payload.data;
    }
    if (storage == "cache") {
        const auto &metadata = base.at("caches").at(record.at("source").string());
        const auto capacity = integer<size_t>(metadata.at("size"));
        auto *start = slice(cache.data, cache.size, integer<size_t>(metadata.at("offset")), capacity);
        return slice(start, capacity, cache_offset(*this, record), 0);
    }
    const auto offset = integer<size_t>(record.at("offset"));
    const auto size = integer<size_t>(record.at("size"));
    if (storage == "arena") {
        return slice(arena.data, arena.size, offset, size);
    }
    if (storage == "constant") {
        return slice(constants.data, constants.size, offset, size);
    }
    if (storage == "weight") {
        return slice(weights.data, weights.size, add(weights_start, offset), size);
    }
    throw std::invalid_argument("unknown storage " + storage);
}

size_t Runtime::span(const std::string &name) {
    const auto &record = *backing(*this, name).second;
    const auto &storage = record.at("storage").string();
    if (storage == "input") {
        return inputs.at(record.at("source").string()).payload.size;
    }
    if (storage == "cache") {
        const auto &metadata = base.at("caches").at(record.at("source").string());
        const auto capacity = integer<size_t>(metadata.at("size"));
        const auto offset = cache_offset(*this, record);
        if (offset > capacity) {
            throw std::out_of_range("cache byte offset exceeds capacity");
        }
        return capacity - offset;
    }
    return integer<size_t>(record.at("size"));
}

std::pair<size_t, size_t> Runtime::storage_shape(const std::string &name) {
    const auto [tensor, root] = backing(*this, name);
    const auto &record = *root;
    const auto &storage = record.at("storage").string();
    if (storage == "cache") {
        const auto &metadata = base.at("caches").at(record.at("source").string());
        return matrix_shape(metadata.at("shape").get<Shape>());
    }
    return matrix_shape(shape(*tensor));
}

View Runtime::view(const std::string &name) {
    View result{address(name), shape(name), strides(name),
                dtype_id(buffer(name).at("dtype").string()), span(name)};
    const auto width = itemsize(result.dtype);
    if (result.shape.size() != result.strides.size()) {
        throw std::invalid_argument("tensor shape and stride ranks disagree");
    }
    if (elements(result.shape)) {
        size_t footprint = 1;
        for (size_t axis = 0; axis < result.shape.size(); ++axis) {
            const auto dimension = result.shape[axis];
            if (dimension > 1) {
                footprint = add(footprint, multiply(extent(dimension - 1), extent(result.strides[axis])));
            }
        }
        if (width && multiply(footprint, width) > result.span) {
            throw std::out_of_range(name + ": tensor view exceeds storage");
        }
    }
    return result;
}

Integer Runtime::scalar_integer(const std::string &name, size_t index) {
    const auto value = view(name);
    const auto dtype = value.dtype == TRILLIM_BF16 ? TRILLIM_UINT16 : value.dtype;
    const auto scalar = read_scalar(value, index);
    if (trillim_float(dtype)) {
        return Integer::from_truncated_float(scalar.f);
    }
    if (trillim_unsigned(dtype)) {
        return Integer{scalar.u};
    }
    return Integer{scalar.i};
}

bool Runtime::truth(const std::string &name) {
    const auto value = view(name);
    if (elements(value.shape) != 1) {
        throw std::invalid_argument(name + ": expected a single tensor value");
    }
    const auto scalar = read_scalar(value, 0);
    if (value.dtype == TRILLIM_BF16 || trillim_unsigned(value.dtype)) {
        return scalar.u != 0;
    }
    if (trillim_float(value.dtype)) {
        return scalar.f != 0;
    }
    return scalar.i != 0;
}

void Runtime::check_cancelled() const {
    if (trillim_cpu_cancelled(pool)) {
        throw Cancelled("model execution cancelled");
    }
}

void Runtime::invoke(
    const Json *identity, const std::string &name, const Json &parameters,
    const std::vector<std::pair<uint8_t *, size_t>> &sources,
    const std::vector<std::pair<uint8_t *, size_t>> &targets, const Json &scratch
) {
    auto &call = calls[identity];
    const auto &kernel = kernels.at(name);
    prepare(call, kernel, parameters, name);
    const auto &schema = *kernel.schema;
    if (sources.size() != schema.input_count || targets.size() != schema.output_count) {
        throw std::invalid_argument(name + ": kernel operand count does not match its schema");
    }
    for (size_t index = 0; index < sources.size(); ++index) {
        validate_slot(sources[index], call.input_sizes[index], schema.input_alignments[index], name, "input");
        call.inputs[index] = sources[index].first;
    }
    for (size_t index = 0; index < targets.size(); ++index) {
        validate_slot(targets[index], call.output_sizes[index], schema.output_alignments[index], name, "output");
        call.outputs[index] = targets[index].first;
    }
    const auto scratch_span = integer<size_t>(scratch.at("size"));
    call.scratch = slice(arena.data, arena.size, integer<size_t>(scratch.at("offset")), scratch_span);
    validate_slot({call.scratch, scratch_span}, call.scratch_size, schema.scratch_alignment, name, "scratch");
    const auto transfer = base.find("transfer_kernel");
    const bool transfer_copy = transfer != nullptr && *transfer == name;
    for (size_t output = 0; output < targets.size(); ++output) {
        auto *target = targets[output].first;
        const auto size = call.output_sizes[output];
        if (overlaps(target, size, weights.data, weights.size) ||
            overlaps(target, size, constants.data, constants.size)) {
            throw std::invalid_argument(name + ": output points into read-only model storage");
        }
        if (overlaps(target, size, call.scratch, call.scratch_size)) {
            throw std::invalid_argument(name + ": scratch overlaps output");
        }
        for (size_t other = 0; other < output; ++other) {
            if (overlaps(target, size, targets[other].first, call.output_sizes[other])) {
                throw std::invalid_argument(name + ": output buffers overlap");
            }
        }
        for (size_t input = 0; input < sources.size(); ++input) {
            const auto *source = sources[input].first;
            if (!transfer_copy && overlaps(target, size, source, call.input_sizes[input]) &&
                (target != source || !schema.can_alias ||
                 !schema.can_alias(call.packed.data, output, input))) {
                throw std::invalid_argument(name + ": kernel does not allow input/output alias");
            }
        }
    }
    for (size_t input = 0; input < sources.size(); ++input) {
        if (overlaps(sources[input].first, call.input_sizes[input], call.scratch, call.scratch_size)) {
            throw std::invalid_argument(name + ": scratch overlaps input");
        }
    }
    invoke_bound(*this, call, name);
}

void Runtime::call(const Json &instruction) {
    const auto &name = instruction.at("kernel").string();
    const auto dependencies = instruction.find("argument_key");
    const bool cache_binding = dependencies != nullptr && dependencies->is_array();
    std::vector<Integer> key;
    if (cache_binding) {
        key.reserve(dependencies->size());
        for (const auto &dependency : *dependencies) {
            key.push_back(evaluate(dependency));
        }
    }
    const auto previous = calls.find(&instruction);
    const bool bound = cache_binding && previous != calls.end() && previous->second.bound;
    if (bound && previous->second.key == key) {
        const auto &call = previous->second;
        for (const auto &[tensor, dimensions] : call.shapes) {
            shapes.insert_or_assign(tensor, dimensions);
        }
        invoke_bound(*this, call, name);
        if (call.probe) {
            note_sample();
        }
        return;
    }

    const auto copies = instruction.find("shape_from");
    if (copies != nullptr) {
        for (const auto &[destination, source] : copies->items()) {
            shapes.insert_or_assign(destination, shape(source.string()));
        }
    }
    const auto &out_names = instruction.at("outputs");
    for (const auto &output : out_names) {
        const auto &tensor = output.string();
        if (copies == nullptr || !copies->contains(tensor)) {
            shapes.erase(tensor);
            shapes.emplace(tensor, shape(tensor));
        }
    }
    Json parameters = bound ? previous->second.parameters : Json::object();
    const auto &fields = instruction.at("parameters");
    const auto &varying = bound ? previous->second.varying : fields;
    for (const auto &[field, value] : varying.items()) {
        parameters[field] = parameter(value);
    }
    auto operands = [&](const Json &tensors) {
        std::vector<std::pair<uint8_t *, size_t>> result;
        result.reserve(tensors.size());
        for (const auto &value : tensors) {
            const auto &tensor = value.string();
            auto *pointer = address(tensor);
            result.emplace_back(pointer, span(tensor));
        }
        return result;
    };
    auto sources = operands(instruction.at("inputs"));
    auto targets = operands(out_names);
    invoke(&instruction, name, parameters, sources, targets, instruction.at("scratch"));

    auto &call = calls.at(&instruction);
    if (cache_binding) {
        if (!call.bound) {
            call.varying = Json::object();
            for (const auto &[field, value] : fields.items()) {
                const bool dynamic = value.is_array()
                    ? std::any_of(value.begin(), value.end(), expression_parameter)
                    : expression_parameter(value);
                if (dynamic) {
                    call.varying[field] = value;
                }
            }
        }
        call.bound = true;
        call.key = std::move(key);
        call.shapes.clear();
        for (const auto &output : out_names) {
            const auto &tensor = output.string();
            call.shapes.emplace(tensor, shapes.at(tensor));
        }
    }
    call.probe = !probe_outputs.empty() && std::any_of(
        out_names.begin(), out_names.end(),
        [&](const Json &output) {
            return probe_outputs.contains(output.string());
        }
    );
    if (call.probe) {
        note_sample();
    }
}

} // namespace trillim::cpu
