#include "runtime.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace trillim::cpu {
namespace {

using Wide = __int128_t;

int64_t expression_integer(Wide value) {
    if (value < std::numeric_limits<int64_t>::min() || value > std::numeric_limits<int64_t>::max()) {
        throw std::overflow_error("integer expression exceeds int64");
    }
    return static_cast<int64_t>(value);
}

int64_t json_integer(const Json &value) {
    return json_integer_exact(value).int64();
}

size_t byte_count(Wide value) {
    if (value < 0 || value > std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("tensor offset or size exceeds address space");
    }
    return static_cast<size_t>(value);
}

size_t axis_index(Integer position, size_t count) {
    if (position < 0) {
        position = position + count;
    }
    if (position < 0 || position >= count) {
        throw std::out_of_range("axis or scalar index is outside tensor dimensions");
    }
    return position.size();
}

const std::string &name_of(const Json &value) {
    return value.string();
}

uint32_t array_dtype(uint32_t dtype) {
    return dtype == TRILLIM_BF16 ? TRILLIM_UINT16 : dtype;
}

std::pair<size_t, size_t> matrix_dimensions(const Shape &shape) {
    size_t rows = shape.size() > 1 ? byte_count(shape[0]) : 1;
    return {rows, rows == 0 ? 0 : elements(shape) / rows};
}

std::pair<uint8_t *, size_t> cache_storage(const Runtime &runtime, const Json &metadata) {
    size_t offset = json_integer_exact(metadata.at("offset")).size();
    size_t bytes = json_integer_exact(metadata.at("size")).size();
    if (offset > runtime.cache.size || bytes > runtime.cache.size - offset) {
        throw std::runtime_error("cache view exceeds storage");
    }
    return {runtime.cache.data + offset, bytes};
}

View payload_view(uint8_t *data, size_t bytes, size_t span) {
    return {data, {Integer{bytes}.int64()}, {1}, TRILLIM_UINT8, span};
}

size_t view_footprint(const View &view, size_t width) {
    if (view.strides.size() != view.shape.size()) {
        throw std::runtime_error("tensor view shape and strides have different ranks");
    }
    size_t span = 1;
    for (size_t axis = 0; axis < view.shape.size(); ++axis) {
        if (view.shape[axis] <= 1) {
            continue;
        }
        if (view.strides[axis] < 0) {
            throw std::runtime_error("tensor view starts before its storage");
        }
        span = byte_count(span + Wide{view.shape[axis] - 1} * view.strides[axis]);
    }
    size_t bytes = byte_count(Wide{span} * width);
    if (bytes > view.span) {
        throw std::runtime_error("tensor view exceeds storage");
    }
    return bytes;
}

View broadcast_source(const View &source, const Shape &target) {
    if (source.shape.size() != source.strides.size()) {
        throw std::runtime_error("tensor view shape and strides have different ranks");
    }
    Shape strides(target.size(), 0);
    size_t source_axis = source.shape.size();
    size_t target_axis = target.size();
    while (source_axis > 0 && target_axis > 0) {
        --source_axis;
        --target_axis;
        if (source.shape[source_axis] == target[target_axis]) {
            strides[target_axis] = source.strides[source_axis];
        } else if (source.shape[source_axis] != 1) {
            throw std::runtime_error("tensor source cannot broadcast to destination shape");
        }
    }
    while (source_axis > 0) {
        if (source.shape[--source_axis] != 1) {
            throw std::runtime_error("tensor source cannot broadcast to destination shape");
        }
    }
    return {source.data, target, std::move(strides), source.dtype, source.span};
}

#if defined(__ARM_NEON)
bool interleaved_three_channels(const View &view) {
    return view.strides[0] == 3 && view.strides[1] == 1;
}

bool planar_three_channels(const View &view) {
    return view.strides[0] == 1 && view.strides[1] >= view.shape[0];
}

template <bool ToInterleaved>
void copy_three_channels(const Runtime &runtime, const View &destination, const View &source) {
    constexpr size_t width = sizeof(uint32_t);
    constexpr size_t rows_per_chunk = (1024 * 1024 / (4 * 3 * width)) * 4;
    const View &planar = ToInterleaved ? source : destination;
    const auto rows = static_cast<size_t>(planar.shape[0]);
    const auto plane_stride = static_cast<size_t>(planar.strides[1]) * width;
    for (size_t row = 0; row < rows;) {
        runtime.check_cancelled();
        const size_t end = row + std::min(rows - row, rows_per_chunk);
        for (; end - row >= 4; row += 4) {
            const size_t offset = row * width;
            if constexpr (ToInterleaved) {
                uint32x4x3_t block;
                block.val[0] = vreinterpretq_u32_u8(vld1q_u8(source.data + offset));
                block.val[1] = vreinterpretq_u32_u8(vld1q_u8(source.data + offset + plane_stride));
                block.val[2] = vreinterpretq_u32_u8(vld1q_u8(source.data + offset + 2 * plane_stride));
                vst3q_u32(reinterpret_cast<uint32_t *>(destination.data + 3 * offset), block);
            } else {
                const auto block = vld3q_u32(reinterpret_cast<const uint32_t *>(source.data + 3 * offset));
                vst1q_u8(destination.data + offset, vreinterpretq_u8_u32(block.val[0]));
                vst1q_u8(destination.data + offset + plane_stride, vreinterpretq_u8_u32(block.val[1]));
                vst1q_u8(destination.data + offset + 2 * plane_stride, vreinterpretq_u8_u32(block.val[2]));
            }
        }
        for (; row < end; ++row) {
            for (size_t channel = 0; channel < 3; ++channel) {
                const size_t planar_offset = row * width + channel * plane_stride;
                const size_t interleaved_offset = (row * 3 + channel) * width;
                const size_t source_offset = ToInterleaved ? planar_offset : interleaved_offset;
                const size_t destination_offset = ToInterleaved ? interleaved_offset : planar_offset;
                std::memcpy(destination.data + destination_offset, source.data + source_offset, width);
            }
        }
    }
}
#endif

View slice(View view, size_t axis, int64_t begin, int64_t end) {
    if (begin < 0 || end < begin || end > view.shape.at(axis)) {
        throw std::runtime_error("cache slice exceeds tensor dimensions");
    }
    view.shape[axis] = end - begin;
    if (elements(view.shape) != 0) {
        size_t offset = byte_count(Wide{begin} * view.strides.at(axis));
        offset = byte_count(Wide{offset} * itemsize(view.dtype));
        if (offset > view.span) {
            throw std::runtime_error("cache slice exceeds storage");
        }
        view.data += offset;
        view.span -= offset;
    }
    return view;
}

void coordinates_for(size_t flat, const Shape &shape, const std::vector<size_t> &axes, Shape &coordinates) {
    for (size_t index = axes.size(); index > 0; --index) {
        size_t axis = axes[index - 1];
        size_t extent = static_cast<size_t>(shape[axis]);
        coordinates[axis] = flat % extent;
        flat /= extent;
    }
}

size_t coordinate_offset(const Shape &coordinates, const Shape &strides) {
    if (coordinates.size() != strides.size()) {
        throw std::runtime_error("tensor coordinates and strides have different ranks");
    }
    size_t offset = 0;
    for (size_t axis = 0; axis < coordinates.size(); ++axis) {
        offset = byte_count(offset + Wide{coordinates[axis]} * strides[axis]);
    }
    return offset;
}

} // namespace

Integer Runtime::evaluate(const Json &value) {
    if (integer_literal(value)) {
        return json_integer_exact(value);
    }
    if (auto reference = value.find("expr"); reference != nullptr) {
        const auto &name = name_of(*reference);
        if (auto memo = expression_values.find(name); memo != expression_values.end()) {
            return memo->second;
        }
        Integer result = evaluate(*expression_index.at(name));
        expression_values.emplace(name, result);
        return result;
    }

    const auto &op = name_of(value.at("op"));
    if (op == "dimension" || op == "scalar") {
        const auto &tensor = name_of(value.at("tensor"));
        Integer axis = value.contains("axis") ? json_integer_exact(value.at("axis")) : 0;
        if (op == "dimension") {
            Shape dimensions = shape(tensor);
            return dimensions[axis_index(axis, dimensions.size())];
        }
        size_t index = axis < 0 ? axis_index(axis, elements(shape(tensor))) : axis.size();
        return scalar_integer(tensor, index);
    }
    if (op == "cache_length") {
        return lengths.at(name_of(value.at("cache")));
    }

    const auto &arguments = value.at("args");
    Integer a = evaluate(arguments.at(0));
    if (op == "broadcast_stride" && a == 1) {
        return 0;
    }
    Integer b = evaluate(arguments.at(1));
    if (op == "broadcast_stride") {
        return b;
    }
    if (op == "broadcast") {
        if (a != b && a != 1 && b != 1) {
            throw std::runtime_error("incompatible broadcast");
        }
        return a == 1 ? b : a;
    }
    if (op == "add") {
        return a + b;
    }
    if (op == "sub") {
        return a - b;
    }
    if (op == "mul") {
        return a * b;
    }
    if (op == "min") {
        return std::min(a, b);
    }
    if (op == "max") {
        return std::max(a, b);
    }
    if (op == "floor_div") {
        return divide(a, b, Division::Floor);
    }
    if (op == "ceil_div") {
        return divide(a, b, Division::Ceil);
    }
    if (op == "mod") {
        return divide(a, b, Division::Remainder);
    }
    throw std::runtime_error("unknown expression " + op);
}

double Runtime::scalar_parameter(const Json &value) {
    if (value.is_number()) {
        return value.get<double>();
    }
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.contains("float")) {
        return std::stod(name_of(value.at("float")));
    }
    return evaluate(value).real();
}

Json Runtime::parameter(const Json &value) {
    if (value.is_array()) {
        Json result = Json::array();
        for (const auto &element : value) {
            result.push_back(parameter(element));
        }
        return result;
    }
    if (value.is_number()) {
        return value;
    }
    if (value.contains("float")) {
        return scalar_parameter(value);
    }
    return integer_json(evaluate(value));
}

void Runtime::copy_view(const View &destination, const View &original) {
    if (array_dtype(destination.dtype) != array_dtype(original.dtype)) {
        throw std::runtime_error("tensor copy requires matching native dtypes");
    }
    size_t width = trillim_itemsize(destination.dtype);
    if (width == 0) {
        throw std::runtime_error("packed tensor copies require a codec");
    }
    const bool broadcasting = destination.shape != original.shape;
    const View broadcast = broadcasting ? broadcast_source(original, destination.shape) : View{};
    const View &source = broadcasting ? broadcast : original;
    size_t count = elements(destination.shape);
    if (count == 0) {
        return;
    }
    size_t logical_bytes = byte_count(Wide{count} * width);
    size_t destination_bytes = view_footprint(destination, width);
    size_t source_bytes = view_footprint(source, width);
    auto target = reinterpret_cast<uintptr_t>(destination.data);
    auto origin = reinterpret_cast<uintptr_t>(source.data);
    bool overlaps = target <= origin ? origin - target < destination_bytes : target - origin < source_bytes;
    if (overlaps) {
        size_t bytes = byte_count(Wide{elements(original.shape)} * width);
        std::unique_ptr<uint8_t[]> snapshot(new uint8_t[bytes]);
        View frozen{snapshot.get(), original.shape, contiguous_strides(original.shape), original.dtype, bytes};
        copy_view(frozen, original);
        copy_view(destination, frozen);
        return;
    }

#if defined(__ARM_NEON)
    if (width == sizeof(uint32_t) && destination.shape.size() == 2 &&
        destination.shape[0] >= 4 && destination.shape[1] == 3 &&
        (target | origin) % alignof(uint32_t) == 0) {
        if (interleaved_three_channels(destination) && planar_three_channels(source)) {
            copy_three_channels<true>(*this, destination, source);
            host_copy_bytes += logical_bytes;
            return;
        }
        if (planar_three_channels(destination) && interleaved_three_channels(source)) {
            copy_three_channels<false>(*this, destination, source);
            host_copy_bytes += logical_bytes;
            return;
        }
    }
#endif

    size_t prefix = destination.shape.size();
    size_t block = 1;
    while (prefix > 0) {
        size_t axis = prefix - 1;
        auto extent = static_cast<size_t>(destination.shape[axis]);
        if (extent != 1 && (static_cast<size_t>(destination.strides[axis]) != block ||
                            static_cast<size_t>(source.strides[axis]) != block)) {
            break;
        }
        block *= extent;
        --prefix;
    }
    constexpr size_t chunk = 1024 * 1024;
    size_t until_check = 0;
    size_t block_bytes = byte_count(Wide{block} * width);
    std::vector<size_t> coordinates(prefix);
    size_t target_offset = 0;
    size_t source_offset = 0;
    for (size_t index = 0; index < count / block; ++index) {
        for (size_t copied = 0; copied < block_bytes;) {
            if (until_check == 0) {
                check_cancelled();
                until_check = chunk;
            }
            size_t bytes = std::min(until_check, block_bytes - copied);
            std::memcpy(destination.data + target_offset + copied, source.data + source_offset + copied, bytes);
            copied += bytes;
            until_check -= bytes;
        }
        for (size_t axis = prefix; axis > 0;) {
            --axis;
            const auto extent = static_cast<size_t>(destination.shape[axis]);
            const auto target_stride = static_cast<size_t>(destination.strides[axis]) * width;
            const auto source_stride = static_cast<size_t>(source.strides[axis]) * width;
            if (++coordinates[axis] < extent) {
                target_offset += target_stride;
                source_offset += source_stride;
                break;
            }
            coordinates[axis] = 0;
            target_offset -= (extent - 1) * target_stride;
            source_offset -= (extent - 1) * source_stride;
        }
    }
    host_copy_bytes += logical_bytes;
}

void Runtime::copy_cache(const std::string &name, bool write, const View &destination, const View &source) {
    if (!profile) {
        copy_view(destination, source);
        return;
    }
    const uint64_t before_bytes = host_copy_bytes;
    const uint64_t started = now_ns();
    copy_view(destination, source);
    const uint64_t elapsed = now_ns() - started;
    const uint64_t bytes = host_copy_bytes - before_bytes;
    if (bytes == 0) {
        return;
    }
    auto &record = copy_by_cache[(write ? "write:" : "read:") + name];
    record.ns += elapsed;
    record.bytes += bytes;
    if (record.count == 0) {
        record.first = {destination.shape, destination.strides, source.shape, source.strides, destination.dtype};
    }
    ++record.count;
    if (record.last.target_shape != destination.shape || record.last.target_strides != destination.strides ||
        record.last.source_shape != source.shape || record.last.source_strides != source.strides ||
        record.last.dtype != destination.dtype) {
        record.last = {destination.shape, destination.strides, source.shape, source.strides, destination.dtype};
    }
}

View Runtime::logical_view(const std::string &name, Allocation &decoded) {
    View source = view(name);
    if (!itemsize(source.dtype)) {
        const auto [rows, columns] = storage_shape(name);
        if (packed_bytes(source.dtype, rows, columns) > source.span) {
            throw std::runtime_error("packed tensor exceeds storage");
        }
        decoded = decode_packed(source.data, source.dtype, rows, columns, *this);
        source.data = decoded.data;
        source.dtype = TRILLIM_FP32;
        source.span = decoded.size;
        if (elements(source.shape)) {
            view_footprint(source, sizeof(float));
        }
    } else if (source.dtype == TRILLIM_BF16) {
        size_t count = elements(source.shape);
        size_t bytes = byte_count(Wide{count} * sizeof(float));
        decoded = Allocation(bytes);
        Shape dense_strides = contiguous_strides(source.shape);
        const uint8_t *encoded = source.data;
        Allocation dense;
        if (source.strides != dense_strides) {
            dense = Allocation(byte_count(Wide{count} * sizeof(uint16_t)));
            copy_view(View{dense.data, source.shape, dense_strides, source.dtype, dense.size}, source);
            encoded = dense.data;
        }
        for (size_t index = 0; index < count; ++index) {
            if (index % 4096 == 0) {
                check_cancelled();
            }
            uint16_t bits;
            std::memcpy(&bits, encoded + index * sizeof(bits), sizeof(bits));
            uint32_t expanded = uint32_t{bits} << 16;
            std::memcpy(decoded.data + index * sizeof(expanded), &expanded, sizeof(expanded));
        }
        source = View{decoded.data, std::move(source.shape), std::move(dense_strides), TRILLIM_FP32, bytes};
    }
    return source;
}

View Runtime::cache_view(const std::string &name, Allocation &decoded, bool capacity) {
    const auto &metadata = base.at("caches").at(name);
    uint32_t dtype = dtype_id(name_of(metadata.at("dtype")));
    auto [data, bytes] = cache_storage(*this, metadata);
    auto dimensions = metadata.at("shape").get<Shape>();
    Shape strides = contiguous_strides(dimensions);
    if (!itemsize(dtype)) {
        const auto [rows, columns] = matrix_dimensions(dimensions);
        if (packed_bytes(dtype, rows, columns) > bytes) {
            throw std::runtime_error("packed cache exceeds storage");
        }
        decoded = decode_packed(data, dtype, rows, columns, *this);
        data = decoded.data;
        bytes = decoded.size;
        dtype = TRILLIM_FP32;
    }
    View view{data, std::move(dimensions), std::move(strides), dtype, bytes};
    if (!capacity && metadata.at("axis").is_number_integer()) {
        size_t axis = axis_index(json_integer(metadata.at("axis")), view.shape.size());
        return slice(std::move(view), axis, 0, lengths.at(name));
    }
    return view;
}

void Runtime::cache_read(const Json &instruction) {
    const auto &name = name_of(instruction.at("cache"));
    const auto &output = name_of(instruction.at("output"));
    const auto &metadata = base.at("caches").at(name);
    auto dimensions = metadata.at("shape").get<Shape>();
    if (metadata.at("axis").is_number_integer()) {
        size_t axis = axis_index(json_integer(metadata.at("axis")), dimensions.size());
        dimensions[axis] = lengths.at(name);
    }
    shapes[output] = std::move(dimensions);
    if (instruction.at("snapshot").get<bool>()) {
        Allocation decoded;
        View source = cache_view(name, decoded);
        uint32_t dtype = dtype_id(name_of(buffer(output).at("dtype")));
        if (itemsize(dtype)) {
            copy_cache(name, false, view(output), source);
            return;
        }
        const auto [rows, columns] = matrix_dimensions(source.shape);
        Shape dense_strides = contiguous_strides(source.shape);
        Allocation dense;
        if (source.strides != dense_strides) {
            dense = Allocation(byte_count(Wide{elements(source.shape)} * sizeof(float)));
            View target{dense.data, source.shape, std::move(dense_strides), TRILLIM_FP32, dense.size};
            copy_cache(name, false, target, source);
            source = std::move(target);
        }
        if (source.dtype != TRILLIM_FP32) {
            throw std::runtime_error("packed cache snapshot requires decoded float32 values");
        }
        const auto packed = encode_packed(reinterpret_cast<const float *>(source.data), dtype, rows, columns, *this);
        copy_cache(name, false, payload_view(address(output), packed.size, span(output)),
                   payload_view(packed.data, packed.size, packed.size));
    }
}

void Runtime::cache_write(const Json &instruction) {
    const auto &name = name_of(instruction.at("cache"));
    const auto &source = name_of(instruction.at("input"));
    const auto &metadata = base.at("caches").at(name);
    auto capacity = metadata.at("shape").get<Shape>();
    Shape dimensions = shape(source);
    bool metadata_only = instruction.at("metadata_only").get<bool>();
    uint32_t dtype = dtype_id(name_of(metadata.at("dtype")));
    bool packed = !itemsize(dtype);
    Allocation decoded_cache;
    Allocation decoded_source;
    View target;
    View values;
    if (!metadata_only) {
        target = cache_view(name, decoded_cache, true);
        values = packed ? logical_view(source, decoded_source) : view(source);
    }
    auto save_packed = [&]() {
        if (packed && !metadata_only) {
            const auto [rows, columns] = matrix_dimensions(target.shape);
            const auto payload = encode_packed(reinterpret_cast<const float *>(target.data), dtype, rows, columns, *this);
            const auto [data, bytes] = cache_storage(*this, metadata);
            copy_cache(name, true, payload_view(data, payload.size, bytes),
                       payload_view(payload.data, payload.size, payload.size));
        }
    };
    if (!metadata.at("axis").is_number_integer()) {
        if (capacity != dimensions) {
            throw std::runtime_error("overwrite cache shape differs from its capacity");
        }
        if (!metadata_only) {
            copy_cache(name, true, target, values);
        }
        save_packed();
        return;
    }

    size_t axis = axis_index(json_integer(metadata.at("axis")), capacity.size());
    int64_t write = dimensions.at(axis);
    bool append = instruction.at("action") == "append";
    int64_t begin = append ? lengths.at(name) : 0;
    int64_t end = expression_integer(Wide{begin} + write);
    bool sliding = append && metadata.at("mode") == "sliding";
    if (end <= capacity[axis]) {
        if (!metadata_only) {
            copy_cache(name, true, slice(target, axis, begin, end), values);
        }
        save_packed();
        lengths.at(name) = end;
        return;
    }
    if (!sliding) {
        throw std::runtime_error("cache capacity exceeded");
    }
    if (!metadata_only) {
        Shape combined_shape = capacity;
        combined_shape[axis] = end;
        size_t bytes = byte_count(Wide{elements(combined_shape)} * itemsize(target.dtype));
        std::unique_ptr<uint8_t[]> snapshot(new uint8_t[bytes]);
        View combined{snapshot.get(), combined_shape, contiguous_strides(combined_shape), target.dtype, bytes};
        copy_cache(name, true, slice(combined, axis, 0, begin), slice(target, axis, 0, begin));
        copy_cache(name, true, slice(combined, axis, begin, end), values);
        copy_cache(name, true, target, slice(std::move(combined), axis, end - capacity[axis], end));
    }
    save_packed();
    lengths.at(name) = capacity[axis];
}

void Runtime::scalar_copy(const std::string &source, const std::string &destination,
                          const Json &transfer, size_t source_offset, size_t destination_offset) {
    uint32_t input_dtype = dtype_id(name_of(buffer(source).at("dtype")));
    uint32_t output_dtype = dtype_id(name_of(buffer(destination).at("dtype")));
    auto [rows, columns] = storage_shape(source);
    size_t bytes = byte_count(Wide{destination_offset} * itemsize(output_dtype));
    size_t capacity = span(destination);
    if (bytes > capacity) {
        throw std::runtime_error("scalar transfer destination exceeds storage");
    }
    Json parameters = {
        {"count", 1}, {"rank", 0}, {"shape", Json::array()}, {"strides", Json::array()},
        {"source_offset", source_offset}, {"source_rows", rows}, {"source_columns", columns},
        {"output_rows", 1}, {"output_columns", 1}, {"input_dtype", input_dtype}, {"output_dtype", output_dtype},
    };
    invoke(&transfer, name_of(transfer.at("kernel")), parameters,
           {{address(source), span(source)}}, {{address(destination) + bytes, capacity - bytes}},
           transfer.at("scratch"));
}

void Runtime::reduce(const Json &instruction) {
    const auto &values = instruction.at("values").get<std::vector<std::string>>();
    const auto &initial = instruction.at("initial").get<std::vector<std::string>>();
    const auto &parameters = instruction.at("parameters").get<std::vector<std::string>>();
    const auto &results = instruction.at("results").get<std::vector<std::string>>();
    const auto &outputs = instruction.at("outputs").get<std::vector<std::string>>();
    const Json &transfer = instruction.at("transfer");
    bool keepdims = instruction.at("keepdims").get<bool>();
    size_t count = values.size();
    if (count == 0 || initial.size() != count || results.size() != count ||
        outputs.size() != count || parameters.size() != 2 * count) {
        throw std::runtime_error("invalid reducer arity");
    }
    Shape dimensions = shape(values.front());
    Shape output_shape;
    Shape reduction_shape;
    std::vector<size_t> kept_axes;
    std::vector<size_t> reduced_axes;
    for (const auto &value : instruction.at("axes")) {
        size_t axis = axis_index(json_integer(value), dimensions.size());
        if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) != reduced_axes.end()) {
            throw std::runtime_error("reduction axes repeat after normalization");
        }
        reduced_axes.push_back(axis);
        reduction_shape.push_back(dimensions[axis]);
    }
    for (size_t axis = 0; axis < dimensions.size(); ++axis) {
        if (std::find(reduced_axes.begin(), reduced_axes.end(), axis) == reduced_axes.end()) {
            kept_axes.push_back(axis);
            output_shape.push_back(dimensions[axis]);
        }
    }
    size_t output_count = elements(output_shape);
    size_t reduction_count = elements(reduction_shape);
    Shape coordinates(dimensions.size());
    Shape output_coordinates(keepdims ? dimensions.size() : kept_axes.size());
    std::vector<std::array<uint8_t, 8>> snapshots(count);
    std::vector<size_t> widths;
    for (const auto &result : results) {
        widths.push_back(itemsize(dtype_id(name_of(buffer(result).at("dtype")))));
    }
    for (size_t output_index = 0; output_index < output_count; ++output_index) {
        check_cancelled();
        coordinates_for(output_index, dimensions, kept_axes, coordinates);
        for (size_t index = 0; index < kept_axes.size(); ++index) {
            size_t axis = kept_axes[index];
            output_coordinates[keepdims ? axis : index] = coordinates[axis];
        }
        for (size_t index = 0; index < count; ++index) {
            scalar_copy(initial[index], parameters[index], transfer);
        }
        for (size_t reduced_index = 0; reduced_index < reduction_count; ++reduced_index) {
            coordinates_for(reduced_index, dimensions, reduced_axes, coordinates);
            for (size_t index = 0; index < count; ++index) {
                size_t offset = coordinate_offset(coordinates, strides(values[index]));
                scalar_copy(values[index], parameters[count + index], transfer, offset);
            }
            program(instruction.at("program"));
            // Identity carries can alias other result slots, so capture the full tuple first.
            for (size_t index = 0; index < count; ++index) {
                std::memcpy(snapshots[index].data(), address(results[index]), widths[index]);
                host_copy_bytes += widths[index];
            }
            for (size_t index = 0; index < count; ++index) {
                std::memcpy(address(parameters[index]), snapshots[index].data(), widths[index]);
                host_copy_bytes += widths[index];
            }
        }
        for (size_t index = 0; index < count; ++index) {
            size_t offset = coordinate_offset(output_coordinates, strides(outputs[index]));
            scalar_copy(parameters[index], outputs[index], transfer, 0, offset);
        }
    }
}

void Runtime::program(const Json &instructions) {
    for (const auto &instruction : instructions) {
        uint64_t started = 0;
        uint64_t previous_instruction_ns = 0;
        if (profile) {
            started = now_ns();
            previous_instruction_ns = profiled_instruction_ns;
        }
        const auto &op = name_of(instruction.at("op"));
        if (op != "call") {
            check_cancelled();
        }
        if (op != "guard") {
            expression_values.clear();
        }
        if (op == "call") {
            call(instruction);
        } else if (op == "guard") {
            Integer a = evaluate(instruction.at("left"));
            Integer b = evaluate(instruction.at("right"));
            const auto &predicate = name_of(instruction.at("predicate"));
            bool valid;
            if (predicate == "eq") {
                valid = a == b;
            } else if (predicate == "le") {
                valid = a <= b;
            } else if (predicate == "ge") {
                valid = a >= b;
            } else if (predicate == "divisible") {
                valid = b != 0 && divide(a, b, Division::Remainder) == 0;
            } else if (predicate == "broadcast") {
                valid = a == b || a == 1 || b == 1;
            } else {
                throw std::runtime_error("unknown shape guard predicate " + predicate);
            }
            if (!valid) {
                const auto &reason = name_of(instruction.at("reason"));
                throw std::runtime_error("shape guard failed: " + reason + " (" +
                                         a.str() + ", " + b.str() + ")");
            }
        } else if (op == "views") {
            for (const auto &value : instruction.at("outputs")) {
                const auto &name = name_of(value);
                shapes.erase(name);
                shapes[name] = shape(name);
            }
        } else if (op == "if") {
            bool condition = truth(name_of(instruction.at("condition")));
            const auto &branch = instruction.at("branches").at(condition ? 0 : 1);
            program(branch.at("program"));
            program(branch.at("copies"));
        } else if (op == "while") {
            const auto &condition = instruction.at("condition");
            const auto &body = instruction.at("body");
            program(instruction.at("initial_copies"));
            emit_loop_tokens(instruction);
            const Integer maximum = json_integer_exact(instruction.at("maximum_iterations"));
            for (Integer iteration = 0; iteration <= maximum; iteration = iteration + 1) {
                check_cancelled();
                program(condition.at("copies"));
                program(condition.at("program"));
                if (!truth(name_of(condition.at("result")))) {
                    break;
                }
                if (iteration == maximum) {
                    throw std::runtime_error("compiled loop iteration bound exceeded");
                }
                program(body.at("copies"));
                program(body.at("program"));
                program(body.at("updates"));
                emit_loop_tokens(instruction);
            }
        } else if (op == "reduce") {
            reduce(instruction);
        } else if (op == "cache_read") {
            cache_read(instruction);
        } else if (op == "cache_write") {
            cache_write(instruction);
        } else {
            throw std::runtime_error("unknown instruction " + op);
        }
        if (op != "guard") {
            expression_values.clear();
        }
        if (profile) {
            const uint64_t nested_ns = profiled_instruction_ns - previous_instruction_ns;
            const uint64_t elapsed_ns = now_ns() - started - nested_ns;
            profiled_instruction_ns += elapsed_ns;
            auto &[total_ns, count] = instruction_ns[op];
            total_ns += elapsed_ns;
            ++count;
            if (op == "cache_read" || op == "cache_write") {
                host_copy_ns += elapsed_ns;
            }
        }
    }
    check_cancelled();
}

} // namespace trillim::cpu
