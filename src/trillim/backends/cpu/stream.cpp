#include "runtime.h"

#include <algorithm>
#include <string_view>

namespace trillim::cpu {
namespace {

const Json *producer(const Json &program, const Json &tensor) {
    for (auto instruction = program.rbegin(); instruction != program.rend(); ++instruction) {
        const auto outputs = instruction->find("outputs");
        if (instruction->at("op") != "views" && outputs != nullptr &&
            std::find(outputs->begin(), outputs->end(), tensor) != outputs->end()) {
            return &*instruction;
        }
    }
    return nullptr;
}

const std::string &name(const Json &value) {
    return value.string();
}

const Json &unref(const Runtime &runtime, const Json &expression) {
    const auto ref = expression.find("expr");
    return ref == nullptr ? expression : *runtime.expression_index.at(name(*ref));
}

std::string source(const Runtime &runtime, const std::string &tensor) {
    const auto &record = runtime.buffer(tensor);
    if (record.at("storage") == "alias") {
        return source(runtime, name(record.at("source")));
    }
    return tensor;
}

std::pair<const Json *, std::string> produced_tensor(
    const Runtime &runtime, const Json &program, std::string tensor
) {
    for (;;) {
        if (const auto *instruction = producer(program, tensor)) {
            return {instruction, std::move(tensor)};
        }
        const auto &record = runtime.buffer(tensor);
        if (record.at("storage") != "alias") {
            return {nullptr, std::move(tensor)};
        }
        tensor = name(record.at("source"));
    }
}

bool token_dtype(uint32_t dtype) {
    return dtype != TRILLIM_BOOL && itemsize(dtype) && !trillim_float(dtype);
}

bool call_is(const Json *instruction, std::string_view kernel) {
    return instruction && instruction->at("op") == "call" &&
           name(instruction->at("kernel")).ends_with(kernel);
}

bool single_coordinate(const Json &parameters) {
    return parameters.value("coordinates", 0) == 1 &&
           parameters.value("slice_rank", -1) == 0 &&
           parameters.value("slice_size", 0) == 1 &&
           parameters.at("indexed_strides") == Json::array({1}) &&
           parameters.at("output_indexed_strides") == Json::array({1});
}

bool prefix_gather(const Runtime &runtime, const Json &gather, const Json &length) {
    const auto &parameters = gather.at("parameters");
    if (!single_coordinate(parameters) || parameters.at("masked") != 0 ||
        parameters.at("index_rank") != 1 ||
        parameters.at("index_strides") != Json::array({1}) ||
        unref(runtime, parameters.at("count")) != length ||
        unref(runtime, parameters.at("items")) != length) {
        return false;
    }
    const auto &indices = name(gather.at("inputs").at(1));
    const auto *iota = produced_tensor(runtime, *runtime.active_program, indices).first;
    if (!call_is(iota, "_scalar_iota")) {
        return false;
    }
    const auto &range = iota->at("parameters");
    return range.at("start") == 0 && range.at("step") == 1 &&
           unref(runtime, range.at("count")) == length;
}

bool reads_parameter(const Runtime &runtime, const Json &program,
                     std::string tensor, const Json &parameter) {
    while (tensor != name(parameter)) {
        const auto &record = runtime.buffer(tensor);
        if (record.at("storage") != "alias" || producer(program, tensor)) {
            return false;
        }
        tensor = name(record.at("source"));
    }
    return true;
}

bool unit_constant(Runtime &runtime, const Json &tensor) {
    const auto &record = runtime.buffer(name(tensor));
    if (!record.at("shape").empty() || !token_dtype(dtype_id(name(record.at("dtype")))) ||
        runtime.buffer(source(runtime, name(tensor))).at("storage") != "constant") {
        return false;
    }
    const auto value = runtime.scalar_integer(name(tensor));
    return value == 0 || value == 1;
}

bool scalar_call(const Json *instruction, std::string_view kernel, uint32_t dtype) {
    if (!call_is(instruction, kernel)) {
        return false;
    }
    const auto &arguments = instruction->at("parameters");
    return arguments.at("rank") == 0 && arguments.at("dtype") == dtype;
}

bool nondecreasing_count(Runtime &runtime, const Json &program,
                         const Json &parameter, const Json &result) {
    const auto dtype = dtype_id(name(runtime.buffer(name(parameter)).at("dtype")));
    const auto *add = produced_tensor(runtime, program, name(result)).first;
    if (!token_dtype(dtype) || !scalar_call(add, "_scalar_add", dtype)) {
        return false;
    }
    const auto &inputs = add->at("inputs");
    const Json *step = &inputs.at(1);
    if (!reads_parameter(runtime, program, name(inputs.at(0)), parameter)) {
        if (!reads_parameter(runtime, program, name(inputs.at(1)), parameter)) {
            return false;
        }
        step = &inputs.at(0);
    }
    if (unit_constant(runtime, *step)) {
        return true;
    }
    const auto *select = produced_tensor(runtime, program, name(*step)).first;
    if (!scalar_call(select, "_scalar_select", dtype)) {
        return false;
    }
    return unit_constant(runtime, select->at("inputs").at(1)) &&
           unit_constant(runtime, select->at("inputs").at(2));
}

bool append_only(Runtime &runtime, const Json &loop,
                 const std::string &tensor, const std::string &length) {
    const auto &outputs = loop.at("outputs");
    const auto tensor_slot = static_cast<size_t>(
        std::find(outputs.begin(), outputs.end(), tensor) - outputs.begin());
    const auto length_slot = static_cast<size_t>(
        std::find(outputs.begin(), outputs.end(), length) - outputs.begin());
    if (tensor_slot == outputs.size() || length_slot == outputs.size()) {
        return false;
    }
    const auto &body = loop.at("body");
    const auto &program = body.at("program");
    const auto &parameters = body.at("parameters");
    const auto &results = body.at("results");
    const auto *update = produced_tensor(runtime, program, name(results.at(tensor_slot))).first;
    if (!call_is(update, "_scalar_scatter") && !call_is(update, "_scalar_scatter_masked")) {
        return false;
    }
    const auto &arguments = update->at("parameters");
    const auto &inputs = update->at("inputs");
    return single_coordinate(arguments) && arguments.at("mode") == 0 &&
           arguments.at("items") == 1 && arguments.at("index_rank") == 0 &&
           reads_parameter(runtime, program, name(inputs.at(0)), parameters.at(tensor_slot)) &&
           reads_parameter(runtime, program, name(inputs.at(1)), parameters.at(length_slot)) &&
           nondecreasing_count(runtime, program, parameters.at(length_slot), results.at(length_slot));
}

} // namespace

void Runtime::begin_token_stream(TokenSink sink) {
    const auto output = base.at("outputs").find("tokens");
    if (output == nullptr) {
        throw std::invalid_argument("token streaming requires a tokens output");
    }
    const auto &tensor = name(*output);
    const auto &record = buffer(tensor);
    if (record.at("shape").size() != 1 || !token_dtype(dtype_id(name(record.at("dtype"))))) {
        throw std::invalid_argument("token streaming requires a one-dimensional integer tokens output");
    }
    token_stream = TokenStream{std::move(sink), nullptr, tensor, {}, {}, 0, false};

    // Stream during the loop if the output selects a prefix of a carried buffer
    // and the body writes at the previous length.
    const auto &length = unref(*this, record.at("shape").at(0));
    if (!length.is_object() || length.value("op", "") != "scalar" ||
        length.value("axis", 0) != 0 || record.at("strides") != Json::array({1})) {
        return;
    }
    auto [count_loop, length_tensor] = produced_tensor(*this, *active_program, name(length.at("tensor")));
    if (!count_loop || count_loop->at("op") != "while") {
        return;
    }
    auto [loop, tokens] = produced_tensor(*this, *active_program, tensor);
    if (call_is(loop, "_scalar_gather") && prefix_gather(*this, *loop, length)) {
        const auto &input = name(loop->at("inputs").at(0));
        if (buffer(input).at("strides") != Json::array({1})) {
            return;
        }
        auto [instruction, input_tensor] = produced_tensor(*this, *active_program, input);
        loop = instruction;
        tokens = std::move(input_tensor);
    }
    const auto &vector = buffer(tokens);
    const auto &counter = buffer(length_tensor);
    const auto count_dtype = dtype_id(name(counter.at("dtype")));
    if (loop != count_loop ||
        vector.at("shape").size() != 1 || vector.at("strides") != Json::array({1}) ||
        vector.at("dtype") != record.at("dtype") || !counter.at("shape").empty() ||
        !token_dtype(count_dtype) ||
        !append_only(*this, *loop, tokens, length_tensor)) {
        return;
    }
    const auto count_bits = itemsize(count_dtype) * 8 - !trillim_unsigned(count_dtype);
    if (vector.at("maximum_shape").at(0).get<uint64_t>() >= (UINT64_MAX >> (64 - count_bits))) {
        return;
    }
    token_stream.loop = loop;
    token_stream.buffer = std::move(tokens);
    token_stream.length = std::move(length_tensor);
}

void Runtime::emit_tokens(const View &tokens, size_t count) {
    const auto stride = tokens.strides.at(0);
    for (size_t index = token_stream.emitted; index < count; ++index) {
        check_cancelled();
        const auto value = trillim_load(tokens.data, index * static_cast<size_t>(stride), tokens.dtype);
        const Json token = trillim_unsigned(tokens.dtype) ? Json(value.u) : Json(value.i);
        token_stream.sink(index, token);
    }
    token_stream.emitted = count;
}

void Runtime::emit_loop_tokens(const Json &instruction) {
    if (token_stream.loop != &instruction) {
        return;
    }
    const auto tokens = view(token_stream.buffer);
    const auto count = scalar_integer(token_stream.length).int64();
    if (count < 0 || count > tokens.shape.at(0)) {
        throw std::runtime_error("generation loop produced an invalid token count");
    }
    const auto size = static_cast<size_t>(count);
    if (token_stream.started &&
        (size < token_stream.emitted || size - token_stream.emitted > 1)) {
        throw std::runtime_error("generation loop must preserve or extend its token prefix by one");
    }
    token_stream.started = true;
    emit_tokens(tokens, size);
}

void Runtime::finish_token_stream() {
    if (!token_stream.sink) {
        return;
    }
    const auto output = view(token_stream.output);
    const auto count = static_cast<size_t>(output.shape.at(0));
    if (token_stream.loop && (!token_stream.started || count != token_stream.emitted)) {
        throw std::runtime_error("streamed token count differs from the model output");
    }
    emit_tokens(output, count);
    token_stream = TokenStream{};
}

} // namespace trillim::cpu
