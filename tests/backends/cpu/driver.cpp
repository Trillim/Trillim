#include "integer.h"

#include <bit>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace trillim::cpu;

namespace {
Integer operation(const Integer &left, const Integer &right, const std::string &name) {
    if (name == "add") return left + right;
    if (name == "sub") return left - right;
    if (name == "mul") return left * right;
    if (name == "floor") return divide(left, right, Division::Floor);
    if (name == "ceil") return divide(left, right, Division::Ceil);
    if (name == "mod") return divide(left, right, Division::Remainder);
    throw std::invalid_argument("unknown integer operation");
}

Json integer_case(const Json &request) {
    const auto left = Integer::parse(request.at("left").string());
    if (request.contains("operations")) {
        Integer value = left;
        Json history = Json::array();
        for (const auto &step : request.at("operations")) {
            value = operation(value, Integer::parse(step.at("value").string()), step.at("op").string());
            history.push_back(value.str());
        }
        return history;
    }
    const auto right = Integer::parse(request.at("right").string());
    if (request.contains("op")) {
        return operation(left, right, request.at("op").string()).str();
    }
    Json real, truncated, signed_value, unsigned_value, size_value;
    try { real = left.real(); } catch (const std::overflow_error &) {}
    try { signed_value = left.int64(); } catch (const std::overflow_error &) {}
    try { unsigned_value = left.uint64(); } catch (const std::overflow_error &) {}
    try { size_value = left.size(); } catch (const std::overflow_error &) {}
    const auto bits = request.at("float_bits").get<uint32_t>();
    try {
        truncated = Integer::from_truncated_float(std::bit_cast<float>(bits)).str();
    } catch (const std::invalid_argument &) {}
    Json divisions = Json::array();
    if (right != 0) {
        for (const auto mode : {Division::Floor, Division::Ceil, Division::Remainder}) {
            divisions.push_back(divide(left, right, mode).str());
        }
    }
    return Json{
        {"left", left.str()}, {"right", right.str()},
        {"roundtrip", json_integer_exact(integer_json(left)).str()},
        {"compare", left < right ? -1 : left > right ? 1 : 0},
        {"sum", (left + right).str()}, {"difference", (left - right).str()},
        {"product", (left * right).str()}, {"divisions", divisions},
        {"real", real}, {"truncated", truncated},
        {"int64", signed_value}, {"uint64", unsigned_value}, {"size", size_value},
    };
}

Json json_api(const Json &request) {
    const auto name = request.at("op").string();
    if (name == "real_bits") {
        return std::bit_cast<double>(request.at("bits").get<uint64_t>());
    }
    if (name == "nest") {
        Json value = 0;
        for (size_t i = 0; i < request.at("depth").get<size_t>(); ++i) {
            value = Json::array({value});
        }
        return value;
    }
    const auto &value = request.at("value");
    if (name == "exact_integer") return json_integer_exact(value).str();
    if (name == "int8") return value.get<int8_t>();
    if (name == "uint8") return value.get<uint8_t>();
    if (name == "int32") return value.get<int32_t>();
    if (name == "uint32") return value.get<uint32_t>();
    if (name == "int64") return value.get<int64_t>();
    if (name == "uint64") return value.get<uint64_t>();
    if (name == "bool") return value.get<bool>();
    if (name == "string") return value.get<std::string>();
    if (name == "double") return value.get<double>();
    if (name != "mutate") throw std::invalid_argument("unknown JSON operation");

    const auto &original = request.at("object");
    auto object = original;
    const auto key = request.at("key").string();
    const auto *found = object.find(key);
    Json result{
        {"found", found != nullptr}, {"contains", object.contains(key)},
        {"before", found ? object.at(key) : Json{}},
        {"fallback", object.value(key, Json("missing"))},
    };
    object[key] = value;
    auto array = request.at("array");
    array.push_back(value);
    Json forward = Json::array(), reverse = Json::array(), rebuilt = Json::object();
    for (const auto &item : array) forward.push_back(item);
    for (auto item = array.rbegin(); item != array.rend(); ++item) reverse.push_back(*item);
    for (const auto &[field, item] : object.items()) rebuilt[field] = item;
    result["object"] = object;
    result["original"] = original;
    result["rebuilt_equal"] = rebuilt == object;
    result["array"] = forward;
    result["reverse"] = reverse;
    result["last"] = array.at(array.size() - 1);
    result["object_size"] = object.size();
    return result;
}

Json execute(char mode, const std::string &text) {
    if (mode == 'p') return Integer::parse(text).str();
    if (mode == 's') return text;
    if (mode == 'j' || mode == 'v') {
        const auto *begin = reinterpret_cast<const uint8_t *>(text.data());
        return mode == 'j' ? Json::parse(text) : Json::parse(begin, begin + text.size());
    }
    std::istringstream input(text);
    const auto request = read_manifest(input);
    if (mode == 'i') return integer_case(request);
    if (mode == 'o') return json_api(request);
    throw std::invalid_argument("unknown driver request");
}
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    for (;;) {
        char header[9];
        std::cin.read(header, sizeof(header));
        if (std::cin.gcount() == 0) return 0;
        if (std::cin.gcount() != sizeof(header)) return 2;
        uint64_t length = 0;
        for (size_t i = 0; i < 8; ++i) {
            length |= uint64_t{static_cast<uint8_t>(header[i + 1])} << (8 * i);
        }
        if (length > 4 * 1024 * 1024) return 3;
        std::string text(length, '\0');
        std::cin.read(text.data(), static_cast<std::streamsize>(length));
        if (static_cast<uint64_t>(std::cin.gcount()) != length) return 4;
        Json response;
        try {
            response = Json{{"ok", true}, {"result", execute(header[0], text).dump()}};
        } catch (const std::overflow_error &error) {
            response = Json{{"ok", false}, {"error", error.what()}, {"kind", "overflow"}};
        } catch (const std::out_of_range &error) {
            response = Json{{"ok", false}, {"error", error.what()}, {"kind", "range"}};
        } catch (const std::invalid_argument &error) {
            response = Json{{"ok", false}, {"error", error.what()}, {"kind", "invalid"}};
        }
        std::cout << response.dump() << std::endl;
    }
}
