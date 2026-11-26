#ifndef TRILLIM_CPU_JSON_H
#define TRILLIM_CPU_JSON_H

#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace trillim::cpu {

class Json {
public:
    using Array = std::vector<Json>;
    using Object = std::vector<std::pair<std::string, Json>>;
    enum class Type { Null, Boolean, Signed, Unsigned, Real, String, Array, Object, Integer };

    Json() = default;
    Json(std::nullptr_t) noexcept;
    Json(bool value) noexcept;
    Json(double value) noexcept;
    Json(const char *value);
    Json(std::string value);
    Json(std::initializer_list<std::pair<std::string, Json>> fields);

    template<std::integral T> requires (!std::same_as<T, bool>)
    Json(T value) noexcept {
        if constexpr (std::is_signed_v<T>) {
            data_ = static_cast<int64_t>(value);
        } else {
            data_ = static_cast<uint64_t>(value);
        }
    }

    template<typename T>
    Json(const std::vector<T> &values) : data_(Array{}) {
        auto &output = std::get<Array>(data_);
        output.reserve(values.size());
        for (const auto &value : values) {
            output.emplace_back(value);
        }
    }

    static Json array(std::initializer_list<Json> values = {});
    static Json object();
    static Json parse(std::string_view text);
    static Json parse(const uint8_t *begin, const uint8_t *end);
    std::string dump() const;

    Type type() const noexcept;
    bool is_null() const noexcept;
    bool is_boolean() const noexcept;
    bool is_number() const noexcept;
    bool is_number_integer() const noexcept;
    bool is_number_unsigned() const noexcept;
    bool is_number_float() const noexcept;
    bool is_big_integer() const noexcept;
    bool is_string() const noexcept;
    bool is_array() const noexcept;
    bool is_object() const noexcept;
    size_t size() const noexcept;
    bool empty() const noexcept;

    const Json *find(std::string_view key) const noexcept;
    bool contains(std::string_view key) const noexcept;
    const Json &at(std::string_view key) const;
    const Json &at(size_t index) const;
    const Json &operator[](size_t index) const;
    Json &operator[](std::string_view key);
    void push_back(Json value);

    const Object &items() const;
    Array::const_iterator begin() const;
    Array::const_iterator end() const;
    Array::const_reverse_iterator rbegin() const;
    Array::const_reverse_iterator rend() const;
    const std::string &string() const;
    const std::string &integer_text() const;

    template<typename T>
    T get() const {
        if constexpr (std::same_as<T, Json>) {
            return *this;
        } else if constexpr (std::same_as<T, std::string>) {
            return string();
        } else if constexpr (std::same_as<T, bool>) {
            return boolean();
        } else if constexpr (std::integral<T>) {
            if constexpr (std::is_signed_v<T>) {
                const int64_t value = signed_integer();
                if (!std::in_range<T>(value)) {
                    throw std::out_of_range("JSON integer is outside the destination range");
                }
                return static_cast<T>(value);
            } else {
                const uint64_t value = unsigned_integer();
                if (!std::in_range<T>(value)) {
                    throw std::out_of_range("JSON integer is outside the destination range");
                }
                return static_cast<T>(value);
            }
        } else if constexpr (std::floating_point<T>) {
            return static_cast<T>(real());
        } else {
            T result;
            result.reserve(size());
            for (const auto &element : *this) {
                result.push_back(element.template get<typename T::value_type>());
            }
            return result;
        }
    }

    template<typename T>
    T value(std::string_view key, T fallback) const {
        const auto *found = find(key);
        return found ? found->get<T>() : std::move(fallback);
    }
    std::string value(std::string_view key, const char *fallback) const;
    bool operator==(const Json &other) const;

private:
    struct IntegerToken {
        std::string text;
        bool operator==(const IntegerToken &) const = default;
    };
    std::variant<std::monostate, bool, int64_t, uint64_t, double, std::string,
                 Array, Object, IntegerToken> data_;

    explicit Json(Array values);
    explicit Json(Object values);
    explicit Json(IntegerToken value);
    const Array &elements() const;
    bool boolean() const;
    int64_t signed_integer() const;
    uint64_t unsigned_integer() const;
    double real() const;
    void append(std::string &output, size_t depth) const;
    friend class JsonParser;
};

class JsonParser {
public:
    explicit JsonParser(std::string_view text);
    Json parse();

private:
    std::string_view text_;
    size_t position_ = 0;

    [[noreturn]] void fail(const char *reason) const;
    char peek() const;
    void whitespace();
    bool take(char character);
    void expect(char character);
    uint32_t hex_quad();
    std::string string();
    Json number();
    Json value(size_t depth);
};

}
#endif
