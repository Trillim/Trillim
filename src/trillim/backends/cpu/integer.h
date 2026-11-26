#ifndef TRILLIM_RUNTIME_INTEGER_H
#define TRILLIM_RUNTIME_INTEGER_H

#include "json.h"

#include <compare>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trillim::cpu {

enum class Division { Floor, Ceil, Remainder };

class Integer {
public:
    Integer(__int128_t value = 0) noexcept : small_(value) {}

    static Integer parse(std::string_view text);
    static Integer from_truncated_float(float value);
    int64_t int64() const;
    uint64_t uint64() const;
    size_t size() const;
    double real() const;
    std::string str() const;

    std::strong_ordering operator<=>(const Integer &other) const noexcept;
    bool operator==(const Integer &other) const noexcept;
    friend Integer operator+(const Integer &left, const Integer &right);
    friend Integer operator-(const Integer &left, const Integer &right);
    friend Integer operator*(const Integer &left, const Integer &right);
    friend Integer divide(const Integer &left, const Integer &right, Division mode);
    friend Json integer_json(const Integer &value);

private:
    struct Big;
    __int128_t small_;
    std::shared_ptr<const Big> large_;

    bool negative() const noexcept;
    std::vector<uint32_t> magnitude() const;
    static Integer from_limbs(bool negative, std::vector<uint32_t> limbs);
    static Integer add(const Integer &left, const Integer &right, bool subtract);
};

bool integer_literal(const Json &value);
Integer json_integer_exact(const Json &value);
Json integer_json(const Integer &value);
Json read_manifest(std::istream &input);
}

#endif
