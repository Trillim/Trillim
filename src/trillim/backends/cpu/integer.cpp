#include "integer.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <istream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace trillim::cpu {
namespace {
using Wide = __int128_t;
using Unsigned = __uint128_t;
using Limbs = std::vector<uint32_t>;

constexpr Unsigned sign_bit = Unsigned{1} << 127;

void trim(Limbs &value) {
    while (!value.empty() && value.back() == 0) {
        value.pop_back();
    }
}

Limbs limbs_of(Unsigned value) {
    Limbs limbs;
    while (value != 0) {
        limbs.push_back(static_cast<uint32_t>(value));
        value >>= 32;
    }
    return limbs;
}

size_t bit_count(const Limbs &limbs) {
    return limbs.empty() ? 0 : (limbs.size() - 1) * 32 + std::bit_width(limbs.back());
}

bool bit_at(const Limbs &limbs, size_t bit) {
    return (limbs[bit / 32] >> (bit % 32)) & 1;
}

std::strong_ordering compare(const Limbs &left, const Limbs &right) noexcept {
    if (left.size() != right.size()) {
        return left.size() <=> right.size();
    }
    for (size_t i = left.size(); i-- > 0;) {
        if (left[i] != right[i]) {
            return left[i] <=> right[i];
        }
    }
    return std::strong_ordering::equal;
}

void multiply_add(Limbs &value, uint32_t factor, uint32_t carry) {
    for (uint32_t &limb : value) {
        const uint64_t product = uint64_t{limb} * factor + carry;
        limb = static_cast<uint32_t>(product);
        carry = static_cast<uint32_t>(product >> 32);
    }
    if (carry != 0) {
        value.push_back(carry);
    }
}

void subtract(Limbs &left, const Limbs &right) {
    uint64_t borrow = 0;
    for (size_t i = 0; i < left.size(); ++i) {
        const uint64_t deduction = (i < right.size() ? right[i] : 0) + borrow;
        borrow = left[i] < deduction;
        left[i] = static_cast<uint32_t>(left[i] - deduction);
    }
    trim(left);
}

Limbs summed(const Limbs &left, const Limbs &right) {
    Limbs result(std::max(left.size(), right.size()));
    uint64_t carry = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        const uint64_t sum = carry + (i < left.size() ? left[i] : 0) + (i < right.size() ? right[i] : 0);
        result[i] = static_cast<uint32_t>(sum);
        carry = sum >> 32;
    }
    if (carry != 0) {
        result.push_back(static_cast<uint32_t>(carry));
    }
    return result;
}

Limbs product(const Limbs &left, const Limbs &right) {
    Limbs result(left.size() + right.size());
    for (size_t i = 0; i < left.size(); ++i) {
        uint64_t carry = 0;
        for (size_t j = 0; j < right.size(); ++j) {
            const uint64_t value = uint64_t{left[i]} * right[j] + result[i + j] + carry;
            result[i + j] = static_cast<uint32_t>(value);
            carry = value >> 32;
        }
        result[i + right.size()] = static_cast<uint32_t>(carry);
    }
    return result;
}

std::pair<Limbs, Limbs> divided(const Limbs &numerator, const Limbs &denominator) {
    Limbs quotient(numerator.size());
    Limbs remainder;
    for (size_t bit = bit_count(numerator); bit-- > 0;) {
        multiply_add(remainder, 2, bit_at(numerator, bit));
        if (compare(remainder, denominator) >= 0) {
            subtract(remainder, denominator);
            quotient[bit / 32] |= uint32_t{1} << (bit % 32);
        }
    }
    return {std::move(quotient), std::move(remainder)};
}

bool digit(char value) {
    return value >= '0' && value <= '9';
}


}

struct Integer::Big {
    bool negative;
    Limbs limbs;
};

bool Integer::negative() const noexcept {
    return large_ ? large_->negative : small_ < 0;
}

Limbs Integer::magnitude() const {
    if (large_) {
        return large_->limbs;
    }
    const Unsigned bits = static_cast<Unsigned>(small_);
    return limbs_of(small_ < 0 ? -bits : bits);
}

Integer Integer::from_limbs(bool negative, Limbs limbs) {
    trim(limbs);
    if (limbs.size() <= 4) {
        Unsigned value = 0;
        for (size_t i = limbs.size(); i-- > 0;) {
            value = (value << 32) | limbs[i];
        }
        if (value < sign_bit || (negative && value == sign_bit)) {
            return static_cast<Wide>(negative ? -value : value);
        }
    }
    Integer result;
    result.large_ = std::make_shared<const Big>(Big{negative, std::move(limbs)});
    return result;
}

Integer Integer::parse(std::string_view text) {
    const bool negative = !text.empty() && text.front() == '-';
    if (negative) {
        text.remove_prefix(1);
    }
    if (text.empty()) {
        throw std::invalid_argument("empty integer literal");
    }
    Unsigned small = 0;
    Limbs limbs;
    for (const char character : text) {
        if (!digit(character)) {
            throw std::invalid_argument("invalid integer literal");
        }
        auto part = static_cast<uint32_t>(character - '0');
        Unsigned multiplied;
        Unsigned next;
        if (limbs.empty() && !__builtin_mul_overflow(small, Unsigned{10}, &multiplied) &&
            !__builtin_add_overflow(multiplied, static_cast<Unsigned>(part), &next)) {
            small = next;
            continue;
        }
        if (limbs.empty()) {
            limbs = limbs_of(small);
        }
        multiply_add(limbs, 10, part);
    }
    if (limbs.empty()) {
        return from_limbs(negative, limbs_of(small));
    }
    return from_limbs(negative, std::move(limbs));
}

Integer Integer::from_truncated_float(float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("cannot convert a nonfinite scalar to integer");
    }
    const auto magnitude = static_cast<Unsigned>(std::abs(value));
    if (magnitude < sign_bit) {
        const auto small = static_cast<Wide>(magnitude);
        return value < 0 ? -small : small;
    }
    return from_limbs(value < 0, limbs_of(magnitude));
}

int64_t Integer::int64() const {
    if (large_ || small_ < INT64_MIN || small_ > INT64_MAX) {
        throw std::overflow_error("integer is outside the int64 range");
    }
    return static_cast<int64_t>(small_);
}

uint64_t Integer::uint64() const {
    if (large_ || small_ < 0 || static_cast<Unsigned>(small_) > UINT64_MAX) {
        throw std::overflow_error("integer is outside the uint64 range");
    }
    return static_cast<uint64_t>(small_);
}

size_t Integer::size() const {
    const uint64_t value = uint64();
    if (value > std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("integer is outside the size_t range");
    }
    return static_cast<size_t>(value);
}

double Integer::real() const {
    if (!large_) {
        return static_cast<double>(small_);
    }
    const auto &limbs = large_->limbs;
    const size_t bits = bit_count(limbs);
    if (bits > 1024) {
        throw std::overflow_error("integer is too large to convert to double");
    }
    const size_t shifted = bits - 53;
    uint64_t mantissa = 0;
    for (size_t bit = bits; bit-- > shifted;) {
        mantissa = (mantissa << 1) | bit_at(limbs, bit);
    }
    if (bit_at(limbs, shifted - 1)) {
        bool sticky = false;
        for (size_t bit = 0; bit + 1 < shifted && !sticky; ++bit) {
            sticky = bit_at(limbs, bit);
        }
        mantissa += sticky || (mantissa & 1);
    }
    const double value = std::ldexp(static_cast<double>(mantissa), static_cast<int>(shifted));
    if (!std::isfinite(value)) {
        throw std::overflow_error("integer is too large to convert to double");
    }
    return negative() ? -value : value;
}

std::string Integer::str() const {
    auto limbs = magnitude();
    if (limbs.empty()) {
        return "0";
    }
    std::vector<uint32_t> groups;
    while (!limbs.empty()) {
        uint64_t carry = 0;
        for (size_t i = limbs.size(); i-- > 0;) {
            const uint64_t value = (carry << 32) | limbs[i];
            limbs[i] = static_cast<uint32_t>(value / 1'000'000'000);
            carry = value % 1'000'000'000;
        }
        groups.push_back(static_cast<uint32_t>(carry));
        trim(limbs);
    }
    std::string text = negative() ? "-" : "";
    text += std::to_string(groups.back());
    for (size_t i = groups.size() - 1; i-- > 0;) {
        const std::string part = std::to_string(groups[i]);
        text.append(9 - part.size(), '0');
        text += part;
    }
    return text;
}

std::strong_ordering Integer::operator<=>(const Integer &other) const noexcept {
    if (!large_ && !other.large_) {
        return small_ <=> other.small_;
    }
    if (negative() != other.negative()) {
        return negative() ? std::strong_ordering::less : std::strong_ordering::greater;
    }
    const auto unsigned_order = large_ && other.large_ ? compare(large_->limbs, other.large_->limbs)
                                                       : bool(large_) <=> bool(other.large_);
    return negative() ? 0 <=> unsigned_order : unsigned_order;
}

bool Integer::operator==(const Integer &other) const noexcept {
    return (*this <=> other) == 0;
}

Integer Integer::add(const Integer &left, const Integer &right, bool subtract_right) {
    if (!left.large_ && !right.large_) {
        Wide value;
        const bool overflow = subtract_right ? __builtin_sub_overflow(left.small_, right.small_, &value)
                                             : __builtin_add_overflow(left.small_, right.small_, &value);
        if (!overflow) {
            return value;
        }
    }
    auto a = left.magnitude();
    auto b = right.magnitude();
    const bool a_negative = left.negative();
    const bool b_negative = right.negative() != subtract_right;
    if (a_negative == b_negative) {
        return from_limbs(a_negative, summed(a, b));
    }
    if (compare(a, b) >= 0) {
        subtract(a, b);
        return from_limbs(a_negative, std::move(a));
    }
    subtract(b, a);
    return from_limbs(b_negative, std::move(b));
}

Integer operator+(const Integer &left, const Integer &right) {
    return Integer::add(left, right, false);
}

Integer operator-(const Integer &left, const Integer &right) {
    return Integer::add(left, right, true);
}

Integer operator*(const Integer &left, const Integer &right) {
    if (!left.large_ && !right.large_) {
        Wide value;
        if (!__builtin_mul_overflow(left.small_, right.small_, &value)) {
            return value;
        }
    }
    return Integer::from_limbs(left.negative() != right.negative(), product(left.magnitude(), right.magnitude()));
}

Integer divide(const Integer &left, const Integer &right, Division mode) {
    if (right == 0) {
        throw std::invalid_argument("integer division by zero");
    }
    const bool opposite = left.negative() != right.negative();
    if (!left.large_ && !right.large_ &&
        !(left.small_ == static_cast<Wide>(sign_bit) && right.small_ == -1)) {
        const Wide quotient = left.small_ / right.small_;
        const Wide remainder = left.small_ % right.small_;
        if (mode == Division::Remainder) {
            return remainder != 0 && opposite ? remainder + right.small_ : remainder;
        }
        const bool round = remainder != 0 && opposite == (mode == Division::Floor);
        return quotient + (round ? (opposite ? -1 : 1) : 0);
    }
    auto denominator = right.magnitude();
    auto [quotient, remainder] = divided(left.magnitude(), denominator);
    if (mode == Division::Remainder) {
        if (opposite && !remainder.empty()) {
            subtract(denominator, remainder);
            remainder = std::move(denominator);
        }
        return Integer::from_limbs(right.negative(), std::move(remainder));
    }
    if (!remainder.empty() && opposite == (mode == Division::Floor)) {
        multiply_add(quotient, 1, 1);
    }
    return Integer::from_limbs(opposite, std::move(quotient));
}

bool integer_literal(const Json &value) {
    return value.is_number_integer() || value.is_boolean();
}

Integer json_integer_exact(const Json &value) {
    if (value.is_big_integer()) {
        return Integer::parse(value.integer_text());
    }
    if (value.is_number_unsigned()) {
        return value.get<uint64_t>();
    }
    if (value.is_number_integer()) {
        return value.get<int64_t>();
    }
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    throw std::invalid_argument("expected manifest integer");
}

Json integer_json(const Integer &value) {
    if (!value.large_) {
        if (value.small_ >= INT64_MIN && value.small_ <= INT64_MAX) {
            return static_cast<int64_t>(value.small_);
        }
        if (value.small_ >= 0 && static_cast<Unsigned>(value.small_) <= UINT64_MAX) {
            return static_cast<uint64_t>(value.small_);
        }
    }
    return Json::parse(value.str());
}

Json read_manifest(std::istream &input) {
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return Json::parse(text);
}
}
