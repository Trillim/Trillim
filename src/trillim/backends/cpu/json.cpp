#include "json.h"

#include <charconv>
#include <cmath>
#include <unordered_set>

namespace trillim::cpu {
namespace {
constexpr size_t maximum_depth = 256;

bool digit(char value) {
    return value >= '0' && value <= '9';
}

// Validate raw UTF-8 as well as escaped Unicode, including overlong encodings.
size_t utf8_width(std::string_view text, size_t position) {
    const auto first = static_cast<uint8_t>(text[position]);
    const size_t width = first >= 0xc2 && first <= 0xdf ? 2 :
                         first >= 0xe0 && first <= 0xef ? 3 :
                         first >= 0xf0 && first <= 0xf4 ? 4 : 0;
    if (!width || text.size() - position < width) {
        throw std::invalid_argument("invalid UTF-8");
    }
    for (size_t offset = 1; offset < width; ++offset) {
        const auto byte = static_cast<uint8_t>(text[position + offset]);
        if (byte < 0x80 || byte > 0xbf ||
            (offset == 1 && ((first == 0xe0 && byte < 0xa0) ||
                             (first == 0xed && byte > 0x9f) ||
                             (first == 0xf0 && byte < 0x90) ||
                             (first == 0xf4 && byte > 0x8f)))) {
            throw std::invalid_argument("invalid UTF-8");
        }
    }
    return width;
}

void append_codepoint(std::string &text, uint32_t value) {
    if (value < 0x80) {
        text += static_cast<char>(value);
    } else if (value < 0x800) {
        text += static_cast<char>(0xc0 | (value >> 6));
        text += static_cast<char>(0x80 | (value & 0x3f));
    } else if (value < 0x10000) {
        text += static_cast<char>(0xe0 | (value >> 12));
        text += static_cast<char>(0x80 | ((value >> 6) & 0x3f));
        text += static_cast<char>(0x80 | (value & 0x3f));
    } else {
        text += static_cast<char>(0xf0 | (value >> 18));
        text += static_cast<char>(0x80 | ((value >> 12) & 0x3f));
        text += static_cast<char>(0x80 | ((value >> 6) & 0x3f));
        text += static_cast<char>(0x80 | (value & 0x3f));
    }
}

void quote(std::string &output, std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    output += '"';
    for (size_t position = 0; position < text.size();) {
        const auto byte = static_cast<uint8_t>(text[position]);
        if (byte >= 0x80) {
            const size_t count = utf8_width(text, position);
            output.append(text.substr(position, count));
            position += count;
            continue;
        }
        ++position;
        switch (byte) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (byte < 0x20) {
                output += "\\u00";
                output += hex[byte >> 4];
                output += hex[byte & 15];
            } else {
                output += static_cast<char>(byte);
            }
        }
    }
    output += '"';
}
}

JsonParser::JsonParser(std::string_view text) : text_(text) {}
Json JsonParser::parse() {
    Json result = value(0);
    whitespace();
    if (position_ != text_.size()) {
        fail("trailing characters");
    }
    return result;
}


[[noreturn]] void JsonParser::fail(const char *reason) const {
    throw std::invalid_argument("invalid JSON at byte " + std::to_string(position_) + ": " + reason);
}
char JsonParser::peek() const {
    return position_ < text_.size() ? text_[position_] : '\0';
}
void JsonParser::whitespace() {
    while (position_ < text_.size() && (peek() == ' ' || peek() == '\t' || peek() == '\r' || peek() == '\n')) {
        ++position_;
    }
}
bool JsonParser::take(char character) {
    if (position_ < text_.size() && peek() == character) {
        ++position_;
        return true;
    }
    return false;
}
void JsonParser::expect(char character) {
    if (!take(character)) {
        fail("unexpected or missing character");
    }
}
uint32_t JsonParser::hex_quad() {
    uint32_t result = 0;
    for (size_t index = 0; index < 4; ++index) {
        if (position_ == text_.size()) {
            fail("incomplete Unicode escape");
        }
        const char character = text_[position_++];
        const int digit = character >= '0' && character <= '9' ? character - '0' :
                          character >= 'a' && character <= 'f' ? character - 'a' + 10 :
                          character >= 'A' && character <= 'F' ? character - 'A' + 10 : -1;
        if (digit < 0) {
            fail("invalid Unicode escape");
        }
        result = (result << 4) | static_cast<uint32_t>(digit);
    }
    return result;
}
std::string JsonParser::string() {
    expect('"');
    std::string result;
    while (position_ < text_.size()) {
        const auto byte = static_cast<uint8_t>(peek());
        if (byte == '"') {
            ++position_;
            return result;
        }
        if (byte < 0x20) {
            fail("unescaped control character");
        }
        if (byte >= 0x80) {
            size_t width;
            try {
                width = utf8_width(text_, position_);
            } catch (const std::invalid_argument &) {
                fail("invalid UTF-8 string");
            }
            result.append(text_.substr(position_, width));
            position_ += width;
            continue;
        }
        ++position_;
        if (byte != '\\') {
            result += static_cast<char>(byte);
            continue;
        }
        if (position_ == text_.size()) {
            fail("incomplete string escape");
        }
        switch (text_[position_++]) {
        case '"': result += '"'; break;
        case '\\': result += '\\'; break;
        case '/': result += '/'; break;
        case 'b': result += '\b'; break;
        case 'f': result += '\f'; break;
        case 'n': result += '\n'; break;
        case 'r': result += '\r'; break;
        case 't': result += '\t'; break;
        case 'u': {
            uint32_t point = hex_quad();
            if (point >= 0xd800 && point <= 0xdbff) {
                expect('\\');
                expect('u');
                const auto low = hex_quad();
                if (low < 0xdc00 || low > 0xdfff) {
                    fail("invalid Unicode surrogate pair");
                }
                point = 0x10000 + ((point - 0xd800) << 10) + low - 0xdc00;
            } else if (point >= 0xdc00 && point <= 0xdfff) {
                fail("unpaired Unicode surrogate");
            }
            append_codepoint(result, point);
            break;
        }
        default: fail("invalid string escape");
        }
    }
    fail("unterminated string");
}
Json JsonParser::number() {
    const size_t start = position_;
    const bool negative = take('-');
    if (!take('0')) {
        if (peek() < '1' || peek() > '9') {
            fail("expected a number");
        }
        do { ++position_; } while (digit(peek()));
    }
    bool real = false;
    if (take('.')) {
        real = true;
        if (!digit(peek())) {
            fail("missing decimal digits");
        }
        do { ++position_; } while (digit(peek()));
    }
    if (take('e') || take('E')) {
        real = true;
        if (!take('+')) {
            (void)take('-');
        }
        if (!digit(peek())) {
            fail("missing exponent digits");
        }
        do { ++position_; } while (digit(peek()));
    }
    const auto token = text_.substr(start, position_ - start);
    const char *end = token.data() + token.size();
    if (real) {
        double number;
        const auto result = std::from_chars(token.data(), end, number, std::chars_format::general);
        if (result.ec != std::errc{} || result.ptr != end || !std::isfinite(number)) {
            fail("floating number is out of range");
        }
        return number;
    }
    if (negative) {
        int64_t number;
        if (std::from_chars(token.data(), end, number).ec == std::errc{}) {
            return number;
        }
    } else {
        uint64_t number;
        if (std::from_chars(token.data(), end, number).ec == std::errc{}) {
            return number;
        }
    }
    return Json(Json::IntegerToken{std::string(token)});
}
Json JsonParser::value(size_t depth) {
    if (depth > maximum_depth) {
        fail("nesting limit exceeded");
    }
    whitespace();
    if (peek() == '"') {
        return string();
    }
    if (peek() == '-' || digit(peek())) {
        return number();
    }
    if (take('[')) {
        Json::Array values;
        whitespace();
        if (!take(']')) {
            do {
                values.push_back(value(depth + 1));
                whitespace();
                if (take(']')) {
                    return Json(std::move(values));
                }
                expect(',');
            } while (true);
        }
        return Json(std::move(values));
    }
    if (take('{')) {
        Json::Object fields;
        std::unordered_set<std::string> names;
        whitespace();
        if (!take('}')) {
            do {
                whitespace();
                std::string key = string();
                if (!names.insert(key).second) {
                    fail("duplicate object key");
                }
                whitespace();
                expect(':');
                fields.emplace_back(std::move(key), value(depth + 1));
                whitespace();
                if (take('}')) {
                    return Json(std::move(fields));
                }
                expect(',');
            } while (true);
        }
        return Json(std::move(fields));
    }
    for (const auto token : {std::string_view("true"), std::string_view("false"), std::string_view("null")}) {
        if (text_.substr(position_, token.size()) == token) {
            position_ += token.size();
            return token == "null" ? Json(nullptr) : Json(token == "true");
        }
    }
    fail("expected a value");
}

Json::Json(std::nullptr_t) noexcept {}
Json::Json(bool value) noexcept : data_(value) {}
Json::Json(double value) noexcept : data_(value) {}
Json::Json(const char *value) : data_(std::string(value)) {}
Json::Json(std::string value) : data_(std::move(value)) {}
Json::Json(std::initializer_list<std::pair<std::string, Json>> fields) : data_(Object(fields)) {}
Json::Json(Array values) : data_(std::move(values)) {}
Json::Json(Object values) : data_(std::move(values)) {}
Json::Json(IntegerToken value) : data_(std::move(value)) {}

Json Json::array(std::initializer_list<Json> values) {
    return Json(Array(values));
}
Json Json::object() { return Json(Object{}); }
Json Json::parse(std::string_view text) {
    return JsonParser(text).parse();
}
Json Json::parse(const uint8_t *begin, const uint8_t *end) {
    return parse(std::string_view(reinterpret_cast<const char *>(begin), static_cast<size_t>(end - begin)));
}
Json::Type Json::type() const noexcept {
    return static_cast<Type>(data_.index());
}
bool Json::is_null() const noexcept {
    return type() == Type::Null;
}
bool Json::is_boolean() const noexcept {
    return type() == Type::Boolean;
}
bool Json::is_number() const noexcept {
    return is_number_integer() || is_number_float();
}
bool Json::is_number_integer() const noexcept {
    return type() == Type::Signed || type() == Type::Unsigned || is_big_integer();
}
bool Json::is_number_unsigned() const noexcept {
    return type() == Type::Unsigned;
}
bool Json::is_number_float() const noexcept {
    return type() == Type::Real;
}
bool Json::is_big_integer() const noexcept {
    return type() == Type::Integer;
}
bool Json::is_string() const noexcept {
    return type() == Type::String;
}
bool Json::is_array() const noexcept {
    return type() == Type::Array;
}
bool Json::is_object() const noexcept {
    return type() == Type::Object;
}

size_t Json::size() const noexcept {
    if (is_null()) {
    return 0;
}
    if (is_array()) {
    return std::get<Array>(data_).size();
}
    if (is_object()) {
    return std::get<Object>(data_).size();
}
    return 1;
}
bool Json::empty() const noexcept {
    return size() == 0;
}
const Json *Json::find(std::string_view key) const noexcept {
    if (is_object()) {
        for (const auto &[name, value] : std::get<Object>(data_)) {
            if (name == key) {
    return &value;
}
        }
    }
    return nullptr;
}
bool Json::contains(std::string_view key) const noexcept {
    return find(key) != nullptr;
}
const Json &Json::at(std::string_view key) const {
    const auto *result = find(key);
    if (!result) {
    throw std::invalid_argument("missing JSON object field: " + std::string(key));
}
    return *result;
}
const Json &Json::at(size_t index) const {
    return elements().at(index);
}
const Json &Json::operator[](size_t index) const {
    return at(index);
}
Json &Json::operator[](std::string_view key) {
    if (is_null()) { data_ = Object{}; }
    if (!is_object()) {
    throw std::invalid_argument("expected JSON object");
}
    for (auto &[name, value] : std::get<Object>(data_)) {
        if (name == key) {
    return value;
}
    }
    auto &fields = std::get<Object>(data_);
    return fields.emplace_back(std::string(key), Json{}).second;
}
void Json::push_back(Json value) {
    if (!is_array()) {
    throw std::invalid_argument("expected JSON array");
}
    std::get<Array>(data_).push_back(std::move(value));
}
const Json::Object &Json::items() const {
    if (!is_object()) {
    throw std::invalid_argument("expected JSON object");
}
    return std::get<Object>(data_);
}
const Json::Array &Json::elements() const {
    if (!is_array()) {
    throw std::invalid_argument("expected JSON array");
}
    return std::get<Array>(data_);
}
Json::Array::const_iterator Json::begin() const {
    return elements().begin();
}
Json::Array::const_iterator Json::end() const {
    return elements().end();
}
Json::Array::const_reverse_iterator Json::rbegin() const {
    return elements().rbegin();
}
Json::Array::const_reverse_iterator Json::rend() const {
    return elements().rend();
}
const std::string &Json::string() const {
    if (!is_string()) {
    throw std::invalid_argument("expected JSON string");
}
    return std::get<std::string>(data_);
}
const std::string &Json::integer_text() const {
    if (!is_big_integer()) {
    throw std::invalid_argument("expected large JSON integer");
}
    return std::get<IntegerToken>(data_).text;
}
bool Json::boolean() const {
    if (!is_boolean()) {
    throw std::invalid_argument("expected JSON boolean");
}
    return std::get<bool>(data_);
}
int64_t Json::signed_integer() const {
    if (type() == Type::Signed) {
    return std::get<int64_t>(data_);
}
    if (is_number_unsigned() && std::get<uint64_t>(data_) <= INT64_MAX) {
        return static_cast<int64_t>(std::get<uint64_t>(data_));
    }
    throw std::out_of_range("expected JSON integer in int64 range");
}
uint64_t Json::unsigned_integer() const {
    if (is_number_unsigned()) {
    return std::get<uint64_t>(data_);
}
    if (type() == Type::Signed && std::get<int64_t>(data_) >= 0) {
        return static_cast<uint64_t>(std::get<int64_t>(data_));
    }
    throw std::out_of_range("expected JSON integer in uint64 range");
}
double Json::real() const {
    if (is_number_float()) {
    return std::get<double>(data_);
}
    if (is_number_unsigned()) {
    return static_cast<double>(std::get<uint64_t>(data_));
}
    if (type() == Type::Signed) {
    return static_cast<double>(std::get<int64_t>(data_));
}
    if (is_big_integer()) {
        const auto &text = integer_text();
        double value;
        const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
        if (result.ec == std::errc{} && std::isfinite(value)) { return value; }
        throw std::out_of_range("JSON integer is too large for double");
    }
    throw std::invalid_argument("expected JSON number");
}
std::string Json::value(std::string_view key, const char *fallback) const {
    const auto *found = find(key);
    return found ? found->string() : std::string(fallback);
}

bool Json::operator==(const Json &other) const {
    if (type() == other.type()) {
    return data_ == other.data_;
}
    if (is_number_float() && other.is_number()) {
    return real() == other.real();
}
    if (other.is_number_float() && is_number()) {
    return real() == other.real();
}
    if (type() == Type::Signed && other.is_number_unsigned()) {
        const auto value = std::get<int64_t>(data_);
        return value >= 0 && static_cast<uint64_t>(value) == other.get<uint64_t>();
    }
    if (is_number_unsigned() && other.type() == Type::Signed) {
    return other == *this;
}
    return false;
}

void Json::append(std::string &output, size_t depth) const {
    if (depth > maximum_depth) {
    throw std::invalid_argument("JSON nesting limit exceeded");
}
    if (is_null()) {
    output += "null";
}
    else if (is_boolean()) {
    output += boolean() ? "true" : "false";
}
    else if (is_string()) {
    quote(output, string());
}
    else if (is_big_integer()) {
    output += integer_text();
}
    else if (is_array() || is_object()) {
        output += is_array() ? '[' : '{';
        bool first = true;
        if (is_array()) {
            for (const auto &value : elements()) {
                if (!first) {
    output += ',';
}
                first = false;
                value.append(output, depth + 1);
            }
        } else {
            for (const auto &[key, value] : items()) {
                if (!first) {
    output += ',';
}
                first = false;
                quote(output, key);
                output += ':';
                value.append(output, depth + 1);
            }
        }
        output += is_array() ? ']' : '}';
    } else {
        char buffer[64];
        char *end;
        if (is_number_float()) {
            if (!std::isfinite(real())) {
    output += "null"; return;
}
            end = std::to_chars(buffer, buffer + sizeof(buffer), real()).ptr;
            output.append(buffer, end);
            const std::string_view text(buffer, static_cast<size_t>(end - buffer));
            if (text.find_first_of(".eE") == std::string_view::npos) {
    output += ".0";
}
        } else {
            end = is_number_unsigned()
                ? std::to_chars(buffer, buffer + sizeof(buffer), get<uint64_t>()).ptr
                : std::to_chars(buffer, buffer + sizeof(buffer), get<int64_t>()).ptr;
            output.append(buffer, end);
        }
    }
}
std::string Json::dump() const {
    std::string result;
    append(result, 0);
    return result;
}
}
