#include "runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>

namespace trillim::cpu {
namespace {

constexpr size_t cancellation_interval = 4096;

void store_scale(uint8_t *output, float scale) {
    const uint16_t bits = trillim_to_bf16(scale);
    std::memcpy(output, &bits, sizeof(bits));
}

int quantize(float value, int minimum, int maximum) {
    return static_cast<int>(std::clamp(std::nearbyint(value),
                                      static_cast<float>(minimum), static_cast<float>(maximum)));
}

float max_abs(const float *values, size_t count) {
    float maximum = 0;
    for (size_t i = 0; i < count; ++i) {
        maximum = std::max(maximum, std::abs(values[i]));
    }
    return maximum;
}

void encode_rows(const float *source, size_t rows, size_t columns, uint8_t *output,
                 const Runtime &runtime) {
    auto *scales = output + rows * columns;
    for (size_t row = 0; row < rows; ++row) {
        float maximum = 1e-5f;
        for (size_t start = 0; start < columns;) {
            runtime.check_cancelled();
            const size_t count = std::min(columns - start, cancellation_interval);
            maximum = std::max(maximum, max_abs(source + start, count));
            start += count;
        }
        store_scale(scales + row * 2, maximum / 127.0f);
        const float multiplier = 127.0f / maximum;
        for (size_t column = 0; column < columns; ++column) {
            if (column % cancellation_interval == 0) {
                runtime.check_cancelled();
            }
            output[column] = static_cast<uint8_t>(quantize(source[column] * multiplier, -127, 127));
        }
        source += columns;
        output += columns;
    }
}

void encode_ternary(const float *source, float scale, bool matrix, uint8_t *output) {
    const float divisor = scale == 0 ? 1.0f : scale;
    for (size_t byte = 0; byte < 32; ++byte) {
        uint8_t packed = 0;
        for (size_t lane = 0; lane < 4; ++lane) {
            const float normalized = source[byte + lane * 32] / divisor;
            const int code = matrix ? quantize(normalized, -1, 1) + 1 :
                (normalized >= 0.5f ? 2 : (normalized <= -0.5f ? 0 : 1));
            packed = static_cast<uint8_t>((packed << 2) | code);
        }
        output[byte] = packed;
    }
}

void encode_binary(const float *source, size_t valid, uint8_t *output, uint8_t *scale) {
    // NumPy's FP32 reduction of a contiguous 128-element block accumulates eight lanes.
    std::array<float, 8> sums;
    for (size_t lane = 0; lane < sums.size(); ++lane) {
        sums[lane] = std::abs(source[lane]);
    }
    for (size_t byte = 0; byte < 16; ++byte) {
        uint8_t signs = 0;
        for (size_t lane = 0; lane < sums.size(); ++lane) {
            const float value = source[byte * sums.size() + lane];
            if (byte != 0) {
                sums[lane] += std::abs(value);
            }
            signs = static_cast<uint8_t>((signs << 1) | (value >= 0));
        }
        output[byte] = signs;
    }
    const float sum = ((sums[0] + sums[1]) + (sums[2] + sums[3])) +
                      ((sums[4] + sums[5]) + (sums[6] + sums[7]));
    store_scale(scale, sum / static_cast<float>(valid));
}

void encode_integers(const float *source, uint32_t dtype, uint8_t *output,
                     uint8_t *scale, uint8_t *high) {
    if (dtype == TRILLIM_GRP_INT8) {
        const float ideal = max_abs(source, 32) / 127.0f;
        const float divisor = ideal == 0 ? 1.0f : ideal;
        store_scale(scale, ideal);
        for (size_t i = 0; i < 32; ++i) {
            output[i] = static_cast<uint8_t>(quantize(source[i] / divisor, -127, 127));
        }
        return;
    }

    const int bound = dtype == TRILLIM_GRP_INT4 ? 8 : 16;
    const auto [minimum, maximum] = std::minmax_element(source, source + 32);
    const float positive = std::max(0.0f, *maximum) / static_cast<float>(bound - 1);
    const float negative = std::max(0.0f, -*minimum) / static_cast<float>(bound);
    const float rounded = trillim_bf16(trillim_to_bf16(std::max(positive, negative)));
    const float divisor = rounded == 0 ? 1.0f : rounded;
    store_scale(scale, rounded);

    std::array<int, 32> codes;
    for (size_t i = 0; i < codes.size(); ++i) {
        codes[i] = quantize(source[i] / divisor, -bound, bound - 1) + bound;
    }
    for (size_t byte = 0; byte < 16; ++byte) {
        output[byte] = static_cast<uint8_t>((codes[byte] & 15) | ((codes[byte + 16] & 15) << 4));
    }
    if (dtype == TRILLIM_GRP_INT5) {
        for (size_t byte = 0; byte < 4; ++byte) {
            uint8_t packed = 0;
            for (size_t bit = 0; bit < 8; ++bit) {
                packed |= static_cast<uint8_t>((codes[byte * 8 + bit] >> 4) << bit);
            }
            high[byte] = packed;
        }
    }
}

void recover_int6(const float *source, float master, int &subscale, int *quantized) {
    constexpr float tolerance = 1e-5f;
    const float divisor = master == 0 ? 1.0f : master;
    std::array<int, 16> products;
    int common = 0;
    for (size_t i = 0; i < products.size(); ++i) {
        products[i] = static_cast<int>(std::nearbyint(source[i] / divisor));
        const float represented = static_cast<float>(products[i]) * master;
        if (std::abs(represented - source[i]) > tolerance) {
            return;
        }
        common = std::gcd(common, products[i]);
    }

    const auto [minimum, maximum] = std::minmax_element(products.begin(), products.end());
    int closest = 0;
    int distance = 128;
    for (int candidate = 1; candidate < 128; ++candidate) {
        const int difference = std::abs(candidate - subscale);
        if (difference < distance && common % candidate == 0 &&
            *minimum >= -32 * candidate && *maximum <= 31 * candidate) {
            closest = candidate;
            distance = difference;
        }
    }
    if (closest != 0) {
        subscale = closest;
        for (size_t i = 0; i < products.size(); ++i) {
            quantized[i] = products[i] / closest;
        }
    }
}

void encode_int6(const float *source, uint8_t *low, uint8_t *scale,
                 uint8_t *high, uint8_t *subscales) {
    std::array<float, 16> ideal;
    for (size_t sub = 0; sub < ideal.size(); ++sub) {
        ideal[sub] = max_abs(source + sub * 16, 16) / 31.0f;
    }
    const auto anchor = std::max_element(ideal.begin(), ideal.end());
    const float master = trillim_bf16(trillim_to_bf16(*anchor / 127.0f));
    const float divisor = master == 0 ? 1.0f : master;
    store_scale(scale, master);

    std::array<int, 256> codes;
    for (size_t sub = 0; sub < ideal.size(); ++sub) {
        int subscale = quantize(ideal[sub] / divisor, 0, 127);
        const float effective = master * static_cast<float>(subscale);
        const float denominator = effective == 0 ? 1.0f : effective;
        const float *chunk = source + sub * 16;
        int *quantized = codes.data() + sub * 16;
        bool changed = false;
        for (size_t i = 0; i < 16; ++i) {
            quantized[i] = quantize(chunk[i] / denominator, -32, 31);
            const float reconstructed = static_cast<float>(quantized[i] * subscale) * master;
            changed |= std::abs(reconstructed - chunk[i]) > 1e-5f;
        }
        // Keep the largest subgroup as the anchor for re-encoding the master scale.
        if (changed && ideal.begin() + sub != anchor) {
            recover_int6(chunk, master, subscale, quantized);
        }
        subscales[sub] = static_cast<uint8_t>(subscale);
        for (size_t i = 0; i < 16; ++i) {
            quantized[i] += 32;
        }
    }

    for (size_t half = 0; half < 2; ++half) {
        const int *block = codes.data() + half * 128;
        for (size_t i = 0; i < 32; ++i) {
            low[half * 64 + i] = static_cast<uint8_t>((block[i] & 15) | ((block[i + 32] & 15) << 4));
            low[half * 64 + 32 + i] = static_cast<uint8_t>((block[i + 64] & 15) | ((block[i + 96] & 15) << 4));
            high[half * 32 + i] = static_cast<uint8_t>((block[i] >> 4) | ((block[i + 32] >> 4) << 2) |
                ((block[i + 64] >> 4) << 4) | ((block[i + 96] >> 4) << 6));
        }
    }
}

void encode_mxfp4(const float *source, uint8_t *output, uint8_t *scale) {
    constexpr std::array<float, 7> midpoints = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    const float maximum = max_abs(source, 32);
    int exponent = 0;
    if (maximum != 0) {
        std::frexp(maximum, &exponent);
        exponent = std::clamp(exponent - 3, -127, 127);
    }
    *scale = static_cast<uint8_t>(exponent + 127);
    for (size_t byte = 0; byte < 16; ++byte) {
        uint8_t packed = 0;
        for (size_t lane = 0; lane < 2; ++lane) {
            const float value = source[byte * 2 + lane];
            const float magnitude = std::abs(std::ldexp(value, -exponent));
            const auto midpoint = std::lower_bound(midpoints.begin(), midpoints.end(), magnitude);
            auto code = static_cast<uint8_t>(midpoint - midpoints.begin());
            if (midpoint != midpoints.end() && *midpoint == magnitude && (code & 1) != 0) {
                ++code;
            }
            code |= static_cast<uint8_t>(std::signbit(value) ? 8 : 0);
            packed |= static_cast<uint8_t>(code << (lane * 4));
        }
        output[byte] = packed;
    }
}

float read_int6(const uint8_t *payload, size_t blocks, size_t block, size_t offset) {
    const auto *scales = payload + blocks * 128;
    const auto *low = payload + block * 128 + offset / 128 * 64;
    const auto *high = scales + blocks * 2 + block * 64 + offset / 128 * 32;
    const auto *subscales = scales + blocks * 66 + block * 16;
    const size_t lane = offset % 128 / 32;
    const size_t byte = offset % 32;
    const int code = ((low[lane / 2 * 32 + byte] >> (lane % 2 * 4)) & 15) |
        (((high[byte] >> (lane * 2)) & 3) << 4);
    uint16_t bits;
    int8_t subscale;
    std::memcpy(&bits, scales + block * 2, sizeof(bits));
    std::memcpy(&subscale, subscales + offset / 16, sizeof(subscale));
    return (static_cast<float>(code - 32) * static_cast<float>(subscale)) * trillim_bf16(bits);
}

} // namespace

Allocation encode_packed(const float *source, uint32_t dtype, size_t rows, size_t columns,
                         const Runtime &runtime) {
    if (itemsize(dtype) != 0) {
        throw std::invalid_argument("packed codec requires a packed dtype");
    }
    Allocation output(packed_bytes(dtype, rows, columns));
    const bool matrix = dtype == TRILLIM_MAT_TERNARY;
    float gamma = 0;
    for (size_t start = 0, size = rows * columns; start < size;) {
        runtime.check_cancelled();
        const size_t count = std::min(size - start, cancellation_interval);
        for (size_t i = 0; i < count; ++i) {
            const float value = source[start + i];
            if (!std::isfinite(value)) {
                throw std::invalid_argument("packed dtypes require finite FP32 values");
            }
            if (matrix) {
                gamma = std::max(gamma, std::abs(value));
            }
        }
        start += count;
    }
    if (dtype == TRILLIM_ROW_INT8) {
        encode_rows(source, rows, columns, output.data, runtime);
        return output;
    }

    const size_t group_size = dtype == TRILLIM_DUBGRP_INT6 ? 256 :
        (matrix || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_BINARY ? 128 : 32);
    const size_t width = dtype == TRILLIM_DUBGRP_INT6 ? 128 :
        (matrix || dtype == TRILLIM_GRP_TERNARY || dtype == TRILLIM_GRP_INT8 ? 32 : 16);
    const size_t groups = columns / group_size + (columns % group_size != 0);
    const size_t blocks = rows * groups;
    auto *scales = output.data + blocks * width;
    if (matrix) {
        std::memcpy(scales, &gamma, sizeof(gamma));
    }

    std::array<float, 256> padded;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t group = 0; group < groups; ++group) {
            const size_t block = row * groups + group;
            if (block % 16 == 0) {
                runtime.check_cancelled();
            }
            const size_t start = group * group_size;
            const size_t valid = std::min(columns - start, group_size);
            const float *values = source + row * columns + start;
            if (valid != group_size) {
                std::copy_n(values, valid, padded.begin());
                std::fill_n(padded.begin() + valid, group_size - valid, 0.0f);
                values = padded.data();
            }
            auto *destination = output.data + block * width;
            switch (dtype) {
            case TRILLIM_MAT_TERNARY:
                encode_ternary(values, gamma, true, destination);
                break;
            case TRILLIM_GRP_TERNARY: {
                const float scale = max_abs(values, 128);
                encode_ternary(values, scale, false, destination);
                store_scale(scales + block * 2, scale);
                break;
            }
            case TRILLIM_GRP_BINARY:
                encode_binary(values, valid, destination, scales + block * 2);
                break;
            case TRILLIM_GRP_INT8:
            case TRILLIM_GRP_INT4:
                encode_integers(values, dtype, destination, scales + block * 2, nullptr);
                break;
            case TRILLIM_GRP_INT5:
                encode_integers(values, dtype, destination, scales + block * 2,
                                scales + blocks * 2 + block * 4);
                break;
            case TRILLIM_DUBGRP_INT6:
                encode_int6(values, destination, scales + block * 2,
                            scales + blocks * 2 + block * 64,
                            scales + blocks * 66 + block * 16);
                break;
            case TRILLIM_MXFP4:
                encode_mxfp4(values, destination, scales + block);
                break;
            }
        }
    }
    return output;
}

Allocation decode_packed(const uint8_t *payload, uint32_t dtype, size_t rows, size_t columns,
                         const Runtime &runtime) {
    if (itemsize(dtype) != 0) {
        throw std::invalid_argument("packed codec requires a packed dtype");
    }
    packed_bytes(dtype, rows, columns);
    const size_t dense_bytes = packed_bytes(TRILLIM_FP32, rows, columns);
    if (dtype == TRILLIM_MAT_TERNARY || dtype == TRILLIM_GRP_TERNARY) {
        const size_t groups = columns / 128 + (columns % 128 != 0);
        const size_t values = rows * groups * 32;
        for (size_t i = 0; i < values; ++i) {
            if (i % cancellation_interval == 0) {
                runtime.check_cancelled();
            }
            if ((payload[i] & (payload[i] >> 1) & 0x55) != 0) {
                throw std::invalid_argument("ternary code 3 is invalid");
            }
        }
    }

    Allocation output(dense_bytes);
    const size_t int6_groups = columns / 256 + (columns % 256 != 0);
    for (size_t i = 0; i < rows * columns; ++i) {
        if (i % cancellation_interval == 0) {
            runtime.check_cancelled();
        }
        // INT6's two FP32 multiplications preserve the sign of a zero subscale product.
        const float value = dtype == TRILLIM_DUBGRP_INT6 ?
            read_int6(payload, rows * int6_groups, i / columns * int6_groups + i % columns / 256,
                      i % columns % 256) :
            trillim_packed_read_float(payload, i, dtype, rows, columns);
        std::memcpy(output.data + i * sizeof(value), &value, sizeof(value));
    }
    return output;
}

} // namespace trillim::cpu
