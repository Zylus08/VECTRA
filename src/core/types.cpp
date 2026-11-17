#include "vectra/core/types.hpp"
#include <cstring>

namespace vectra {

uint16_t float32_to_float16(float val) {
    uint32_t f;
    std::memcpy(&f, &val, sizeof(float));

    uint32_t sign = (f >> 31) & 0x1;
    int32_t exp = static_cast<int32_t>((f >> 23) & 0xFF) - 127;
    uint32_t frac = f & 0x7FFFFF;

    uint16_t h_sign = static_cast<uint16_t>(sign << 15);

    if (exp > 15) {
        // Overflow to infinity
        return h_sign | 0x7C00;
    } else if (exp >= -14) {
        // Normal half float
        uint16_t h_exp = static_cast<uint16_t>((exp + 15) << 10);
        uint16_t h_frac = static_cast<uint16_t>(frac >> 13);
        // Round to nearest even
        if ((frac & 0x1000) && ((frac & 0xFFF) || (h_frac & 1))) {
            h_frac++;
        }
        return h_sign | (h_exp + h_frac);
    } else if (exp >= -24) {
        // Subnormal half float
        frac |= 0x800000;
        int shift = -14 - exp + 13;
        uint16_t h_frac = static_cast<uint16_t>(frac >> shift);
        return h_sign | h_frac;
    } else {
        // Underflow to zero
        return h_sign;
    }
}

float float16_to_float32(uint16_t val) {
    uint32_t sign = (val >> 15) & 0x1;
    uint32_t exp = (val >> 10) & 0x1F;
    uint32_t frac = val & 0x3FF;

    uint32_t f_sign = sign << 31;
    uint32_t f_exp;
    uint32_t f_frac;

    if (exp == 0) {
        if (frac == 0) {
            // Zero
            f_exp = 0;
            f_frac = 0;
        } else {
            // Subnormal in FP16, normalize for FP32
            int shift = 0;
            while ((frac & 0x400) == 0) {
                frac <<= 1;
                shift++;
            }
            frac &= 0x3FF;
            f_exp = (127 - 15 - shift + 1) << 23;
            f_frac = frac << 13;
        }
    } else if (exp == 31) {
        // Inf or NaN
        f_exp = 0xFF << 23;
        f_frac = frac << 13;
    } else {
        // Normalized
        f_exp = (exp - 15 + 127) << 23;
        f_frac = frac << 13;
    }

    uint32_t f = f_sign | f_exp | f_frac;
    float res;
    std::memcpy(&res, &f, sizeof(float));
    return res;
}

} // namespace vectra
