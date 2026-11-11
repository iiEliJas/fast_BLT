#ifndef BLT_OPS_FP8_CONVERT_H
#define BLT_OPS_FP8_CONVERT_H

// Bit-exact scalar conversions between FP32 and the narrow float formats.
// These mirror the __nv_cvt_* / __half intrinsics used by the CUDA backend so
// CPU emulation and GPU kernels agree bit-for-bit for finite inputs.
//
// E4M3 follows OCP FP8 (1-4-3, bias 7, max finite 448, encoding 0x7F is NaN).
// E5M2 follows OCP FP8 (1-5-2, bias 15, max finite 57344).

#include <math.h>
#include <stdint.h>
#include <string.h>

static inline uint8_t blt_f32_to_e4m3(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    uint8_t sign = (uint8_t)((b >> 31) & 1u);
    uint32_t be = (b >> 23) & 0xFFu;
    uint32_t bm = b & 0x7FFFFFu;

    if (be == 0xFFu) {
        // Inf saturates to the largest finite value; NaN stays NaN.
        return bm ? (uint8_t)0x7Fu : (uint8_t)((sign << 7) | 0x7Eu);
    }
    if (be == 0u && bm == 0u) {
        return (uint8_t)(sign << 7);
    }

    int e = (int)be - 127;
    float mag = fabsf(f);

    if (e < -6) {
        uint32_t m = (uint32_t)rintf(mag * 512.0f); // quantum 2^-9
        if (m > 7u) {
            return (uint8_t)((sign << 7) | (1u << 3));
        }
        return (uint8_t)((sign << 7) | m);
    }
    if (e > 8) {
        return (uint8_t)((sign << 7) | 0x7Eu);
    }

    uint32_t m3 = bm >> 20;
    uint32_t rem = bm & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m3 & 1u))) {
        m3++;
    }
    int ef = e + 7;
    if (m3 == 8u) {
        m3 = 0;
        ef++;
    }
    if (ef >= 15) {
        // Saturation: ef 15 with mantissa 7 would collide with the NaN code.
        if (ef > 15 || m3 > 6u) {
            return (uint8_t)((sign << 7) | 0x7Eu);
        }
    }
    return (uint8_t)((sign << 7) | ((uint32_t)ef << 3) | m3);
}

static inline float blt_e4m3_to_f32(uint8_t h) {
    uint32_t sign = (h >> 7) & 1u;
    uint32_t e = (h >> 3) & 0xFu;
    uint32_t m = h & 0x7u;
    float v;
    if (e == 0u) {
        v = ldexpf((float)m, -9);
    } else if (e == 15u) {
        v = (m == 7u) ? NAN : ldexpf(1.0f + (float)m / 8.0f, 8);
    } else {
        v = ldexpf(1.0f + (float)m / 8.0f, (int)e - 7);
    }
    return sign ? -v : v;
}

static inline uint8_t blt_f32_to_e5m2(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    uint8_t sign = (uint8_t)((b >> 31) & 1u);
    uint32_t be = (b >> 23) & 0xFFu;
    uint32_t bm = b & 0x7FFFFFu;

    if (be == 0xFFu) {
        return bm ? (uint8_t)0x7Fu : (uint8_t)((sign << 7) | 0x7Cu);
    }
    if (be == 0u && bm == 0u) {
        return (uint8_t)(sign << 7);
    }

    int e = (int)be - 127;
    float mag = fabsf(f);

    if (e < -14) {
        uint32_t m = (uint32_t)rintf(mag * 65536.0f); // quantum 2^-16
        if (m > 3u) {
            return (uint8_t)((sign << 7) | (1u << 2));
        }
        return (uint8_t)((sign << 7) | m);
    }
    if (e > 15) {
        return (uint8_t)((sign << 7) | 0x7Bu);
    }

    uint32_t m2 = bm >> 21;
    uint32_t rem = bm & 0x1FFFFFu;
    if (rem > 0x100000u || (rem == 0x100000u && (m2 & 1u))) {
        m2++;
    }
    int ef = e + 15;
    if (m2 == 4u) {
        m2 = 0;
        ef++;
    }
    if (ef >= 31) {
        return (uint8_t)((sign << 7) | 0x7Bu);
    }
    return (uint8_t)((sign << 7) | ((uint32_t)ef << 2) | m2);
}

static inline float blt_e5m2_to_f32(uint8_t h) {
    uint32_t sign = (h >> 7) & 1u;
    uint32_t e = (h >> 2) & 0x1Fu;
    uint32_t m = h & 0x3u;
    float v;
    if (e == 0u) {
        v = ldexpf((float)m, -16);
    } else if (e == 31u) {
        v = (m != 0u) ? NAN : INFINITY;
    } else {
        v = ldexpf(1.0f + (float)m / 4.0f, (int)e - 15);
    }
    return sign ? -v : v;
}

static inline uint16_t blt_f32_to_fp16(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    uint16_t sign = (uint16_t)((b >> 31) & 1u);
    uint32_t be = (b >> 23) & 0xFFu;
    uint32_t bm = b & 0x7FFFFFu;

    if (be == 0xFFu) {
        return (uint16_t)((sign << 15) | 0x7C00u | (bm ? 0x200u : 0u));
    }

    int he = (int)be - 112; // fp32 bias 127 minus fp16 bias 15
    if (he >= 0x1F) {
        return (uint16_t)((sign << 15) | 0x7C00u);
    }
    if (he <= 0) {
        if (he < -10) {
            return (uint16_t)(sign << 15);
        }
        bm |= 0x800000u;
        int shift = 14 - he;
        uint32_t rem = bm & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        uint32_t out = bm >> shift;
        if (rem > half || (rem == half && (out & 1u))) {
            out++;
        }
        return (uint16_t)((sign << 15) | out);
    }

    uint16_t out = (uint16_t)((sign << 15) | ((uint32_t)he << 10) | (bm >> 13));
    uint32_t rem = bm & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1u))) {
        out++;
    }
    return out;
}

static inline float blt_fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    float v;
    if (e == 0u) {
        v = ldexpf((float)m, -24);
    } else if (e == 31u) {
        v = (m != 0u) ? NAN : INFINITY;
    } else {
        v = ldexpf(1.0f + (float)m / 1024.0f, (int)e - 15);
    }
    return sign ? -v : v;
}

#endif // BLT_OPS_FP8_CONVERT_H
