#include "core/backend.h"
#include "ops/cast.h"
#include "ops/fp8_convert.h"

#include <string.h>
#include <stdint.h>

// fp32 -> bf16 -> fp32 round-trip. Deterministic, bit-exact on all
// platforms — this is the CPU ground truth for CUDA parity tests.

static uint16_t fp32_to_bf16_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    // Round-to-nearest-even bias.
    uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    return (uint16_t)((bits + rounding_bias) >> 16);
}

static float bf16_bits_to_fp32(uint16_t h) {
    // bf16 and fp32 share the same exponent layout (8-bit, bias 127).
    // Mantissa width differs (7 vs 23 bits), so conversion is just a
    // 16-bit left shift — sign, exponent, and mantissa land in place.
    uint32_t bits = (uint32_t)h << 16;
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

void blt_cast_cpu(const blt_tensor *in, blt_tensor *out) {
    BLT_REQUIRE(in != NULL && out != NULL, "blt_cast: in and out must not be NULL");
    BLT_REQUIRE(in->numel == out->numel, "blt_cast: in and out must have the same numel");

    if (in->dtype == out->dtype) {
        if (in->data != out->data) memcpy(out->data, in->data, in->numel * blt_dtype_sizeof(in->dtype));
        return;
    }

    if (in->dtype == BLT_DTYPE_FP32 && out->dtype == BLT_DTYPE_BF16) {
        const float *src = (const float *)in->data;
        uint16_t *dst = (uint16_t *)out->data;
        for (size_t i = 0; i < in->numel; ++i) dst[i] = fp32_to_bf16_bits(src[i]);
        return;
    }

    if (in->dtype == BLT_DTYPE_BF16 && out->dtype == BLT_DTYPE_FP32) {
        const uint16_t *src = (const uint16_t *)in->data;
        float *dst = (float *)out->data;
        for (size_t i = 0; i < in->numel; ++i) dst[i] = bf16_bits_to_fp32(src[i]);
        return;
    }

    if (in->dtype == BLT_DTYPE_FP32 && out->dtype == BLT_DTYPE_FP16) {
        const float *src = (const float *)in->data;
        uint16_t *dst = (uint16_t *)out->data;
        for (size_t i = 0; i < in->numel; ++i) dst[i] = blt_f32_to_fp16(src[i]);
        return;
    }
    if (in->dtype == BLT_DTYPE_FP16 && out->dtype == BLT_DTYPE_FP32) {
        const uint16_t *src = (const uint16_t *)in->data;
        float *dst = (float *)out->data;
        for (size_t i = 0; i < in->numel; ++i) dst[i] = blt_fp16_to_f32(src[i]);
        return;
    }

    if (in->dtype == BLT_DTYPE_FP32 && out->dtype == BLT_DTYPE_FP8_E4M3) {
        const float *src = (const float *)in->data;
        uint8_t *dst = (uint8_t *)out->data;
        for (size_t i = 0; i < in->numel; ++i) dst[i] = blt_f32_to_e4m3(src[i]);
        return;
    }
    if (in->dtype == BLT_DTYPE_FP8_E4M3 && out->dtype == BLT_DTYPE_FP32) {
        const uint8_t *src = (const uint8_t *)in->data;
        float *dst = (float *)out->data;
        for (size_t i = 0; i < in->numel; ++i) dst[i] = blt_e4m3_to_f32(src[i]);
        return;
    }

    if (in->dtype == BLT_DTYPE_FP32 && out->dtype == BLT_DTYPE_FP8_E5M2) {
        const float *src = (const float *)in->data;
        uint8_t *dst = (uint8_t *)out->data;
        for (size_t i = 0; i < in->numel; ++i) dst[i] = blt_f32_to_e5m2(src[i]);
        return;
    }
    if (in->dtype == BLT_DTYPE_FP8_E5M2 && out->dtype == BLT_DTYPE_FP32) {
        const uint8_t *src = (const uint8_t *)in->data;
        float *dst = (float *)out->data;
        for (size_t i = 0; i < in->numel; ++i) dst[i] = blt_e5m2_to_f32(src[i]);
        return;
    }

    BLT_FATAL("blt_cast: unsupported conversion from dtype %d to %d", in->dtype, out->dtype);
}
