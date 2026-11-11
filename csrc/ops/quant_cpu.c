#include "core/backend.h"
#include "ops/quant.h"
#include "ops/fp8_convert.h"

#include <math.h>
#include <stdint.h>

static float scale_value(const blt_tensor *scale) {
    if (scale == NULL) {
        return 1.0f;
    }
    return ((const float *)scale->data)[0];
}

static uint8_t to_fp8(blt_dtype dtype, float v) {
    if (dtype == BLT_DTYPE_FP8_E5M2) {
        return blt_f32_to_e5m2(v);
    }
    return blt_f32_to_e4m3(v);
}

static float from_fp8(blt_dtype dtype, uint8_t v) {
    if (dtype == BLT_DTYPE_FP8_E5M2) {
        return blt_e5m2_to_f32(v);
    }
    return blt_e4m3_to_f32(v);
}

void blt_amax_cpu(const blt_tensor *x, blt_tensor *out) {
    BLT_REQUIRE(x != NULL && out != NULL, "blt_amax: x and out must not be NULL");
    BLT_REQUIRE(x->dtype == BLT_DTYPE_FP32, "blt_amax: x must be FP32");
    BLT_REQUIRE(out->numel >= 1 && out->dtype == BLT_DTYPE_FP32, "blt_amax: out must hold at least one FP32 value");

    const float *src = (const float *)x->data;
    float amax = 0.0f;
    for (size_t i = 0; i < x->numel; ++i) {
        float a = fabsf(src[i]);
        if (a > amax) {
            amax = a;
        }
    }
    ((float *)out->data)[0] = amax;
}

void blt_quant_scale_cpu(const blt_tensor *amax_in, float max_value, float min_scale, blt_tensor *scale_out) {
    BLT_REQUIRE(amax_in != NULL && scale_out != NULL, "blt_quant_scale: inputs must not be NULL");
    BLT_REQUIRE(amax_in->numel >= 1 && scale_out->numel >= 1, "blt_quant_scale: inputs must hold one value");
    BLT_REQUIRE(max_value > 0.0f, "blt_quant_scale: max_value must be positive");

    float s = ((const float *)amax_in->data)[0] / max_value;
    if (!(s >= min_scale)) {
        s = min_scale;
    }
    ((float *)scale_out->data)[0] = s;
}

void blt_fp8_quantize_cpu(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose) {
    BLT_REQUIRE(x != NULL && out != NULL, "blt_fp8_quantize: x and out must not be NULL");
    BLT_REQUIRE(x->ndim == 2 && out->ndim == 2, "blt_fp8_quantize: x and out must be 2D");
    BLT_REQUIRE(x->dtype == BLT_DTYPE_FP32, "blt_fp8_quantize: x must be FP32");
    BLT_REQUIRE(blt_dtype_is_fp8(out->dtype), "blt_fp8_quantize: out must be an FP8 dtype");

    const size_t rows = x->shape[0];
    const size_t cols = x->shape[1];
    const float inv = 1.0f / scale_value(scale);

    const float *src = (const float *)x->data;
    uint8_t *dst = (uint8_t *)out->data;
    if (transpose) {
        BLT_REQUIRE(out->shape[0] == cols && out->shape[1] == rows, "blt_fp8_quantize: transposed out shape mismatch");
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                dst[c * rows + r] = to_fp8(out->dtype, src[r * cols + c] * inv);
            }
        }
    } else {
        BLT_REQUIRE(out->shape[0] == rows && out->shape[1] == cols, "blt_fp8_quantize: out shape mismatch");
        for (size_t i = 0; i < rows * cols; ++i) {
            dst[i] = to_fp8(out->dtype, src[i] * inv);
        }
    }
}

void blt_fp8_dequantize_cpu(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose) {
    BLT_REQUIRE(x != NULL && out != NULL, "blt_fp8_dequantize: x and out must not be NULL");
    BLT_REQUIRE(x->ndim == 2 && out->ndim == 2, "blt_fp8_dequantize: x and out must be 2D");
    BLT_REQUIRE(blt_dtype_is_fp8(x->dtype), "blt_fp8_dequantize: x must be an FP8 dtype");
    BLT_REQUIRE(out->dtype == BLT_DTYPE_FP32, "blt_fp8_dequantize: out must be FP32");

    const size_t rows = transpose ? x->shape[1] : x->shape[0];
    const size_t cols = transpose ? x->shape[0] : x->shape[1];
    BLT_REQUIRE(out->shape[0] == rows && out->shape[1] == cols, "blt_fp8_dequantize: out shape mismatch");

    const float s = scale_value(scale);
    const uint8_t *src = (const uint8_t *)x->data;
    float *dst = (float *)out->data;
    if (transpose) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                dst[r * cols + c] = from_fp8(x->dtype, src[c * rows + r]) * s;
            }
        }
    } else {
        for (size_t i = 0; i < rows * cols; ++i) {
            dst[i] = from_fp8(x->dtype, src[i]) * s;
        }
    }
}

void blt_fp8_roundtrip_cpu(const blt_tensor *x, const blt_tensor *scale, blt_dtype fmt, blt_tensor *out) {
    BLT_REQUIRE(x != NULL && out != NULL, "blt_fp8_roundtrip: x and out must not be NULL");
    BLT_REQUIRE(x->numel == out->numel, "blt_fp8_roundtrip: x and out must have the same numel");
    BLT_REQUIRE(blt_dtype_is_fp8(fmt), "blt_fp8_roundtrip: fmt must be an FP8 dtype");

    const float s = scale_value(scale);
    const float inv = 1.0f / s;
    const float *src = (const float *)x->data;
    float *dst = (float *)out->data;
    for (size_t i = 0; i < x->numel; ++i) {
        dst[i] = from_fp8(fmt, to_fp8(fmt, src[i] * inv)) * s;
    }
}

void blt_fp8_transpose_cpu(const blt_tensor *src, blt_tensor *dst) {
    BLT_REQUIRE(src != NULL && dst != NULL, "blt_fp8_transpose: src and dst must not be NULL");
    BLT_REQUIRE(blt_dtype_is_fp8(src->dtype) && src->dtype == dst->dtype,
                "blt_fp8_transpose: src and dst must share an FP8 dtype");
    const size_t rows = src->shape[0];
    const size_t cols = src->shape[1];
    BLT_REQUIRE(dst->shape[0] == cols && dst->shape[1] == rows, "blt_fp8_transpose: dst shape must be src^T");
    const uint8_t *s = (const uint8_t *)src->data;
    uint8_t *d = (uint8_t *)dst->data;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            d[c * rows + r] = s[r * cols + c];
        }
    }
}

int blt_fp8_matmul_eligible(size_t m, size_t k, size_t n) {
    // Tensor-core FP8 wants a 16-element contraction axis, and below roughly
    // 16k MACs the format conversion costs more than the GEMM saves.
    if (k % 16 != 0 || m < 16 || n < 16) {
        return 0;
    }
    return (m * k * n) >= (size_t)16384;
}
