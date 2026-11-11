#ifndef BLT_CORE_DTYPE_H
#define BLT_CORE_DTYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef enum {
    BLT_DTYPE_FP32 = 0,
    BLT_DTYPE_BF16 = 1,
    BLT_DTYPE_INT32 = 2,
    BLT_DTYPE_UINT8 = 3,
    BLT_DTYPE_FP8_E4M3 = 4,
    BLT_DTYPE_FP16 = 5,
    BLT_DTYPE_FP8_E5M2 = 6
} blt_dtype;

static inline size_t blt_dtype_sizeof(blt_dtype dtype) {
    switch (dtype) {
    case BLT_DTYPE_FP32:
        return 4;
    case BLT_DTYPE_BF16:
        return 2;
    case BLT_DTYPE_INT32:
        return 4;
    case BLT_DTYPE_UINT8:
        return 1;
    case BLT_DTYPE_FP8_E4M3:
        return 1;
    case BLT_DTYPE_FP16:
        return 2;
    case BLT_DTYPE_FP8_E5M2:
        return 1;
    default:
        return 0;
    }
}

static inline int blt_dtype_is_fp8(blt_dtype dtype) {
    return dtype == BLT_DTYPE_FP8_E4M3 || dtype == BLT_DTYPE_FP8_E5M2;
}

static inline int blt_dtype_is_fp16(blt_dtype dtype) { return dtype == BLT_DTYPE_BF16 || dtype == BLT_DTYPE_FP16; }

// Short name for logs and CLI errors.
static inline const char *blt_dtype_name(blt_dtype dtype) {
    switch (dtype) {
    case BLT_DTYPE_FP32:
        return "fp32";
    case BLT_DTYPE_BF16:
        return "bf16";
    case BLT_DTYPE_FP16:
        return "fp16";
    case BLT_DTYPE_FP8_E4M3:
        return "fp8-e4m3";
    case BLT_DTYPE_FP8_E5M2:
        return "fp8-e5m2";
    case BLT_DTYPE_INT32:
        return "int32";
    case BLT_DTYPE_UINT8:
        return "uint8";
    default:
        return "unknown";
    }
}

// Largest finite magnitude representable by an FP8 encoding. Used to derive
// the per-tensor scale (amax / max_value).
static inline float blt_fp8_max(blt_dtype dtype) { return dtype == BLT_DTYPE_FP8_E5M2 ? 57344.0f : 448.0f; }

#ifdef __cplusplus
}
#endif

#endif // BLT_CORE_DTYPE_H
