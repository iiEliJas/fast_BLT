#ifndef BLT_OPS_CAST_H
#define BLT_OPS_CAST_H

#ifdef __cplusplus
extern "C" {
#endif

#include "core/tensor.h"

// Elementwise dtype conversion. in and out must have the same shape and
// numel. Supported: FP32 <-> BF16, FP32 <-> FP16, FP32 <-> FP8 E4M3/E5M2.
// FP8 casts are unscaled; use ops/quant.h for scale-aware quantization.
void blt_cast(const blt_tensor *in, blt_tensor *out);

#ifdef __cplusplus
}
#endif
#endif
