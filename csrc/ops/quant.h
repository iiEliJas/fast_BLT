#ifndef BLT_OPS_QUANT_H
#define BLT_OPS_QUANT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "core/tensor.h"

// Per-tensor symmetric FP8 quantization used by the mixed-precision matmul
// path. Scales live in 1-element FP32 tensors on the same backend as the data.

// amax_out[0] = max |x| over all elements. x must be FP32.
void blt_amax(const blt_tensor *x, blt_tensor *amax_out);

// scale_out[0] = max(amax_in[0] / max_value, min_scale). Both are 1-element FP32.
void blt_quant_scale(const blt_tensor *amax_in, float max_value, float min_scale, blt_tensor *scale_out);

// out[i] = fp8(x[i] / scale). out->dtype selects E4M3 or E5M2. When transpose
// is set, x is [rows, cols] and out is [cols, rows] holding x^T.
void blt_fp8_quantize(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose);

// out = fp32(x) * scale. When transpose is set, x is stored as [cols, rows] and
// out is [rows, cols] = x^T.
void blt_fp8_dequantize(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose);

// out = dequantize(quantize(x)) with FP32 in and out. `fmt` selects the FP8
// encoding. This is the CPU ground truth for a tensor a CUDA FP8 kernel would
// consume.
void blt_fp8_roundtrip(const blt_tensor *x, const blt_tensor *scale, blt_dtype fmt, blt_tensor *out);

// dst = src^T for FP8 buffers. src is [rows, cols], dst is [cols, rows]; the
// bytes are copied verbatim (no requantization). dst->scale is left untouched.
void blt_fp8_transpose(const blt_tensor *src, blt_tensor *dst);

// Size/alignment gate for the tensor-core FP8 GEMM. Below the threshold the
// dequantize + SGEMM fallback is both cheaper and more accurate.
int blt_fp8_matmul_eligible(size_t m, size_t k, size_t n);

#ifdef __cplusplus
}
#endif

#endif // BLT_OPS_QUANT_H
