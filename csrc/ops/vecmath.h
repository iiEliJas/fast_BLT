#ifndef BLT_OPS_VECMATH_H
#define BLT_OPS_VECMATH_H

#include <math.h>
#include <stddef.h>
#include <stdbool.h>
#include "core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// sum(a[i] * b[i]) for i in [0, n).
float blt_vec_dot(blt_backend backend, const float *a, const float *b, size_t n);

// Same as blt_vec_dot, but writes the result to *out (host or device pointer
// matching backend) instead of returning it. Avoids a host sync when the
// caller accumulates several results before downloading.
void blt_vec_dot_into(blt_backend backend, const float *a, const float *b, size_t n, float *out);

// Numerically stable softmax over a single FP32 row, in place.
//
// For each column col:
//   - if mask_row != NULL:  row[col] = row[col]*scale + mask_row[col]
//     (mask_row holds additive mask values, 0 or -INFINITY)
//   - else if is_causal and col > row_idx: row[col] = -INFINITY
//   - else:                 row[col] *= scale
//
// Non-finite entries after masking are zeroed; if every entry is non-finite
// the row is left all-zero.
void blt_softmax_masked_row_inplace(blt_backend backend, float *row, size_t row_len, size_t row_idx, bool is_causal,
                                    const float *mask_row, float scale);

// Copies a rows x cols tile between strided fp32 buffers (row strides in
// elements). Used for QKV head-slice gather and cache-row scatter.
void blt_strided_copy(blt_backend backend, float *dst, size_t dst_stride, const float *src, size_t src_stride,
                      size_t rows, size_t cols);

void blt_fill_uniform(blt_backend backend, float *data, size_t n, uint64_t *rng_state);
void blt_fill_constant(blt_backend backend, float *data, size_t n, float v);

// Element-wise scale: data[i] *= s for i in [0, n). CUDA-aware.
void blt_cuda_vec_scale(float *data, float s, size_t n);

// In-place SGD: w[i] -= lr * (g[i] * grad_scale) for i in [0, n). Device pointers.
void blt_cuda_sgd_step(float *w, const float *g, float lr, float grad_scale, size_t n);

// In-place AdamW step on device pointers. Computes bias-corrected
// AdamW update for each element using g[i] * grad_scale: w, m, v are all
// device pointers.
void blt_cuda_adamw_step(float *w, const float *g, float *m, float *v, float lr, float b1, float b2, float eps,
                         float wd, float bc1, float bc2, float grad_scale, size_t n);

// Reduction: returns sum of squared AdamW updates over [0, n), computed from
// g[i] * grad_scale. All pointers are device pointers.
float blt_cuda_adamw_sqnorm(const float *w, const float *g, const float *m, const float *v, float b1, float b2,
                            float eps, float wd, float bc1, float bc2, float grad_scale, size_t n);

#ifdef __cplusplus
}
#endif

#endif // BLT_OPS_VECMATH_H
