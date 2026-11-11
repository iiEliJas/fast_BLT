#ifndef BLT_CORE_TENSOR_H
#define BLT_CORE_TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include "core/dtype.h"

#define BLT_MAX_NDIM 4

typedef enum { BLT_BACKEND_CPU = 0, BLT_BACKEND_CUDA = 1 } blt_backend;

// Runtime flag: when nonzero, CUDA dispatches single-threaded deterministic
// scatter_add kernels and sets CUBLAS_PEDANTIC_MATH. CPU backend is always
// deterministic; this flag has no effect there. Single-GPU only -- not
// validated for multi-GPU/NCCL.
extern int g_blt_deterministic;

typedef struct blt_tensor {
    void *data;
    size_t shape[BLT_MAX_NDIM];
    size_t strides[BLT_MAX_NDIM];
    size_t ndim;
    size_t numel;
    blt_dtype dtype;
    blt_backend backend;
    bool is_view;
    // Optional per-tensor dequantization scale for FP8 data. The stored value
    // is (original / scale) and must be multiplied by scale to recover it.
    // NULL means 1.0 (and is the only valid value for non-FP8 tensors).
    const struct blt_tensor *scale;
    // Set when an FP8 shadow stores the transpose of its logical shape, so a
    // [K, N] weight is held as [N, K] for the forward GEMM. Ignored otherwise.
    bool transposed;
} blt_tensor;

size_t blt_tensor_compute_numel(const size_t *shape, size_t ndim);
void blt_tensor_compute_row_major_strides(const size_t *shape, size_t ndim, size_t *strides_out);
size_t blt_tensor_bytes(const blt_tensor *t);
void view_1d(blt_tensor *view, void *data, size_t len, blt_dtype dtype, blt_backend backend);
// Creates a 1D view of `len` elements starting `offset_elems` elements into
// src's data (dtype/backend inherited from src).
void view_1d_offset(blt_tensor *view, const blt_tensor *src, size_t offset_elems, size_t len);
void blt_tensor_view_2d(blt_tensor *t, void *data, size_t rows, size_t cols, blt_backend backend);
void blt_tensor_view_3d(blt_tensor *t, void *data, size_t d0, size_t d1, size_t d2, blt_backend backend);

void zero_tensor(blt_tensor *t);

// H2D for CUDA, memcpy for CPU.
void blt_tensor_copy_from_host(blt_tensor *t, const void *host_src, size_t bytes);

// D2H for CUDA, memcpy for CPU.
void blt_tensor_copy_to_host(const blt_tensor *t, void *host_dst, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif // BLT_CORE_TENSOR_H
