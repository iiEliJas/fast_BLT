#include "core/backend.h"
#include "core/allocator.h"
#include "core/cuda_shim.h"
#include "ops/matmul.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

extern "C" void blt_cast_cuda(const blt_tensor *in, blt_tensor *out);

// fp32 GEMMs on cuBLAS. cuBLAS is column-major; row-major C = A @ B maps to
// the standard swapped-operand formulation C' = B' @ A' (primes denote the
// col-major views of the same row-major buffers), so no transposes or
// packing copies are needed anywhere.

static cublasHandle_t blt_cublas_handle(void) {
    // Lazily-created per-process handle; never destroyed (process teardown reclaims it).
    static cublasHandle_t handle = NULL;
    if (!handle) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            BLT_FATAL("blt_matmul: cublasCreate failed");
        }
        if (g_blt_deterministic) {
            cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
        }
    }
    return handle;
}

static cublasLtHandle_t blt_cublaslt_handle(void) {
    static cublasLtHandle_t handle = NULL;
    if (!handle) {
        if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            BLT_FATAL("blt_matmul: cublasLtCreate failed");
        }
    }
    return handle;
}

static void blt_cublas_check(cublasStatus_t st, const char *what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        BLT_FATAL("%s failed with cuBLAS status %d", what, (int)st);
    }
}

extern "C" void blt_matmul_cuda(const blt_tensor *a, const blt_tensor *b, blt_tensor *out) {
    const size_t m = a->shape[0];
    const size_t k = a->shape[1];
    const size_t n = b->shape[1];
    if (m == 0 || k == 0 || n == 0) {
        return;
    }

    // Mixed precision: if one operand is fp8 and the other fp32, cast the fp32
    // operand to fp8 in scratch and recurse with both fp8.
    if ((a->dtype == BLT_DTYPE_FP32 && b->dtype == BLT_DTYPE_FP8_E4M3) ||
        (a->dtype == BLT_DTYPE_FP8_E4M3 && b->dtype == BLT_DTYPE_FP32)) {
        const blt_tensor *fp8_op = (a->dtype == BLT_DTYPE_FP8_E4M3) ? a : b;
        const blt_tensor *fp32_op = (a->dtype == BLT_DTYPE_FP32) ? a : b;
        // cast fp32_op to fp8 scratch
        void *scratch = blt_arena_alloc(blt_cuda_get_scratch_arena(), fp32_op->numel, 64);
        int nn = (int)fp32_op->numel;
        int th = 256, bl = (nn + th - 1) / th;
        // inline cast kernel
        // (declared below via a lambda? use a helper kernel)
        // We'll call a small kernel defined above.
        // For simplicity, cast on host via cudaMemcpy? No, stay on device.
        // Use blt_cast_cuda with temp tensors.
        // Build temp tensor for fp32 (view) and fp8 (scratch).
        blt_tensor t32 = *fp32_op;
        t32.dtype = BLT_DTYPE_FP32;
        t32.data = (void *)fp32_op->data;
        blt_tensor t8 = *fp32_op;
        t8.dtype = BLT_DTYPE_FP8_E4M3;
        t8.data = scratch;
        t8.backend = BLT_BACKEND_CUDA;
        extern void blt_cast_cuda(const blt_tensor *, blt_tensor *);
        blt_cast_cuda(&t32, &t8);
        blt_tensor a8 = *a;
        blt_tensor b8 = *b;
        if (a->dtype == BLT_DTYPE_FP32) {
            a8 = t8;
            a8.shape[0] = a->shape[0];
            a8.shape[1] = a->shape[1];
        }
        if (b->dtype == BLT_DTYPE_FP32) {
            b8 = t8;
            b8.shape[0] = b->shape[0];
            b8.shape[1] = b->shape[1];
        }
        blt_matmul_cuda(&a8, &b8, out);
        return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t h = blt_cublas_handle();

    // Mixed precision: if one operand is bf16 and the other fp32, cast the fp32
    // operand to bf16 in scratch and recurse with both bf16.
    if ((a->dtype == BLT_DTYPE_FP32 && b->dtype == BLT_DTYPE_BF16) ||
        (a->dtype == BLT_DTYPE_BF16 && b->dtype == BLT_DTYPE_FP32)) {
        const blt_tensor *fp32_op = (a->dtype == BLT_DTYPE_FP32) ? a : b;
        void *scratch =
            blt_arena_alloc(blt_cuda_get_scratch_arena(), fp32_op->numel * blt_dtype_sizeof(BLT_DTYPE_BF16), 64);
        extern void blt_cast_cuda(const blt_tensor *, blt_tensor *);

        blt_tensor bf16_view = *fp32_op;
        bf16_view.dtype = BLT_DTYPE_BF16;
        bf16_view.data = scratch;
        bf16_view.backend = BLT_BACKEND_CUDA;
        blt_cast_cuda(fp32_op, &bf16_view);

        // Both operands are bf16 after the cast, so the recursive call takes the
        // tensor-core branch below and never re-enters this one.
        blt_tensor a_bf16 = *a;
        blt_tensor b_bf16 = *b;
        if (a->dtype == BLT_DTYPE_FP32) {
            a_bf16 = bf16_view;
        } else {
            b_bf16 = bf16_view;
        }
        blt_matmul_cuda(&a_bf16, &b_bf16, out);
        return;
    }

    // bf16 inputs: use tensor cores via cublasGemmEx, accumulate in fp32
    if (a->dtype == BLT_DTYPE_BF16 && b->dtype == BLT_DTYPE_BF16) {
        blt_cublas_check(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha, b->data, CUDA_R_16BF,
                                      (int)n, a->data, CUDA_R_16BF, (int)k, &beta, out->data, CUDA_R_32F, (int)n,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                         "blt_matmul: cublasGemmEx bf16");
        return;
    }

    // fp8 (E4M3) inputs: use cuBLASLt tensor cores, accumulate in fp32.
    if (a->dtype == BLT_DTYPE_FP8_E4M3 && b->dtype == BLT_DTYPE_FP8_E4M3) {
        cublasLtHandle_t lth = blt_cublaslt_handle();
        cublasLtMatmulDesc_t desc = NULL;
        cublasLtMatrixLayout_t layoutA = NULL, layoutB = NULL, layoutC = NULL;
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, (int)k, (int)m, (int)k);
        cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, (int)n, (int)k, (int)n);
        cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, (int)n, (int)m, (int)n);
        size_t ws_size = 4 * 1024 * 1024;
        void *ws = NULL;
        cudaMalloc(&ws, ws_size);
        const float alpha = 1.0f, beta = 0.0f;
        // Row-major C=A@B  <=>  col-major C'=B'@A'. Pass b->data as A-slot, a->data as B-slot.
        cublasLtMatmul(lth, desc, &alpha, b->data, layoutB, a->data, layoutA, &beta, out->data, layoutC, out->data,
                       layoutC, NULL, ws, ws_size, 0);
        cudaFree(ws);
        cublasLtMatmulDescDestroy(desc);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        return;
    }

    blt_cublas_check(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha, (const float *)b->data,
                                 (int)n, (const float *)a->data, (int)k, &beta, (float *)out->data, (int)n),
                     "blt_matmul: cublasSgemm");
}

// a: [M,K], b: [K,N], grad_out: [M,N]
// grad_a: [M,K] = grad_out @ b^T   (overwritten when non-NULL)
// grad_b: [K,N] = a^T @ grad_out   (overwritten when non-NULL)
// a or b may be BF16 shadows: the narrowed operand runs on tensor cores and
// both gradients still accumulate in FP32.
extern "C" void blt_matmul_backward_cuda(const blt_tensor *a, const blt_tensor *b, const blt_tensor *grad_out,
                                         blt_tensor *grad_a, blt_tensor *grad_b) {
    const size_t m = a->shape[0];
    const size_t k = a->shape[1];
    const size_t n = b->shape[1];
    if (m == 0 || k == 0 || n == 0) {
        return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t h = blt_cublas_handle();

    const bool a_bf16 = a->dtype == BLT_DTYPE_BF16;
    const bool b_bf16 = b->dtype == BLT_DTYPE_BF16;

    // grad_a and grad_b both consume grad_out; when either operand is BF16,
    // stage one BF16 copy of grad_out and issue tensor-core GEMMs with FP32
    // accumulation. grad_b accumulates in FP32 regardless.
    blt_tensor go_bf16 = *grad_out;
    if ((a_bf16 || b_bf16) && (grad_a != NULL || grad_b != NULL)) {
        void *scratch = blt_arena_alloc(blt_cuda_get_scratch_arena(), (size_t)m * n * sizeof(uint16_t), 64);
        go_bf16.dtype = BLT_DTYPE_BF16;
        go_bf16.data = scratch;
        go_bf16.backend = BLT_BACKEND_CUDA;
        blt_cast_cuda(grad_out, &go_bf16);
    }

    if (grad_a != NULL) {
        // Rc[K,M] = Bc^T @ Gc : op(T) on b, op(N) on grad_out
        if (b_bf16) {
            blt_cublas_check(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)k, (int)m, (int)n, &alpha, b->data,
                                          CUDA_R_16BF, (int)n, go_bf16.data, CUDA_R_16BF, (int)n, &beta, grad_a->data,
                                          CUDA_R_32F, (int)k, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                             "blt_matmul_backward: cublasGemmEx bf16 (grad_a)");
        } else {
            blt_cublas_check(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)k, (int)m, (int)n, &alpha,
                                         (const float *)b->data, (int)n, (const float *)grad_out->data, (int)n, &beta,
                                         (float *)grad_a->data, (int)k),
                             "blt_matmul_backward: cublasSgemm (grad_a)");
        }
    }

    if (grad_b != NULL) {
        // Sc[N,K] = Gc @ Ac^T : op(N) on grad_out, op(T) on a
        if (a_bf16) {
            blt_cublas_check(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, (int)n, (int)k, (int)m, &alpha, go_bf16.data,
                                          CUDA_R_16BF, (int)n, a->data, CUDA_R_16BF, (int)k, &beta, grad_b->data,
                                          CUDA_R_32F, (int)n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                             "blt_matmul_backward: cublasGemmEx bf16 (grad_b)");
        } else {
            blt_cublas_check(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, (int)n, (int)k, (int)m, &alpha,
                                         (const float *)grad_out->data, (int)n, (const float *)a->data, (int)k, &beta,
                                         (float *)grad_b->data, (int)n),
                             "blt_matmul_backward: cublasSgemm (grad_b)");
        }
    }
}
