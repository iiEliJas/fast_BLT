#include "core/backend.h"
#include "core/allocator.h"
#include "core/cuda_shim.h"
#include "core/tensor.h"
#include "ops/matmul.h"
#include "ops/cast.h"
#include "ops/quant.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

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

static void *blt_lt_workspace(void) {
    static void *ws = NULL;
    const size_t bytes = 32u << 20;
    if (ws == NULL && cudaMalloc(&ws, bytes) != cudaSuccess) {
        BLT_FATAL("blt_matmul: failed to allocate cuBLASLt workspace");
    }
    return ws;
}

static blt_tensor blt_scratch_2d(size_t rows, size_t cols, blt_dtype dtype) {
    size_t shape[2] = {rows, cols};
    return blt_tensor_create_uninit(blt_cuda_get_scratch_arena(), shape, 2, dtype);
}

static blt_tensor blt_scratch_1d(blt_dtype dtype) {
    size_t shape[1] = {1};
    return blt_tensor_create_uninit(blt_cuda_get_scratch_arena(), shape, 1, dtype);
}

// Shared 1.0 scale for FP8 buffers that were never given an explicit scale.
static const float *blt_unit_scale(void) {
    static float *p = NULL;
    if (p == NULL) {
        float one = 1.0f;
        if (cudaMalloc(&p, sizeof(float)) != cudaSuccess ||
            cudaMemcpy(p, &one, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            BLT_FATAL("blt_matmul: failed to allocate unit scale");
        }
    }
    return p;
}

typedef struct {
    blt_tensor data;
    blt_tensor scale;
} blt_quant_t;

// Quantize an FP32 tensor with a fresh per-tensor amax scale. transpose stores
// the result as [cols, rows]. Both buffers come from the CUDA scratch arena.
static blt_quant_t blt_quantize_scratch(const blt_tensor *src, blt_dtype fmt, int transpose) {
    blt_quant_t r;
    const size_t rows = src->shape[0];
    const size_t cols = src->shape[1];
    r.data = blt_scratch_2d(transpose ? cols : rows, transpose ? rows : cols, fmt);
    r.scale = blt_scratch_1d(BLT_DTYPE_FP32);
    blt_tensor amax = blt_scratch_1d(BLT_DTYPE_FP32);
    blt_amax(src, &amax);
    blt_quant_scale(&amax, blt_fp8_max(fmt), 1e-30f, &r.scale);
    blt_fp8_quantize(src, &r.scale, &r.data, transpose);
    return r;
}

typedef struct {
    blt_tensor data;
    const float *scale;
} blt_fp8_operand;

// Produce an FP8 view of `t` in the requested orientation. FP8 tensors keep
// their own scale; FP32 tensors are quantized with a fresh one.
static blt_fp8_operand blt_as_fp8(const blt_tensor *t, blt_dtype fmt, int transpose) {
    blt_fp8_operand r;
    if (blt_dtype_is_fp8(t->dtype)) {
        if ((t->transposed ? 1 : 0) == transpose) {
            r.data = *t;
        } else {
            r.data = blt_scratch_2d(t->shape[1], t->shape[0], t->dtype);
            blt_fp8_transpose(t, &r.data);
        }
        r.scale = t->scale ? (const float *)t->scale->data : blt_unit_scale();
    } else {
        blt_quant_t q = blt_quantize_scratch(t, fmt, transpose);
        r.data = q.data;
        r.scale = (const float *)q.scale.data;
    }
    return r;
}

// Materialize a logical FP32 copy of an FP8 or 16-bit operand.
static const blt_tensor *blt_to_f32(const blt_tensor *t, blt_tensor *slot) {
    if (t->dtype == BLT_DTYPE_FP32) {
        return t;
    }
    if (blt_dtype_is_fp8(t->dtype)) {
        const size_t rows = t->transposed ? t->shape[1] : t->shape[0];
        const size_t cols = t->transposed ? t->shape[0] : t->shape[1];
        *slot = blt_scratch_2d(rows, cols, BLT_DTYPE_FP32);
        blt_fp8_dequantize(t, t->scale, slot, t->transposed ? 1 : 0);
    } else {
        *slot = blt_scratch_2d(t->shape[0], t->shape[1], BLT_DTYPE_FP32);
        blt_cast(t, slot);
    }
    return slot;
}

// cuBLASLt requires every FP8/FP32 operand pointer to be 16-byte aligned. Arena
// tensors are 64-byte aligned, but a matmul output can be a view into a packed
// buffer at an arbitrary byte offset, so the check is positional rather than
// structural.
static int blt_ptr_16b(const void *p) { return (((uintptr_t)p) & 15u) == 0; }

// C[m,n] = A_slot^T @ B_slot with A_slot [n,k] and B_slot [m,k], both row-major
// FP8. This is the only orientation cuBLASLt exposes for FP8 on pre-Blackwell
// parts, and it covers forward and both gradient GEMMs.
static void blt_fp8_gemm(const blt_tensor *a_slot, const float *a_scale, const blt_tensor *b_slot, const float *b_scale,
                         void *out, int m, int k, int n, int e5m2) {
    cublasLtHandle_t lt = blt_cublaslt_handle();
    cublasLtMatmulDesc_t desc = NULL;
    cublasLtMatrixLayout_t la = NULL, lb = NULL, lc = NULL;
    const cudaDataType_t dt = e5m2 ? CUDA_R_8F_E5M2 : CUDA_R_8F_E4M3;

    blt_cublas_check(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
    cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    blt_cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)),
                     "cublasLt desc TRANSA");
    blt_cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)),
                     "cublasLt desc TRANSB");
    blt_cublas_check(
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)),
        "cublasLt desc A scale");
    blt_cublas_check(
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)),
        "cublasLt desc B scale");

    blt_cublas_check(cublasLtMatrixLayoutCreate(&la, dt, k, n, k), "cublasLt layout A");
    blt_cublas_check(cublasLtMatrixLayoutCreate(&lb, dt, k, m, k), "cublasLt layout B");
    blt_cublas_check(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, n, m, n), "cublasLt layout C");

    // An unaligned destination is staged through scratch and copied back; the
    // C layout is plain row-major [m,n], so the copy is a straight byte copy.
    void *c_ptr = out;
    blt_tensor c_stage = {0};
    const size_t c_bytes = (size_t)m * (size_t)n * sizeof(float);
    if (!blt_ptr_16b(out)) {
        c_stage = blt_scratch_2d((size_t)m, (size_t)n, BLT_DTYPE_FP32);
        c_ptr = c_stage.data;
    }

    const float alpha = 1.0f, beta = 0.0f;
    const size_t ws_bytes = 32u << 20;
    blt_cublas_check(cublasLtMatmul(lt, desc, &alpha, a_slot->data, la, b_slot->data, lb, &beta, c_ptr, lc, c_ptr, lc,
                                    NULL, blt_lt_workspace(), ws_bytes, 0),
                     "cublasLtMatmul fp8");
    if (c_stage.data != NULL) {
        if (cudaMemcpy(out, c_stage.data, c_bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
            BLT_FATAL("blt_matmul: failed to copy staged FP8 GEMM output");
        }
    }

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
}

static void blt_gemm_lowp(const blt_tensor *a, const blt_tensor *b, blt_tensor *out, size_t m, size_t k, size_t n) {
    const cudaDataType_t dt = (a->dtype == BLT_DTYPE_BF16) ? CUDA_R_16BF : CUDA_R_16F;
    const float alpha = 1.0f, beta = 0.0f;
    blt_cublas_check(cublasGemmEx(blt_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha,
                                  b->data, dt, (int)n, a->data, dt, (int)k, &beta, out->data, CUDA_R_32F, (int)n,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                     "blt_matmul: cublasGemmEx lowp");
}

static void blt_sgemm(const blt_tensor *a, const blt_tensor *b, blt_tensor *out, size_t m, size_t k, size_t n) {
    const float alpha = 1.0f, beta = 0.0f;
    blt_cublas_check(cublasSgemm(blt_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m, (int)k, &alpha,
                                 (const float *)b->data, (int)n, (const float *)a->data, (int)k, &beta,
                                 (float *)out->data, (int)n),
                     "blt_matmul: cublasSgemm");
}

extern "C" void blt_matmul_cuda(const blt_tensor *a, const blt_tensor *b, blt_tensor *out) {
    const size_t m = a->shape[0];
    const size_t k = a->shape[1];
    const size_t n = b->transposed ? b->shape[0] : b->shape[1];
    if (m == 0 || k == 0 || n == 0) {
        return;
    }

    const int a_fp8 = blt_dtype_is_fp8(a->dtype);
    const int b_fp8 = blt_dtype_is_fp8(b->dtype);

    if (a_fp8 || b_fp8) {
        const blt_dtype fmt = b_fp8 ? b->dtype : a->dtype;
        // The activation operand is quantized into aligned scratch, so only the
        // weight pointer can be off; a misaligned destination is staged instead.
        if (b_fp8 && a->dtype == BLT_DTYPE_FP32 && blt_fp8_matmul_eligible(m, k, n) && blt_ptr_16b(b->data)) {
            blt_fp8_operand aq = blt_as_fp8(a, fmt, 0);
            blt_fp8_operand wq = blt_as_fp8(b, fmt, 1);
            blt_fp8_gemm(&wq.data, wq.scale, &aq.data, aq.scale, out->data, (int)m, (int)k, (int)n,
                         fmt == BLT_DTYPE_FP8_E5M2);
            return;
        }
        blt_tensor a_store = {0}, b_store = {0};
        const blt_tensor *a_use = blt_to_f32(a, &a_store);
        const blt_tensor *b_use = blt_to_f32(b, &b_store);
        blt_sgemm(a_use, b_use, out, m, k, n);
        return;
    }

    const int a_low = blt_dtype_is_fp16(a->dtype);
    const int b_low = blt_dtype_is_fp16(b->dtype);
    if (a_low || b_low) {
        blt_tensor a_store = {0}, b_store = {0};
        const blt_tensor *a_use = a, *b_use = b;
        const blt_dtype dt = a_low ? a->dtype : b->dtype;
        if (!a_low) {
            a_store = blt_scratch_2d(m, k, dt);
            blt_cast(a, &a_store);
            a_use = &a_store;
        }
        if (!b_low) {
            b_store = blt_scratch_2d(k, n, dt);
            blt_cast(b, &b_store);
            b_use = &b_store;
        } else if (b_use->dtype != a_use->dtype) {
            blt_tensor mixed = blt_scratch_2d(k, n, a_use->dtype);
            blt_cast(b, &mixed);
            b_store = mixed;
            b_use = &b_store;
        }
        blt_gemm_lowp(a_use, b_use, out, m, k, n);
        return;
    }

    blt_sgemm(a, b, out, m, k, n);
}

// a: [M,K], b: [K,N], grad_out: [M,N]
// grad_a: [M,K] = grad_out @ b^T
// grad_b: [K,N] = a^T @ grad_out
extern "C" void blt_matmul_backward_cuda(const blt_tensor *a, const blt_tensor *b, const blt_tensor *grad_out,
                                         blt_tensor *grad_a, blt_tensor *grad_b) {
    const size_t m = a->shape[0];
    const size_t k = a->shape[1];
    const size_t n = b->transposed ? b->shape[0] : b->shape[1];
    if (m == 0 || k == 0 || n == 0) {
        return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t h = blt_cublas_handle();

    const int a_fp8 = blt_dtype_is_fp8(a->dtype);
    const int b_fp8 = blt_dtype_is_fp8(b->dtype);

    if (a_fp8 || b_fp8) {
        const blt_dtype fmt = b_fp8 ? b->dtype : a->dtype;
        const int e5m2 = fmt == BLT_DTYPE_FP8_E5M2;

        if (grad_a != NULL) {
            const int fast = b_fp8 && blt_fp8_matmul_eligible(m, n, k) && blt_ptr_16b(b->data);
            blt_fp8_operand wq = blt_as_fp8(b, fmt, 0);
            if (fast) {
                blt_fp8_operand gq = blt_as_fp8(grad_out, fmt, 0);
                blt_fp8_gemm(&wq.data, wq.scale, &gq.data, gq.scale, grad_a->data, (int)m, (int)n, (int)k, e5m2);
            } else {
                blt_tensor w_store = {0}, g_store = {0};
                const blt_tensor *w_use = blt_to_f32(b, &w_store);
                const blt_tensor *g_use = blt_to_f32(grad_out, &g_store);
                blt_cublas_check(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)k, (int)m, (int)n, &alpha,
                                             (const float *)w_use->data, (int)n, (const float *)g_use->data, (int)n,
                                             &beta, (float *)grad_a->data, (int)k),
                                 "blt_matmul_backward: SGEMM fp8 (grad_a)");
            }
        }

        if (grad_b != NULL) {
            const int fast = b_fp8 && blt_fp8_matmul_eligible(k, m, n) && blt_ptr_16b(a->data);
            blt_fp8_operand gt = blt_as_fp8(grad_out, fmt, 1);
            blt_fp8_operand at = blt_as_fp8(a, fmt, 1);
            if (fast) {
                blt_fp8_gemm(&gt.data, gt.scale, &at.data, at.scale, grad_b->data, (int)k, (int)m, (int)n, e5m2);
            } else {
                blt_tensor a_store = {0}, g_store = {0};
                const blt_tensor *a_use = blt_to_f32(a, &a_store);
                const blt_tensor *g_use = blt_to_f32(grad_out, &g_store);
                blt_cublas_check(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, (int)n, (int)k, (int)m, &alpha,
                                             (const float *)g_use->data, (int)n, (const float *)a_use->data, (int)k,
                                             &beta, (float *)grad_b->data, (int)n),
                                 "blt_matmul_backward: SGEMM fp8 (grad_b)");
            }
        }
        return;
    }

    const int a_low = blt_dtype_is_fp16(a->dtype);
    const int b_low = blt_dtype_is_fp16(b->dtype);
    const blt_dtype low_dt = a_low ? a->dtype : (b_low ? b->dtype : BLT_DTYPE_FP32);

    // Stage low-precision copies of the operands and of grad_out so every
    // tensor-core GEMM sees values rounded exactly like the forward pass. A
    // gradient only uses tensor cores when the operand it multiplies is the
    // low-precision one; the other side keeps full FP32.
    blt_tensor go_lp = {0};
    const blt_tensor *go_use = grad_out;
    if (a_low || b_low) {
        go_lp = blt_scratch_2d(m, n, low_dt);
        blt_cast(grad_out, &go_lp);
        go_use = &go_lp;
    }

    if (grad_a != NULL) {
        if (b_low) {
            const cudaDataType_t dt = (b->dtype == BLT_DTYPE_BF16) ? CUDA_R_16BF : CUDA_R_16F;
            blt_cublas_check(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)k, (int)m, (int)n, &alpha, b->data, dt,
                                          (int)n, go_use->data, dt, (int)n, &beta, grad_a->data, CUDA_R_32F, (int)k,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                             "blt_matmul_backward: cublasGemmEx lowp (grad_a)");
        } else {
            blt_cublas_check(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)k, (int)m, (int)n, &alpha,
                                         (const float *)b->data, (int)n, (const float *)grad_out->data, (int)n, &beta,
                                         (float *)grad_a->data, (int)k),
                             "blt_matmul_backward: cublasSgemm (grad_a)");
        }
    }

    if (grad_b != NULL) {
        if (a_low) {
            const cudaDataType_t dt = (a->dtype == BLT_DTYPE_BF16) ? CUDA_R_16BF : CUDA_R_16F;
            blt_cublas_check(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, (int)n, (int)k, (int)m, &alpha, go_use->data, dt,
                                          (int)n, a->data, dt, (int)k, &beta, grad_b->data, CUDA_R_32F, (int)n,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                             "blt_matmul_backward: cublasGemmEx lowp (grad_b)");
        } else {
            blt_cublas_check(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, (int)n, (int)k, (int)m, &alpha,
                                         (const float *)grad_out->data, (int)n, (const float *)a->data, (int)k, &beta,
                                         (float *)grad_b->data, (int)n),
                             "blt_matmul_backward: cublasSgemm (grad_b)");
        }
    }
}
