#include "core/backend.h"
#include "core/allocator.h"
#include "ops/matmul.h"
#include "ops/cast.h"
#include "ops/quant.h"

#include <stdlib.h>
#include <string.h>

static void matmul_fp32(const float *a_data, const float *b_data, float *out_data, size_t m, size_t k, size_t n) {
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (size_t inner = 0; inner < k; ++inner) {
                sum += a_data[row * k + inner] * b_data[inner * n + col];
            }
            out_data[row * n + col] = sum;
        }
    }
}

// Scratch for the FP8 emulation path. FP8 tensors never reach the CPU in
// training (the trainer rejects `--dtype fp8` without CUDA), but parity tests
// and direct callers still need the same numeric behaviour as the GPU path.
static blt_arena *cpu_quant_arena(void) {
    static blt_arena *arena = NULL;
    if (arena == NULL) {
        arena = blt_arena_create(128ull * 1024 * 1024, BLT_BACKEND_CPU);
    }
    return arena;
}

static blt_tensor cpu_quant_2d(blt_arena *arena, size_t rows, size_t cols, blt_dtype dtype) {
    size_t shape[2] = {rows, cols};
    return blt_tensor_create_uninit(arena, shape, 2, dtype);
}

static blt_tensor cpu_quant_1d(blt_arena *arena, blt_dtype dtype) {
    size_t shape[1] = {1};
    return blt_tensor_create_uninit(arena, shape, 1, dtype);
}

// Dequantize an FP8 tensor into its logical [rows, cols] FP32 form.
static blt_tensor cpu_fp8_to_f32(const blt_tensor *t, blt_arena *arena) {
    size_t rows = t->transposed ? t->shape[1] : t->shape[0];
    size_t cols = t->transposed ? t->shape[0] : t->shape[1];
    blt_tensor out = cpu_quant_2d(arena, rows, cols, BLT_DTYPE_FP32);
    blt_fp8_dequantize(t, t->scale, &out, t->transposed ? 1 : 0);
    return out;
}

// Round an FP32 tensor through FP8 with a fresh amax-derived scale.
static blt_tensor cpu_fp8_roundtrip(const blt_tensor *x, blt_dtype fmt, blt_arena *arena) {
    blt_tensor amax = cpu_quant_1d(arena, BLT_DTYPE_FP32);
    blt_tensor scale = cpu_quant_1d(arena, BLT_DTYPE_FP32);
    blt_amax(x, &amax);
    blt_quant_scale(&amax, blt_fp8_max(fmt), 1e-30f, &scale);
    blt_tensor out = cpu_quant_2d(arena, x->shape[0], x->shape[1], BLT_DTYPE_FP32);
    blt_fp8_roundtrip(x, &scale, fmt, &out);
    return out;
}

static blt_tensor cpu_lowp_to_f32(const blt_tensor *t, blt_arena *arena) {
    blt_tensor out = cpu_quant_2d(arena, t->shape[0], t->shape[1], BLT_DTYPE_FP32);
    blt_cast(t, &out);
    return out;
}

// Round an FP32 tensor through `fmt` and back. The tensor-core backward feeds
// a narrowed grad_out into the GEMM, so the CPU ground truth has to narrow it
// the same way before accumulating.
static blt_tensor cpu_lowp_roundtrip(const blt_tensor *x, blt_dtype fmt, blt_arena *arena) {
    blt_tensor enc = cpu_quant_2d(arena, x->shape[0], x->shape[1], fmt);
    blt_tensor out = cpu_quant_2d(arena, x->shape[0], x->shape[1], BLT_DTYPE_FP32);
    blt_cast(x, &enc);
    blt_cast(&enc, &out);
    return out;
}

static void matmul_fp8_cpu(const blt_tensor *a, const blt_tensor *b, blt_tensor *out, size_t m, size_t k, size_t n) {
    const int a_fp8 = blt_dtype_is_fp8(a->dtype);
    const int b_fp8 = blt_dtype_is_fp8(b->dtype);

    blt_arena *ar = cpu_quant_arena();
    blt_arena_reset(ar);

    blt_tensor a_store = {0};
    blt_tensor b_store = {0};
    const blt_tensor *a_use = a;
    const blt_tensor *b_use = b;

    if (b_fp8 && a->dtype == BLT_DTYPE_FP32 && blt_fp8_matmul_eligible(m, k, n)) {
        // CUDA tensor-core path: quantize the activation, dequantize the weight.
        a_store = cpu_fp8_roundtrip(a, blt_fp8_pair_fmt(b->dtype), ar);
        b_store = cpu_fp8_to_f32(b, ar);
        a_use = &a_store;
        b_use = &b_store;
    } else {
        if (a_fp8) {
            a_store = cpu_fp8_to_f32(a, ar);
            a_use = &a_store;
        } else if (a->dtype != BLT_DTYPE_FP32) {
            a_store = cpu_lowp_to_f32(a, ar);
            a_use = &a_store;
        }
        if (b_fp8) {
            b_store = cpu_fp8_to_f32(b, ar);
            b_use = &b_store;
        } else if (b->dtype != BLT_DTYPE_FP32) {
            b_store = cpu_lowp_to_f32(b, ar);
            b_use = &b_store;
        }
    }

    matmul_fp32((const float *)a_use->data, (const float *)b_use->data, (float *)out->data, m, k, n);
}

void blt_matmul_cpu(const blt_tensor *a, const blt_tensor *b, blt_tensor *out) {
    BLT_REQUIRE(a != NULL && b != NULL && out != NULL, "matmul: inputs must not be NULL");
    BLT_REQUIRE(a->ndim == 2 && b->ndim == 2 && out->ndim == 2, "matmul: all tensors must be 2D");
    BLT_REQUIRE(a->dtype == BLT_DTYPE_FP32 || a->dtype == BLT_DTYPE_BF16 || a->dtype == BLT_DTYPE_FP16 ||
                    blt_dtype_is_fp8(a->dtype),
                "matmul: input a must be FP32, BF16, FP16 or FP8");
    BLT_REQUIRE(b->dtype == BLT_DTYPE_FP32 || b->dtype == BLT_DTYPE_BF16 || b->dtype == BLT_DTYPE_FP16 ||
                    blt_dtype_is_fp8(b->dtype),
                "matmul: input b must be FP32, BF16, FP16 or FP8");
    BLT_REQUIRE(out->dtype == BLT_DTYPE_FP32, "matmul: output must be FP32");

    const size_t b_rows = b->transposed ? b->shape[1] : b->shape[0];
    const size_t b_cols = b->transposed ? b->shape[0] : b->shape[1];
    BLT_REQUIRE(a->shape[1] == b_rows, "matmul inner dimensions do not match");
    BLT_REQUIRE(out->shape[0] == a->shape[0] && out->shape[1] == b_cols, "matmul output shape is incorrect");

    size_t m = a->shape[0];
    size_t k = a->shape[1];
    size_t n = b_cols;
    if (m == 0 || k == 0 || n == 0) return;

    if (blt_dtype_is_fp8(a->dtype) || blt_dtype_is_fp8(b->dtype)) {
        matmul_fp8_cpu(a, b, out, m, k, n);
        return;
    }

    if (a->dtype == BLT_DTYPE_FP32 && b->dtype == BLT_DTYPE_FP32) {
        matmul_fp32((const float *)a->data, (const float *)b->data, (float *)out->data, m, k, n);
        return;
    }

    // Not faster — exists so CPU output matches CUDA tensor-core rounding.
    float *a_fp32 = (float *)malloc(m * k * sizeof(float));
    float *b_fp32 = (float *)malloc(k * n * sizeof(float));
    BLT_REQUIRE(a_fp32 && b_fp32, "matmul: failed to allocate lowp->fp32 temporaries");

    blt_tensor a_tmp = {0}, b_tmp = {0};
    blt_tensor_view_2d(&a_tmp, a_fp32, m, k, BLT_BACKEND_CPU);
    a_tmp.dtype = BLT_DTYPE_FP32;
    blt_tensor_view_2d(&b_tmp, b_fp32, k, n, BLT_BACKEND_CPU);
    b_tmp.dtype = BLT_DTYPE_FP32;

    blt_cast(a, &a_tmp);
    blt_cast(b, &b_tmp);

    matmul_fp32(a_fp32, b_fp32, (float *)out->data, m, k, n);

    free(a_fp32);
    free(b_fp32);
}

// grad_a[m,k] = sum_n grad_out[m,n] * w[k,n]
static void grad_a_fp32(const float *go, const float *w, float *ga, size_t m, size_t k, size_t n) {
    for (size_t r = 0; r < m; r++) {
        const float *go_row = go + r * n;
        float *ga_row = ga + r * k;
        for (size_t c = 0; c < k; c++) {
            float acc = 0.0f;
            for (size_t i = 0; i < n; i++) {
                acc += go_row[i] * w[c * n + i];
            }
            ga_row[c] = acc;
        }
    }
}

// grad_b[k,n] = sum_m a[m,k] * grad_out[m,n]
static void grad_b_fp32(const float *a, const float *go, float *gb, size_t m, size_t k, size_t n) {
    memset(gb, 0, k * n * sizeof(float));
    for (size_t r = 0; r < m; r++) {
        const float *a_row = a + r * k;
        const float *go_row = go + r * n;
        for (size_t c = 0; c < k; c++) {
            float av = a_row[c];
            if (av == 0.0f) continue;
            float *gb_row = gb + c * n;
            for (size_t i = 0; i < n; i++) {
                gb_row[i] += av * go_row[i];
            }
        }
    }
}

static void matmul_backward_fp8_cpu(const blt_tensor *a, const blt_tensor *b, const blt_tensor *grad_out,
                                    blt_tensor *grad_a, blt_tensor *grad_b, size_t M, size_t K, size_t N) {
    const int a_fp8 = blt_dtype_is_fp8(a->dtype);
    const int b_fp8 = blt_dtype_is_fp8(b->dtype);
    const blt_dtype fmt = b_fp8 ? b->dtype : a->dtype;
    const blt_dtype dyn_fmt = blt_fp8_pair_fmt(fmt);

    blt_arena *ar = cpu_quant_arena();
    blt_arena_reset(ar);

    if (grad_a != NULL) {
        blt_tensor w_store = {0};
        blt_tensor g_store = {0};
        const blt_tensor *w_use = b;
        const blt_tensor *g_use = grad_out;

        if (b_fp8) {
            w_store = cpu_fp8_to_f32(b, ar);
            w_use = &w_store;
        } else if (b->dtype != BLT_DTYPE_FP32) {
            w_store = cpu_lowp_to_f32(b, ar);
            w_use = &w_store;
        }
        if (b_fp8 && blt_fp8_matmul_eligible(M, N, K)) {
            g_store = cpu_fp8_roundtrip(grad_out, dyn_fmt, ar);
            g_use = &g_store;
        }
        grad_a_fp32((const float *)g_use->data, (const float *)w_use->data, (float *)grad_a->data, M, K, N);
    }

    if (grad_b != NULL) {
        blt_tensor a_store = {0};
        blt_tensor g_store = {0};
        const blt_tensor *a_use = a;
        const blt_tensor *g_use = grad_out;
        const int fast = b_fp8 && blt_fp8_matmul_eligible(K, M, N);

        if (a_fp8) {
            a_store = cpu_fp8_to_f32(a, ar);
            a_use = &a_store;
        } else if (a->dtype != BLT_DTYPE_FP32) {
            a_store = cpu_lowp_to_f32(a, ar);
            a_use = &a_store;
        } else if (fast) {
            a_store = cpu_fp8_roundtrip(a, dyn_fmt, ar);
            a_use = &a_store;
        }
        if (fast) {
            g_store = cpu_fp8_roundtrip(grad_out, dyn_fmt, ar);
            g_use = &g_store;
        }
        grad_b_fp32((const float *)a_use->data, (const float *)g_use->data, (float *)grad_b->data, M, K, N);
    }
}

// a: [M,K], b: [K,N], grad_out: [M,N]
// grad_a: [M,K] = grad_out @ b^T
// grad_b: [K,N] = a^T @ grad_out
void blt_matmul_backward_cpu(const blt_tensor *a, const blt_tensor *b, const blt_tensor *grad_out, blt_tensor *grad_a,
                             blt_tensor *grad_b) {
    BLT_REQUIRE(a != NULL && b != NULL && grad_out != NULL, "blt_matmul_backward: a, b, grad_out must not be NULL");
    BLT_REQUIRE(a->ndim == 2 && b->ndim == 2 && grad_out->ndim == 2, "blt_matmul_backward: a, b, grad_out must be 2D");
    BLT_REQUIRE(grad_out->dtype == BLT_DTYPE_FP32, "blt_matmul_backward: grad_out must be FP32");

    size_t M = a->shape[0];
    size_t K = a->shape[1];
    const size_t b_rows = b->transposed ? b->shape[1] : b->shape[0];
    const size_t b_cols = b->transposed ? b->shape[0] : b->shape[1];
    BLT_REQUIRE(b_rows == K, "blt_matmul_backward: a.shape[1] must equal b.shape[0]");
    size_t N = b_cols;
    BLT_REQUIRE(grad_out->shape[0] == M && grad_out->shape[1] == N,
                "blt_matmul_backward: grad_out shape must be [M, N]");
    if (M == 0 || K == 0 || N == 0) return;

    if (blt_dtype_is_fp8(a->dtype) || blt_dtype_is_fp8(b->dtype)) {
        matmul_backward_fp8_cpu(a, b, grad_out, grad_a, grad_b, M, K, N);
        return;
    }

    const int lowp = (a->dtype != BLT_DTYPE_FP32) || (b->dtype != BLT_DTYPE_FP32);
    if (lowp) {
        blt_arena *ar = cpu_quant_arena();
        blt_arena_reset(ar);
        blt_tensor a_store = {0};
        blt_tensor b_store = {0};
        const blt_dtype fmt = a->dtype != BLT_DTYPE_FP32 ? a->dtype : b->dtype;
        const blt_tensor *a_use = a;
        const blt_tensor *b_use = b;
        if (a->dtype != BLT_DTYPE_FP32) {
            a_store = cpu_lowp_to_f32(a, ar);
            a_use = &a_store;
        }
        if (b->dtype != BLT_DTYPE_FP32) {
            b_store = cpu_lowp_to_f32(b, ar);
            b_use = &b_store;
        }
        blt_tensor g_store = cpu_lowp_roundtrip(grad_out, fmt, ar);
        const float *g = (const float *)g_store.data;

        if (grad_a != NULL) {
            grad_a_fp32(g, (const float *)b_use->data, (float *)grad_a->data, M, K, N);
        }
        if (grad_b != NULL) {
            grad_b_fp32((const float *)a_use->data, g, (float *)grad_b->data, M, K, N);
        }
        return;
    }

    if (grad_a != NULL) {
        BLT_REQUIRE(grad_a->dtype == BLT_DTYPE_FP32 && grad_a->ndim == 2 && grad_a->shape[0] == M &&
                        grad_a->shape[1] == K,
                    "blt_matmul_backward: grad_a must be [M, K] FP32");
        grad_a_fp32((const float *)grad_out->data, (const float *)b->data, (float *)grad_a->data, M, K, N);
    }

    if (grad_b != NULL) {
        BLT_REQUIRE(grad_b->dtype == BLT_DTYPE_FP32 && grad_b->ndim == 2 && grad_b->shape[0] == K &&
                        grad_b->shape[1] == N,
                    "blt_matmul_backward: grad_b must be [K, N] FP32");
        grad_b_fp32((const float *)a->data, (const float *)grad_out->data, (float *)grad_b->data, M, K, N);
    }
}
