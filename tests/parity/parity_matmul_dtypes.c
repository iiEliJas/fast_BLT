// CPU-vs-GPU parity for the FP16 and scaled-FP8 matmul paths.

#include "core/allocator.h"
#include "core/backend.h"
#include "core/tensor.h"
#include "ops/cast.h"
#include "ops/matmul.h"
#include "ops/quant.h"

#include "test_helpers.h"
#include "test_suite.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef BLT_WITH_CUDA

int run_cuda_parity_matmul_dtypes(void) { return 1; }

#else

static uint32_t g_rng = 0xC0FFEEu;

static float next_uniform(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float)((g_rng >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}

static blt_tensor rand_2d(blt_arena *arena, size_t rows, size_t cols, blt_dtype dtype) {
    size_t shape[2] = {rows, cols};
    blt_tensor t = blt_tensor_create(arena, shape, 2, dtype);
    blt_tensor tmp = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    float *d = (float *)tmp.data;
    for (size_t i = 0; i < tmp.numel; ++i) d[i] = next_uniform();
    if (dtype == BLT_DTYPE_FP32) {
        memcpy(t.data, tmp.data, tmp.numel * sizeof(float));
    } else {
        blt_cast(&tmp, &t);
    }
    return t;
}

static void upload(blt_tensor *dst, const blt_tensor *src) { blt_tensor_upload(dst, src->data, blt_tensor_bytes(src)); }

int run_cuda_parity_matmul_dtypes(void) {
    blt_arena *host = blt_arena_create(8 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *dev = blt_arena_create(64 * 1024 * 1024, BLT_BACKEND_CUDA);
    TEST_ASSERT(host != NULL && dev != NULL);

    const size_t M = 32, K = 16, N = 32;

    // ---- FP16 forward and backward ----
    {
        blt_tensor a = rand_2d(host, M, K, BLT_DTYPE_FP16);
        blt_tensor b = rand_2d(host, K, N, BLT_DTYPE_FP16);
        blt_tensor go = rand_2d(host, M, N, BLT_DTYPE_FP32);

        blt_tensor h_out = blt_tensor_create(host, (size_t[]){M, N}, 2, BLT_DTYPE_FP32);
        blt_tensor h_ga = blt_tensor_create(host, (size_t[]){M, K}, 2, BLT_DTYPE_FP32);
        blt_tensor h_gb = blt_tensor_create(host, (size_t[]){K, N}, 2, BLT_DTYPE_FP32);
        blt_matmul(&a, &b, &h_out);
        blt_matmul_backward(&a, &b, &go, &h_ga, &h_gb);

        blt_tensor d_a = blt_tensor_create(dev, (size_t[]){M, K}, 2, BLT_DTYPE_FP16);
        blt_tensor d_b = blt_tensor_create(dev, (size_t[]){K, N}, 2, BLT_DTYPE_FP16);
        blt_tensor d_go = blt_tensor_create(dev, (size_t[]){M, N}, 2, BLT_DTYPE_FP32);
        upload(&d_a, &a);
        upload(&d_b, &b);
        upload(&d_go, &go);

        blt_tensor d_out = blt_tensor_create(dev, (size_t[]){M, N}, 2, BLT_DTYPE_FP32);
        blt_tensor d_ga = blt_tensor_create(dev, (size_t[]){M, K}, 2, BLT_DTYPE_FP32);
        blt_tensor d_gb = blt_tensor_create(dev, (size_t[]){K, N}, 2, BLT_DTYPE_FP32);
        blt_matmul(&d_a, &d_b, &d_out);
        blt_matmul_backward(&d_a, &d_b, &d_go, &d_ga, &d_gb);

        blt_tensor g_out = blt_tensor_to_host(&d_out, host);
        blt_tensor g_ga = blt_tensor_to_host(&d_ga, host);
        blt_tensor g_gb = blt_tensor_to_host(&d_gb, host);

        TEST_ASSERT_CLOSE(&g_out, &h_out, 1e-3f);
        TEST_ASSERT_CLOSE(&g_ga, &h_ga, 1e-3f);
        TEST_ASSERT_CLOSE(&g_gb, &h_gb, 1e-3f);
    }

    // ---- FP8 forward and backward through a transposed weight shadow ----
    {
        blt_tensor a = rand_2d(host, M, K, BLT_DTYPE_FP32);
        blt_tensor w = rand_2d(host, K, N, BLT_DTYPE_FP32);
        blt_tensor go = rand_2d(host, M, N, BLT_DTYPE_FP32);

        blt_tensor shadow = blt_tensor_create(host, (size_t[]){N, K}, 2, BLT_DTYPE_FP8_E4M3);
        blt_tensor amax = blt_tensor_create(host, (size_t[]){1}, 1, BLT_DTYPE_FP32);
        blt_tensor scale = blt_tensor_create(host, (size_t[]){1}, 1, BLT_DTYPE_FP32);
        blt_amax(&w, &amax);
        blt_quant_scale(&amax, blt_fp8_max(BLT_DTYPE_FP8_E4M3), 1e-30f, &scale);
        blt_fp8_quantize(&w, &scale, &shadow, 1);
        shadow.scale = &scale;
        shadow.transposed = true;

        blt_tensor h_out = blt_tensor_create(host, (size_t[]){M, N}, 2, BLT_DTYPE_FP32);
        blt_tensor h_ga = blt_tensor_create(host, (size_t[]){M, K}, 2, BLT_DTYPE_FP32);
        blt_tensor h_gb = blt_tensor_create(host, (size_t[]){K, N}, 2, BLT_DTYPE_FP32);
        blt_matmul(&a, &shadow, &h_out);
        blt_matmul_backward(&a, &shadow, &go, &h_ga, &h_gb);

        blt_tensor d_a = blt_tensor_create(dev, (size_t[]){M, K}, 2, BLT_DTYPE_FP32);
        blt_tensor d_go = blt_tensor_create(dev, (size_t[]){M, N}, 2, BLT_DTYPE_FP32);
        blt_tensor d_shadow = blt_tensor_create(dev, (size_t[]){N, K}, 2, BLT_DTYPE_FP8_E4M3);
        blt_tensor d_scale = blt_tensor_create(dev, (size_t[]){1}, 1, BLT_DTYPE_FP32);
        upload(&d_a, &a);
        upload(&d_go, &go);
        upload(&d_shadow, &shadow);
        upload(&d_scale, &scale);
        d_shadow.scale = &d_scale;
        d_shadow.transposed = true;

        blt_tensor d_out = blt_tensor_create(dev, (size_t[]){M, N}, 2, BLT_DTYPE_FP32);
        blt_tensor d_ga = blt_tensor_create(dev, (size_t[]){M, K}, 2, BLT_DTYPE_FP32);
        blt_tensor d_gb = blt_tensor_create(dev, (size_t[]){K, N}, 2, BLT_DTYPE_FP32);
        blt_matmul(&d_a, &d_shadow, &d_out);
        blt_matmul_backward(&d_a, &d_shadow, &d_go, &d_ga, &d_gb);

        blt_tensor g_out = blt_tensor_to_host(&d_out, host);
        blt_tensor g_ga = blt_tensor_to_host(&d_ga, host);
        blt_tensor g_gb = blt_tensor_to_host(&d_gb, host);

        TEST_ASSERT_CLOSE(&g_out, &h_out, 2e-2f);
        TEST_ASSERT_CLOSE(&g_ga, &h_ga, 2e-2f);
        TEST_ASSERT_CLOSE(&g_gb, &h_gb, 2e-2f);
    }

    blt_arena_destroy(dev);
    blt_arena_destroy(host);
    return 1;
}

#endif
