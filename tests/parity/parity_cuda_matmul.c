// CPU-vs-GPU parity for cuBLAS-backed matmul forward/backward.

#include "core/allocator.h"
#include "core/backend.h"
#include "ops/cast.h"
#include "ops/matmul.h"

#include "test_helpers.h"
#include "test_suite.h"

#include <stdint.h>

#ifndef BLT_WITH_CUDA

int run_cuda_parity_matmul(void) { return 1; }
int run_cuda_parity_bf16_matmul(void) { return 1; }

#else

static uint32_t prng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void fill_random(blt_tensor *t, uint32_t *state) {
    float *d = (float *)t->data;
    for (size_t i = 0; i < t->numel; i++) {
        d[i] = ((float)(prng_next(state) & 0xFFFF) / 32768.0f - 1.0f);
    }
}

static blt_tensor make_2d(blt_arena *arena, size_t rows, size_t cols, uint32_t *state) {
    size_t shape[2] = {rows, cols};
    blt_tensor t = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    if (state) {
        fill_random(&t, state);
    }
    return t;
}

int run_cuda_parity_matmul(void) {
    blt_arena *host = blt_arena_create(4 << 20, BLT_BACKEND_CPU);
    blt_arena *dev = blt_arena_create(4 << 20, BLT_BACKEND_CUDA);
    TEST_ASSERT(host != NULL && dev != NULL);

    uint32_t rng = 0xCAFEBABEu;

    const size_t cases[4][3] = {
        {1, 1, 1},      // degenerate single element
        {17, 33, 5},    // skinny, odd dims
        {64, 128, 256}, // typical transformer shapes
        {129, 65, 257}, // primes everywhere
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const size_t M = cases[c][0];
        const size_t K = cases[c][1];
        const size_t N = cases[c][2];

        blt_tensor a = make_2d(host, M, K, &rng);
        blt_tensor b = make_2d(host, K, N, &rng);
        blt_tensor go = make_2d(host, M, N, &rng);

        // ---- forward ----
        size_t out_shape[2] = {M, N};
        blt_tensor exp_out = blt_tensor_create(host, out_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&a, &b, &exp_out);

        blt_tensor da = blt_tensor_to_device(&a, dev);
        blt_tensor db = blt_tensor_to_device(&b, dev);
        blt_tensor d_out = blt_tensor_create(dev, out_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&da, &db, &d_out);
        blt_tensor got_out = blt_tensor_to_host(&d_out, host);
        TEST_ASSERT_CLOSE(&got_out, &exp_out, 1e-4f);

        // ---- backward with both grads ----
        blt_tensor exp_ga = make_2d(host, M, K, NULL);
        blt_tensor exp_gb = make_2d(host, K, N, NULL);
        blt_matmul_backward(&a, &b, &go, &exp_ga, &exp_gb);

        blt_tensor dgo = blt_tensor_to_device(&go, dev);
        blt_tensor d_ga = blt_tensor_create(dev, a.shape, 2, BLT_DTYPE_FP32);
        blt_tensor d_gb = blt_tensor_create(dev, b.shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&da, &db, &dgo, &d_ga, &d_gb);
        blt_tensor got_ga = blt_tensor_to_host(&d_ga, host);
        blt_tensor got_gb = blt_tensor_to_host(&d_gb, host);
        TEST_ASSERT_CLOSE(&got_ga, &exp_ga, 1e-4f);
        TEST_ASSERT_CLOSE(&got_gb, &exp_gb, 1e-4f);

        // ---- backward with only grad_b (NULL grad_a path) ----
        blt_tensor exp_gb2 = make_2d(host, K, N, NULL);
        blt_matmul_backward(&a, &b, &go, NULL, &exp_gb2);
        d_gb = blt_tensor_create(dev, b.shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&da, &db, &dgo, NULL, &d_gb);
        got_gb = blt_tensor_to_host(&d_gb, host);
        TEST_ASSERT_CLOSE(&got_gb, &exp_gb2, 1e-4f);

        // ---- backward with only grad_a (NULL grad_b path) ----
        blt_tensor exp_ga2 = make_2d(host, M, K, NULL);
        blt_matmul_backward(&a, &b, &go, &exp_ga2, NULL);
        d_ga = blt_tensor_create(dev, a.shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&da, &db, &dgo, &d_ga, NULL);
        got_ga = blt_tensor_to_host(&d_ga, host);
        TEST_ASSERT_CLOSE(&got_ga, &exp_ga2, 1e-4f);
    }

    blt_arena_destroy(dev);
    blt_arena_destroy(host);
    return 1;
}

int run_cuda_parity_bf16_matmul(void) {
    blt_arena *host = blt_arena_create(4 << 20, BLT_BACKEND_CPU);
    blt_arena *dev = blt_arena_create(4 << 20, BLT_BACKEND_CUDA);
    TEST_ASSERT(host != NULL && dev != NULL);

    uint32_t rng = 0x51F15EEDu;
    const size_t cases[3][3] = {
        {1, 7, 3},
        {17, 33, 5},
        {64, 128, 256},
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const size_t M = cases[c][0];
        const size_t K = cases[c][1];
        const size_t N = cases[c][2];

        blt_tensor a_fp32 = make_2d(host, M, K, &rng);
        blt_tensor b_fp32 = make_2d(host, K, N, &rng);

        size_t a_shape[2] = {M, K};
        size_t b_shape[2] = {K, N};
        blt_tensor a_bf16 = blt_tensor_create(host, a_shape, 2, BLT_DTYPE_BF16);
        blt_tensor b_bf16 = blt_tensor_create(host, b_shape, 2, BLT_DTYPE_BF16);
        blt_cast(&a_fp32, &a_bf16);
        blt_cast(&b_fp32, &b_bf16);

        size_t out_shape[2] = {M, N};
        blt_tensor expected = blt_tensor_create(host, out_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&a_bf16, &b_bf16, &expected);

        // fp32 activation x bf16 weight
        blt_tensor da = blt_tensor_to_device(&a_fp32, dev);
        blt_tensor db = blt_tensor_to_device(&b_bf16, dev);
        blt_tensor d_out = blt_tensor_create(dev, out_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&da, &db, &d_out);
        blt_tensor got = blt_tensor_to_host(&d_out, host);
        TEST_ASSERT_CLOSE(&got, &expected, 2e-3f);

        // bf16 activation x fp32 weight (symmetric path)
        da = blt_tensor_to_device(&a_bf16, dev);
        db = blt_tensor_to_device(&b_fp32, dev);
        d_out = blt_tensor_create(dev, out_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&da, &db, &d_out);
        got = blt_tensor_to_host(&d_out, host);
        TEST_ASSERT_CLOSE(&got, &expected, 2e-3f);

        // ---- backward: bf16 weight, fp32 activation ----
        // Round grad_out once so the CUDA path's BF16 cast is exact and the CPU
        // reference sees the same values.
        blt_tensor go = make_2d(host, M, N, &rng);
        blt_tensor go_bf16 = blt_tensor_create(host, out_shape, 2, BLT_DTYPE_BF16);
        blt_tensor go_rounded = blt_tensor_create(host, out_shape, 2, BLT_DTYPE_FP32);
        blt_cast(&go, &go_bf16);
        blt_cast(&go_bf16, &go_rounded);

        blt_tensor exp_ga = blt_tensor_create(host, a_shape, 2, BLT_DTYPE_FP32);
        blt_tensor exp_gb = blt_tensor_create(host, b_shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&a_fp32, &b_bf16, &go_rounded, &exp_ga, &exp_gb);

        blt_tensor d_go = blt_tensor_to_device(&go_rounded, dev);
        da = blt_tensor_to_device(&a_fp32, dev);
        db = blt_tensor_to_device(&b_bf16, dev);
        blt_tensor d_ga = blt_tensor_create(dev, a_shape, 2, BLT_DTYPE_FP32);
        blt_tensor d_gb = blt_tensor_create(dev, b_shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&da, &db, &d_go, &d_ga, &d_gb);
        blt_tensor got_ga = blt_tensor_to_host(&d_ga, host);
        blt_tensor got_gb = blt_tensor_to_host(&d_gb, host);
        TEST_ASSERT_CLOSE(&got_ga, &exp_ga, 5e-3f);
        TEST_ASSERT_CLOSE(&got_gb, &exp_gb, 1e-4f);

        // ---- backward: bf16 activation, fp32 weight ----
        blt_matmul_backward(&a_bf16, &b_fp32, &go_rounded, &exp_ga, &exp_gb);
        da = blt_tensor_to_device(&a_bf16, dev);
        db = blt_tensor_to_device(&b_fp32, dev);
        d_ga = blt_tensor_create(dev, a_shape, 2, BLT_DTYPE_FP32);
        d_gb = blt_tensor_create(dev, b_shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&da, &db, &d_go, &d_ga, &d_gb);
        got_ga = blt_tensor_to_host(&d_ga, host);
        got_gb = blt_tensor_to_host(&d_gb, host);
        TEST_ASSERT_CLOSE(&got_ga, &exp_ga, 1e-4f);
        TEST_ASSERT_CLOSE(&got_gb, &exp_gb, 5e-3f);
    }

    blt_arena_destroy(dev);
    blt_arena_destroy(host);
    return 1;
}

#endif // BLT_WITH_CUDA
