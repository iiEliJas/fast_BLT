#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/allocator.h"
#include "core/backend.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "ops/cast.h"
#include "ops/matmul.h"
#include "ops/quant.h"

#include "test_helpers.h"
#include "test_suite.h"

static uint32_t g_rng = 0x12345678u;

static float next_uniform(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float)((g_rng >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}

static blt_tensor rand_2d(blt_arena *arena, size_t rows, size_t cols, blt_dtype dtype) {
    size_t shape[2] = {rows, cols};
    blt_tensor t = blt_tensor_create(arena, shape, 2, dtype);
    if (dtype == BLT_DTYPE_FP32) {
        float *d = (float *)t.data;
        for (size_t i = 0; i < t.numel; ++i) d[i] = next_uniform();
    } else {
        blt_tensor tmp = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
        float *d = (float *)tmp.data;
        for (size_t i = 0; i < tmp.numel; ++i) d[i] = next_uniform();
        blt_cast(&tmp, &t);
    }
    return t;
}

static void ref_matmul(const float *a, const float *b, float *out, size_t m, size_t k, size_t n) {
    for (size_t r = 0; r < m; ++r) {
        for (size_t c = 0; c < n; ++c) {
            double acc = 0.0;
            for (size_t i = 0; i < k; ++i) acc += (double)a[r * k + i] * (double)b[i * n + c];
            out[r * n + c] = (float)acc;
        }
    }
}

static float max_abs(const float *v, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) m = fmaxf(m, fabsf(v[i]));
    return m;
}

// Builds a transposed FP8 weight shadow [N,K] from an FP32 [K,N] master.
// The scale tensor is stored in the arena so shadow.scale stays valid after
// this helper returns.
static blt_tensor make_fp8_weight(blt_arena *arena, const blt_tensor *w_f32, blt_dtype fmt) {
    size_t shape[2] = {w_f32->shape[1], w_f32->shape[0]};
    blt_tensor shadow = blt_tensor_create(arena, shape, 2, fmt);
    size_t s1[1] = {1};
    blt_tensor amax = blt_tensor_create(arena, s1, 1, BLT_DTYPE_FP32);
    blt_tensor scale = blt_tensor_create(arena, s1, 1, BLT_DTYPE_FP32);
    blt_amax(w_f32, &amax);
    blt_quant_scale(&amax, blt_fp8_max(fmt), 1e-30f, &scale);
    blt_fp8_quantize(w_f32, &scale, &shadow, 1);
    blt_tensor *scale_slot = (blt_tensor *)blt_container_alloc(arena, sizeof(blt_tensor));
    *scale_slot = scale;
    shadow.scale = scale_slot;
    shadow.transposed = true;
    return shadow;
}

int run_matmul_dtypes_tests(void) {
    blt_arena *arena = blt_arena_create(4 * 1024 * 1024, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    const size_t M = 32, K = 16, N = 32;

    // ---- FP16 x FP16 ----
    {
        blt_tensor a_f32 = rand_2d(arena, M, K, BLT_DTYPE_FP32);
        blt_tensor b_f32 = rand_2d(arena, K, N, BLT_DTYPE_FP32);
        size_t as[2] = {M, K}, bs[2] = {K, N}, os[2] = {M, N};
        blt_tensor a16 = blt_tensor_create(arena, as, 2, BLT_DTYPE_FP16);
        blt_tensor b16 = blt_tensor_create(arena, bs, 2, BLT_DTYPE_FP16);
        blt_tensor out = blt_tensor_create(arena, os, 2, BLT_DTYPE_FP32);
        blt_cast(&a_f32, &a16);
        blt_cast(&b_f32, &b16);

        blt_tensor a_back = blt_tensor_create(arena, as, 2, BLT_DTYPE_FP32);
        blt_tensor b_back = blt_tensor_create(arena, bs, 2, BLT_DTYPE_FP32);
        blt_cast(&a16, &a_back);
        blt_cast(&b16, &b_back);

        blt_tensor ref = blt_tensor_create(arena, os, 2, BLT_DTYPE_FP32);
        ref_matmul((const float *)a_back.data, (const float *)b_back.data, (float *)ref.data, M, K, N);

        blt_matmul(&a16, &b16, &out);
        TEST_ASSERT_CLOSE(&out, &ref, 1e-6f);
    }

    // ---- FP8 forward through a transposed weight shadow ----
    {
        blt_tensor a = rand_2d(arena, M, K, BLT_DTYPE_FP32);
        blt_tensor w = rand_2d(arena, K, N, BLT_DTYPE_FP32);
        blt_tensor shadow = make_fp8_weight(arena, &w, BLT_DTYPE_FP8_E4M3);

        size_t os[2] = {M, N};
        blt_tensor out = blt_tensor_create(arena, os, 2, BLT_DTYPE_FP32);
        blt_matmul(&a, &shadow, &out);

        // Emulated reference: quantize the activation, dequantize the weight.
        blt_tensor amax = blt_tensor_create(arena, (size_t[]){1}, 1, BLT_DTYPE_FP32);
        blt_tensor scale = blt_tensor_create(arena, (size_t[]){1}, 1, BLT_DTYPE_FP32);
        blt_amax(&a, &amax);
        blt_quant_scale(&amax, blt_fp8_max(BLT_DTYPE_FP8_E4M3), 1e-30f, &scale);
        blt_tensor aq = blt_tensor_create(arena, (size_t[]){M, K}, 2, BLT_DTYPE_FP32);
        blt_fp8_roundtrip(&a, &scale, BLT_DTYPE_FP8_E4M3, &aq);
        blt_tensor w_dq = blt_tensor_create(arena, (size_t[]){K, N}, 2, BLT_DTYPE_FP32);
        blt_fp8_dequantize(&shadow, shadow.scale, &w_dq, 1);

        blt_tensor ref = blt_tensor_create(arena, (size_t[]){M, N}, 2, BLT_DTYPE_FP32);
        ref_matmul((const float *)aq.data, (const float *)w_dq.data, (float *)ref.data, M, K, N);
        TEST_ASSERT_CLOSE(&out, &ref, 1e-5f);

        // ... and within FP8 quantization error of the unquantized product.
        blt_tensor exact = blt_tensor_create(arena, (size_t[]){M, N}, 2, BLT_DTYPE_FP32);
        ref_matmul((const float *)a.data, (const float *)w.data, (float *)exact.data, M, K, N);
        float lim = 0.15f * fmaxf(1.0f, max_abs((const float *)exact.data, M * N));
        TEST_ASSERT_CLOSE(&out, &exact, lim);
    }

    // ---- FP8 backward ----
    {
        blt_tensor a = rand_2d(arena, M, K, BLT_DTYPE_FP32);
        blt_tensor w = rand_2d(arena, K, N, BLT_DTYPE_FP32);
        blt_tensor go = rand_2d(arena, M, N, BLT_DTYPE_FP32);
        blt_tensor shadow = make_fp8_weight(arena, &w, BLT_DTYPE_FP8_E4M3);

        blt_tensor ga = blt_tensor_create(arena, (size_t[]){M, K}, 2, BLT_DTYPE_FP32);
        blt_tensor gb = blt_tensor_create(arena, (size_t[]){K, N}, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&a, &shadow, &go, &ga, &gb);

        // Reference from the same quantized operands.
        blt_tensor aq = blt_tensor_create(arena, (size_t[]){M, K}, 2, BLT_DTYPE_FP32);
        blt_tensor gq = blt_tensor_create(arena, (size_t[]){M, N}, 2, BLT_DTYPE_FP32);
        blt_tensor w_dq = blt_tensor_create(arena, (size_t[]){K, N}, 2, BLT_DTYPE_FP32);
        blt_tensor amax = blt_tensor_create(arena, (size_t[]){1}, 1, BLT_DTYPE_FP32);
        blt_tensor scale = blt_tensor_create(arena, (size_t[]){1}, 1, BLT_DTYPE_FP32);

        blt_amax(&a, &amax);
        blt_quant_scale(&amax, blt_fp8_max(BLT_DTYPE_FP8_E4M3), 1e-30f, &scale);
        blt_fp8_roundtrip(&a, &scale, BLT_DTYPE_FP8_E4M3, &aq);
        blt_amax(&go, &amax);
        blt_quant_scale(&amax, blt_fp8_max(BLT_DTYPE_FP8_E4M3), 1e-30f, &scale);
        blt_fp8_roundtrip(&go, &scale, BLT_DTYPE_FP8_E4M3, &gq);
        blt_fp8_dequantize(&shadow, shadow.scale, &w_dq, 1);

        // grad_a = G @ W^T
        blt_tensor ref_ga = blt_tensor_create(arena, (size_t[]){M, K}, 2, BLT_DTYPE_FP32);
        const float *g = (const float *)gq.data;
        const float *wd = (const float *)w_dq.data;
        float *rga = (float *)ref_ga.data;
        for (size_t r = 0; r < M; ++r) {
            for (size_t c = 0; c < K; ++c) {
                double acc = 0.0;
                for (size_t i = 0; i < N; ++i) acc += (double)g[r * N + i] * (double)wd[c * N + i];
                rga[r * K + c] = (float)acc;
            }
        }
        // grad_b = A^T @ G
        blt_tensor ref_gb = blt_tensor_create(arena, (size_t[]){K, N}, 2, BLT_DTYPE_FP32);
        const float *aqd = (const float *)aq.data;
        float *rgb = (float *)ref_gb.data;
        memset(rgb, 0, K * N * sizeof(float));
        for (size_t r = 0; r < M; ++r) {
            for (size_t c = 0; c < K; ++c) {
                float av = aqd[r * K + c];
                for (size_t i = 0; i < N; ++i) rgb[c * N + i] += av * g[r * N + i];
            }
        }

        TEST_ASSERT_CLOSE(&ga, &ref_ga, 1e-5f);
        TEST_ASSERT_CLOSE(&gb, &ref_gb, 1e-5f);
    }

    // ---- Ineligible shape falls back to dequantize + FP32 ----
    {
        const size_t KM = 8;
        blt_tensor a = rand_2d(arena, M, KM, BLT_DTYPE_FP32);
        blt_tensor w = rand_2d(arena, KM, N, BLT_DTYPE_FP32);
        blt_tensor shadow = make_fp8_weight(arena, &w, BLT_DTYPE_FP8_E4M3);

        blt_tensor out = blt_tensor_create(arena, (size_t[]){M, N}, 2, BLT_DTYPE_FP32);
        blt_matmul(&a, &shadow, &out);

        blt_tensor w_dq = blt_tensor_create(arena, (size_t[]){KM, N}, 2, BLT_DTYPE_FP32);
        blt_fp8_dequantize(&shadow, shadow.scale, &w_dq, 1);
        blt_tensor ref = blt_tensor_create(arena, (size_t[]){M, N}, 2, BLT_DTYPE_FP32);
        ref_matmul((const float *)a.data, (const float *)w_dq.data, (float *)ref.data, M, KM, N);
        TEST_ASSERT_CLOSE(&out, &ref, 1e-5f);
    }

    blt_arena_destroy(arena);
    return 1;
}
