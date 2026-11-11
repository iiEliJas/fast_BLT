#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/allocator.h"
#include "core/backend.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "ops/cast.h"
#include "ops/quant.h"

#include "test_helpers.h"
#include "test_suite.h"

static blt_tensor make_fp32(blt_arena *arena, size_t rows, size_t cols, const float *values) {
    size_t shape[2] = {rows, cols};
    blt_tensor t = blt_tensor_create(arena, shape, 2, BLT_DTYPE_FP32);
    memcpy(t.data, values, rows * cols * sizeof(float));
    return t;
}

static blt_tensor make_1d(blt_arena *arena, blt_dtype dtype, size_t n) {
    size_t shape[1] = {n};
    return blt_tensor_create(arena, shape, 1, dtype);
}

// fp32 -> fmt -> fp32 for a single scalar, using the generic cast op.
static float roundtrip_scalar(blt_arena *arena, blt_dtype fmt, float value) {
    blt_tensor src = make_fp32(arena, 1, 1, &value);
    blt_tensor enc = make_1d(arena, fmt, 1);
    blt_tensor dec = make_1d(arena, BLT_DTYPE_FP32, 1);
    blt_cast(&src, &enc);
    blt_cast(&enc, &dec);
    return ((float *)dec.data)[0];
}

static int check_scalar(blt_arena *arena, blt_dtype fmt, float in, float expected, float tol) {
    float got = roundtrip_scalar(arena, fmt, in);
    if (fabsf(got - expected) > tol) {
        fprintf(stderr, "  cast dtype %d: %g -> %g (expected %g)\n", (int)fmt, (double)in, (double)got,
                (double)expected);
        return 0;
    }
    return 1;
}

int run_quant_tests(void) {
    blt_arena *arena = blt_arena_create(65536, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    // ---- E4M3 representable values ----
    {
        struct {
            float in, expected;
        } cases[] = {
            {0.0f, 0.0f},
            {1.0f, 1.0f},
            {-1.0f, -1.0f},
            {0.5f, 0.5f},
            {448.0f, 448.0f},
            {-448.0f, -448.0f},
            {0.015625f, 0.015625f},       // 2^-6, smallest normal
            {0.013671875f, 0.013671875f}, // 7 * 2^-9, largest subnormal
            {0.0001f, 0.0f},
            {500.0f, 448.0f},
            {480.0f, 448.0f},
            {464.0f, 448.0f},
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            TEST_ASSERT(check_scalar(arena, BLT_DTYPE_FP8_E4M3, cases[i].in, cases[i].expected, 0.0f));
        }
    }

    // ---- E5M2 representable values (2 mantissa bits) ----
    {
        struct {
            float in, expected;
        } cases[] = {
            {0.0f, 0.0f},
            {1.0f, 1.0f},
            {-2.0f, -2.0f},
            {0.5f, 0.5f},
            {57344.0f, 57344.0f},
            {-57344.0f, -57344.0f},
            {1.52587890625e-05f, 1.52587890625e-05f}, // 2^-16, smallest subnormal
            {6.103515625e-05f, 6.103515625e-05f},     // 2^-14, smallest normal
            {1e-8f, 0.0f},
            {100000.0f, 57344.0f},
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            TEST_ASSERT(check_scalar(arena, BLT_DTYPE_FP8_E5M2, cases[i].in, cases[i].expected, 0.0f));
        }
    }

    // ---- FP16 exact values ----
    {
        struct {
            float in, expected;
        } cases[] = {
            {0.0f, 0.0f},
            {1.0f, 1.0f},
            {-1.0f, -1.0f},
            {2.0f, 2.0f},
            {0.5f, 0.5f},
            {65504.0f, 65504.0f},                             // max finite
            {5.960464477539063e-08f, 5.960464477539063e-08f}, // 2^-24, smallest subnormal
            {6.103515625e-05f, 6.103515625e-05f},             // 2^-14, smallest normal
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            TEST_ASSERT(check_scalar(arena, BLT_DTYPE_FP16, cases[i].in, cases[i].expected, 0.0f));
        }
        // 0.1 is not exact in fp16 but must round to within 1 ulp (~5e-5).
        float got = roundtrip_scalar(arena, BLT_DTYPE_FP16, 0.1f);
        TEST_ASSERT(fabsf(got - 0.1f) < 1e-4f);
    }

    // ---- amax / scale / quantize / dequantize ----
    {
        float values[4] = {1.0f, -3.0f, 2.0f, 0.5f};
        blt_tensor x = make_fp32(arena, 2, 2, values);
        blt_tensor amax = make_1d(arena, BLT_DTYPE_FP32, 1);
        blt_tensor scale = make_1d(arena, BLT_DTYPE_FP32, 1);
        blt_tensor q = blt_tensor_create(arena, (size_t[]){2, 2}, 2, BLT_DTYPE_FP8_E4M3);
        blt_tensor dq = blt_tensor_create(arena, (size_t[]){2, 2}, 2, BLT_DTYPE_FP32);

        blt_amax(&x, &amax);
        TEST_ASSERT(fabsf(((float *)amax.data)[0] - 3.0f) < 1e-6f);

        blt_quant_scale(&amax, blt_fp8_max(BLT_DTYPE_FP8_E4M3), 1e-30f, &scale);
        float s = ((float *)scale.data)[0];
        TEST_ASSERT(fabsf(s - 3.0f / 448.0f) < 1e-9f);

        blt_fp8_quantize(&x, &scale, &q, 0);
        blt_fp8_dequantize(&q, &scale, &dq, 0);

        const float *out = (const float *)dq.data;
        for (size_t i = 0; i < 4; ++i) {
            float rel = fabsf(out[i] - values[i]) / fmaxf(fabsf(values[i]), 1e-6f);
            TEST_ASSERT(rel <= 0.0625f + 1e-5f);
        }
        // The largest magnitude quantizes to the format maximum exactly.
        TEST_ASSERT(fabsf(out[1] - (-3.0f)) < 1e-5f);
    }

    // ---- transpose quantize / dequantize round-trip ----
    {
        float values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        blt_tensor x = make_fp32(arena, 2, 3, values);
        blt_tensor scale = make_1d(arena, BLT_DTYPE_FP32, 1);
        ((float *)scale.data)[0] = 6.0f / 448.0f;
        blt_tensor qt = blt_tensor_create(arena, (size_t[]){3, 2}, 2, BLT_DTYPE_FP8_E4M3);
        blt_tensor dqt = blt_tensor_create(arena, (size_t[]){2, 3}, 2, BLT_DTYPE_FP32);

        blt_fp8_quantize(&x, &scale, &qt, 1);
        blt_fp8_dequantize(&qt, &scale, &dqt, 1);

        const float *out = (const float *)dqt.data;
        for (size_t r = 0; r < 2; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                float want = values[r * 3 + c];
                TEST_ASSERT(fabsf(out[r * 3 + c] - want) <= 0.0625f * fabsf(want) + 1e-5f);
            }
        }
    }

    // ---- roundtrip emulation is idempotent ----
    {
        float values[3] = {-1.5f, 0.25f, 7.0f};
        blt_tensor x = make_fp32(arena, 1, 3, values);
        blt_tensor scale = make_1d(arena, BLT_DTYPE_FP32, 1);
        ((float *)scale.data)[0] = 7.0f / 448.0f;
        blt_tensor once = blt_tensor_create(arena, (size_t[]){1, 3}, 2, BLT_DTYPE_FP32);
        blt_tensor twice = blt_tensor_create(arena, (size_t[]){1, 3}, 2, BLT_DTYPE_FP32);
        blt_fp8_roundtrip(&x, &scale, BLT_DTYPE_FP8_E4M3, &once);
        blt_fp8_roundtrip(&once, &scale, BLT_DTYPE_FP8_E4M3, &twice);
        for (size_t i = 0; i < 3; ++i) {
            TEST_ASSERT(fabsf(((float *)once.data)[i] - ((float *)twice.data)[i]) < 1e-6f);
        }
    }

    // ---- Dynamic-operand encoding paired with a weight encoding ----
    {
        TEST_ASSERT(blt_fp8_pair_fmt(BLT_DTYPE_FP8_E4M3) == BLT_DTYPE_FP8_E4M3);
        TEST_ASSERT(blt_fp8_pair_fmt(BLT_DTYPE_FP8_E5M2) == BLT_DTYPE_FP8_E4M3);
        TEST_ASSERT(blt_fp8_pair_fmt(BLT_DTYPE_BF16) == BLT_DTYPE_BF16);
    }

    // ---- FP8 GEMM eligibility gate ----
    {
        TEST_ASSERT(blt_fp8_matmul_eligible(64, 64, 64) == 1);
        TEST_ASSERT(blt_fp8_matmul_eligible(32, 32, 32) == 1);
        TEST_ASSERT(blt_fp8_matmul_eligible(64, 40, 64) == 0); // k not 16-aligned
        TEST_ASSERT(blt_fp8_matmul_eligible(8, 64, 64) == 0);  // too small
        TEST_ASSERT(blt_fp8_matmul_eligible(16, 16, 16) == 0); // below the MAC threshold
    }

    blt_arena_destroy(arena);
    return 1;
}
