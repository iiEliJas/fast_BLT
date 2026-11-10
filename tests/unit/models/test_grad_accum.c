#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_helpers.h"
#include "test_suite.h"

#include "core/allocator.h"
#include "models/model.h"
#include "models/model_builder.h"
#include "models/param_visitor.h"
#include "models/patcher.h"

#define GRAD_ACCUM_EMBED 16
#define GRAD_ACCUM_HIDDEN 32
#define GRAD_ACCUM_LAYERS 2
#define GRAD_ACCUM_MAX_SEQ 64
#define GRAD_ACCUM_SEQ 32
#define GRAD_ACCUM_PATCH_LEN 4

// Deterministic xorshift32, independent of libc rand() so the test does not
// depend on global RNG state left behind by other suites.
static uint32_t next_u32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

struct weight_fill_ctx {
    uint32_t state;
    float scale;
};

static void fill_weights_fn(float *t, const blt_param_info *info, void *ctx) {
    struct weight_fill_ctx *c = (struct weight_fill_ctx *)ctx;
    for (size_t i = 0; i < info->numel; i++) {
        if (info->is_norm) {
            t[i] = 1.0f;
        } else {
            const float r = (float)(next_u32(&c->state) & 0xFFFFu) / 65535.0f;
            t[i] = (r * 2.0f - 1.0f) * c->scale;
        }
    }
}

struct grad_fill_ctx {
    float value;
    int vary;
};

// With vary == 0 every element gets `value`; with vary == 1 the value also
// depends on the parameter index and element offset so a missing tensor or a
// mixed-up tensor shows up as a mismatch.
static void fill_grads_fn(float *t, const blt_param_info *info, void *ctx) {
    struct grad_fill_ctx *c = (struct grad_fill_ctx *)ctx;
    for (size_t i = 0; i < info->numel; i++) {
        if (c->vary) {
            const int32_t r = (int32_t)((info->idx * 131 + (int)i * 17) % 2001) - 1000;
            t[i] = (float)r / 1000.0f;
        } else {
            t[i] = c->value;
        }
    }
}

struct flat_ctx {
    float *data;
    size_t total;
    size_t offset;
};

static void flat_count_fn(float *t, const blt_param_info *info, void *ctx) {
    (void)t;
    ((struct flat_ctx *)ctx)->total += info->numel;
}

static void flat_save_fn(float *t, const blt_param_info *info, void *ctx) {
    struct flat_ctx *c = (struct flat_ctx *)ctx;
    memcpy(c->data + c->offset, t, info->numel * sizeof(float));
    c->offset += info->numel;
}

static size_t count_grads(const blt_model *model, blt_model_grad *grad) {
    struct flat_ctx count = {0};
    blt_model_visit_params(model, grad, flat_count_fn, &count, 1);
    return count.total;
}

static float *snapshot_grads(const blt_model *model, blt_model_grad *grad, size_t *total_out) {
    const size_t total = count_grads(model, grad);
    float *data = (float *)malloc(total * sizeof(float));
    if (!data) return NULL;
    struct flat_ctx save = {data, total, 0};
    blt_model_visit_params(model, grad, flat_save_fn, &save, 1);
    *total_out = total;
    return data;
}

struct compare_ctx {
    const float *reference;
    size_t offset;
    float scale;
    size_t mismatches;
};

static void compare_fn(float *t, const blt_param_info *info, void *ctx) {
    struct compare_ctx *c = (struct compare_ctx *)ctx;
    for (size_t i = 0; i < info->numel; i++) {
        const float expected = c->reference[c->offset + i] * c->scale;
        if (t[i] != expected) c->mismatches++;
    }
    c->offset += info->numel;
}

static size_t count_mismatches(const blt_model *model, blt_model_grad *grad, const float *reference, float scale) {
    struct compare_ctx cmp = {reference, 0, scale, 0};
    blt_model_visit_params(model, grad, compare_fn, &cmp, 1);
    return cmp.mismatches;
}

static blt_model *make_model(blt_arena *arena) {
    blt_model_config cfg;
    blt_model_config_defaults(&cfg, GRAD_ACCUM_EMBED, GRAD_ACCUM_HIDDEN, GRAD_ACCUM_LAYERS, GRAD_ACCUM_LAYERS,
                              GRAD_ACCUM_LAYERS, GRAD_ACCUM_MAX_SEQ, 0);
    return blt_model_create(arena, &cfg);
}

static size_t build_fixed_patches(size_t seq_len, size_t patch_len, blt_patch_info *patches) {
    size_t n = 0;
    size_t start = 0;
    while (start < seq_len) {
        const size_t len = (start + patch_len <= seq_len) ? patch_len : (seq_len - start);
        patches[n].start_idx = start;
        patches[n].length = len;
        patches[n].peak_entropy = 0.0f;
        n++;
        start += len;
    }
    return n;
}

// Mirrors one micro-batch of the trainer: reset the scratch arena, upload the
// window, zero the gradient, then run backward.
static void compute_window_grad(const blt_model *model, blt_model_grad *grad, const uint8_t *bytes, size_t len,
                                const blt_patch_info *patches, size_t num_patches, blt_arena *scratch) {
    blt_arena_reset(scratch);
    size_t shape[1] = {len};
    blt_tensor bytes_in = blt_tensor_create(scratch, shape, 1, BLT_DTYPE_UINT8);
    memcpy(bytes_in.data, bytes, len);
    blt_model_grad_zero(model, grad);
    blt_model_backward(model, &bytes_in, NULL, patches, num_patches, NULL, 0, grad, scratch);
}

// Every gradient tensor must be reached by zero/add/scale. The implementation
// keeps explicit parameter lists, so a new tensor added without updating those
// lists leaves stale values behind; the visitor is the ground truth.
static int test_grad_ops_touch_every_tensor(void) {
    blt_arena *arena = blt_arena_create(32u << 20, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);
    blt_model *model = make_model(arena);
    blt_model_grad *src = blt_model_grad_create(arena, model);
    blt_model_grad *dst = blt_model_grad_create(arena, model);
    TEST_ASSERT(model != NULL && src != NULL && dst != NULL);

    const size_t total = count_grads(model, src);
    TEST_ASSERT(total > 0);

    struct grad_fill_ctx src_fill = {0.0f, 1};
    blt_model_visit_params(model, src, fill_grads_fn, &src_fill, 1);

    // Zero must reach every tensor: seed dst with a non-zero sentinel first.
    struct grad_fill_ctx sentinel = {12345.0f, 0};
    blt_model_visit_params(model, dst, fill_grads_fn, &sentinel, 1);
    blt_model_grad_zero(model, dst);

    float *zeros = (float *)calloc(total, sizeof(float));
    TEST_ASSERT(zeros != NULL);
    TEST_ASSERT(count_mismatches(model, dst, zeros, 1.0f) == 0);
    free(zeros);

    // Add must copy every tensor exactly.
    blt_model_grad_add(model, dst, src);
    size_t src_total = 0;
    float *src_flat = snapshot_grads(model, src, &src_total);
    TEST_ASSERT(src_flat != NULL && src_total == total);
    TEST_ASSERT(count_mismatches(model, dst, src_flat, 1.0f) == 0);

    // Scale must hit every tensor with the exact multiplier.
    const float scale = 0.25f;
    blt_model_grad_scale(model, dst, scale);
    TEST_ASSERT(count_mismatches(model, dst, src_flat, scale) == 0);

    // Scaling by 1 is an exact no-op even on a populated buffer.
    size_t scaled_total = 0;
    float *scaled_flat = snapshot_grads(model, dst, &scaled_total);
    TEST_ASSERT(scaled_flat != NULL && scaled_total == total);
    blt_model_grad_scale(model, dst, 1.0f);
    TEST_ASSERT(count_mismatches(model, dst, scaled_flat, 1.0f) == 0);

    free(scaled_flat);
    free(src_flat);
    blt_arena_destroy(arena);
    return 1;
}

// Two micro-batches accumulated and scaled by 1/2 must equal the explicit
// average of the two single-window gradients. Reusing the same micro gradient
// buffer on the second window also covers the zero-before-backward contract.
static int test_two_window_gradient_average(void) {
    blt_arena *arena = blt_arena_create(32u << 20, BLT_BACKEND_CPU);
    blt_arena *scratch = blt_arena_create(16u << 20, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL && scratch != NULL);

    blt_model *model = make_model(arena);
    blt_model_grad *grad = blt_model_grad_create(arena, model);
    blt_model_grad *window_a = blt_model_grad_create(arena, model);
    blt_model_grad *window_b = blt_model_grad_create(arena, model);
    blt_model_grad *accum = blt_model_grad_create(arena, model);
    TEST_ASSERT(model != NULL && grad != NULL && window_a != NULL && window_b != NULL && accum != NULL);

    struct weight_fill_ctx wfill = {0x12345678u, 0.1f};
    blt_model_visit_params(model, NULL, fill_weights_fn, &wfill, 0);

    uint8_t text_a[GRAD_ACCUM_SEQ];
    uint8_t text_b[GRAD_ACCUM_SEQ];
    for (size_t i = 0; i < GRAD_ACCUM_SEQ; i++) {
        text_a[i] = (uint8_t)((i * 37 + 11) & 0xFF);
        text_b[i] = (uint8_t)((i * 53 + 29) & 0xFF);
    }

    blt_patch_info patches[16];
    const size_t num_patches = build_fixed_patches(GRAD_ACCUM_SEQ, GRAD_ACCUM_PATCH_LEN, patches);
    TEST_ASSERT(num_patches >= 2);

    compute_window_grad(model, window_a, text_a, GRAD_ACCUM_SEQ, patches, num_patches, scratch);
    compute_window_grad(model, window_b, text_b, GRAD_ACCUM_SEQ, patches, num_patches, scratch);

    // Reference: (grad_a + grad_b) / 2.
    blt_model_grad_zero(model, accum);
    blt_model_grad_add(model, accum, window_a);
    blt_model_grad_add(model, accum, window_b);
    blt_model_grad_scale(model, accum, 0.5f);
    size_t expected_total = 0;
    float *expected = snapshot_grads(model, accum, &expected_total);
    TEST_ASSERT(expected != NULL && expected_total > 0);

    // Trainer path: one reusable micro gradient buffer, accumulate, then
    // scale by the number of completed micro-batches.
    blt_model_grad_zero(model, accum);
    compute_window_grad(model, grad, text_a, GRAD_ACCUM_SEQ, patches, num_patches, scratch);
    blt_model_grad_add(model, accum, grad);
    compute_window_grad(model, grad, text_b, GRAD_ACCUM_SEQ, patches, num_patches, scratch);
    blt_model_grad_add(model, accum, grad);
    blt_model_grad_scale(model, accum, 0.5f);

    TEST_ASSERT(count_mismatches(model, accum, expected, 1.0f) == 0);

    free(expected);
    blt_arena_destroy(scratch);
    blt_arena_destroy(arena);
    return 1;
}

int run_grad_accum_tests(void) {
    int ok = 1;
    ok &= test_grad_ops_touch_every_tensor();
    ok &= test_two_window_gradient_average();
    return ok;
}
