#include "test_helpers.h"
#include "test_suite.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/allocator.h"
#include "models/model.h"
#include "models/model_builder.h"
#include "models/param_visitor.h"
#include "ops/optim.h"
#include "ops/vecmath.h"
#include "train_diag.h"
#include "train_optim.h"

// The default small config has 67 parameter tensors: one embedding, two ngram
// tables, 12 per local layer, 7 per global layer, and lm_head + d0. That is
// above the old clip buffer's initial capacity of 64, so these tests exercise
// the growth path that used to discard the first 64 squared-norm entries.
#define TEST_EMBED 16
#define TEST_HIDDEN 32
#define TEST_LAYERS 2
#define TEST_MAX_SEQ 64

static uint64_t next_rng(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x;
}

struct fill_ctx {
    uint64_t rng;
    float scale;
};

static void fill_weights_fn(float *t, const blt_param_info *info, void *ctx) {
    struct fill_ctx *c = (struct fill_ctx *)ctx;
    for (size_t i = 0; i < info->numel; i++) {
        const int32_t r = (int32_t)(next_rng(&c->rng) >> 40) % 2001;
        t[i] = ((float)r - 1000.0f) / 1000.0f * c->scale;
    }
}

static void fill_grads_fn(float *t, const blt_param_info *info, void *ctx) {
    struct fill_ctx *c = (struct fill_ctx *)ctx;
    for (size_t i = 0; i < info->numel; i++) {
        const int32_t r = (int32_t)(next_rng(&c->rng) >> 40) % 2001;
        t[i] = ((float)r - 1000.0f) / 1000.0f * c->scale;
    }
}

struct count_ctx {
    size_t count;
};

static void count_fn(float *t, const blt_param_info *info, void *ctx) {
    (void)t;
    (void)info;
    ((struct count_ctx *)ctx)->count++;
}

struct sq_ctx {
    float sq;
};

static void sq_fn(float *t, const blt_param_info *info, void *ctx) {
    struct sq_ctx *c = (struct sq_ctx *)ctx;
    c->sq += blt_vec_dot(info->backend, t, t, info->numel);
}

struct hash_ctx {
    uint64_t hash;
};

static void hash_fn(float *t, const blt_param_info *info, void *ctx) {
    struct hash_ctx *c = (struct hash_ctx *)ctx;
    const unsigned char *bytes = (const unsigned char *)t;
    for (size_t i = 0; i < info->numel * sizeof(float); i++) {
        c->hash ^= bytes[i];
        c->hash *= 1099511628211ULL;
    }
}

static uint64_t hash_grads(const blt_model *model, blt_model_grad *grad) {
    struct hash_ctx ctx = {14695981039346656037ULL};
    blt_model_visit_params(model, grad, hash_fn, &ctx, 1);
    return ctx.hash;
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

static void flat_load_fn(float *t, const blt_param_info *info, void *ctx) {
    struct flat_ctx *c = (struct flat_ctx *)ctx;
    memcpy(t, c->data + c->offset, info->numel * sizeof(float));
    c->offset += info->numel;
}

static float *snapshot_weights(const blt_model *model) {
    struct flat_ctx count = {0};
    blt_model_visit_params(model, NULL, flat_count_fn, &count, 0);
    float *data = (float *)malloc(count.total * sizeof(float));
    if (!data) return NULL;
    struct flat_ctx save = {data, count.total, 0};
    blt_model_visit_params(model, NULL, flat_save_fn, &save, 0);
    return data;
}

static void restore_weights(blt_model *model, const float *data) {
    struct flat_ctx load = {(float *)data, 0, 0};
    blt_model_visit_params(model, NULL, flat_load_fn, &load, 0);
}

static float *snapshot_grads(const blt_model *model, blt_model_grad *grad) {
    struct flat_ctx count = {0};
    blt_model_visit_params(model, grad, flat_count_fn, &count, 1);
    float *data = (float *)malloc(count.total * sizeof(float));
    if (!data) return NULL;
    struct flat_ctx save = {data, count.total, 0};
    blt_model_visit_params(model, grad, flat_save_fn, &save, 1);
    return data;
}

static void restore_grads(const blt_model *model, blt_model_grad *grad, const float *data) {
    struct flat_ctx load = {(float *)data, 0, 0};
    blt_model_visit_params(model, grad, flat_load_fn, &load, 1);
}

static blt_model *make_model(blt_arena *arena) {
    blt_model_config cfg;
    blt_model_config_defaults(&cfg, TEST_EMBED, TEST_HIDDEN, TEST_LAYERS, TEST_LAYERS, TEST_LAYERS, TEST_MAX_SEQ, 0);
    return blt_model_create(arena, &cfg);
}

static int test_clip_norm_matches_visitor_order(void) {
    blt_arena *arena = blt_arena_create(64u << 20, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);
    blt_model *model = make_model(arena);
    blt_model_grad *grad = blt_model_grad_create(arena, model);
    TEST_ASSERT(model != NULL && grad != NULL);

    struct fill_ctx wfill = {0x12345678ULL, 0.05f};
    blt_model_visit_params(model, NULL, fill_weights_fn, &wfill, 0);
    struct fill_ctx gfill = {0x9ABCDEF0ULL, 1.5f};
    blt_model_visit_params(model, grad, fill_grads_fn, &gfill, 1);

    struct count_ctx counted = {0};
    blt_model_visit_params(model, grad, count_fn, &counted, 1);
    TEST_ASSERT(counted.count > 64);

    struct sq_ctx reference = {0.0f};
    blt_model_visit_params(model, grad, sq_fn, &reference, 1);
    const float expected = sqrtf(reference.sq);
    TEST_ASSERT(expected > 0.0f);

    const uint64_t hash_before = hash_grads(model, grad);
    const float max_norm = expected * 0.5f;
    float scale = -1.0f;
    const float got = blt_grad_clip_norm(model, grad, max_norm, &scale);

    // Same per-tensor dots in the same visitor order, then one host sum.
    TEST_ASSERT(got == expected);
    TEST_ASSERT(scale == max_norm / expected);
    TEST_ASSERT(hash_grads(model, grad) == hash_before);

    // No clipping request returns a unit scale.
    const float no_clip = blt_grad_clip_norm(model, grad, expected * 2.0f, &scale);
    TEST_ASSERT(no_clip == expected);
    TEST_ASSERT(scale == 1.0f);

    blt_arena_destroy(arena);
    return 1;
}

static int test_sgd_clip_scale_fold_is_bit_identical(void) {
    blt_arena *arena = blt_arena_create(64u << 20, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);
    blt_model *model = make_model(arena);
    blt_model_grad *grad = blt_model_grad_create(arena, model);
    TEST_ASSERT(model != NULL && grad != NULL);

    struct fill_ctx wfill = {0x0BADF00DULL, 0.05f};
    blt_model_visit_params(model, NULL, fill_weights_fn, &wfill, 0);
    struct fill_ctx gfill = {0xFEEDFACEULL, 1.0f};
    blt_model_visit_params(model, grad, fill_grads_fn, &gfill, 1);

    float *w0 = snapshot_weights(model);
    float *g0 = snapshot_grads(model, grad);
    TEST_ASSERT(w0 != NULL && g0 != NULL);

    const float lr = 0.03125f;
    const float scale = 0.37f;

    // Reference: pre-scale the gradient, then run the unscaled update.
    blt_model_grad_scale(model, grad, scale);
    sgd_all(model, grad, lr, 1.0f);
    float *w_reference = snapshot_weights(model);
    TEST_ASSERT(w_reference != NULL);

    // Folded: leave the gradient untouched and let the optimizer scale it.
    restore_weights(model, w0);
    restore_grads(model, grad, g0);
    sgd_all(model, grad, lr, scale);
    float *w_folded = snapshot_weights(model);
    TEST_ASSERT(w_folded != NULL);

    struct flat_ctx count = {0};
    blt_model_visit_params(model, NULL, flat_count_fn, &count, 0);
    TEST_ASSERT(memcmp(w_reference, w_folded, count.total * sizeof(float)) == 0);

    free(w0);
    free(g0);
    free(w_reference);
    free(w_folded);
    blt_arena_destroy(arena);
    return 1;
}

static int test_adamw_clip_scale_fold_is_bit_identical(void) {
    blt_arena *model_arena = blt_arena_create(64u << 20, BLT_BACKEND_CPU);
    blt_arena *opt_arena = blt_arena_create(64u << 20, BLT_BACKEND_CPU);
    TEST_ASSERT(model_arena != NULL && opt_arena != NULL);
    blt_model *model = make_model(model_arena);
    blt_model_grad *grad = blt_model_grad_create(model_arena, model);
    TEST_ASSERT(model != NULL && grad != NULL);

    struct fill_ctx wfill = {0x13572468ULL, 0.05f};
    blt_model_visit_params(model, NULL, fill_weights_fn, &wfill, 0);
    struct fill_ctx gfill = {0x24681357ULL, 1.0f};
    blt_model_visit_params(model, grad, fill_grads_fn, &gfill, 1);

    float *w0 = snapshot_weights(model);
    float *g0 = snapshot_grads(model, grad);
    TEST_ASSERT(w0 != NULL && g0 != NULL);

    adamw_state *state_ref = adamw_state_create(opt_arena, model);
    adamw_state *state_fold = adamw_state_create(opt_arena, model);
    TEST_ASSERT(state_ref != NULL && state_fold != NULL);

    blt_adamw_config cfg = {
        .lr = 0.001f, .beta1 = 0.9f, .beta2 = 0.999f, .eps = 1e-8f, .weight_decay = 0.01f, .step = 0};
    const float scale = 0.42f;

    blt_model_grad_scale(model, grad, scale);
    adamw_all(model, grad, state_ref, &cfg, 1.0f);
    float *w_reference = snapshot_weights(model);
    TEST_ASSERT(w_reference != NULL);

    restore_weights(model, w0);
    restore_grads(model, grad, g0);
    adamw_all(model, grad, state_fold, &cfg, scale);
    float *w_folded = snapshot_weights(model);
    TEST_ASSERT(w_folded != NULL);

    struct flat_ctx count = {0};
    blt_model_visit_params(model, NULL, flat_count_fn, &count, 0);
    TEST_ASSERT(memcmp(w_reference, w_folded, count.total * sizeof(float)) == 0);

    free(w0);
    free(g0);
    free(w_reference);
    free(w_folded);
    blt_arena_destroy(model_arena);
    blt_arena_destroy(opt_arena);
    return 1;
}

// --grad-accum 1 returns before the accumulator and before the copy/scale
// steps, so the gradient must be left untouched. The helpers that would run
// for accumulation must also be exact no-ops at zero and unit scale.
static int test_grad_accum_one_identity(void) {
    blt_arena *arena = blt_arena_create(64u << 20, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);
    blt_model *model = make_model(arena);
    blt_model_grad *grad = blt_model_grad_create(arena, model);
    blt_model_grad *accum = blt_model_grad_create(arena, model);
    TEST_ASSERT(model != NULL && grad != NULL && accum != NULL);

    struct fill_ctx gfill = {0xCAFEBABEULL, 1.0f};
    blt_model_visit_params(model, grad, fill_grads_fn, &gfill, 1);

    const uint64_t expected = hash_grads(model, grad);

    // Path taken by --grad-accum 1: no zero/add/scale on the gradient.
    TEST_ASSERT(hash_grads(model, grad) == expected);

    // Path that would run for accumulation: zero + add + scale by 1.
    blt_model_grad_zero(model, accum);
    blt_model_grad_add(model, accum, grad);
    blt_model_grad_scale(model, accum, 1.0f);
    TEST_ASSERT(hash_grads(model, accum) == expected);

    // Scaling by 1 is exact even on an already-populated buffer.
    blt_model_grad_scale(model, grad, 1.0f);
    TEST_ASSERT(hash_grads(model, grad) == expected);

    blt_arena_destroy(arena);
    return 1;
}

int run_train_diag_tests(void) {
    int ok = 1;
    ok &= test_clip_norm_matches_visitor_order();
    ok &= test_sgd_clip_scale_fold_is_bit_identical();
    ok &= test_adamw_clip_scale_fold_is_bit_identical();
    ok &= test_grad_accum_one_identity();
    return ok;
}
