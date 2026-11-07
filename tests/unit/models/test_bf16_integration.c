#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/allocator.h"
#include "core/backend.h"
#include "models/checkpoint.h"
#include "models/model.h"
#include "models/model_builder.h"
#include "models/param_visitor.h"

#include "test_helpers.h"
#include "test_suite.h"

typedef struct {
    uint64_t state;
} fill_ctx;

static float fill_next(fill_ctx *c) {
    c->state = c->state * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint32_t x = (uint32_t)(c->state >> 32);
    return ((float)(x & 0xFFFFu) / 32768.0f - 1.0f) * 0.2f;
}

static void fill_params(float *w, const blt_param_info *info, void *ctx) {
    fill_ctx *c = (fill_ctx *)ctx;
    for (size_t i = 0; i < info->numel; i++) w[i] = info->is_norm ? 1.0f : fill_next(c);
}

static blt_model *make_model(blt_arena *arena, int use_bf16, uint64_t seed) {
    blt_model_config cfg;
    blt_model_config_defaults(&cfg, 16, 32, 1, 1, 1, 32, 0);
    cfg.encoder_config.use_bf16 = use_bf16;
    cfg.decoder_config.use_bf16 = use_bf16;
    cfg.global_config.use_bf16 = use_bf16;

    blt_model *m = blt_model_create(arena, &cfg);
    if (m == NULL) return NULL;

    fill_ctx c = {seed ? seed : 1};
    blt_model_visit_params(m, NULL, fill_params, &c, 0);
    blt_model_refresh_bf16(m);
    return m;
}

typedef struct {
    float lr;
    int finite;
} update_ctx;

static void sgd_update(float *w, float *g, const blt_param_info *info, void *ctx) {
    update_ctx *u = (update_ctx *)ctx;
    for (size_t i = 0; i < info->numel; i++) {
        const float nw = w[i] - u->lr * g[i];
        if (!isfinite(nw)) u->finite = 0;
        w[i] = nw;
    }
}

static void check_finite(float *w, const blt_param_info *info, void *ctx) {
    int *ok = (int *)ctx;
    for (size_t i = 0; i < info->numel; i++) {
        if (!isfinite(w[i])) {
            *ok = 0;
            return;
        }
    }
}

static int make_patches(size_t seq_len, blt_patch_info *patches, size_t cap, size_t *num_patches) {
    size_t n = 0;
    for (size_t s = 0; s < seq_len; s += 4) {
        if (n >= cap) return 0;
        const size_t len = (seq_len - s < 4) ? seq_len - s : 4;
        patches[n].start_idx = s;
        patches[n].length = len;
        patches[n].peak_entropy = 0.0f;
        n++;
    }
    *num_patches = n;
    return 1;
}

static int run_steps(blt_model *model, size_t steps, float *losses) {
    const size_t seq_len = 24;
    const float lr = 0.05f;

    blt_arena *grad_arena = blt_arena_create(32u << 20, BLT_BACKEND_CPU);
    blt_arena *scratch = blt_arena_create(128u << 20, BLT_BACKEND_CPU);
    TEST_ASSERT(grad_arena != NULL && scratch != NULL);

    blt_model_grad *grad = blt_model_grad_create(grad_arena, model);
    TEST_ASSERT(grad != NULL);

    blt_patch_info patches[8];
    size_t num_patches = 0;
    TEST_ASSERT(make_patches(seq_len, patches, 8, &num_patches));

    size_t byte_shape[1] = {seq_len};
    size_t logits_shape[2] = {seq_len, 256};
    size_t scalar_shape[1] = {1};

    for (size_t step = 0; step < steps; step++) {
        blt_arena_reset(scratch);

        uint8_t host_bytes[24];
        for (size_t i = 0; i < seq_len; i++) {
            host_bytes[i] = (uint8_t)((i * 37u + step * 11u + 3u) & 0xFFu);
        }

        blt_tensor bytes = blt_tensor_create_uninit(scratch, byte_shape, 1, BLT_DTYPE_UINT8);
        blt_tensor_upload(&bytes, host_bytes, seq_len);
        blt_tensor logits = blt_tensor_create(scratch, logits_shape, 2, BLT_DTYPE_FP32);
        blt_tensor loss = blt_tensor_create(scratch, scalar_shape, 1, BLT_DTYPE_FP32);

        blt_model_grad_zero(model, grad);
        blt_model_forward(model, &bytes, NULL, patches, num_patches, NULL, 0, &logits, &loss, scratch);

        float lv = 0.0f;
        blt_tensor_download(&loss, &lv, sizeof(float));
        TEST_ASSERT(isfinite(lv));
        losses[step] = lv;

        blt_model_grad_zero(model, grad);
        blt_model_backward(model, &bytes, NULL, patches, num_patches, NULL, 0, grad, scratch);

        update_ctx u = {.lr = lr, .finite = 1};
        blt_model_visit_param_pairs(model, grad, sgd_update, &u);
        TEST_ASSERT(u.finite);
        blt_model_refresh_bf16(model);
    }

    int finite = 1;
    blt_model_visit_params(model, NULL, check_finite, &finite, 0);
    TEST_ASSERT(finite);

    blt_arena_destroy(scratch);
    blt_arena_destroy(grad_arena);
    return 1;
}

static int compare_tensors(const blt_model *a, const blt_model *b) {
    const size_t n = blt_model_num_tensors(a);
    TEST_ASSERT(n == blt_model_num_tensors(b));

    for (size_t i = 0; i < n; i++) {
        const char *na = NULL;
        const char *nb = NULL;
        blt_tensor *ta = NULL;
        blt_tensor *tb = NULL;
        blt_model_tensor_at(a, i, &na, &ta);
        blt_model_tensor_at(b, i, &nb, &tb);
        TEST_ASSERT(strcmp(na, nb) == 0);
        TEST_ASSERT(ta->dtype == BLT_DTYPE_FP32 && tb->dtype == BLT_DTYPE_FP32);
        TEST_ASSERT(ta->numel == tb->numel);
        TEST_ASSERT(memcmp(ta->data, tb->data, ta->numel * sizeof(float)) == 0);
    }
    return 1;
}

static int checkpoint_roundtrip(blt_model *trained) {
    const char *path = "bf16_ckpt_test.fblt";

    blt_model_save(trained, path);

    blt_arena *arena = blt_arena_create(48u << 20, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    blt_model *plain = make_model(arena, 0, 999);
    TEST_ASSERT(plain != NULL);
    blt_model_load(plain, path);
    TEST_ASSERT(compare_tensors(trained, plain));

    blt_model *bf16 = make_model(arena, 1, 999);
    TEST_ASSERT(bf16 != NULL);
    blt_model_load(bf16, path);
    blt_model_refresh_bf16(bf16);

    const float *master = (const float *)bf16->decoder->layers[0].ffn_down_w.data;
    const uint16_t *shadow = (const uint16_t *)bf16->decoder->layers_bf16[0].ffn_down_w.data;
    for (size_t i = 0; i < bf16->decoder->layers[0].ffn_down_w.numel; i++) {
        uint32_t bits;
        memcpy(&bits, &master[i], 4);
        const uint16_t want = (uint16_t)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
        TEST_ASSERT(shadow[i] == want);
    }

    remove(path);
    blt_arena_destroy(arena);
    return 1;
}

int run_bf16_integration_tests(void) {
    const size_t steps = 6;

    blt_arena *fp32_arena = blt_arena_create(48u << 20, BLT_BACKEND_CPU);
    blt_arena *bf16_arena = blt_arena_create(48u << 20, BLT_BACKEND_CPU);
    TEST_ASSERT(fp32_arena != NULL && bf16_arena != NULL);

    blt_model *fp32 = make_model(fp32_arena, 0, 4242);
    blt_model *bf16 = make_model(bf16_arena, 1, 4242);
    TEST_ASSERT(fp32 != NULL && bf16 != NULL);

    float fp32_losses[16];
    float bf16_losses[16];
    TEST_ASSERT(run_steps(fp32, steps, fp32_losses));
    TEST_ASSERT(run_steps(bf16, steps, bf16_losses));

    TEST_ASSERT(fabsf(fp32_losses[0] - bf16_losses[0]) <= 0.02f);
    for (size_t i = 0; i < steps; i++) {
        const float tol = 0.05f + 0.05f * fabsf(fp32_losses[i]);
        TEST_ASSERT(fabsf(fp32_losses[i] - bf16_losses[i]) <= tol);
    }

    TEST_ASSERT(checkpoint_roundtrip(bf16));

    blt_arena_destroy(bf16_arena);
    blt_arena_destroy(fp32_arena);
    return 1;
}
