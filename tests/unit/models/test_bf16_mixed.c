#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/allocator.h"
#include "core/backend.h"
#include "models/model.h"
#include "models/model_builder.h"
#include "models/param_visitor.h"
#include "ops/cast.h"
#include "ops/matmul.h"

#include "test_helpers.h"
#include "test_suite.h"

static uint16_t ref_fp32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    return (uint16_t)((bits + rounding_bias) >> 16);
}

static int tensor_matches_bf16(const blt_tensor *master, const blt_tensor *shadow) {
    if (master->numel != shadow->numel) return 0;
    if (shadow->dtype != BLT_DTYPE_BF16) return 0;
    const float *m = (const float *)master->data;
    const uint16_t *s = (const uint16_t *)shadow->data;
    for (size_t i = 0; i < master->numel; i++) {
        const uint16_t want = ref_fp32_to_bf16(m[i]);
        if (s[i] != want) {
            fprintf(stderr, "  bf16 mismatch at %zu: got %u want %u (master=%g)\n", i, (unsigned)s[i], (unsigned)want,
                    (double)m[i]);
            return 0;
        }
    }
    return 1;
}

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

static void set_constant(blt_tensor *t, float v) {
    float *p = (float *)t->data;
    for (size_t i = 0; i < t->numel; i++) p[i] = v;
}

int run_bf16_refresh_tests(void) {
    blt_arena *arena = blt_arena_create(32u << 20, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    blt_model *m = make_model(arena, 1, 12345);
    TEST_ASSERT(m != NULL);

    TEST_ASSERT(m->encoder->layers_bf16 != NULL);
    TEST_ASSERT(m->decoder->layers_bf16 != NULL);
    TEST_ASSERT(m->global->stack.layer_weights_bf16 != NULL);

    TEST_ASSERT(tensor_matches_bf16(&m->encoder->layers[0].attn_qkv_w, &m->encoder->layers_bf16[0].attn_qkv_w));
    TEST_ASSERT(tensor_matches_bf16(&m->encoder->layers[0].cross_weight_q, &m->encoder->layers_bf16[0].cross_weight_q));
    TEST_ASSERT(tensor_matches_bf16(&m->global->stack.layer_storage[0].ffn_down_w,
                                    &m->global->stack.layer_storage[0].ffn_down_w_bf16));
    TEST_ASSERT(tensor_matches_bf16(&m->decoder->layers[0].attn_proj_w, &m->decoder->layers_bf16[0].attn_proj_w));
    TEST_ASSERT(
        tensor_matches_bf16(&m->decoder->layers[0].cross_weight_proj, &m->decoder->layers_bf16[0].cross_weight_proj));

    set_constant(&m->encoder->layers[0].ffn_up_w, -0.375f);
    set_constant(&m->global->stack.layer_storage[0].attn_qkv_w, 0.125f);
    set_constant(&m->decoder->layers[0].cross_weight_k, -0.0625f);
    blt_model_refresh_bf16(m);

    TEST_ASSERT(tensor_matches_bf16(&m->encoder->layers[0].ffn_up_w, &m->encoder->layers_bf16[0].ffn_up_w));
    TEST_ASSERT(tensor_matches_bf16(&m->global->stack.layer_storage[0].attn_qkv_w,
                                    &m->global->stack.layer_storage[0].attn_qkv_w_bf16));
    TEST_ASSERT(tensor_matches_bf16(&m->decoder->layers[0].cross_weight_k, &m->decoder->layers_bf16[0].cross_weight_k));

    blt_model *plain = make_model(arena, 0, 7);
    TEST_ASSERT(plain != NULL);
    TEST_ASSERT(plain->encoder->layers_bf16 == NULL);
    TEST_ASSERT(plain->decoder->layers_bf16 == NULL);
    TEST_ASSERT(plain->global->stack.layer_weights_bf16 == NULL);
    blt_model_refresh_bf16(plain);

    blt_arena_destroy(arena);
    return 1;
}

// FP32 activations times BF16 weights must match rounding the weight operand
// and staying in FP32; this guards the mixed branch in blt_matmul_cpu, which
// stages both operands in FP32 scratch.
int run_bf16_mixed_matmul(void) {
    blt_arena *arena = blt_arena_create(1u << 20, BLT_BACKEND_CPU);
    TEST_ASSERT(arena != NULL);

    const size_t M = 5, K = 7, N = 3;
    size_t a_shape[2] = {M, K};
    size_t b_shape[2] = {K, N};
    size_t o_shape[2] = {M, N};

    blt_tensor a = blt_tensor_create(arena, a_shape, 2, BLT_DTYPE_FP32);
    blt_tensor b = blt_tensor_create(arena, b_shape, 2, BLT_DTYPE_FP32);
    float *ap = (float *)a.data;
    float *bp = (float *)b.data;
    for (size_t i = 0; i < M * K; i++) ap[i] = (float)((int)(i * 13 % 17) - 8) * 0.125f;
    for (size_t i = 0; i < K * N; i++) bp[i] = (float)((int)(i * 7 % 11) - 5) * 0.25f;

    blt_tensor b_bf16 = blt_tensor_create(arena, b_shape, 2, BLT_DTYPE_BF16);
    blt_cast(&b, &b_bf16);
    blt_tensor b_rounded = blt_tensor_create(arena, b_shape, 2, BLT_DTYPE_FP32);
    blt_cast(&b_bf16, &b_rounded);

    blt_tensor expected = blt_tensor_create(arena, o_shape, 2, BLT_DTYPE_FP32);
    blt_tensor got = blt_tensor_create(arena, o_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&a, &b_rounded, &expected);
    blt_matmul(&a, &b_bf16, &got);
    TEST_ASSERT_CLOSE(&got, &expected, 1e-6f);

    blt_arena_destroy(arena);
    return 1;
}
