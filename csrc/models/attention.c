#include "models/attention.h"
#include <stdlib.h>
#include "core/backend.h"
#include "core/tensor.h"
#include "core/allocator.h"
#include "ops/matmul.h"
#include "ops/vecmath.h"
#include "ops/attn_core.h"
#include "ops/elementwise.h"
#include "ops/rope.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int is_fp32_or_bf16(blt_dtype d) { return d == BLT_DTYPE_FP32 || d == BLT_DTYPE_BF16; }

static void validate_attention_call(const blt_tensor *input, const blt_tensor *weight_qkv,
                                    const blt_tensor *weight_proj, const blt_tensor *output,
                                    const blt_attention_config *config, const blt_arena *arena, size_t *out_seq_len,
                                    size_t *out_head_dim) {
    BLT_REQUIRE(config != NULL && arena != NULL, "blt_multihead_attention: config and arena cannot be NULL");

    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;
    BLT_REQUIRE(num_heads != 0 && embed_dim % num_heads == 0,
                "blt_multihead_attention: embed_dim must be divisible by num_heads");

    blt_check_nd_fp32(input, 2, (const size_t[]){0, embed_dim},
                      "blt_multihead_attention: input must be [seq_len, embed_dim] FP32");
    size_t seq_len = input->shape[0];

    BLT_REQUIRE(weight_qkv != NULL && weight_qkv->ndim == 2 && weight_qkv->shape[0] == embed_dim &&
                    weight_qkv->shape[1] == 3 * embed_dim,
                "blt_multihead_attention: weight_qkv must be [embed_dim, 3*embed_dim]");
    BLT_REQUIRE(is_fp32_or_bf16(weight_qkv->dtype), "blt_multihead_attention: weight_qkv must be FP32 or BF16");
    BLT_REQUIRE(weight_proj != NULL && weight_proj->ndim == 2 && weight_proj->shape[0] == embed_dim &&
                    weight_proj->shape[1] == embed_dim,
                "blt_multihead_attention: weight_proj must be [embed_dim, embed_dim]");
    BLT_REQUIRE(is_fp32_or_bf16(weight_proj->dtype), "blt_multihead_attention: weight_proj must be FP32 or BF16");
    blt_check_nd_fp32(output, 2, (const size_t[]){seq_len, embed_dim},
                      "blt_multihead_attention: output must be [seq_len, embed_dim] FP32");

    size_t head_dim = (config->head_dim != 0) ? config->head_dim : (embed_dim / num_heads);
    BLT_REQUIRE(head_dim * num_heads == embed_dim,
                "blt_multihead_attention: head_dim * num_heads must equal embed_dim");

    if (config->use_rope) {
        BLT_REQUIRE(head_dim % 2 == 0, "blt_multihead_attention: RoPE requires an even head_dim");
        BLT_REQUIRE(config->rope_theta > 0.0f,
                    "blt_multihead_attention: rope_theta must be positive when use_rope is true");
    }

    *out_seq_len = seq_len;
    *out_head_dim = head_dim;
}

static const float *build_mask(const blt_attention_config *config, size_t seq_len, blt_arena *arena) {
    if (config->mask_config == NULL) {
        return NULL;
    }

    BLT_REQUIRE(config->mask_config->seq_len_q == seq_len && config->mask_config->seq_len_kv == seq_len,
                "blt_multihead_attention: mask_config seq_len_q/seq_len_kv must match input seq_len ");

    blt_tensor mask_tensor = {0};
    blt_build_attention_mask(config->mask_config, &mask_tensor, arena);

    return (const float *)mask_tensor.data;
}

static void apply_rope_to_all_heads(float *qkv_data, size_t seq_len, size_t embed_dim, size_t num_heads,
                                    size_t head_dim, const blt_tensor *rope_cos, const blt_tensor *rope_sin,
                                    float *q_head_buf, float *q_rot_buf, float *k_head_buf, float *k_rot_buf,
                                    blt_backend backend) {
    size_t qkv_stride = 3 * embed_dim;

    for (size_t head = 0; head < num_heads; ++head) {
        size_t q_offset = head * head_dim;
        size_t k_offset = embed_dim + head * head_dim;

        // Fused RoPE on packed layout: eliminates strided_copy + rope_apply + strided_copy
        blt_rope_apply_packed(qkv_data, qkv_stride, q_offset, seq_len, head_dim, (const float *)rope_cos->data,
                              (const float *)rope_sin->data, backend);
        blt_rope_apply_packed(qkv_data, qkv_stride, k_offset, seq_len, head_dim, (const float *)rope_cos->data,
                              (const float *)rope_sin->data, backend);
    }
}

static void attention_head(blt_backend backend, const float *qkv_data, size_t head_idx, size_t seq_len,
                           size_t embed_dim, size_t head_dim, bool is_causal, const float *mask, float scale,
                           float *scores_buf, float *lse_buf, float *combined_out) {
    size_t qkv_stride = 3 * embed_dim;
    size_t q_offset = head_idx * head_dim;
    size_t k_offset = embed_dim + head_idx * head_dim;
    size_t v_offset = 2 * embed_dim + head_idx * head_dim;

    blt_attention_head_args a;
    memset(&a, 0, sizeof(a));
    a.q = qkv_data + q_offset;
    a.q_stride = qkv_stride;
    a.k = qkv_data + k_offset;
    a.k_stride = qkv_stride;
    a.v = qkv_data + v_offset;
    a.v_stride = qkv_stride;
    a.combined = combined_out;
    a.combined_stride = embed_dim;
    a.combined_col_offset = q_offset;
    const char *flash_env = getenv("BLT_FLASH_ATTN");
    int flash_on = (flash_env && flash_env[0] == '1');
    if (flash_on) {
        a.weights_out = NULL;
        a.lse_out = lse_buf;
    } else {
        a.weights_out = scores_buf;
        a.lse_out = NULL;
    }
    a.mask = mask;
    a.nq = seq_len;
    a.nk = seq_len;
    a.head_dim = head_dim;
    a.is_causal = is_causal;
    a.scale = scale;

    blt_attention_head_core(backend, &a);
}

void blt_multihead_attention(const blt_tensor *input, const blt_tensor *weight_qkv, const blt_tensor *weight_proj,
                             blt_tensor *output, const blt_attention_config *config, blt_arena *arena) {
    size_t seq_len, head_dim;
    validate_attention_call(input, weight_qkv, weight_proj, output, config, arena, &seq_len, &head_dim);

    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    size_t qkv_numel = seq_len * 3 * embed_dim;
    size_t combined_numel = seq_len * embed_dim;
    size_t scores_numel = seq_len * seq_len;
    size_t head_numel = seq_len * head_dim;

    float *qkv_data = (float *)blt_arena_alloc(arena, qkv_numel * sizeof(float), sizeof(float));
    float *combined_data = (float *)blt_arena_alloc(arena, combined_numel * sizeof(float), sizeof(float));
    float *scores_buf = (float *)blt_arena_alloc(arena, scores_numel * sizeof(float), sizeof(float));

    {
        blt_tensor zt;
        blt_tensor_view_2d(&zt, combined_data, combined_numel, 1, input->backend);
        zero_tensor(&zt);
    }

    blt_tensor qkv_tensor;
    blt_tensor_view_2d(&qkv_tensor, qkv_data, seq_len, 3 * embed_dim, input->backend);
    blt_matmul(input, weight_qkv, &qkv_tensor);

    if (config->use_rope) {
        size_t half = head_dim / 2;

        const blt_tensor *rope_cos_t;
        const blt_tensor *rope_sin_t;
        blt_tensor computed_cos_t, computed_sin_t; // fallback

        bool have_cache = (config->rope_cos_cache != NULL && config->rope_sin_cache != NULL);

        if (have_cache) {
            blt_check_nd_fp32(config->rope_cos_cache, 2, (const size_t[]){seq_len, half},
                              "blt_multihead_attention: rope_cos_cache must be [seq_len, head_dim/2] FP32");
            blt_check_nd_fp32(config->rope_sin_cache, 2, (const size_t[]){seq_len, half},
                              "blt_multihead_attention: rope_sin_cache must be [seq_len, head_dim/2] FP32");
            rope_cos_t = config->rope_cos_cache;
            rope_sin_t = config->rope_sin_cache;
        } else {
            float *rope_cos_data = (float *)blt_arena_alloc(arena, seq_len * half * sizeof(float), sizeof(float));
            float *rope_sin_data = (float *)blt_arena_alloc(arena, seq_len * half * sizeof(float), sizeof(float));

            blt_tensor_view_2d(&computed_cos_t, rope_cos_data, seq_len, half, input->backend);
            blt_tensor_view_2d(&computed_sin_t, rope_sin_data, seq_len, half, input->backend);

            blt_rope_config rope_config = {.theta = config->rope_theta, .head_dim = head_dim};
            blt_rope_precompute(seq_len, &rope_config, &computed_cos_t, &computed_sin_t);

            rope_cos_t = &computed_cos_t;
            rope_sin_t = &computed_sin_t;
        }

        float *q_head_buf = (float *)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));
        float *k_head_buf = (float *)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));
        float *q_rot_buf = (float *)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));
        float *k_rot_buf = (float *)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));

        apply_rope_to_all_heads(qkv_data, seq_len, embed_dim, num_heads, head_dim, rope_cos_t, rope_sin_t, q_head_buf,
                                q_rot_buf, k_head_buf, k_rot_buf, input->backend);
    }

    const float *mask_data = build_mask(config, seq_len, arena);

    for (size_t head = 0; head < num_heads; ++head) {
        attention_head(input->backend, qkv_data, head, seq_len, embed_dim, head_dim, config->is_causal, mask_data,
                       scale, scores_buf, scores_buf, combined_data);
    }

    blt_tensor combined_tensor;
    blt_tensor_view_2d(&combined_tensor, combined_data, seq_len, embed_dim, input->backend);
    blt_matmul(&combined_tensor, weight_proj, output);
}

typedef struct {
    float *qkv_data;      // [seq_len, 3*embed_dim], post-RoPE
    float *combined_data; // [seq_len, embed_dim]
    float *weights_all;   // [num_heads, seq_len, seq_len] softmax weights (legacy)
    float *lse_all;       // [num_heads, seq_len, 2] LSE (m, l) per row (flash)
} attn_bwd_forward_cache;

static void attn_bwd_recompute_forward(const blt_tensor *input, const blt_tensor *weight_qkv,
                                       const blt_attention_config *config, blt_arena *arena, size_t seq_len,
                                       size_t embed_dim, size_t num_heads, size_t head_dim, float scale,
                                       attn_bwd_forward_cache *cache, const blt_tensor **out_rope_cos,
                                       const blt_tensor **out_rope_sin) {

    size_t qkv_numel = seq_len * 3 * embed_dim;
    size_t combined_numel = seq_len * embed_dim;
    size_t weights_numel = num_heads * seq_len * seq_len;

    cache->qkv_data = (float *)blt_arena_alloc(arena, qkv_numel * sizeof(float), sizeof(float));
    cache->combined_data = (float *)blt_arena_alloc(arena, combined_numel * sizeof(float), sizeof(float));
    cache->weights_all = (float *)blt_arena_alloc(arena, weights_numel * sizeof(float), sizeof(float));
    size_t lse_numel = num_heads * seq_len * 2;
    cache->lse_all = (float *)blt_arena_alloc(arena, lse_numel * sizeof(float), sizeof(float));
    {
        blt_tensor zt;
        blt_tensor_view_2d(&zt, cache->combined_data, combined_numel, 1, input->backend);
        zero_tensor(&zt);
    }

    blt_tensor qkv_tensor;
    blt_tensor_view_2d(&qkv_tensor, cache->qkv_data, seq_len, 3 * embed_dim, input->backend);
    blt_matmul(input, weight_qkv, &qkv_tensor);

    *out_rope_cos = NULL;
    *out_rope_sin = NULL;

    if (config->use_rope) {
        size_t half = head_dim / 2;

        const blt_tensor *rope_cos_t;
        const blt_tensor *rope_sin_t;
        blt_tensor computed_cos_t, computed_sin_t;

        bool have_cache = (config->rope_cos_cache != NULL && config->rope_sin_cache != NULL);
        if (have_cache) {
            blt_check_nd_fp32(config->rope_cos_cache, 2, (const size_t[]){seq_len, half},
                              "blt_multihead_attention_backward: rope_cos_cache must be [seq_len, head_dim/2] FP32");
            blt_check_nd_fp32(config->rope_sin_cache, 2, (const size_t[]){seq_len, half},
                              "blt_multihead_attention_backward: rope_sin_cache must be [seq_len, head_dim/2] FP32");
            rope_cos_t = config->rope_cos_cache;
            rope_sin_t = config->rope_sin_cache;
        } else {
            float *rope_cos_data = (float *)blt_arena_alloc(arena, seq_len * half * sizeof(float), sizeof(float));
            float *rope_sin_data = (float *)blt_arena_alloc(arena, seq_len * half * sizeof(float), sizeof(float));
            blt_tensor_view_2d(&computed_cos_t, rope_cos_data, seq_len, half, input->backend);
            blt_tensor_view_2d(&computed_sin_t, rope_sin_data, seq_len, half, input->backend);
            blt_rope_config rope_config = {.theta = config->rope_theta, .head_dim = head_dim};
            blt_rope_precompute(seq_len, &rope_config, &computed_cos_t, &computed_sin_t);
            rope_cos_t = &computed_cos_t;
            rope_sin_t = &computed_sin_t;
        }
        *out_rope_cos = rope_cos_t;
        *out_rope_sin = rope_sin_t;

        size_t head_numel = seq_len * head_dim;
        float *q_head_buf = (float *)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));
        float *k_head_buf = (float *)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));
        float *q_rot_buf = (float *)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));
        float *k_rot_buf = (float *)blt_arena_alloc(arena, head_numel * sizeof(float), sizeof(float));

        apply_rope_to_all_heads(cache->qkv_data, seq_len, embed_dim, num_heads, head_dim, rope_cos_t, rope_sin_t,
                                q_head_buf, q_rot_buf, k_head_buf, k_rot_buf, input->backend);
    }

    const float *mask_data = build_mask(config, seq_len, arena);

    for (size_t head = 0; head < num_heads; head++) {
        float *W = cache->weights_all + head * seq_len * seq_len;
        attention_head(input->backend, cache->qkv_data, head, seq_len, embed_dim, head_dim, config->is_causal,
                       mask_data, scale, W, cache->lse_all + head * seq_len * 2, cache->combined_data);
    }
}

void blt_multihead_attention_backward(const blt_tensor *input, const blt_tensor *weight_qkv,
                                      const blt_tensor *weight_proj, const blt_tensor *grad_out, blt_tensor *grad_input,
                                      blt_tensor *grad_weight_qkv, blt_tensor *grad_weight_proj,
                                      const blt_attention_config *config, blt_arena *arena) {
    BLT_REQUIRE(input != NULL && weight_qkv != NULL && weight_proj != NULL && grad_out != NULL,
                "blt_multihead_attention_backward: input/weight_qkv/weight_proj/grad_out must not be NULL");
    BLT_REQUIRE(config != NULL && arena != NULL, "blt_multihead_attention_backward: config and arena must not be NULL");

    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;
    BLT_REQUIRE(num_heads != 0 && embed_dim % num_heads == 0,
                "blt_multihead_attention_backward: embed_dim must be divisible by num_heads");
    size_t head_dim = (config->head_dim != 0) ? config->head_dim : (embed_dim / num_heads);
    BLT_REQUIRE(head_dim * num_heads == embed_dim,
                "blt_multihead_attention_backward: head_dim * num_heads must equal embed_dim");

    blt_check_nd_fp32(input, 2, (const size_t[]){0, embed_dim},
                      "blt_multihead_attention_backward: input must be [seq_len, embed_dim] FP32");
    size_t seq_len = input->shape[0];
    BLT_REQUIRE(weight_qkv != NULL && weight_qkv->ndim == 2 && weight_qkv->shape[0] == embed_dim &&
                    weight_qkv->shape[1] == 3 * embed_dim,
                "blt_multihead_attention_backward: weight_qkv must be [embed_dim, 3*embed_dim]");
    BLT_REQUIRE(is_fp32_or_bf16(weight_qkv->dtype),
                "blt_multihead_attention_backward: weight_qkv must be FP32 or BF16");
    BLT_REQUIRE(weight_proj != NULL && weight_proj->ndim == 2 && weight_proj->shape[0] == embed_dim &&
                    weight_proj->shape[1] == embed_dim,
                "blt_multihead_attention_backward: weight_proj must be [embed_dim, embed_dim]");
    BLT_REQUIRE(is_fp32_or_bf16(weight_proj->dtype),
                "blt_multihead_attention_backward: weight_proj must be FP32 or BF16");
    blt_check_nd_fp32(grad_out, 2, (const size_t[]){seq_len, embed_dim},
                      "blt_multihead_attention_backward: grad_out must be [seq_len, embed_dim] FP32");

    float scale = 1.0f / sqrtf((float)head_dim);

    attn_bwd_forward_cache cache = {0};
    const blt_tensor *rope_cos_t = NULL;
    const blt_tensor *rope_sin_t = NULL;
    attn_bwd_recompute_forward(input, weight_qkv, config, arena, seq_len, embed_dim, num_heads, head_dim, scale, &cache,
                               &rope_cos_t, &rope_sin_t);

    blt_tensor combined_tensor;
    blt_tensor_view_2d(&combined_tensor, cache.combined_data, seq_len, embed_dim, input->backend);

    float *grad_combined_data = (float *)blt_arena_alloc(arena, seq_len * embed_dim * sizeof(float), sizeof(float));
    blt_tensor grad_combined_tensor;
    blt_tensor_view_2d(&grad_combined_tensor, grad_combined_data, seq_len, embed_dim, input->backend);

    blt_matmul_backward(&combined_tensor, weight_proj, grad_out, &grad_combined_tensor, grad_weight_proj);

    size_t qkv_stride = 3 * embed_dim;
    float *grad_qkv_data = (float *)blt_arena_alloc(arena, seq_len * qkv_stride * sizeof(float), sizeof(float));
    {
        blt_tensor zt;
        blt_tensor_view_2d(&zt, grad_qkv_data, seq_len * qkv_stride, 1, input->backend);
        zero_tensor(&zt);
    }

    float *grad_scores_buf = (float *)blt_arena_alloc(arena, seq_len * seq_len * sizeof(float), sizeof(float));
    const float *mask_data = build_mask(config, seq_len, arena);

    for (size_t head = 0; head < num_heads; head++) {
        size_t q_offset = head * head_dim;
        size_t k_offset = embed_dim + head * head_dim;
        size_t v_offset = 2 * embed_dim + head * head_dim;

        blt_attention_head_bwd_args a;
        memset(&a, 0, sizeof(a));
        a.q = cache.qkv_data + q_offset;
        a.q_stride = qkv_stride;
        a.k = cache.qkv_data + k_offset;
        a.k_stride = qkv_stride;
        a.v = cache.qkv_data + v_offset;
        a.v_stride = qkv_stride;
        const char *flash_env2 = getenv("BLT_FLASH_ATTN");
        int flash_on2 = (flash_env2 && flash_env2[0] == '1');
        if (flash_on2) {
            a.weights = cache.weights_all + head * seq_len * seq_len;
            a.lse = cache.lse_all + head * seq_len * 2;
        } else {
            a.weights = cache.weights_all + head * seq_len * seq_len;
            a.lse = NULL;
        }
        a.mask = (const float *)mask_data; // may be NULL
        a.is_causal = config->is_causal;
        a.grad_combined = grad_combined_data;
        a.gc_stride = embed_dim;
        a.gc_col_offset = q_offset;
        a.scores_scratch = grad_scores_buf;
        a.grad_q = grad_qkv_data + q_offset;
        a.gq_stride = qkv_stride;
        a.grad_k = grad_qkv_data + k_offset;
        a.gk_stride = qkv_stride;
        a.grad_v = grad_qkv_data + v_offset;
        a.gv_stride = qkv_stride;
        a.nq = seq_len;
        a.nk = seq_len;
        a.head_dim = head_dim;
        a.scale = scale;

        blt_attention_head_core_backward(input->backend, &a);
    }

    if (config->use_rope) {
        size_t qkv_stride = 3 * embed_dim;
        for (size_t head = 0; head < num_heads; head++) {
            size_t q_offset = head * head_dim;
            size_t k_offset = embed_dim + head * head_dim;

            // Fused RoPE backward on packed layout - no intermediate buffers
            blt_rope_apply_packed_backward(grad_qkv_data, qkv_stride, q_offset, seq_len, head_dim,
                                           (const float *)rope_cos_t->data, (const float *)rope_sin_t->data,
                                           input->backend);
            blt_rope_apply_packed_backward(grad_qkv_data, qkv_stride, k_offset, seq_len, head_dim,
                                           (const float *)rope_cos_t->data, (const float *)rope_sin_t->data,
                                           input->backend);
        }
    }

    blt_tensor grad_qkv_tensor;
    blt_tensor_view_2d(&grad_qkv_tensor, grad_qkv_data, seq_len, 3 * embed_dim, input->backend);
    blt_matmul_backward(input, weight_qkv, &grad_qkv_tensor, grad_input, grad_weight_qkv);
}