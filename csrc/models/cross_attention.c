#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "models/cross_attention.h"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/allocator.h"
#include "ops/attn_core.h"
#include "ops/matmul.h"
#include "ops/vecmath.h"
#include "ops/elementwise.h"
#include "ops/mask_builder.h"

static void check_cross_weight(const blt_tensor *w, size_t embed_dim, const char *name) {
    BLT_REQUIRE(w != NULL && w->ndim == 2 && w->shape[0] == embed_dim && w->shape[1] == embed_dim,
                "%s must be [embed_dim, embed_dim]", name);
    BLT_REQUIRE(w->dtype == BLT_DTYPE_FP32 || w->dtype == BLT_DTYPE_BF16 || w->dtype == BLT_DTYPE_FP16 ||
                    blt_dtype_is_fp8(w->dtype),
                "%s must be FP32, BF16, FP16 or FP8", name);
}

static void validate_cross_attention_call(const blt_tensor *query_in, const blt_tensor *kv_in,
                                          const blt_cross_attention_weights *weights,
                                          const blt_cross_attention_config *config, const blt_arena *arena,
                                          size_t *out_num_patches, size_t *out_seq_len, size_t *out_head_dim) {
    BLT_REQUIRE(config != NULL && arena != NULL, "blt_cross_attention: config and arena cannot be NULL");
    BLT_REQUIRE(config->mask_config != NULL,
                "blt_cross_attention: mask_config is required (block-diagonal patch mask)");

    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;
    BLT_REQUIRE(num_heads != 0 && embed_dim % num_heads == 0,
                "blt_cross_attention: embed_dim must be divisible by num_heads");

    blt_check_nd_fp32(query_in, 2, (const size_t[]){0, embed_dim},
                      "blt_cross_attention: query_in must be [num_patches, embed_dim] FP32");
    size_t num_patches = query_in->shape[0];

    blt_check_nd_fp32(kv_in, 2, (const size_t[]){0, embed_dim},
                      "blt_cross_attention: kv_in must be [seq_len, embed_dim] FP32");
    size_t seq_len = kv_in->shape[0];

    check_cross_weight(weights->weight_q, embed_dim, "blt_cross_attention: weight_q");
    check_cross_weight(weights->weight_k, embed_dim, "blt_cross_attention: weight_k");
    check_cross_weight(weights->weight_v, embed_dim, "blt_cross_attention: weight_v");
    check_cross_weight(weights->weight_proj, embed_dim, "blt_cross_attention: weight_proj");

    size_t head_dim = (config->head_dim != 0) ? config->head_dim : (embed_dim / num_heads);
    BLT_REQUIRE(head_dim * num_heads == embed_dim, "blt_cross_attention: head_dim * num_heads must equal embed_dim");

    *out_num_patches = num_patches;
    *out_seq_len = seq_len;
    *out_head_dim = head_dim;
}

static const float *build_mask(const blt_cross_attention_config *config, size_t num_patches, size_t seq_len,
                               blt_arena *arena) {
    BLT_REQUIRE(config->mask_config->seq_len_q == num_patches && config->mask_config->seq_len_kv == seq_len,
                "blt_cross_attention: mask_config seq_len_q/seq_len_kv must match "
                "num_patches (query rows) and seq_len (kv rows)");

    blt_tensor mask_tensor = {0};
    blt_build_attention_mask(config->mask_config, &mask_tensor, arena);

    return (const float *)mask_tensor.data;
}

// Head h occupies column block [h*head_dim, (h+1)*head_dim) of each
// [rows, embed_dim] buffer. Queries come from patches, K/V from bytes.
static void cross_attention_head(blt_backend backend, const float *q_data, const float *k_data, const float *v_data,
                                 size_t head_idx, size_t num_patches, size_t seq_len, size_t embed_dim, size_t head_dim,
                                 const float *mask, float scale, float *scores_buf, float *combined_out) {
    BLT_REQUIRE(mask != NULL, "blt_cross_attention: attention mask is required (block-diagonal patch mask)");
    size_t head_offset = head_idx * head_dim;

    blt_attention_head_args a;
    memset(&a, 0, sizeof(a));
    a.q = q_data + head_offset;
    a.q_stride = embed_dim;
    a.k = k_data + head_offset;
    a.k_stride = embed_dim;
    a.v = v_data + head_offset;
    a.v_stride = embed_dim;
    a.combined = combined_out;
    a.combined_stride = embed_dim;
    a.combined_col_offset = head_offset;
    a.weights_out = scores_buf;
    a.mask = mask;
    a.nq = num_patches;
    a.nk = seq_len;
    a.head_dim = head_dim;
    a.is_causal = false; // dense group mask fully defines visibility
    a.scale = scale;

    blt_attention_head_core(backend, &a);
}

// Q/K/V projections, per-head scaled dot-product attention with block-diagonal
// patch mask, then output projection. No positional embeddings (paper-explicit),
// no residual — caller computes P_l = P_{l-1} + output.
void blt_cross_attention_forward(const blt_tensor *query_in, const blt_tensor *kv_in,
                                 const blt_cross_attention_weights *weights, blt_tensor *output,
                                 const blt_cross_attention_config *config, blt_arena *arena) {
    size_t num_patches, seq_len, head_dim;
    validate_cross_attention_call(query_in, kv_in, weights, config, arena, &num_patches, &seq_len, &head_dim);

    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;

    blt_check_nd_fp32(output, 2, (const size_t[]){num_patches, embed_dim},
                      "blt_cross_attention: output must be [num_patches, embed_dim] FP32");

    float scale = 1.0f / sqrtf((float)head_dim);

    // Arena allocations for Q/K/V projections, concatenated multi-head output,
    // and per-head [num_patches, seq_len] score matrix.
    size_t q_numel = num_patches * embed_dim;
    size_t kv_numel = seq_len * embed_dim;
    size_t scores_numel = num_patches * seq_len;

    float *q_data = (float *)blt_arena_alloc(arena, q_numel * sizeof(float), sizeof(float));
    float *k_data = (float *)blt_arena_alloc(arena, kv_numel * sizeof(float), sizeof(float));
    float *v_data = (float *)blt_arena_alloc(arena, kv_numel * sizeof(float), sizeof(float));
    float *combined_data = (float *)blt_arena_alloc(arena, q_numel * sizeof(float), sizeof(float));
    float *scores_buf = (float *)blt_arena_alloc(arena, scores_numel * sizeof(float), sizeof(float));

    {
        blt_tensor zt;
        blt_tensor_view_2d(&zt, combined_data, q_numel, 1, query_in->backend);
        zero_tensor(&zt);
    }

    // Q and K/V come from different inputs, so three separate matmuls.
    blt_tensor q_tensor, k_tensor, v_tensor;
    blt_tensor_view_2d(&q_tensor, q_data, num_patches, embed_dim, query_in->backend);
    blt_tensor_view_2d(&k_tensor, k_data, seq_len, embed_dim, kv_in->backend);
    blt_tensor_view_2d(&v_tensor, v_data, seq_len, embed_dim, kv_in->backend);

    blt_matmul(query_in, weights->weight_q, &q_tensor);
    blt_matmul(kv_in, weights->weight_k, &k_tensor);
    blt_matmul(kv_in, weights->weight_v, &v_tensor);

    // No RoPE in cross attention

    const float *mask_data = build_mask(config, num_patches, seq_len, arena);

    for (size_t head = 0; head < num_heads; ++head) {
        cross_attention_head(query_in->backend, q_data, k_data, v_data, head, num_patches, seq_len, embed_dim, head_dim,
                             mask_data, scale, scores_buf, combined_data);
    }

    blt_tensor combined_tensor;
    blt_tensor_view_2d(&combined_tensor, combined_data, num_patches, embed_dim, query_in->backend);
    blt_matmul(&combined_tensor, weights->weight_proj, output);

    // Arena cleanup is the caller's responsibility.
}

typedef struct {
    float *q_data;        // [num_patches, embed_dim]
    float *k_data;        // [seq_len, embed_dim]
    float *v_data;        // [seq_len, embed_dim]
    float *combined_data; // [num_patches, embed_dim]
    float *weights_all;   // [num_heads, num_patches, seq_len] softmax weights
} attn_bwd_forward_cache;

static void attn_bwd_recompute_forward(const blt_tensor *query_in, const blt_tensor *kv_in,
                                       const blt_cross_attention_weights *weights,
                                       const blt_cross_attention_config *config, blt_arena *arena, size_t num_patches,
                                       size_t seq_len, size_t embed_dim, size_t num_heads, size_t head_dim, float scale,
                                       attn_bwd_forward_cache *cache) {

    size_t q_numel = num_patches * embed_dim;
    size_t kv_numel = seq_len * embed_dim;
    size_t weights_numel = num_heads * num_patches * seq_len;

    cache->q_data = (float *)blt_arena_alloc(arena, q_numel * sizeof(float), sizeof(float));
    cache->k_data = (float *)blt_arena_alloc(arena, kv_numel * sizeof(float), sizeof(float));
    cache->v_data = (float *)blt_arena_alloc(arena, kv_numel * sizeof(float), sizeof(float));
    cache->combined_data = (float *)blt_arena_alloc(arena, q_numel * sizeof(float), sizeof(float));
    cache->weights_all = (float *)blt_arena_alloc(arena, weights_numel * sizeof(float), sizeof(float));
    {
        blt_tensor zt;
        blt_tensor_view_2d(&zt, cache->combined_data, q_numel, 1, query_in->backend);
        zero_tensor(&zt);
    }

    blt_tensor q_tensor, k_tensor, v_tensor;
    blt_tensor_view_2d(&q_tensor, cache->q_data, num_patches, embed_dim, query_in->backend);
    blt_tensor_view_2d(&k_tensor, cache->k_data, seq_len, embed_dim, kv_in->backend);
    blt_tensor_view_2d(&v_tensor, cache->v_data, seq_len, embed_dim, kv_in->backend);

    blt_matmul(query_in, weights->weight_q, &q_tensor);
    blt_matmul(kv_in, weights->weight_k, &k_tensor);
    blt_matmul(kv_in, weights->weight_v, &v_tensor);

    const float *mask_data = build_mask(config, num_patches, seq_len, arena);

    // Each head's slice of weights_all doubles as the scores buffer;
    // cross_attention_head writes softmaxed weights into it.
    for (size_t head = 0; head < num_heads; head++) {
        float *W = cache->weights_all + head * num_patches * seq_len;
        cross_attention_head(query_in->backend, cache->q_data, cache->k_data, cache->v_data, head, num_patches, seq_len,
                             embed_dim, head_dim, mask_data, scale, W, cache->combined_data);
    }
}

void blt_cross_attention_backward(const blt_tensor *query_in, const blt_tensor *kv_in,
                                  const blt_cross_attention_weights *weights, const blt_tensor *grad_out,
                                  blt_tensor *grad_query_in, blt_tensor *grad_kv_in,
                                  blt_cross_attention_grad *grad_weights, const blt_cross_attention_config *config,
                                  blt_arena *arena) {

    BLT_REQUIRE(query_in != NULL && kv_in != NULL && weights != NULL && grad_out != NULL && grad_query_in != NULL &&
                    grad_kv_in != NULL && grad_weights != NULL,
                "blt_cross_attention_backward: arguments must not be NULL");
    BLT_REQUIRE(config != NULL && arena != NULL, "blt_cross_attention_backward: config and arena must not be NULL");
    BLT_REQUIRE(config->mask_config != NULL,
                "blt_cross_attention_backward: mask_config is required (block-diagonal patch mask)");

    size_t embed_dim = config->embed_dim;
    size_t num_heads = config->num_heads;
    BLT_REQUIRE(num_heads != 0 && embed_dim % num_heads == 0,
                "blt_cross_attention_backward: embed_dim must be divisible by num_heads");
    size_t head_dim = (config->head_dim != 0) ? config->head_dim : (embed_dim / num_heads);
    BLT_REQUIRE(head_dim * num_heads == embed_dim,
                "blt_cross_attention_backward: head_dim * num_heads must equal embed_dim");

    blt_check_nd_fp32(query_in, 2, (const size_t[]){0, embed_dim},
                      "blt_cross_attention_backward: query_in must be [num_patches, embed_dim] FP32");
    size_t num_patches = query_in->shape[0];
    blt_check_nd_fp32(kv_in, 2, (const size_t[]){0, embed_dim},
                      "blt_cross_attention_backward: kv_in must be [seq_len, embed_dim] FP32");
    size_t seq_len = kv_in->shape[0];

    check_cross_weight(weights->weight_q, embed_dim, "blt_cross_attention_backward: weight_q");
    check_cross_weight(weights->weight_k, embed_dim, "blt_cross_attention_backward: weight_k");
    check_cross_weight(weights->weight_v, embed_dim, "blt_cross_attention_backward: weight_v");
    check_cross_weight(weights->weight_proj, embed_dim, "blt_cross_attention_backward: weight_proj");

    blt_check_nd_fp32(grad_out, 2, (const size_t[]){num_patches, embed_dim},
                      "blt_cross_attention_backward: grad_out must be [num_patches, embed_dim] FP32");
    blt_check_nd_fp32(grad_query_in, 2, (const size_t[]){num_patches, embed_dim},
                      "blt_cross_attention_backward: grad_query_in must be [num_patches, embed_dim] FP32");
    blt_check_nd_fp32(grad_kv_in, 2, (const size_t[]){seq_len, embed_dim},
                      "blt_cross_attention_backward: grad_kv_in must be [seq_len, embed_dim] FP32");
    blt_check_nd_fp32(&grad_weights->grad_weight_q, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: grad_weight_q must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(&grad_weights->grad_weight_k, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: grad_weight_k must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(&grad_weights->grad_weight_v, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: grad_weight_v must be [embed_dim, embed_dim] FP32");
    blt_check_nd_fp32(&grad_weights->grad_weight_proj, 2, (const size_t[]){embed_dim, embed_dim},
                      "blt_cross_attention_backward: grad_weight_proj must be [embed_dim, embed_dim] FP32");

    float scale = 1.0f / sqrtf((float)head_dim);

    // Recompute forward intermediates (activations not kept for cross-attention).
    attn_bwd_forward_cache cache = {0};
    attn_bwd_recompute_forward(query_in, kv_in, weights, config, arena, num_patches, seq_len, embed_dim, num_heads,
                               head_dim, scale, &cache);

    // Backward through output projection: output = combined @ weight_proj
    blt_tensor combined_tensor;
    blt_tensor_view_2d(&combined_tensor, cache.combined_data, num_patches, embed_dim, query_in->backend);

    float *grad_combined_data = (float *)blt_arena_alloc(arena, num_patches * embed_dim * sizeof(float), sizeof(float));
    blt_tensor grad_combined_tensor;
    blt_tensor_view_2d(&grad_combined_tensor, grad_combined_data, num_patches, embed_dim, query_in->backend);

    blt_matmul_backward(&combined_tensor, weights->weight_proj, grad_out, &grad_combined_tensor,
                        &grad_weights->grad_weight_proj);

    // Per-head backward through softmax and QK^T.
    float *grad_q_data = (float *)blt_arena_alloc(arena, num_patches * embed_dim * sizeof(float), sizeof(float));
    float *grad_k_data = (float *)blt_arena_alloc(arena, seq_len * embed_dim * sizeof(float), sizeof(float));
    float *grad_v_data = (float *)blt_arena_alloc(arena, seq_len * embed_dim * sizeof(float), sizeof(float));
    {
        blt_tensor zt;
        blt_tensor_view_2d(&zt, grad_q_data, num_patches * embed_dim, 1, query_in->backend);
        zero_tensor(&zt);
    }
    {
        blt_tensor zt;
        blt_tensor_view_2d(&zt, grad_k_data, seq_len * embed_dim, 1, query_in->backend);
        zero_tensor(&zt);
    }
    {
        blt_tensor zt;
        blt_tensor_view_2d(&zt, grad_v_data, seq_len * embed_dim, 1, query_in->backend);
        zero_tensor(&zt);
    }

    float *grad_scores_buf = (float *)blt_arena_alloc(arena, num_patches * seq_len * sizeof(float), sizeof(float));

    for (size_t head = 0; head < num_heads; head++) {
        size_t head_offset = head * head_dim;

        blt_attention_head_bwd_args a;
        memset(&a, 0, sizeof(a));
        a.q = cache.q_data + head_offset;
        a.q_stride = embed_dim;
        a.k = cache.k_data + head_offset;
        a.k_stride = embed_dim;
        a.v = cache.v_data + head_offset;
        a.v_stride = embed_dim;
        a.weights = cache.weights_all + head * num_patches * seq_len;
        a.grad_combined = grad_combined_data;
        a.gc_stride = embed_dim;
        a.gc_col_offset = head_offset;
        a.scores_scratch = grad_scores_buf;
        a.grad_q = grad_q_data + head_offset;
        a.gq_stride = embed_dim;
        a.grad_k = grad_k_data + head_offset;
        a.gk_stride = embed_dim;
        a.grad_v = grad_v_data + head_offset;
        a.gv_stride = embed_dim;
        a.nq = num_patches;
        a.nk = seq_len;
        a.head_dim = head_dim;
        a.scale = scale;

        blt_attention_head_core_backward(query_in->backend, &a);
    }

    // Backward through input projections.
    blt_tensor grad_q_tensor;
    blt_tensor_view_2d(&grad_q_tensor, grad_q_data, num_patches, embed_dim, query_in->backend);
    blt_matmul_backward(query_in, weights->weight_q, &grad_q_tensor, grad_query_in, &grad_weights->grad_weight_q);

    // K and V share the same input, so grad_kv_in is the sum of both matmul
    // input gradients.
    blt_tensor grad_k_tensor, grad_v_tensor;
    blt_tensor_view_2d(&grad_k_tensor, grad_k_data, seq_len, embed_dim, kv_in->backend);
    blt_tensor_view_2d(&grad_v_tensor, grad_v_data, seq_len, embed_dim, kv_in->backend);

    float *grad_kv_k_data = (float *)blt_arena_alloc(arena, seq_len * embed_dim * sizeof(float), sizeof(float));
    float *grad_kv_v_data = (float *)blt_arena_alloc(arena, seq_len * embed_dim * sizeof(float), sizeof(float));
    blt_tensor grad_kv_k_tensor, grad_kv_v_tensor;
    blt_tensor_view_2d(&grad_kv_k_tensor, grad_kv_k_data, seq_len, embed_dim, kv_in->backend);
    blt_tensor_view_2d(&grad_kv_v_tensor, grad_kv_v_data, seq_len, embed_dim, kv_in->backend);

    blt_matmul_backward(kv_in, weights->weight_k, &grad_k_tensor, &grad_kv_k_tensor, &grad_weights->grad_weight_k);
    blt_matmul_backward(kv_in, weights->weight_v, &grad_v_tensor, &grad_kv_v_tensor, &grad_weights->grad_weight_v);

    blt_add(&grad_kv_k_tensor, &grad_kv_v_tensor, grad_kv_in);

    // Arena cleanup is the caller's responsibility.
}
