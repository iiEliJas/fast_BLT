#include "models/local_encoder.h"

#include "core/backend.h"
#include "ops/vecmath.h"
#include "core/tensor.h"
#include "core/allocator.h"
#include "models/attention.h"
#include "models/transformer.h"
#include "models/transformer_stack.h"
#include "models/byte_embedding.h"
#include "models/cross_attention.h"
#include "ops/elementwise.h"
#include "ops/matmul.h"
#include "ops/rmsnorm.h"
#include "ops/swiglu.h"
#include "ops/rope.h"
#include "ops/mask_builder.h"
#include "ops/patch_pool.h"

#include <string.h>

static blt_byte_embedding byte_embedding_view(const blt_local_encoder *model) {
    blt_byte_embedding emb;
    emb.vocab_size = 256;
    emb.weight = model->byte_embedding_weight;
    emb.embed_dim = model->config.embed_dim;
    return emb;
}

static void validate_call(const blt_local_encoder *model, const blt_tensor *bytes_in, const blt_patch_info *patches,
                          size_t num_patches, const blt_arena *arena, size_t *out_seq_len) {
    BLT_REQUIRE(model != NULL && bytes_in != NULL && arena != NULL,
                "blt_local_encoder: model, bytes_in and arena cannot be NULL");
    BLT_REQUIRE(patches != NULL && num_patches >= 1, "blt_local_encoder: need at least one patch");
    BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8,
                "blt_local_encoder: bytes_in must be 1D UINT8 [seq_len]");
    size_t seq_len = bytes_in->shape[0];
    BLT_REQUIRE(seq_len >= 1 && seq_len <= model->config.max_seq_len,
                "blt_local_encoder: seq_len must be in [1, max_seq_len]");
    *out_seq_len = seq_len;
}

// Forward and backward both need these things, so build them once here
typedef struct {
    blt_mask_config local_mask; // byte self-attention: block-causal, window w_E
    blt_mask_config cross_mask; // patch byte cross-attention: block-diagonal
    blt_tensor rope_cos_view;
    blt_tensor rope_sin_view;
} le_context;

static le_context make_context(const blt_local_encoder *model, size_t seq_len, const blt_patch_info *patches,
                               size_t num_patches, const size_t *doc_boundaries, size_t num_docs, blt_arena *arena) {
    // Group-id arrays are host metadata: the patcher-side builder writes
    // them and the mask builder uploads them from host pointers, so they
    // must not land in device memory when arena is a device arena.
    size_t *query_group_ids = (size_t *)blt_container_alloc(arena, num_patches * sizeof(size_t));
    size_t *kv_group_ids = (size_t *)blt_container_alloc(arena, seq_len * sizeof(size_t));
    blt_patch_build_group_ids(patches, num_patches, seq_len, query_group_ids, kv_group_ids);

    le_context ctx = {0};

    // sliding window w_E (0 == full causal), causal
    ctx.local_mask.seq_len_q = seq_len;
    ctx.local_mask.seq_len_kv = seq_len;
    ctx.local_mask.sliding_window = model->config.local_window;
    ctx.local_mask.doc_boundaries = doc_boundaries;
    ctx.local_mask.num_docs = num_docs;
    ctx.local_mask.is_causal = true;

    ctx.cross_mask.seq_len_q = num_patches;
    ctx.cross_mask.seq_len_kv = seq_len;
    ctx.cross_mask.query_group_ids = query_group_ids;
    ctx.cross_mask.kv_group_ids = kv_group_ids;
    ctx.cross_mask.bidirectional_within_group = true;

    size_t head_dim = model->config.embed_dim / model->config.num_heads;
    size_t half = head_dim / 2;
    blt_tensor_view_2d(&ctx.rope_cos_view, model->rope_cos_cache.data, seq_len, half, model->rope_cos_cache.backend);
    blt_tensor_view_2d(&ctx.rope_sin_view, model->rope_sin_cache.data, seq_len, half, model->rope_sin_cache.backend);

    return ctx;
}

static blt_cross_attention_config make_cross_attn_config(const blt_local_encoder_config *config,
                                                         const le_context *ctx) {
    blt_cross_attention_config c = {0};
    c.embed_dim = config->embed_dim;
    c.num_heads = config->cross_attn_heads;
    c.mask_config = &ctx->cross_mask;
    return c;
}

// K-split (SPLIT_QUERY): expands each patch's query group id to its k
// sub-token rows [num_patches*k, E] and points cross_config->mask_config at
// an overridden copy stored in *split_mask_cfg (caller keeps it alive).
static void setup_query_split_mask(blt_cross_attention_config *cross_config, blt_mask_config *split_mask_cfg,
                                   size_t num_patches, size_t k, const le_context *ctx, blt_arena *arena) {
    if (k == 1) {
        return;
    }
    // patch structure is fixed across all layers so build once
    size_t *expanded_query_group_ids = blt_arena_alloc(arena, num_patches * k * sizeof(size_t), 64);
    blt_patch_expand_group_ids(ctx->cross_mask.query_group_ids, num_patches, k, expanded_query_group_ids);

    *split_mask_cfg = *cross_config->mask_config;
    split_mask_cfg->seq_len_q = num_patches * k;
    split_mask_cfg->query_group_ids = expanded_query_group_ids;
    cross_config->mask_config = split_mask_cfg;
}

blt_local_encoder *blt_local_encoder_create(blt_arena *arena, const blt_local_encoder_config *config) {
    BLT_REQUIRE(arena != NULL && config != NULL, "blt_local_encoder_create: arena and config cannot be NULL");

    size_t E = config->embed_dim;
    size_t hidden = config->hidden_dim;
    BLT_REQUIRE(E > 0 && hidden > 0 && config->num_layers >= 1 && config->max_seq_len > 0,
                "blt_local_encoder_create: embed_dim/hidden_dim/num_layers/max_seq_len must be positive");
    BLT_REQUIRE(config->num_heads != 0 && E % config->num_heads == 0,
                "blt_local_encoder_create: embed_dim must be divisible by num_heads");
    size_t head_dim = E / config->num_heads;
    BLT_REQUIRE(head_dim % 2 == 0, "blt_local_encoder_create: RoPE requires an even head_dim");
    BLT_REQUIRE(config->cross_attn_heads != 0 && E % config->cross_attn_heads == 0,
                "blt_local_encoder_create: embed_dim must be divisible by cross_attn_heads");
    BLT_REQUIRE(config->ngram_config.embed_dim == E,
                "blt_local_encoder_create: ngram_config.embed_dim must equal embed_dim");
    BLT_REQUIRE(config->rope_theta > 0.0f, "blt_local_encoder_create: rope_theta must be positive");

    BLT_REQUIRE(config->patch_dim % config->embed_dim == 0,
                "blt_local_encoder_create: patch_dim must be divisible by embed_dim");

    blt_local_encoder *m = (blt_local_encoder *)blt_container_alloc(arena, sizeof(blt_local_encoder));
    m->config = *config;

    size_t emb_shape[2] = {256, E};
    m->byte_embedding_weight = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);

    m->ngram_weights = blt_hash_ngram_create(arena, &config->ngram_config);

    size_t half = head_dim / 2;
    size_t rope_shape[2] = {config->max_seq_len, half};
    m->rope_cos_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    m->rope_sin_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    blt_rope_config rope_cfg = {.theta = config->rope_theta, .head_dim = head_dim};
    blt_rope_precompute(config->max_seq_len, &rope_cfg, &m->rope_cos_cache, &m->rope_sin_cache);

    m->layers = (blt_local_encoder_layer_storage *)blt_container_alloc(
        arena, config->num_layers * sizeof(blt_local_encoder_layer_storage));

    for (size_t l = 0; l < config->num_layers; ++l) {
        blt_local_layer_storage_alloc(arena, &m->layers[l], E, hidden);
    }

    m->layers_bf16 = NULL;
    if (config->use_bf16) {
        m->layers_bf16 =
            (blt_local_layer_bf16 *)blt_container_alloc(arena, config->num_layers * sizeof(blt_local_layer_bf16));
        for (size_t l = 0; l < config->num_layers; ++l) {
            blt_local_layer_bf16_alloc(arena, &m->layers_bf16[l], E, hidden);
        }
    }
    return m;
}

void blt_local_encoder_refresh_bf16(blt_local_encoder *model) {
    if (model == NULL || !model->config.use_bf16 || model->layers_bf16 == NULL) {
        return;
    }
    for (size_t l = 0; l < model->config.num_layers; ++l) {
        blt_local_layer_bf16_refresh(&model->layers_bf16[l], &model->layers[l]);
    }
}

blt_local_encoder_grad *blt_local_encoder_grad_create(blt_arena *arena, const blt_local_encoder *model) {
    BLT_REQUIRE(arena != NULL && model != NULL, "blt_local_encoder_grad_create: arena and model cannot be NULL");

    size_t E = model->config.embed_dim;
    size_t hidden = model->config.hidden_dim;
    size_t num_layers = model->config.num_layers;

    blt_local_encoder_grad *g = (blt_local_encoder_grad *)blt_container_alloc(arena, sizeof(blt_local_encoder_grad));

    size_t emb_shape[2] = {256, E};
    g->embedding_grad = blt_tensor_create(arena, emb_shape, 2, BLT_DTYPE_FP32);

    g->ngram_grads.num_tables = model->ngram_weights.num_tables;
    for (size_t t = 0; t < model->ngram_weights.num_tables; ++t) {
        size_t vocab = model->ngram_weights.tables[t].shape[0];
        size_t table_shape[2] = {vocab, E};
        g->ngram_grads.tables[t] = blt_tensor_create(arena, table_shape, 2, BLT_DTYPE_FP32);
    }

    g->layer_grads =
        (blt_local_encoder_layer_grad *)blt_container_alloc(arena, num_layers * sizeof(blt_local_encoder_layer_grad));

    for (size_t l = 0; l < num_layers; ++l) {
        blt_local_layer_grad_alloc(arena, &g->layer_grads[l], E, hidden);
    }
    return g;
}

// Forward path (BLT 3.2):
// 1. e_0     = byte_embedding(bytes) + hash_ngram_embeddings(bytes)
// 2. P_0     = pool(e_0, patches)
// 3. for l in 1..num_layers:
//      h_l = byte_transformer_layer_l(h_{l-1})     (local block-causal, window w_E)
//      if cross_attn_fires(l): P_l = P_{l-1} + cross_attn_l(RMSNorm(P_{l-1}), h_l)
//      else:                   P_l = P_{l-1}
// Output: P_final -> global patch transformer
//         h_final -> kept for the decoder (FBLT paper)

void blt_local_encoder_forward(const blt_local_encoder *model, const blt_tensor *bytes_in,
                               const blt_patch_info *patches, size_t num_patches, const size_t *doc_boundaries,
                               size_t num_docs, blt_tensor *patch_out, blt_tensor *byte_hidden_out, blt_arena *arena) {
    size_t seq_len;
    validate_call(model, bytes_in, patches, num_patches, arena, &seq_len);
    BLT_REQUIRE(patch_out != NULL && byte_hidden_out != NULL,
                "blt_local_encoder_forward: patch_out and byte_hidden_out cannot be NULL");

    const blt_local_encoder_config *config = &model->config;
    size_t E = config->embed_dim;

    size_t patch_dim = (config->patch_dim == 0) ? E : config->patch_dim;
    BLT_REQUIRE(patch_dim % E == 0, "blt_local_encoder_forward: patch_dim must be a multiple of embed_dim");
    size_t k = patch_dim / E;

    blt_check_nd_fp32(patch_out, 2, (const size_t[]){num_patches, patch_dim},
                      "blt_local_encoder_forward: patch_out must be [num_patches, patch_dim] FP32");
    blt_check_nd_fp32(byte_hidden_out, 2, (const size_t[]){seq_len, E},
                      "blt_local_encoder_forward: byte_hidden_out must be [seq_len, embed_dim] FP32");

    le_context ctx = make_context(model, seq_len, patches, num_patches, doc_boundaries, num_docs, arena);
    blt_transformer_config layer_config =
        blt_local_byte_layer_config(config->embed_dim, config->num_heads, config->rope_theta, config->hidden_dim,
                                    &ctx.local_mask, &ctx.rope_cos_view, &ctx.rope_sin_view);
    blt_cross_attention_config cross_config = make_cross_attn_config(config, &ctx);
    cross_config.embed_dim = E;
    cross_config.patch_dim = patch_dim;
    cross_config.split_mode = (k == 1) ? BLT_CROSS_ATTN_NO_SPLIT : BLT_CROSS_ATTN_SPLIT_QUERY;

    blt_mask_config split_mask_cfg;
    setup_query_split_mask(&cross_config, &split_mask_cfg, num_patches, k, &ctx, arena);

    size_t byte_shape[2] = {seq_len, E};
    size_t patch_shape[2] = {num_patches, patch_dim};

    // e_0 = byte_embedding + hash_ngram_embeddings
    blt_byte_embedding emb = byte_embedding_view(model);
    blt_tensor byte_emb = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_byte_embedding_forward(&emb, bytes_in, &byte_emb);

    blt_tensor h = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_hash_ngram_forward(&model->ngram_weights, &config->ngram_config, bytes_in, &byte_emb, &h);

    // P_0 = pool(e_0, patches)
    blt_tensor pooled = blt_tensor_create(arena, (size_t[]){num_patches, E}, 2, BLT_DTYPE_FP32);
    blt_patch_pool_forward(&h, patches, num_patches, config->pool_type, &pooled);

    blt_tensor p = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    for (size_t sj = 0; sj < k; sj++) {
        // pooled row j lands at column block sj of output row j
        blt_strided_copy(pooled.backend, (float *)p.data + sj * E, patch_dim, (const float *)pooled.data, E,
                         num_patches, E);
    }

    // layer loop
    for (size_t l = 0; l < config->num_layers; ++l) {
        const blt_local_encoder_layer_storage *s = &model->layers[l];
        const blt_local_layer_bf16 *b = model->config.use_bf16 ? &model->layers_bf16[l] : NULL;

        blt_transformer_weights w = blt_local_byte_weights_view_mixed(s, b);
        blt_tensor h_next = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
        blt_transformer_forward(&h, &w, &h_next, &layer_config, arena);
        h = h_next;

        if (blt_local_cross_attn_fires(config->cross_attn_placement, config->cross_attn_all_layers, config->num_layers,
                                       l)) {
            blt_tensor p_view;
            blt_tensor_view_2d(&p_view, p.data, num_patches * k, E, p.backend);

            blt_tensor query_view = blt_tensor_create(arena, (size_t[]){num_patches * k, E}, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&p_view, &s->cross_norm_weight, &query_view);

            blt_cross_attention_weights xw = blt_local_cross_weights_view_mixed(s, b);
            blt_tensor cross_out_split = blt_tensor_create(arena, (size_t[]){num_patches * k, E}, 2, BLT_DTYPE_FP32);
            blt_cross_attention_forward(&query_view, &h, &xw, &cross_out_split, &cross_config, arena);

            blt_tensor cross_out_concat;
            blt_tensor_view_2d(&cross_out_concat, cross_out_split.data, num_patches, patch_dim,
                               cross_out_split.backend);

            blt_tensor p_next = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_add(&p, &cross_out_concat, &p_next);
            p = p_next;
        }
    }

    blt_strided_copy(p.backend, (float *)patch_out->data, num_patches * patch_dim, (const float *)p.data,
                     num_patches * patch_dim, 1, num_patches * patch_dim);
    blt_strided_copy(h.backend, (float *)byte_hidden_out->data, seq_len * E, (const float *)h.data, seq_len * E, 1,
                     seq_len * E);
}

// Backward path -- mirrors forwards call order in reverse.
// Forward doesn't save intermediates, so each layer's forward is
// recomputed here with caching.
typedef struct {
    bool has_cross;
    blt_tensor p_in;
    blt_tensor normed_p;
    blt_transformer_layer_cache *byte_cache;
} layer_cache;

void blt_local_encoder_backward(const blt_local_encoder *model, const blt_tensor *bytes_in,
                                const blt_patch_info *patches, size_t num_patches, const size_t *doc_boundaries,
                                size_t num_docs, const blt_tensor *grad_patch_out,
                                const blt_tensor *grad_byte_hidden_out, blt_local_encoder_grad *grad,
                                blt_arena *arena) {
    size_t seq_len;
    validate_call(model, bytes_in, patches, num_patches, arena, &seq_len);
    BLT_REQUIRE(grad_patch_out != NULL && grad != NULL && grad->layer_grads != NULL,
                "blt_local_encoder_backward: grad_patch_out and grad cannot be NULL");

    const blt_local_encoder_config *config = &model->config;
    const size_t E = config->embed_dim;
    const size_t L = config->num_layers;

    size_t patch_dim = (config->patch_dim == 0) ? E : config->patch_dim;
    BLT_REQUIRE(patch_dim % E == 0, "blt_local_encoder_backward: patch_dim must be a multiple of embed_dim");
    size_t k = patch_dim / E;

    blt_check_nd_fp32(grad_patch_out, 2, (const size_t[]){num_patches, patch_dim},
                      "blt_local_encoder_backward: grad_patch_out must be [num_patches, patch_dim] FP32");
    if (grad_byte_hidden_out != NULL) {
        blt_check_nd_fp32(grad_byte_hidden_out, 2, (const size_t[]){seq_len, E},
                          "blt_local_encoder_backward: grad_byte_hidden_out must be [seq_len, embed_dim] FP32");
    }
    blt_check_nd_fp32(&grad->embedding_grad, 2, (const size_t[]){256, E},
                      "blt_local_encoder_backward: embedding_grad must be [256, embed_dim] FP32");

    le_context ctx = make_context(model, seq_len, patches, num_patches, doc_boundaries, num_docs, arena);
    blt_transformer_config layer_config =
        blt_local_byte_layer_config(config->embed_dim, config->num_heads, config->rope_theta, config->hidden_dim,
                                    &ctx.local_mask, &ctx.rope_cos_view, &ctx.rope_sin_view);
    blt_cross_attention_config cross_config = make_cross_attn_config(config, &ctx);
    cross_config.embed_dim = E;
    cross_config.patch_dim = patch_dim;
    cross_config.split_mode = (k == 1) ? BLT_CROSS_ATTN_NO_SPLIT : BLT_CROSS_ATTN_SPLIT_QUERY;

    blt_mask_config split_mask_cfg;
    setup_query_split_mask(&cross_config, &split_mask_cfg, num_patches, k, &ctx, arena);

    size_t byte_shape[2] = {seq_len, E};
    size_t patch_shape[2] = {num_patches, patch_dim};

    // Recompute forward, caching layer intermediates
    //   h[0..L] - byte states entering each layer (h[0] = e_0)
    //   caches[] - per-layer FFN/norm intermediates + cross-attn inputs
    blt_byte_embedding emb = byte_embedding_view(model);
    blt_tensor byte_emb = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_byte_embedding_forward(&emb, bytes_in, &byte_emb);

    blt_tensor *h = (blt_tensor *)blt_container_alloc(arena, (L + 1) * sizeof(blt_tensor));
    layer_cache *caches = (layer_cache *)blt_container_alloc(arena, L * sizeof(layer_cache));
    memset(caches, 0, L * sizeof(layer_cache));

    h[0] = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_hash_ngram_forward(&model->ngram_weights, &config->ngram_config, bytes_in, &byte_emb, &h[0]);

    blt_tensor pooled = blt_tensor_create(arena, (size_t[]){num_patches, E}, 2, BLT_DTYPE_FP32);
    blt_patch_pool_forward(&h[0], patches, num_patches, config->pool_type, &pooled);

    blt_tensor p = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    for (size_t sj = 0; sj < k; sj++) {
        blt_strided_copy(pooled.backend, (float *)p.data + sj * E, patch_dim, (const float *)pooled.data, E,
                         num_patches, E);
    }

    for (size_t l = 0; l < L; ++l) {
        const blt_local_encoder_layer_storage *s = &model->layers[l];
        layer_cache *c = &caches[l];
        const blt_local_layer_bf16 *b = model->config.use_bf16 ? &model->layers_bf16[l] : NULL;

        blt_transformer_weights w = blt_local_byte_weights_view_mixed(s, b);
        c->byte_cache = blt_transformer_layer_forward_cached(&h[l], &w, &layer_config, arena, seq_len, E,
                                                             config->hidden_dim, &h[l + 1]);

        if (blt_local_cross_attn_fires(config->cross_attn_placement, config->cross_attn_all_layers, config->num_layers,
                                       l)) {
            c->has_cross = true;
            c->p_in = p;

            blt_tensor p_in_view;
            blt_tensor_view_2d(&p_in_view, c->p_in.data, num_patches * k, E, c->p_in.backend);

            // Store normed_p internally as the split shape [num_patches*k, E]
            c->normed_p = blt_tensor_create(arena, (size_t[]){num_patches * k, E}, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&p_in_view, &s->cross_norm_weight, &c->normed_p);

            blt_cross_attention_weights xw = blt_local_cross_weights_view_mixed(s, b);
            blt_tensor cross_out_split = blt_tensor_create(arena, (size_t[]){num_patches * k, E}, 2, BLT_DTYPE_FP32);
            blt_cross_attention_forward(&c->normed_p, &h[l + 1], &xw, &cross_out_split, &cross_config, arena);

            blt_tensor cross_out_concat;
            blt_tensor_view_2d(&cross_out_concat, cross_out_split.data, num_patches, patch_dim,
                               cross_out_split.backend);

            blt_tensor p_next = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_add(&c->p_in, &cross_out_concat, &p_next);
            p = p_next;
        }
    }

    // Terminal gradients
    //   dh — running dL/d(h_{l+1}), starts at dL/d(h_final)
    //   dP — running dL/d(P_{l+1}), starts at dL/d(P_final)
    blt_tensor dh = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    if (grad_byte_hidden_out != NULL) {
        blt_tensor_upload(&dh, grad_byte_hidden_out->data, seq_len * E * sizeof(float));
    }

    blt_tensor dP = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_tensor_upload(&dP, grad_patch_out->data, num_patches * patch_dim * sizeof(float));

    // Reverse layer loop
    for (size_t li = L; li-- > 0;) {
        const blt_local_encoder_layer_storage *s = &model->layers[li];
        blt_local_encoder_layer_grad *lg = &grad->layer_grads[li];
        const layer_cache *c = &caches[li];
        const blt_local_layer_bf16 *b = model->config.use_bf16 ? &model->layers_bf16[li] : NULL;

        if (c->has_cross) {
            blt_cross_attention_weights xw = blt_local_cross_weights_view_mixed(s, b);
            blt_cross_attention_grad xg = {0};
            xg.grad_weight_q = lg->cross_weight_q;
            xg.grad_weight_k = lg->cross_weight_k;
            xg.grad_weight_v = lg->cross_weight_v;
            xg.grad_weight_proj = lg->cross_weight_proj;

            blt_tensor grad_out_split;
            blt_tensor_view_2d(&grad_out_split, dP.data, num_patches * k, E, dP.backend);

            blt_tensor grad_normed_p_split =
                blt_tensor_create(arena, (size_t[]){num_patches * k, E}, 2, BLT_DTYPE_FP32);
            blt_tensor grad_kv = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);

            blt_cross_attention_backward(&c->normed_p, &h[li + 1], &xw, &grad_out_split, &grad_normed_p_split, &grad_kv,
                                         &xg, &cross_config, arena);

            blt_tensor dh_new = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_add(&dh, &grad_kv, &dh_new);
            dh = dh_new;

            blt_tensor p_in_view;
            blt_tensor_view_2d(&p_in_view, c->p_in.data, num_patches * k, E, c->p_in.backend);

            blt_tensor grad_p_norm_split = blt_tensor_create(arena, (size_t[]){num_patches * k, E}, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_backward(&grad_normed_p_split, &p_in_view, &s->cross_norm_weight, &grad_p_norm_split,
                                 &lg->cross_norm_weight);

            blt_tensor grad_p_norm;
            blt_tensor_view_2d(&grad_p_norm, grad_p_norm_split.data, num_patches, patch_dim, grad_p_norm_split.backend);

            blt_tensor dP_new = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
            blt_add(&dP, &grad_p_norm, &dP_new);
            dP = dP_new;
        }

        // byte transformer layer backward: dh becomes dL/d(h[li])
        blt_transformer_weights w = blt_local_byte_weights_view_mixed(s, b);
        blt_transformer_layer_grad lgv = blt_local_byte_grad_view(lg);
        blt_tensor dh_next;
        blt_transformer_layer_backward(c->byte_cache, &w, &layer_config, arena, seq_len, E, config->hidden_dim, &dh,
                                       &lgv, &dh_next);
        dh = dh_next;
    }

    // P_0 = pool(h_0) -- scatter pool gradient into h_0 grad
    blt_tensor dP_pooled = blt_tensor_create(arena, (size_t[]){num_patches, E}, 2, BLT_DTYPE_FP32);
    // Sum the k column blocks of each grad row into the pooled gradient.
    for (size_t sj = 0; sj < k; sj++) {
        blt_tensor block_view, acc_view;
        blt_tensor_view_2d(&block_view, (float *)dP.data + sj * E, num_patches, E, dP.backend);
        if (sj == 0) {
            blt_strided_copy(dP.backend, (float *)dP_pooled.data, E, (const float *)block_view.data, patch_dim,
                             num_patches, E);
        } else {
            acc_view = dP_pooled;
            blt_tensor tmp = blt_tensor_create(arena, (size_t[]){num_patches, E}, 2, BLT_DTYPE_FP32);
            blt_strided_copy(dP.backend, (float *)tmp.data, E, (const float *)block_view.data, patch_dim, num_patches,
                             E);
            blt_add(&acc_view, &tmp, &dP_pooled);
        }
    }

    blt_tensor grad_h0_pool = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_patch_pool_backward(&dP_pooled, &h[0], patches, num_patches, config->pool_type, &grad_h0_pool);

    blt_tensor grad_e0 = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_add(&dh, &grad_h0_pool, &grad_e0);

    // e_0 = byte_embedding + hash n-grams
    blt_tensor grad_byte_emb = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_hash_ngram_backward(&config->ngram_config, bytes_in, &grad_e0, &grad_byte_emb, grad->ngram_grads.tables);

    // scatter-adds into grad->embedding_grad (zeroed by grad_create)
    blt_byte_embedding_backward(&emb, bytes_in, &grad_byte_emb, &grad->embedding_grad);
}
