#include "models/local_decoder.h"
#include "core/backend.h"
#include "ops/vecmath.h"
#include "ops/gather_scatter.h"
#include "core/tensor.h"
#include "core/allocator.h"
#include "models/attention.h"
#include "models/transformer.h"
#include "models/transformer_stack.h"
#include "models/cross_attention.h"
#include "ops/elementwise.h"
#include "ops/matmul.h"
#include "ops/rmsnorm.h"
#include "ops/swiglu.h"
#include "ops/rope.h"
#include "ops/mask_builder.h"
#include "ops/patch_pool.h"
#include "ops/cross_entropy.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

static void validate_call(const blt_local_decoder *model, const blt_tensor *byte_hidden_in, const blt_tensor *patch_in,
                          const blt_patch_info *patches, size_t num_patches, const blt_tensor *bytes_in,
                          const blt_arena *arena, size_t *out_seq_len) {
    BLT_REQUIRE(model != NULL && byte_hidden_in != NULL && patch_in != NULL && arena != NULL,
                "blt_local_decoder: model, byte_hidden_in, patch_in and arena cannot be NULL");
    BLT_REQUIRE(patches != NULL && num_patches >= 1, "blt_local_decoder: need at least one patch");

    size_t E = model->config.embed_dim;
    blt_check_nd_fp32(byte_hidden_in, 2, (const size_t[]){0, E},
                      "blt_local_decoder: byte_hidden_in must be [seq_len, embed_dim] FP32");

    size_t seq_len = byte_hidden_in->shape[0];
    BLT_REQUIRE(seq_len >= 2 && seq_len <= model->config.max_seq_len,
                "blt_local_decoder: seq_len must be in [2, max_seq_len]");

    size_t patch_dim = (model->config.patch_dim == 0) ? E : model->config.patch_dim;
    blt_check_nd_fp32(patch_in, 2, (const size_t[]){num_patches, patch_dim},
                      "blt_local_decoder: patch_in must be [num_patches, patch_dim] FP32");

    // bytes_in is optional (NULL = logits-only inference); when present it
    // supplies the loss targets and must match the sequence length
    if (bytes_in != NULL) {
        BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8 && bytes_in->shape[0] == seq_len,
                    "blt_local_decoder: bytes_in must be 1D UINT8 [seq_len] matching byte_hidden_in");
    }

    *out_seq_len = seq_len;
}

// Setup context for forward/backward: masks, RoPE views, group ids.
typedef struct {
    blt_mask_config local_mask;
    blt_mask_config cross_mask;
    blt_tensor rope_cos_view;
    blt_tensor rope_sin_view;
    size_t k; // patch_dim / embed_dim split factor (1 = no split)
    size_t patch_dim;
} ld_context;

static ld_context make_context(const blt_local_decoder *model, size_t seq_len, size_t num_hfinal_rows,
                               const blt_patch_info *patches, size_t num_patches, const size_t *doc_boundaries,
                               size_t num_docs, int trailing_closed, blt_arena *arena) {
    BLT_REQUIRE(num_hfinal_rows >= 1 && num_hfinal_rows <= seq_len,
                "blt_local_decoder: num_hfinal_rows must be in [1, seq_len]");
    size_t E = model->config.embed_dim;
    size_t patch_dim = model->config.patch_dim ? model->config.patch_dim : E;
    size_t k = patch_dim / E;

    // patch_identity_ids[j] = j; byte_patch_ids[i] = paper-rule decoder latent
    // index for byte i (Fast-BLT 3.1.1): final byte of patch j -> j, non-final
    // -> (j == 0 ? 0 : j-1), uncovered -> num_patches-1.
    // Group-id arrays are host metadata (built on host, uploaded to the
    // mask builder from host pointers); never allocate them into device
    // memory.
    size_t *patch_identity_ids = (size_t *)blt_container_alloc(arena, num_patches * sizeof(size_t));
    size_t *byte_patch_ids = (size_t *)blt_container_alloc(arena, seq_len * sizeof(size_t));
    blt_patch_build_group_ids(patches, num_patches, num_hfinal_rows, patch_identity_ids, byte_patch_ids);
    for (size_t i = 0; i < seq_len; i++) {
        byte_patch_ids[i] = blt_patch_decoder_latent_at(patches, num_patches, i);
    }
    // When the caller knows a boundary fires at seq_len (a new patch starts
    // there), the last byte IS final and reads its own latent. blt_patch_info
    // tiles [0,seq_len) and cannot express that, so the caller passes the bit.
    if (trailing_closed && num_patches >= 1) {
        byte_patch_ids[seq_len - 1] = num_patches - 1;
    }

    // expand the kv (patch) side group ids by k -- each patch's k sub-tokens share its group id
    size_t *expanded_kv_group_ids = (size_t *)blt_container_alloc(arena, num_patches * k * sizeof(size_t));
    blt_patch_expand_group_ids(patch_identity_ids, num_patches, k, expanded_kv_group_ids);

    ld_context ctx = {0};
    ctx.k = k;
    ctx.patch_dim = patch_dim;

    // Byte self-attention: causal, window w_D
    ctx.local_mask.seq_len_q = seq_len;
    ctx.local_mask.seq_len_kv = seq_len;
    ctx.local_mask.sliding_window = model->config.local_window;
    ctx.local_mask.doc_boundaries = doc_boundaries;
    ctx.local_mask.num_docs = num_docs;
    ctx.local_mask.is_causal = true;

    // Cross mask: bytes are queries, patches are kv, split into k sub-tokens each
    ctx.cross_mask.seq_len_q = seq_len;
    ctx.cross_mask.seq_len_kv = num_patches * k;
    ctx.cross_mask.query_group_ids = byte_patch_ids;
    ctx.cross_mask.kv_group_ids = expanded_kv_group_ids;
    ctx.cross_mask.bidirectional_within_group = true;
    ctx.cross_mask.is_causal = false;

    size_t head_dim = E / model->config.num_heads;
    size_t half = head_dim / 2;
    blt_tensor_view_2d(&ctx.rope_cos_view, model->rope_cos_cache.data, seq_len, half, model->rope_cos_cache.backend);
    blt_tensor_view_2d(&ctx.rope_sin_view, model->rope_sin_cache.data, seq_len, half, model->rope_sin_cache.backend);

    return ctx;
}

static blt_cross_attention_config make_cross_attn_config(const blt_local_decoder_config *config,
                                                         const ld_context *ctx) {
    blt_cross_attention_config c = {0};
    c.embed_dim = config->embed_dim;
    c.patch_dim = config->patch_dim;
    c.split_mode = BLT_CROSS_ATTN_SPLIT_KV;
    c.num_heads = config->cross_attn_heads;
    c.mask_config = &ctx->cross_mask;
    return c;
}

// K-split (SPLIT_KV): the kv-side group ids are already expanded in ctx
// ([num_patches*k] entries, one per sub-token); point cross_config->mask_config
// at an overridden copy stored in *split_mask_cfg (caller keeps it alive).
static void setup_kv_split_mask(blt_cross_attention_config *cross_config, blt_mask_config *split_mask_cfg,
                                size_t num_patches, size_t k, const ld_context *ctx) {
    if (k == 1) {
        return;
    }
    *split_mask_cfg = *cross_config->mask_config;
    split_mask_cfg->seq_len_kv = num_patches * k;
    split_mask_cfg->kv_group_ids = ctx->cross_mask.kv_group_ids;
    cross_config->mask_config = split_mask_cfg;
}

blt_local_decoder *blt_local_decoder_create(blt_arena *arena, const blt_local_decoder_config *config) {
    BLT_REQUIRE(arena != NULL && config != NULL, "blt_local_decoder_create: arena and config cannot be NULL");

    size_t E = config->embed_dim;
    size_t hidden = config->hidden_dim;
    BLT_REQUIRE(E > 0 && hidden > 0 && config->num_layers >= 1 && config->max_seq_len > 0,
                "blt_local_decoder_create: embed_dim/hidden_dim/num_layers/max_seq_len must be positive");
    BLT_REQUIRE(config->num_heads != 0 && E % config->num_heads == 0,
                "blt_local_decoder_create: embed_dim must be divisible by num_heads");
    size_t head_dim = E / config->num_heads;
    BLT_REQUIRE(head_dim % 2 == 0, "blt_local_decoder_create: RoPE requires an even head_dim");
    BLT_REQUIRE(config->cross_attn_heads != 0 && E % config->cross_attn_heads == 0,
                "blt_local_decoder_create: embed_dim must be divisible by cross_attn_heads");
    BLT_REQUIRE(config->rope_theta > 0.0f, "blt_local_decoder_create: rope_theta must be positive");
    BLT_REQUIRE(config->vocab_size > 0, "blt_local_decoder_create: vocab_size must be positive");

    BLT_REQUIRE(config->patch_dim % config->embed_dim == 0,
                "blt_local_decoder_create: patch_dim must be divisible by embed_dim");

    blt_local_decoder *m = (blt_local_decoder *)blt_container_alloc(arena, sizeof(blt_local_decoder));
    m->config = *config;

    size_t half = head_dim / 2;
    size_t rope_shape[2] = {config->max_seq_len, half};
    m->rope_cos_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    m->rope_sin_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    blt_rope_config rope_cfg = {.theta = config->rope_theta, .head_dim = head_dim};
    blt_rope_precompute(config->max_seq_len, &rope_cfg, &m->rope_cos_cache, &m->rope_sin_cache);

    m->layers = (blt_local_decoder_layer_storage *)blt_container_alloc(
        arena, config->num_layers * sizeof(blt_local_decoder_layer_storage));

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

    size_t lm_head_shape[2] = {E, config->vocab_size};
    m->lm_head_weight = blt_tensor_create(arena, lm_head_shape, 2, BLT_DTYPE_FP32);

    // D_0 table for draft/MASK rows (BLT_D0_LEARNED); zero-initialized.
    // The diffusion backward accumulates into d0_embed_grad; the legacy
    // decoder backward leaves it zero, so SGD steps on it are no-ops there.
    size_t d0_embed_shape[2] = {BLT_D0_VOCAB, E};
    m->d0_embed_weight = blt_tensor_create(arena, d0_embed_shape, 2, BLT_DTYPE_FP32);

    return m;
}

void blt_local_decoder_refresh_bf16(blt_local_decoder *model) {
    if (model == NULL || !model->config.use_bf16 || model->layers_bf16 == NULL) {
        return;
    }
    for (size_t l = 0; l < model->config.num_layers; ++l) {
        blt_local_layer_bf16_refresh(&model->layers_bf16[l], &model->layers[l]);
    }
}

blt_local_decoder_grad *blt_local_decoder_grad_create(blt_arena *arena, const blt_local_decoder *model) {
    BLT_REQUIRE(arena != NULL && model != NULL, "blt_local_decoder_grad_create: arena and model cannot be NULL");

    size_t E = model->config.embed_dim;
    size_t hidden = model->config.hidden_dim;
    size_t V = model->config.vocab_size;
    size_t num_layers = model->config.num_layers;

    blt_local_decoder_grad *g = (blt_local_decoder_grad *)blt_container_alloc(arena, sizeof(blt_local_decoder_grad));

    g->layer_grads =
        (blt_local_decoder_layer_grad *)blt_container_alloc(arena, num_layers * sizeof(blt_local_decoder_layer_grad));

    for (size_t l = 0; l < num_layers; ++l) {
        blt_local_layer_grad_alloc(arena, &g->layer_grads[l], E, hidden);
    }

    size_t lm_head_shape[2] = {E, V};
    g->lm_head_grad = blt_tensor_create(arena, lm_head_shape, 2, BLT_DTYPE_FP32);

    size_t d0_embed_shape[2] = {BLT_D0_VOCAB, E};
    g->d0_embed_grad = blt_tensor_create(arena, d0_embed_shape, 2, BLT_DTYPE_FP32);

    return g;
}

// Forward path (BLT 3.3, Eq. 10-12; FBLT Checkpoint D)
//
// D_0 = byte_hidden_in (h_final from the local encoder -- not a fresh embedding)
// for l in 1..num_layers:
//   if cross_attn_fires(l): B = D_{l-1} + cross_attn_l(RMSNorm(D_{l-1}), patch_in)
//   else:                   B = D_{l-1}
//   D_l = byte_transformer_layer_l(B)
// logits = D_final @ lm_head_weight
// loss   = shifted next-byte cross-entropy (targets[t] = bytes_in[t+1])

void blt_local_decoder_forward_ext(const blt_local_decoder *model, const blt_tensor *byte_hidden_in,
                                   const blt_tensor *patch_in, const blt_patch_info *patches, size_t num_patches,
                                   const blt_tensor *bytes_in, const blt_tensor *targets, const size_t *doc_boundaries,
                                   size_t num_docs, const blt_local_decoder_d0_opts *d0_opts, int trailing_closed,
                                   blt_tensor *logits_out, blt_tensor *loss_out, blt_arena *arena) {
    BLT_REQUIRE(logits_out != NULL, "blt_local_decoder_forward_ext: logits_out cannot be NULL");
    BLT_REQUIRE(loss_out != NULL || bytes_in == NULL,
                "blt_local_decoder_forward_ext: bytes_in is only used for the loss; pass NULL when loss_out is NULL");
    if (loss_out != NULL) {
        BLT_REQUIRE(bytes_in != NULL,
                    "blt_local_decoder_forward_ext: bytes_in cannot be NULL when loss_out is requested");
    }
    size_t seq_len;
    validate_call(model, byte_hidden_in, patch_in, patches, num_patches, bytes_in, arena, &seq_len);

    // D_0 policy validation
    size_t num_hfinal_rows = seq_len;
    if (d0_opts != NULL && d0_opts->d0_mode != BLT_D0_HFINAL) {
        num_hfinal_rows = d0_opts->num_hfinal_rows;
        BLT_REQUIRE(num_hfinal_rows >= 1 && num_hfinal_rows <= seq_len,
                    "blt_local_decoder_forward_ext: d0_opts->num_hfinal_rows must be in [1, seq_len]");
        if (d0_opts->d0_mode == BLT_D0_LEARNED) {
            BLT_REQUIRE(d0_opts->d0_extra_tokens != NULL,
                        "blt_local_decoder_forward_ext: d0_extra_tokens required for BLT_D0_LEARNED");
            for (size_t i = 0; i < seq_len - num_hfinal_rows; i++) {
                BLT_REQUIRE(d0_opts->d0_extra_tokens[i] < BLT_D0_VOCAB,
                            "blt_local_decoder_forward_ext: d0_extra_tokens[%zu] must be < BLT_D0_VOCAB", i);
            }
        }
    }

    const blt_local_decoder_config *config = &model->config;
    size_t E = config->embed_dim;
    size_t V = config->vocab_size;
    blt_check_nd_fp32(logits_out, 2, (const size_t[]){seq_len, V},
                      "blt_local_decoder_forward_ext: logits_out must be [seq_len, vocab_size] FP32");

    size_t patch_dim = (config->patch_dim == 0) ? E : config->patch_dim;
    BLT_REQUIRE(patch_dim % E == 0, "blt_local_decoder_forward_ext: patch_dim must be a multiple of embed_dim");
    size_t k = patch_dim / E;
    BLT_REQUIRE(patch_in->shape[1] == patch_dim,
                "blt_local_decoder_forward_ext: patch_in must be [num_patches, patch_dim] FP32");

    ld_context ctx = make_context(model, seq_len, num_hfinal_rows, patches, num_patches, doc_boundaries, num_docs,
                                  trailing_closed, arena);
    blt_transformer_config layer_config =
        blt_local_byte_layer_config(config->embed_dim, config->num_heads, config->rope_theta, config->hidden_dim,
                                    &ctx.local_mask, &ctx.rope_cos_view, &ctx.rope_sin_view);
    blt_cross_attention_config cross_config = make_cross_attn_config(config, &ctx);
    cross_config.embed_dim = E;
    cross_config.patch_dim = patch_dim;
    cross_config.split_mode = (k == 1) ? BLT_CROSS_ATTN_NO_SPLIT : BLT_CROSS_ATTN_SPLIT_KV;

    blt_mask_config split_mask_cfg;
    setup_kv_split_mask(&cross_config, &split_mask_cfg, num_patches, k, &ctx);

    blt_tensor patch_in_split;
    blt_tensor_view_2d(&patch_in_split, patch_in->data, num_patches * k, E, patch_in->backend);

    size_t byte_shape[2] = {seq_len, E};

    // D_0 construction per policy.
    // Legacy / HFINAL: D_0 aliases byte_hidden_in (no copy). ZEROS /
    // LEARNED: rows [0, num_hfinal_rows) are copied verbatim from
    // byte_hidden_in; rows beyond it are zero-filled and optionally
    // overwritten from the decoder-owned d0_embed_weight table.
    blt_tensor d;
    if (d0_opts == NULL || d0_opts->d0_mode == BLT_D0_HFINAL) {
        d = *byte_hidden_in;
    } else {
        d = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32); // zero-init
        blt_strided_copy(byte_hidden_in->backend, (float *)d.data, E, (const float *)byte_hidden_in->data, E,
                         num_hfinal_rows, E);

        if (d0_opts->d0_mode == BLT_D0_LEARNED) {
            const size_t n_extra = seq_len - num_hfinal_rows;
            size_t extra_shape[2] = {n_extra, E};
            blt_tensor extra = blt_tensor_create(arena, extra_shape, 2, BLT_DTYPE_FP32);
            // d0_extra_tokens is a host id array; the lookup handles either
            // memory space for the table.
            blt_tensor table_view;
            blt_tensor_view_2d(&table_view, model->d0_embed_weight.data, model->d0_embed_weight.numel / E, E,
                               model->d0_embed_weight.backend);
            blt_embedding_lookup(&table_view, d0_opts->d0_extra_tokens, &extra);
            blt_strided_copy(extra.backend, (float *)d.data + num_hfinal_rows * E, E, (const float *)extra.data, E,
                             n_extra, E);
        }
    }

    // layer loop: cross-attn, byte transformer block
    for (size_t l = 0; l < config->num_layers; ++l) {
        const blt_local_decoder_layer_storage *s = &model->layers[l];
        const blt_local_layer_bf16 *bf = model->config.use_bf16 ? &model->layers_bf16[l] : NULL;
        blt_tensor b;

        if (blt_local_cross_attn_fires(config->cross_attn_placement, config->cross_attn_all_layers, config->num_layers,
                                       l)) {
            blt_tensor normed_d = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&d, &s->cross_norm_weight, &normed_d);

            blt_cross_attention_weights xw = blt_local_cross_weights_view_mixed(s, bf);
            blt_tensor cross_out = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_cross_attention_forward(&normed_d, &patch_in_split, &xw, &cross_out, &cross_config, arena);

            b = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_add(&d, &cross_out, &b);
        } else {
            b = d;
        }

        blt_transformer_weights w = blt_local_byte_weights_view_mixed(s, bf);
        blt_tensor d_next = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
        blt_transformer_forward(&b, &w, &d_next, &layer_config, arena);
        d = d_next;
    }

    // LM head + optional shifted next-byte cross-entropy
    blt_matmul(&d, &model->lm_head_weight, logits_out);

    if (loss_out != NULL) {
        const blt_tensor *ce_targets = (targets != NULL) ? targets : bytes_in;
        blt_tensor logits_view;
        blt_tensor_view_2d(&logits_view, logits_out->data, seq_len - 1, V, logits_out->backend);

        blt_tensor targets_view;
        view_1d_offset(&targets_view, ce_targets, 1, seq_len - 1);

        blt_cross_entropy_forward(&logits_view, &targets_view, loss_out);
    }
}

void blt_local_decoder_forward(const blt_local_decoder *model, const blt_tensor *byte_hidden_in,
                               const blt_tensor *patch_in, const blt_patch_info *patches, size_t num_patches,
                               const blt_tensor *bytes_in, const blt_tensor *targets, const size_t *doc_boundaries,
                               size_t num_docs, blt_tensor *logits_out, blt_tensor *loss_out, blt_arena *arena) {
    BLT_REQUIRE(logits_out != NULL && loss_out != NULL,
                "blt_local_decoder_forward: logits_out and loss_out cannot be NULL");
    blt_local_decoder_forward_ext(model, byte_hidden_in, patch_in, patches, num_patches, bytes_in, targets,
                                  doc_boundaries, num_docs, NULL, 0, logits_out, loss_out, arena);
}

typedef struct {
    bool has_cross;
    blt_tensor d_in;
    blt_tensor normed_d;
    blt_transformer_layer_cache *byte_cache;
} dec_layer_cache;

void blt_local_decoder_backward(const blt_local_decoder *model, const blt_tensor *byte_hidden_in,
                                const blt_tensor *patch_in, const blt_patch_info *patches, size_t num_patches,
                                const blt_tensor *bytes_in, const blt_tensor *targets, const size_t *doc_boundaries,
                                size_t num_docs, blt_tensor *grad_byte_hidden_in, blt_tensor *grad_patch_in,
                                blt_local_decoder_grad *grad, blt_arena *arena) {
    size_t seq_len;
    validate_call(model, byte_hidden_in, patch_in, patches, num_patches, bytes_in, arena, &seq_len);
    BLT_REQUIRE(grad_byte_hidden_in != NULL && grad_patch_in != NULL && grad != NULL && grad->layer_grads != NULL,
                "blt_local_decoder_backward: grad_byte_hidden_in, grad_patch_in and grad cannot be NULL");

    const blt_local_decoder_config *config = &model->config;
    const size_t E = config->embed_dim;
    const size_t V = config->vocab_size;
    const size_t L = config->num_layers;

    size_t patch_dim = (config->patch_dim == 0) ? E : config->patch_dim;
    BLT_REQUIRE(patch_dim % E == 0, "blt_local_decoder_backward: patch_dim must be a multiple of embed_dim");
    size_t k = patch_dim / E;

    blt_check_nd_fp32(grad_byte_hidden_in, 2, (const size_t[]){seq_len, E},
                      "blt_local_decoder_backward: grad_byte_hidden_in must be [seq_len, embed_dim] FP32");
    blt_check_nd_fp32(grad_patch_in, 2, (const size_t[]){num_patches, patch_dim},
                      "blt_local_decoder_backward: grad_patch_in must be [num_patches, patch_dim] FP32");
    blt_check_nd_fp32(&grad->lm_head_grad, 2, (const size_t[]){E, V},
                      "blt_local_decoder_backward: lm_head_grad must be [embed_dim, vocab_size] FP32");

    ld_context ctx = make_context(model, seq_len, seq_len, patches, num_patches, doc_boundaries, num_docs, 0, arena);
    blt_transformer_config layer_config =
        blt_local_byte_layer_config(config->embed_dim, config->num_heads, config->rope_theta, config->hidden_dim,
                                    &ctx.local_mask, &ctx.rope_cos_view, &ctx.rope_sin_view);
    blt_cross_attention_config cross_config = make_cross_attn_config(config, &ctx);
    cross_config.embed_dim = E;
    cross_config.patch_dim = patch_dim;
    cross_config.split_mode = (k == 1) ? BLT_CROSS_ATTN_NO_SPLIT : BLT_CROSS_ATTN_SPLIT_KV;

    blt_mask_config split_mask_cfg;
    setup_kv_split_mask(&cross_config, &split_mask_cfg, num_patches, k, &ctx);

    blt_tensor patch_in_split;
    blt_tensor_view_2d(&patch_in_split, patch_in->data, num_patches * k, E, patch_in->backend);

    size_t byte_shape[2] = {seq_len, E};
    size_t logits_shape[2] = {seq_len, V};

    // Recompute forward, caching per-layer intermediates
    //   d[0..L] - byte states entering each layer (d[0] = byte_hidden_in)
    //   caches[] - per-layer cross-attn + byte-transformer-block intermediates
    blt_tensor *d = (blt_tensor *)blt_container_alloc(arena, (L + 1) * sizeof(blt_tensor));
    dec_layer_cache *caches = (dec_layer_cache *)blt_container_alloc(arena, L * sizeof(dec_layer_cache));
    memset(caches, 0, L * sizeof(dec_layer_cache));

    d[0] = *byte_hidden_in;

    for (size_t l = 0; l < L; ++l) {
        const blt_local_decoder_layer_storage *s = &model->layers[l];
        dec_layer_cache *c = &caches[l];
        const blt_local_layer_bf16 *bf = model->config.use_bf16 ? &model->layers_bf16[l] : NULL;
        c->d_in = d[l];

        blt_tensor b;
        if (blt_local_cross_attn_fires(config->cross_attn_placement, config->cross_attn_all_layers, config->num_layers,
                                       l)) {
            c->has_cross = true;

            c->normed_d = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&d[l], &s->cross_norm_weight, &c->normed_d);

            blt_cross_attention_weights xw = blt_local_cross_weights_view_mixed(s, bf);
            blt_tensor cross_out = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_cross_attention_forward(&c->normed_d, &patch_in_split, &xw, &cross_out, &cross_config, arena);

            b = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_add(&d[l], &cross_out, &b);
        } else {
            c->has_cross = false;
            b = d[l];
        }

        blt_transformer_weights w = blt_local_byte_weights_view_mixed(s, bf);
        c->byte_cache = blt_transformer_layer_forward_cached(&b, &w, &layer_config, arena, seq_len, E,
                                                             config->hidden_dim, &d[l + 1]);
    }

    // LM head, recomputed to obtain the terminal gradient
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&d[L], &model->lm_head_weight, &logits);

    blt_tensor logits_view;
    blt_tensor_view_2d(&logits_view, logits.data, seq_len - 1, V, logits.backend);

    const blt_tensor *ce_targets = (targets != NULL) ? targets : bytes_in;
    blt_tensor targets_view;
    view_1d_offset(&targets_view, ce_targets, 1, seq_len - 1);

    // Terminal gradient dL/dlogits -- only the first seq_len-1 rows were
    // used in the loss; the last position's row gets zero gradient.
    size_t grad_logits_view_shape[2] = {seq_len - 1, V};
    blt_tensor grad_logits_view = blt_tensor_create(arena, grad_logits_view_shape, 2, BLT_DTYPE_FP32);
    blt_cross_entropy_backward(&logits_view, &targets_view, &grad_logits_view);

    blt_tensor grad_logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32); // zero-init
    blt_strided_copy(grad_logits_view.backend, (float *)grad_logits.data, V, (const float *)grad_logits_view.data, V,
                     seq_len - 1, V);

    blt_tensor dh = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_matmul_backward(&d[L], &model->lm_head_weight, &grad_logits, &dh, &grad->lm_head_grad);

    // Reverse layer loop.
    //   dh - running dL/d(d[l+1]), starts at dL/d(d[L])
    //   dP - running dL/d(patch_in), summed across every firing layer
    blt_tensor dP_split = blt_tensor_create(arena, (size_t[]){num_patches * k, E}, 2, BLT_DTYPE_FP32); // zero-init

    for (size_t li = L; li-- > 0;) {
        const blt_local_decoder_layer_storage *s = &model->layers[li];
        blt_local_decoder_layer_grad *lg = &grad->layer_grads[li];
        const dec_layer_cache *c = &caches[li];
        const blt_local_layer_bf16 *b = model->config.use_bf16 ? &model->layers_bf16[li] : NULL;

        // byte transformer block backward: dh becomes dL/d(b) for this layer
        blt_transformer_weights w = blt_local_byte_weights_view_mixed(s, b);
        blt_transformer_layer_grad lgv = blt_local_byte_grad_view(lg);
        blt_tensor grad_b;
        blt_transformer_layer_backward(c->byte_cache, &w, &layer_config, arena, seq_len, E, config->hidden_dim, &dh,
                                       &lgv, &grad_b);

        if (c->has_cross) {
            // b = d_in + cross_out  =>  residual: d(d_in) += grad_b, d(cross_out) = grad_b
            blt_cross_attention_weights xw = blt_local_cross_weights_view_mixed(s, b);
            blt_cross_attention_grad xg = {0};
            xg.grad_weight_q = lg->cross_weight_q;
            xg.grad_weight_k = lg->cross_weight_k;
            xg.grad_weight_v = lg->cross_weight_v;
            xg.grad_weight_proj = lg->cross_weight_proj;

            blt_tensor grad_normed_d = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_tensor grad_kv_split = blt_tensor_create(arena, (size_t[]){num_patches * k, E}, 2, BLT_DTYPE_FP32);
            blt_cross_attention_backward(&c->normed_d, &patch_in_split, &xw, &grad_b, &grad_normed_d, &grad_kv_split,
                                         &xg, &cross_config, arena);

            // patch_in gradient accumulates across firing layers (still in split form)
            blt_tensor dP_split_new = blt_tensor_create(arena, (size_t[]){num_patches * k, E}, 2, BLT_DTYPE_FP32);
            blt_add(&dP_split, &grad_kv_split, &dP_split_new);
            dP_split = dP_split_new;

            blt_tensor grad_d_in_norm = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_backward(&grad_normed_d, &c->d_in, &s->cross_norm_weight, &grad_d_in_norm,
                                 &lg->cross_norm_weight);

            // d(d_in) = grad_b (residual) + grad through the norm
            blt_tensor grad_d_in = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
            blt_add(&grad_b, &grad_d_in_norm, &grad_d_in);
            dh = grad_d_in;
        } else {
            dh = grad_b;
        }
    }

    // Write terminal gradients
    blt_strided_copy(dh.backend, (float *)grad_byte_hidden_in->data, seq_len * E, (const float *)dh.data, seq_len * E,
                     1, seq_len * E);

    blt_tensor dP;
    blt_tensor_view_2d(&dP, dP_split.data, num_patches, patch_dim, dP_split.backend);
    blt_strided_copy(dP.backend, (float *)grad_patch_in->data, num_patches * patch_dim, (const float *)dP.data,
                     num_patches * patch_dim, 1, num_patches * patch_dim);
}
