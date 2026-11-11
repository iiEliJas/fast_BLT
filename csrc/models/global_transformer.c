#include "core/backend.h"
#include "core/allocator.h"
#include "ops/vecmath.h"
#include "models/global_transformer.h"
#include "models/transformer_stack.h"
#include "ops/elementwise.h"
#include "ops/mask_builder.h"

// Builds per-call config: RoPE view + block-causal patch mask
// (mask depends on num_patches/doc_boundaries which vary per call)
static blt_transformer_config make_patch_layer_config(const blt_global_transformer *model, size_t num_patches,
                                                      blt_tensor *cos_view, blt_tensor *sin_view, blt_tensor *mask_out,
                                                      const size_t *doc_boundaries, size_t num_docs, blt_arena *arena) {
    blt_transformer_config cfg = blt_transformer_stack_call_config(&model->stack, num_patches, cos_view, sin_view);

    blt_mask_config *mask_cfg = (blt_mask_config *)blt_container_alloc(arena, sizeof(blt_mask_config));
    *mask_cfg = (blt_mask_config){0};

    mask_cfg->seq_len_q = num_patches;
    mask_cfg->seq_len_kv = num_patches;
    mask_cfg->sliding_window = 0;
    mask_cfg->doc_boundaries = doc_boundaries;
    mask_cfg->num_docs = num_docs;
    mask_cfg->query_group_ids = NULL;
    mask_cfg->kv_group_ids = NULL;
    mask_cfg->bidirectional_within_group = false;
    mask_cfg->is_causal = true;
    blt_build_attention_mask(mask_cfg, mask_out, arena);

    cfg.attn_config.mask_config = mask_cfg;
    return cfg;
}

blt_global_transformer *blt_global_transformer_create(blt_arena *arena, const blt_global_transformer_config *config) {
    BLT_REQUIRE(arena != NULL && config != NULL, "blt_global_transformer_create: arena and config cannot be NULL");

    blt_global_transformer *model =
        (blt_global_transformer *)blt_container_alloc(arena, sizeof(blt_global_transformer));
    model->config = *config;

    // patches arrive as vectors from the local encoder, no embedding table needed
    blt_transformer_stack_config stack_cfg = {
        .num_layers = config->num_layers,
        .embed_dim = config->embed_dim,
        .hidden_dim = config->hidden_dim,
        .num_heads = config->num_heads,
        .max_seq_len = config->max_seq_len,
        .rope_theta = config->rope_theta,
        .mixed_dtype = config->mixed_dtype,
    };
    blt_transformer_stack_init(arena, &model->stack, &stack_cfg);

    return model;
}

blt_global_transformer_grad *blt_global_transformer_grad_create(blt_arena *arena, const blt_global_transformer *model) {
    BLT_REQUIRE(arena != NULL && model != NULL, "blt_global_transformer_grad_create: arena and model cannot be NULL");

    blt_global_transformer_grad *grad =
        (blt_global_transformer_grad *)blt_container_alloc(arena, sizeof(blt_global_transformer_grad));

    grad->stack_grad = blt_transformer_stack_grad_create(arena, &model->stack);

    return grad;
}

// Block-causal patch transformer: patch j attends to patches <= j in its document.
// No output projection — stack output goes directly to patch_out.
void blt_global_transformer_forward(const blt_global_transformer *model, const blt_tensor *patch_in,
                                    const size_t *doc_boundaries, size_t num_docs, blt_tensor *patch_out,
                                    blt_arena *arena) {
    BLT_REQUIRE(model != NULL && patch_in != NULL && patch_out != NULL && arena != NULL,
                "blt_global_transformer_forward: arguments cannot be NULL");
    BLT_REQUIRE(num_docs == 0 || doc_boundaries != NULL,
                "blt_global_transformer_forward: doc_boundaries cannot be NULL when num_docs > 0");
    BLT_REQUIRE(patch_in->ndim == 2 && patch_in->dtype == BLT_DTYPE_FP32,
                "blt_global_transformer_forward: patch_in must be a 2D FP32 tensor [num_patches, embed_dim]");

    size_t num_patches = patch_in->shape[0];
    size_t embed_dim = model->config.embed_dim;
    BLT_REQUIRE(patch_in->shape[1] == embed_dim, "blt_global_transformer_forward: patch_in embed_dim mismatch");
    BLT_REQUIRE(num_patches >= 1, "blt_global_transformer_forward: num_patches must be >= 1");
    BLT_REQUIRE(num_patches <= model->config.max_seq_len,
                "blt_global_transformer_forward: num_patches exceeds the model's max_seq_len (RoPE cache too small)");
    BLT_REQUIRE(patch_out->ndim == 2 && patch_out->dtype == BLT_DTYPE_FP32 && patch_out->shape[0] == num_patches &&
                    patch_out->shape[1] == embed_dim,
                "blt_global_transformer_forward: patch_out must be [num_patches, embed_dim] FP32");

    blt_tensor rope_cos_view, rope_sin_view, mask;
    blt_transformer_config layer_cfg = make_patch_layer_config(model, num_patches, &rope_cos_view, &rope_sin_view,
                                                               &mask, doc_boundaries, num_docs, arena);

    blt_tensor x;
    blt_transformer_stack_forward(&model->stack, patch_in, &layer_cfg, num_patches, &x, arena);

    blt_strided_copy(x.backend, (float *)patch_out->data, x.numel, (const float *)x.data, x.numel, 1, x.numel);
}

void blt_global_transformer_backward(const blt_global_transformer *model, const blt_tensor *patch_in,
                                     const size_t *doc_boundaries, size_t num_docs, const blt_tensor *grad_patch_out,
                                     blt_tensor *grad_patch_in, blt_global_transformer_grad *grad, blt_arena *arena) {
    BLT_REQUIRE(model != NULL && patch_in != NULL && grad_patch_out != NULL && grad_patch_in != NULL && grad != NULL &&
                    arena != NULL,
                "blt_global_transformer_backward: arguments cannot be NULL");
    BLT_REQUIRE(num_docs == 0 || doc_boundaries != NULL,
                "blt_global_transformer_backward: doc_boundaries cannot be NULL when num_docs > 0");
    BLT_REQUIRE(patch_in->ndim == 2 && patch_in->dtype == BLT_DTYPE_FP32,
                "blt_global_transformer_backward: patch_in must be a 2D FP32 tensor [num_patches, embed_dim]");

    size_t num_patches = patch_in->shape[0];
    size_t embed_dim = model->config.embed_dim;

    BLT_REQUIRE(patch_in->shape[1] == embed_dim, "blt_global_transformer_backward: patch_in embed_dim mismatch");
    BLT_REQUIRE(num_patches >= 1, "blt_global_transformer_backward: num_patches must be >= 1");
    BLT_REQUIRE(num_patches <= model->config.max_seq_len,
                "blt_global_transformer_backward: num_patches exceeds the model's max_seq_len (RoPE cache too small)");
    BLT_REQUIRE(grad_patch_out->ndim == 2 && grad_patch_out->shape[0] == num_patches &&
                    grad_patch_out->shape[1] == embed_dim,
                "blt_global_transformer_backward: grad_patch_out must be [num_patches, embed_dim] FP32");
    BLT_REQUIRE(grad_patch_in->ndim == 2 && grad_patch_in->dtype == BLT_DTYPE_FP32 &&
                    grad_patch_in->shape[0] == num_patches && grad_patch_in->shape[1] == embed_dim,
                "blt_global_transformer_backward: grad_patch_in must be [num_patches, embed_dim] FP32");

    blt_tensor rope_cos_view, rope_sin_view, mask;
    blt_transformer_config layer_cfg = make_patch_layer_config(model, num_patches, &rope_cos_view, &rope_sin_view,
                                                               &mask, doc_boundaries, num_docs, arena);

    // no downstream head — grad_patch_out is the incoming gradient directly
    blt_tensor final_x;
    blt_transformer_stack_cache *stack_cache =
        blt_transformer_stack_forward_cached(&model->stack, patch_in, &layer_cfg, num_patches, arena, &final_x);

    blt_tensor grad_x;
    blt_transformer_stack_backward(&model->stack, stack_cache, &layer_cfg, num_patches, grad_patch_out,
                                   grad->stack_grad, &grad_x, arena);

    BLT_REQUIRE(grad_x.ndim == 2 && grad_x.shape[0] == num_patches && grad_x.shape[1] == embed_dim,
                "blt_global_transformer_backward: internal shape mismatch on grad_patch_in");
    blt_strided_copy(grad_x.backend, (float *)grad_patch_in->data, grad_x.numel, (const float *)grad_x.data,
                     grad_x.numel, 1, grad_x.numel);
}