#include "core/backend.h"
#include "core/allocator.h"
#include "models/model.h"

// Model APIs store one start offset per document, including document zero.
// Mask consumers store only the starts of documents one through num_docs - 1.
static const size_t *blt_mask_doc_boundaries(const size_t *doc_boundaries, size_t num_docs) {
    BLT_REQUIRE(num_docs == 0 || doc_boundaries != NULL, "blt_model: doc_boundaries cannot be NULL when num_docs > 0");
    if (num_docs > 0) {
        BLT_REQUIRE(doc_boundaries[0] == 0, "blt_model: document zero must start at offset 0");
        for (size_t d = 1; d < num_docs; d++) {
            BLT_REQUIRE(doc_boundaries[d] >= doc_boundaries[d - 1], "blt_model: document boundaries must be ordered");
        }
    }
    return num_docs > 1 ? doc_boundaries + 1 : doc_boundaries;
}

static void map_byte_boundaries_to_patch_boundaries(const size_t *byte_doc_boundaries, size_t num_docs,
                                                    const blt_patch_info *patches, size_t num_patches,
                                                    size_t *patch_doc_boundaries_out) {
    size_t patch_idx = 0;
    for (size_t d = 0; d < num_docs; d++) {
        size_t byte_boundary = byte_doc_boundaries[d];
        while (patch_idx < num_patches && patches[patch_idx].start_idx < byte_boundary) {
            patch_idx++;
        }
        BLT_REQUIRE(patch_idx < num_patches,
                    "map_byte_boundaries_to_patch_boundaries: doc boundary does not align to a patch start");
        BLT_REQUIRE(patches[patch_idx].start_idx == byte_boundary,
                    "map_byte_boundaries_to_patch_boundaries: doc boundary is not a patch boundary");
        patch_doc_boundaries_out[d] = patch_idx;
    }
}

blt_model *blt_model_create(blt_arena *arena, const blt_model_config *config) {
    BLT_REQUIRE(arena != NULL && config != NULL, "blt_model_create: arena and config cannot be NULL");
    BLT_REQUIRE(config->encoder_config.embed_dim == config->global_config.embed_dim &&
                    config->global_config.embed_dim == config->decoder_config.embed_dim,
                "blt_model_create: encoder/global/decoder embed_dim must match (shared hidden width)");

    blt_model *model = (blt_model *)blt_container_alloc(arena, sizeof(blt_model));
    model->config = *config;

    model->encoder = blt_local_encoder_create(arena, &config->encoder_config);
    model->global = blt_global_transformer_create(arena, &config->global_config);
    model->decoder = blt_local_decoder_create(arena, &config->decoder_config);

    return model;
}

void blt_model_refresh_mixed(blt_model *model) {
    if (model == NULL) {
        return;
    }
    if (model->encoder != NULL) {
        blt_local_encoder_refresh_mixed(model->encoder);
    }
    if (model->global != NULL) {
        blt_transformer_stack_refresh_mixed(&model->global->stack);
    }
    if (model->decoder != NULL) {
        blt_local_decoder_refresh_mixed(model->decoder);
    }
}

blt_model_grad *blt_model_grad_create(blt_arena *arena, const blt_model *model) {
    BLT_REQUIRE(arena != NULL && model != NULL, "blt_model_grad_create: arena and model cannot be NULL");

    blt_model_grad *grad = (blt_model_grad *)blt_container_alloc(arena, sizeof(blt_model_grad));

    grad->encoder_grad = blt_local_encoder_grad_create(arena, model->encoder);
    grad->global_grad = blt_global_transformer_grad_create(arena, model->global);
    grad->decoder_grad = blt_local_decoder_grad_create(arena, model->decoder);

    return grad;
}

// Forward path:
//   Encoder(bytes, patches) -> patch_out, byte_hidden_out
//   Global(patch_out, doc_boundaries) -> O
//   Decoder(byte_hidden_out, O, bytes) -> logits, loss

void blt_model_encode(const blt_model *model, const blt_tensor *bytes_in, const blt_patch_info *patches,
                      size_t num_patches, const size_t *doc_boundaries, size_t num_docs, blt_model_enc_out *out,
                      blt_arena *arena) {
    BLT_REQUIRE(model != NULL && bytes_in != NULL && patches != NULL && out != NULL && arena != NULL,
                "blt_model_encode: arguments cannot be NULL");
    BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8,
                "blt_model_encode: bytes_in must be a 1D UINT8 tensor [seq_len]");
    BLT_REQUIRE(num_docs == 0 || doc_boundaries != NULL,
                "blt_model_encode: doc_boundaries cannot be NULL when num_docs > 0");
    BLT_REQUIRE(num_patches >= 1, "blt_model_encode: num_patches must be >= 1");

    size_t seq_len = bytes_in->shape[0];
    size_t embed_dim = model->config.encoder_config.embed_dim;
    size_t patch_shape[2] = {num_patches, embed_dim};
    size_t byte_shape[2] = {seq_len, embed_dim};

    // 1. Local encoder
    out->patch_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    out->byte_hidden_out = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);

    blt_local_encoder_forward(model->encoder, bytes_in, patches, num_patches,
                              blt_mask_doc_boundaries(doc_boundaries, num_docs), num_docs, &out->patch_out,
                              &out->byte_hidden_out, arena);

    // 2. Remap doc boundaries byte -> patch indices
    out->patch_doc_boundaries = NULL;
    if (num_docs > 0) {
        out->patch_doc_boundaries = (size_t *)blt_container_alloc(arena, num_docs * sizeof(size_t));
        map_byte_boundaries_to_patch_boundaries(doc_boundaries, num_docs, patches, num_patches,
                                                out->patch_doc_boundaries);
    }

    // 3. Global transformer
    out->global_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_global_transformer_forward(model->global, &out->patch_out,
                                   blt_mask_doc_boundaries(out->patch_doc_boundaries, num_docs), num_docs,
                                   &out->global_out, arena);
}

void blt_model_decode(const blt_model *model, const blt_model_enc_out *enc, const blt_patch_info *patches,
                      size_t num_patches, const blt_tensor *bytes_in, const blt_tensor *targets,
                      const size_t *doc_boundaries, size_t num_docs, const blt_local_decoder_d0_opts *d0_opts,
                      blt_tensor *logits_out, blt_tensor *loss_out, blt_arena *arena) {
    BLT_REQUIRE(model != NULL && enc != NULL && patches != NULL && logits_out != NULL && arena != NULL,
                "blt_model_decode: model, enc, patches, logits_out and arena cannot be NULL");
    BLT_REQUIRE(num_patches >= 1, "blt_model_decode: num_patches must be >= 1");
    BLT_REQUIRE(num_patches == enc->patch_out.shape[0],
                "blt_model_decode: num_patches must match the encode-stage patch count");
    BLT_REQUIRE(loss_out != NULL || bytes_in == NULL,
                "blt_model_decode: bytes_in is only used for the loss; pass NULL when loss_out is NULL");
    if (loss_out != NULL) {
        BLT_REQUIRE(bytes_in != NULL, "blt_model_decode: bytes_in cannot be NULL when loss_out is requested");
    }

    size_t seq_len = enc->byte_hidden_out.shape[0];
    if (bytes_in != NULL) {
        BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8 && bytes_in->shape[0] == seq_len,
                    "blt_model_decode: bytes_in must be 1D UINT8 [seq_len] matching enc->byte_hidden_out");
    }

    blt_local_decoder_forward_ext(model->decoder, &enc->byte_hidden_out, &enc->global_out, patches, num_patches,
                                  bytes_in, targets, blt_mask_doc_boundaries(doc_boundaries, num_docs), num_docs,
                                  d0_opts, 0, logits_out, loss_out, arena);
}

void blt_model_forward(const blt_model *model, const blt_tensor *bytes_in, const blt_tensor *targets,
                       const blt_patch_info *patches, size_t num_patches, const size_t *doc_boundaries, size_t num_docs,
                       blt_tensor *logits_out, blt_tensor *loss_out, blt_arena *arena) {
    BLT_REQUIRE(model != NULL && bytes_in != NULL && patches != NULL && logits_out != NULL && arena != NULL,
                "blt_model_forward: arguments cannot be NULL");

    blt_model_enc_out enc;
    blt_model_encode(model, bytes_in, patches, num_patches, doc_boundaries, num_docs, &enc, arena);
    blt_model_decode(model, &enc, patches, num_patches, bytes_in, targets, doc_boundaries, num_docs, NULL, logits_out,
                     loss_out, arena);
    blt_backend_pass_sync(bytes_in->backend);
}

// Backward path — recomputes forward intermediates (memory over speed tradeoff).
void blt_model_backward(const blt_model *model, const blt_tensor *bytes_in, const blt_tensor *targets,
                        const blt_patch_info *patches, size_t num_patches, const size_t *doc_boundaries,
                        size_t num_docs, blt_model_grad *grad, blt_arena *arena) {
    BLT_REQUIRE(model != NULL && bytes_in != NULL && patches != NULL && grad != NULL && arena != NULL,
                "blt_model_backward: arguments cannot be NULL");
    BLT_REQUIRE(bytes_in->ndim == 1 && bytes_in->dtype == BLT_DTYPE_UINT8,
                "blt_model_backward: bytes_in must be a 1D UINT8 tensor [seq_len]");
    BLT_REQUIRE(num_docs == 0 || doc_boundaries != NULL,
                "blt_model_backward: doc_boundaries cannot be NULL when num_docs > 0");
    BLT_REQUIRE(num_patches >= 1, "blt_model_backward: num_patches must be >= 1");

    size_t seq_len = bytes_in->shape[0];
    size_t embed_dim = model->config.encoder_config.embed_dim;
    size_t patch_shape[2] = {num_patches, embed_dim};
    size_t byte_shape[2] = {seq_len, embed_dim};

    // Recompute forward intermediates needed by submodule backward calls.
    blt_tensor patch_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_tensor byte_hidden_out = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_local_encoder_forward(model->encoder, bytes_in, patches, num_patches,
                              blt_mask_doc_boundaries(doc_boundaries, num_docs), num_docs, &patch_out, &byte_hidden_out,
                              arena);

    size_t *patch_doc_boundaries = NULL;
    if (num_docs > 0) {
        patch_doc_boundaries = (size_t *)blt_container_alloc(arena, num_docs * sizeof(size_t));
        map_byte_boundaries_to_patch_boundaries(doc_boundaries, num_docs, patches, num_patches, patch_doc_boundaries);
    }

    blt_tensor global_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_global_transformer_forward(model->global, &patch_out, blt_mask_doc_boundaries(patch_doc_boundaries, num_docs),
                                   num_docs, &global_out, arena);

    // 1. Local decoder backward
    blt_tensor grad_byte_hidden = blt_tensor_create(arena, byte_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_global_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);

    blt_local_decoder_backward(model->decoder, &byte_hidden_out, &global_out, patches, num_patches, bytes_in, targets,
                               blt_mask_doc_boundaries(doc_boundaries, num_docs), num_docs, &grad_byte_hidden,
                               &grad_global_out, grad->decoder_grad, arena);

    // 2. Global transformer backward
    blt_tensor grad_patch_out = blt_tensor_create(arena, patch_shape, 2, BLT_DTYPE_FP32);
    blt_global_transformer_backward(model->global, &patch_out, blt_mask_doc_boundaries(patch_doc_boundaries, num_docs),
                                    num_docs, &grad_global_out, &grad_patch_out, grad->global_grad, arena);

    // 3. Local encoder backward
    blt_local_encoder_backward(model->encoder, bytes_in, patches, num_patches,
                               blt_mask_doc_boundaries(doc_boundaries, num_docs), num_docs, &grad_patch_out,
                               &grad_byte_hidden, grad->encoder_grad, arena);

    blt_backend_pass_sync(bytes_in->backend);
}
static void blt_model_grad_zero_one(blt_model_grad *grad, const blt_model *model) {
    blt_local_encoder_grad *eg = grad->encoder_grad;
    blt_global_transformer_grad *gg = grad->global_grad;
    blt_local_decoder_grad *dg = grad->decoder_grad;

    zero_tensor(&eg->embedding_grad);
    for (size_t k = 0; k < model->encoder->ngram_weights.num_tables; k++) zero_tensor(&eg->ngram_grads.tables[k]);
    for (size_t l = 0; l < model->encoder->config.num_layers; l++) {
        blt_local_layer_grad *t = &eg->layer_grads[l];
        blt_tensor *all[12] = {&t->norm1_weight,   &t->attn_qkv_w,     &t->attn_proj_w,    &t->norm2_weight,
                               &t->ffn_up_w,       &t->ffn_gate_w,     &t->ffn_down_w,     &t->cross_norm_weight,
                               &t->cross_weight_q, &t->cross_weight_k, &t->cross_weight_v, &t->cross_weight_proj};
        for (int i = 0; i < 12; i++) zero_tensor(all[i]);
    }
    for (size_t l = 0; l < model->global->stack.num_layers; l++) {
        blt_transformer_layer_grad *t = &gg->stack_grad->layer_grads[l];
        blt_tensor *all[7] = {&t->norm1_weight, &t->attn_qkv_w, &t->attn_proj_w, &t->norm2_weight,
                              &t->ffn_up_w,     &t->ffn_gate_w, &t->ffn_down_w};
        for (int i = 0; i < 7; i++) zero_tensor(all[i]);
    }
    for (size_t l = 0; l < model->decoder->config.num_layers; l++) {
        blt_local_layer_grad *t = &dg->layer_grads[l];
        blt_tensor *all[12] = {&t->cross_norm_weight, &t->cross_weight_q, &t->cross_weight_k, &t->cross_weight_v,
                               &t->cross_weight_proj, &t->norm1_weight,   &t->attn_qkv_w,     &t->attn_proj_w,
                               &t->norm2_weight,      &t->ffn_up_w,       &t->ffn_gate_w,     &t->ffn_down_w};
        for (int i = 0; i < 12; i++) zero_tensor(all[i]);
    }
    zero_tensor(&dg->lm_head_grad);
    zero_tensor(&dg->d0_embed_grad);
}

void blt_model_grad_zero(const blt_model *model, blt_model_grad *grad) {
    BLT_REQUIRE(model != NULL && grad != NULL, "blt_model_grad_zero: model and grad cannot be NULL");
    blt_model_grad_zero_one(grad, model);
}

static void blt_model_grad_add_one(blt_model_grad *dst, const blt_model_grad *src, const blt_model *model) {
    blt_local_encoder_grad *de = dst->encoder_grad, *se = (blt_local_encoder_grad *)src->encoder_grad;
    blt_global_transformer_grad *dg = dst->global_grad, *sg = (blt_global_transformer_grad *)src->global_grad;
    blt_local_decoder_grad *dd = dst->decoder_grad, *sd = (blt_local_decoder_grad *)src->decoder_grad;

    blt_add(&de->embedding_grad, &se->embedding_grad, &de->embedding_grad);
    for (size_t k = 0; k < model->encoder->ngram_weights.num_tables; k++)
        blt_add(&de->ngram_grads.tables[k], &se->ngram_grads.tables[k], &de->ngram_grads.tables[k]);
    for (size_t l = 0; l < model->encoder->config.num_layers; l++) {
        blt_local_layer_grad *dt = &de->layer_grads[l];
        blt_local_layer_grad *st = &se->layer_grads[l];
        blt_tensor *da[12] = {&dt->norm1_weight,   &dt->attn_qkv_w,     &dt->attn_proj_w,    &dt->norm2_weight,
                              &dt->ffn_up_w,       &dt->ffn_gate_w,     &dt->ffn_down_w,     &dt->cross_norm_weight,
                              &dt->cross_weight_q, &dt->cross_weight_k, &dt->cross_weight_v, &dt->cross_weight_proj};
        blt_tensor *sa[12] = {&st->norm1_weight,   &st->attn_qkv_w,     &st->attn_proj_w,    &st->norm2_weight,
                              &st->ffn_up_w,       &st->ffn_gate_w,     &st->ffn_down_w,     &st->cross_norm_weight,
                              &st->cross_weight_q, &st->cross_weight_k, &st->cross_weight_v, &st->cross_weight_proj};
        for (int i = 0; i < 12; i++) blt_add(da[i], sa[i], da[i]);
    }
    for (size_t l = 0; l < model->global->stack.num_layers; l++) {
        blt_transformer_layer_grad *dt = &dg->stack_grad->layer_grads[l];
        blt_transformer_layer_grad *st = &sg->stack_grad->layer_grads[l];
        blt_tensor *da[7] = {&dt->norm1_weight, &dt->attn_qkv_w, &dt->attn_proj_w, &dt->norm2_weight,
                             &dt->ffn_up_w,     &dt->ffn_gate_w, &dt->ffn_down_w};
        blt_tensor *sa[7] = {&st->norm1_weight, &st->attn_qkv_w, &st->attn_proj_w, &st->norm2_weight,
                             &st->ffn_up_w,     &st->ffn_gate_w, &st->ffn_down_w};
        for (int i = 0; i < 7; i++) blt_add(da[i], sa[i], da[i]);
    }
    for (size_t l = 0; l < model->decoder->config.num_layers; l++) {
        blt_local_layer_grad *dt = &dd->layer_grads[l];
        blt_local_layer_grad *st = &sd->layer_grads[l];
        blt_tensor *da[12] = {&dt->cross_norm_weight, &dt->cross_weight_q, &dt->cross_weight_k, &dt->cross_weight_v,
                              &dt->cross_weight_proj, &dt->norm1_weight,   &dt->attn_qkv_w,     &dt->attn_proj_w,
                              &dt->norm2_weight,      &dt->ffn_up_w,       &dt->ffn_gate_w,     &dt->ffn_down_w};
        blt_tensor *sa[12] = {&st->cross_norm_weight, &st->cross_weight_q, &st->cross_weight_k, &st->cross_weight_v,
                              &st->cross_weight_proj, &st->norm1_weight,   &st->attn_qkv_w,     &st->attn_proj_w,
                              &st->norm2_weight,      &st->ffn_up_w,       &st->ffn_gate_w,     &st->ffn_down_w};
        for (int i = 0; i < 12; i++) blt_add(da[i], sa[i], da[i]);
    }
    blt_add(&dd->lm_head_grad, &sd->lm_head_grad, &dd->lm_head_grad);
    blt_add(&dd->d0_embed_grad, &sd->d0_embed_grad, &dd->d0_embed_grad);
}

void blt_model_grad_add(const blt_model *model, blt_model_grad *dst, const blt_model_grad *src) {
    BLT_REQUIRE(model != NULL && dst != NULL && src != NULL, "blt_model_grad_add: arguments cannot be NULL");
    blt_model_grad_add_one(dst, src, model);
}

static void blt_model_grad_scale_one(blt_model_grad *grad, const blt_model *model, float scale) {
    blt_local_encoder_grad *eg = grad->encoder_grad;
    blt_global_transformer_grad *gg = grad->global_grad;
    blt_local_decoder_grad *dg = grad->decoder_grad;

    blt_scale(&eg->embedding_grad, scale);
    for (size_t k = 0; k < model->encoder->ngram_weights.num_tables; k++) blt_scale(&eg->ngram_grads.tables[k], scale);
    for (size_t l = 0; l < model->encoder->config.num_layers; l++) {
        blt_local_layer_grad *t = &eg->layer_grads[l];
        blt_tensor *all[12] = {&t->norm1_weight,   &t->attn_qkv_w,     &t->attn_proj_w,    &t->norm2_weight,
                               &t->ffn_up_w,       &t->ffn_gate_w,     &t->ffn_down_w,     &t->cross_norm_weight,
                               &t->cross_weight_q, &t->cross_weight_k, &t->cross_weight_v, &t->cross_weight_proj};
        for (int i = 0; i < 12; i++) blt_scale(all[i], scale);
    }
    for (size_t l = 0; l < model->global->stack.num_layers; l++) {
        blt_transformer_layer_grad *t = &gg->stack_grad->layer_grads[l];
        blt_tensor *all[7] = {&t->norm1_weight, &t->attn_qkv_w, &t->attn_proj_w, &t->norm2_weight,
                              &t->ffn_up_w,     &t->ffn_gate_w, &t->ffn_down_w};
        for (int i = 0; i < 7; i++) blt_scale(all[i], scale);
    }
    for (size_t l = 0; l < model->decoder->config.num_layers; l++) {
        blt_local_layer_grad *t = &dg->layer_grads[l];
        blt_tensor *all[12] = {&t->cross_norm_weight, &t->cross_weight_q, &t->cross_weight_k, &t->cross_weight_v,
                               &t->cross_weight_proj, &t->norm1_weight,   &t->attn_qkv_w,     &t->attn_proj_w,
                               &t->norm2_weight,      &t->ffn_up_w,       &t->ffn_gate_w,     &t->ffn_down_w};
        for (int i = 0; i < 12; i++) blt_scale(all[i], scale);
    }
    blt_scale(&dg->lm_head_grad, scale);
    blt_scale(&dg->d0_embed_grad, scale);
}

void blt_model_grad_scale(const blt_model *model, blt_model_grad *grad, float scale) {
    BLT_REQUIRE(model != NULL && grad != NULL, "blt_model_grad_scale: model and grad cannot be NULL");
    blt_model_grad_scale_one(grad, model, scale);
}
