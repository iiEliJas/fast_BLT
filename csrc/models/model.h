#ifndef BLT_MODELS_BLT_MODEL_H
#define BLT_MODELS_BLT_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "core/allocator.h"
#include "core/tensor.h"
#include "ops/elementwise.h"
#include "models/local_encoder.h"
#include "models/global_transformer.h"
#include "models/local_decoder.h"
#include "models/patcher.h"

typedef struct {
    blt_local_encoder_config encoder_config;
    blt_global_transformer_config global_config;
    blt_local_decoder_config decoder_config;
} blt_model_config;

typedef struct {
    blt_model_config config;
    blt_local_encoder *encoder;
    blt_global_transformer *global;
    blt_local_decoder *decoder;
} blt_model;

typedef struct {
    blt_local_encoder_grad *encoder_grad;
    blt_global_transformer_grad *global_grad;
    blt_local_decoder_grad *decoder_grad;
} blt_model_grad;

blt_model *blt_model_create(blt_arena *arena, const blt_model_config *config);
blt_model_grad *blt_model_grad_create(blt_arena *arena, const blt_model *model);

void blt_model_grad_zero(const blt_model *model, blt_model_grad *grad);
void blt_model_grad_add(const blt_model *model, blt_model_grad *dst, const blt_model_grad *src);
void blt_model_grad_scale(const blt_model *model, blt_model_grad *grad, float scale);

// Refreshes every BF16 shadow weight from its FP32 master. Call once per
// step after the optimizer update and before the next forward pass.
void blt_model_refresh_mixed(blt_model *model);

typedef struct {
    blt_tensor patch_out;         // P_final [num_patches, patch_dim]
    blt_tensor global_out;        // O       [num_patches, embed_dim]
    blt_tensor byte_hidden_out;   // h_final [seq_len, embed_dim]
    size_t *patch_doc_boundaries; // [num_docs] model-style patch starts; doc 0 starts at 0
} blt_model_enc_out;

// Model document boundaries use one entry per document: entry[d] is the
// byte offset where document d starts, entry[0] is 0, and entries are ordered.
// The model converts this representation to the mask builder's convention
// before invoking the encoder, global transformer, or decoder.
//
// Stage 1: entropy-free encoding. Runs local encoder + global transformer
// once, freezes the latents in *out. Inference controllers (BLT-S drafting)
// call this once per round then invoke blt_model_decode repeatedly.
void blt_model_encode(const blt_model *model,
                      const blt_tensor *bytes_in, // [seq_len] UINT8
                      const blt_patch_info *patches, size_t num_patches,
                      const size_t *doc_boundaries, // byte-indexed
                      size_t num_docs, blt_model_enc_out *out, blt_arena *arena);

// Stage 2: decoder-only pass over frozen encoder/global latents.
// bytes_in may be NULL iff loss_out is NULL (logits-only inference).
// d0_opts may be NULL for legacy behavior (D_0 = h_final everywhere).
// patches must describe the same segmentation the encode stage used,
// tiling [0, num_hfinal_rows) when extra rows are present.
void blt_model_decode(const blt_model *model, const blt_model_enc_out *enc, const blt_patch_info *patches,
                      size_t num_patches,
                      const blt_tensor *bytes_in,   // [seq_len] UINT8 or NULL (iff loss_out == NULL)
                      const blt_tensor *targets,    // [seq_len] UINT8 or NULL (for loss against alternate targets)
                      const size_t *doc_boundaries, // byte-indexed
                      size_t num_docs, const blt_local_decoder_d0_opts *d0_opts,
                      blt_tensor *logits_out, // [seq_len, vocab_size]
                      blt_tensor *loss_out,   // scalar or NULL
                      blt_arena *arena);

void blt_model_forward(const blt_model *model,
                       const blt_tensor *bytes_in, // [seq_len] UINT8
                       const blt_tensor *targets,  // [seq_len] UINT8 or NULL
                       const blt_patch_info *patches, size_t num_patches,
                       const size_t *doc_boundaries, // byte-indexed
                       size_t num_docs,
                       blt_tensor *logits_out, // [seq_len, 256]
                       blt_tensor *loss_out,   // scalar
                       blt_arena *arena);

void blt_model_backward(const blt_model *model, const blt_tensor *bytes_in, const blt_tensor *targets,
                        const blt_patch_info *patches, size_t num_patches, const size_t *doc_boundaries,
                        size_t num_docs, blt_model_grad *grad, blt_arena *arena);

#ifdef __cplusplus
}
#endif

#endif // BLT_MODELS_BLT_MODEL_H
