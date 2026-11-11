#ifndef BLT_MODELS_GLOBAL_TRANSFORMER_H
#define BLT_MODELS_GLOBAL_TRANSFORMER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "core/tensor.h"
#include "core/allocator.h"
#include "models/transformer.h"
#include "models/transformer_stack.h"

// BLT §3.1 — block-causal transformer over the patch sequence.
// Patch j attends to patches <= j within the same document, never across
// documents. Structurally just blt_transformer_forward stacked num_layers
// times with a block-causal patch mask (same as entropy_lm's byte mask,
// applied to patches).

typedef struct {
    size_t embed_dim;  // h_G
    size_t num_layers; // l_G
    size_t hidden_dim; // FFN width
    size_t num_heads;
    float rope_theta;
    size_t max_seq_len;    // max on num_patches
    blt_dtype mixed_dtype; // shadow dtype for matmul weights (FP32 = none)
} blt_global_transformer_config;

typedef struct {
    blt_global_transformer_config config;
    blt_transformer_stack stack;
} blt_global_transformer;

typedef struct {
    blt_transformer_stack_grad *stack_grad;
} blt_global_transformer_grad;

// Allocates weights zero-initialized. Shared RoPE cache is precomputed
// (via blt_rope_precompute, sized to max_seq_len) and shared by pointer
// across every layer's attn_config. Caller fills weight data afterward.
blt_global_transformer *blt_global_transformer_create(blt_arena *arena, const blt_global_transformer_config *config);

blt_global_transformer_grad *blt_global_transformer_grad_create(blt_arena *arena, const blt_global_transformer *model);

// patch_in:        [num_patches, embed_dim] FP32 — P_final from local encoder.
// doc_boundaries:   patch-index starts for documents 1..num_docs-1; document 0
//                   starts implicitly at patch 0. This is the mask-builder
//                   convention, not the model API's one-entry-per-document
//                   representation.
// patch_out:        [num_patches, embed_dim] FP32 — O, contextualized patch
//                   representations. No separate output projection.
//
// Builds a block-causal patch mask once (document-scoped via doc_boundaries)
// and runs num_layers transformer blocks over it.
void blt_global_transformer_forward(const blt_global_transformer *model, const blt_tensor *patch_in,
                                    const size_t *doc_boundaries, size_t num_docs, blt_tensor *patch_out,
                                    blt_arena *arena);

// Recompute-then-reverse: reruns forward caching per-layer intermediates,
// then walks layers in reverse.
void blt_global_transformer_backward(const blt_global_transformer *model, const blt_tensor *patch_in,
                                     const size_t *doc_boundaries, size_t num_docs, const blt_tensor *grad_patch_out,
                                     blt_tensor *grad_patch_in, blt_global_transformer_grad *grad, blt_arena *arena);

#ifdef __cplusplus
}
#endif

#endif // BLT_MODELS_GLOBAL_TRANSFORMER_H
