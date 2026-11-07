#ifndef BLT_LOCAL_DECODER_H
#define BLT_LOCAL_DECODER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

#include "core/tensor.h"
#include "core/allocator.h"
#include "ops/patch_pool.h"
#include "models/patcher.h"
#include "models/local_common.h"

typedef struct {
    size_t embed_dim;  // h_D
    size_t patch_dim;  // h_G
    size_t num_layers; // l_D
    size_t hidden_dim;
    size_t num_heads; // byte self-attention heads
    size_t cross_attn_heads;
    size_t local_window;                      // 0 = full causal, else causal sliding window
    bool cross_attn_all_layers;               // paper: decoder wants "All Layers" (Table 7)
    blt_xattn_placement cross_attn_placement; // sweep option; DEFAULT = use the bool above
    float rope_theta;
    size_t max_seq_len;
    size_t vocab_size; // 256, for the LM head
    bool use_bf16;     // allocate BF16 shadow copies of matmul weights
} blt_local_decoder_config;

// D_0 policy for rows without an encoder h_final state.
//
// The decoder's canonical input D_0 is the encoder's per-byte hidden state
// h_final. Speculative draft bytes (BLT-S) and masked block rows (BLT-D/DV)
// have no h_final — they sit beyond the encoded prefix — so their D_0 row
// is chosen by this policy. Only affects draft quality / acceptance rate:
// the verification step guarantees output equivalence regardless.

#define BLT_D0_VOCAB 257 // 256 byte values + MASK token at index 256

typedef enum {
    BLT_D0_HFINAL = 0, // every row of byte_hidden_in is real h_final
    BLT_D0_ZEROS,      // rows >= num_hfinal_rows get zero vectors
    BLT_D0_LEARNED     // rows >= num_hfinal_rows read d0_embed_weight[token]
} blt_d0_mode;

typedef struct {
    blt_d0_mode d0_mode;             // policy for rows >= num_hfinal_rows
    size_t num_hfinal_rows;          // rows [0, num_hfinal_rows) are real h_final;
                                     // ignored when d0_mode == BLT_D0_HFINAL
    const uint32_t *d0_extra_tokens; // [seq_len - num_hfinal_rows] token ids in [0, BLT_D0_VOCAB);
                                     // required iff d0_mode == BLT_D0_LEARNED
} blt_local_decoder_d0_opts;

// Per-layer weights: shared layout (cross-attn runs first in decoder layers)
typedef blt_local_layer_storage blt_local_decoder_layer_storage;

typedef struct {
    blt_local_decoder_config config;
    blt_tensor rope_cos_cache, rope_sin_cache;
    blt_local_decoder_layer_storage *layers; // [num_layers]
    blt_local_layer_bf16 *layers_bf16;       // [num_layers], NULL when use_bf16 is false
    blt_tensor lm_head_weight;               // [embed_dim, vocab_size]
    blt_tensor d0_embed_weight;              // [BLT_D0_VOCAB, embed_dim]
                                             // D_0 rows for draft/MASK positions
                                             // when d0_mode == BLT_D0_LEARNED.
                                             // Row 256 doubles as the MASK embedding.
                                             // Gradient written by diffusion backward;
                                             // legacy decoder backward leaves it zero.
} blt_local_decoder;

typedef blt_local_layer_grad blt_local_decoder_layer_grad;

typedef struct {
    blt_local_decoder_layer_grad *layer_grads; // [num_layers]
    blt_tensor lm_head_grad;
    blt_tensor d0_embed_grad; // [BLT_D0_VOCAB, embed_dim] (zero except under diffusion backward)
} blt_local_decoder_grad;

// Allocates zero-initialized weights, precomputes shared RoPE cache.
blt_local_decoder *blt_local_decoder_create(blt_arena *arena, const blt_local_decoder_config *config);

blt_local_decoder_grad *blt_local_decoder_grad_create(blt_arena *arena, const blt_local_decoder *model);

// Copies every layer's FP32 matmul weights into its BF16 shadow.
// No-op when the model was not created with use_bf16.
void blt_local_decoder_refresh_bf16(blt_local_decoder *model);

// doc_boundaries uses the mask-builder convention: entry[d] starts document
// d+1 and document 0 starts implicitly at byte 0; num_docs is the document
// count. The high-level model API converts its one-entry-per-document form.
//
// Legacy entry point: D_0 = byte_hidden_in for every row, loss always
// computed. Thin wrapper around blt_local_decoder_forward_ext with NULL opts.
void blt_local_decoder_forward(const blt_local_decoder *model,
                               const blt_tensor *byte_hidden_in, // [seq_len, embed_dim] — h_final from local encoder
                               const blt_tensor *patch_in,       // [num_patches, embed_dim] — O from global transformer
                               const blt_patch_info *patches, size_t num_patches,
                               const blt_tensor *bytes_in,   // [seq_len] UINT8 — needed only for the loss targets
                               const blt_tensor *targets,    // [seq_len] UINT8 — CE targets; NULL = use bytes_in
                               const size_t *doc_boundaries, // mask-builder convention; see above
                               size_t num_docs,
                               blt_tensor *logits_out, // [seq_len, vocab_size]
                               blt_tensor *loss_out,   // scalar
                               blt_arena *arena);

// Extended decoder forward (Fast-BLT).
// - loss_out may be NULL: skips shifted cross-entropy; bytes_in must also be NULL.
// - d0_opts may be NULL: legacy behavior (D_0 = byte_hidden_in everywhere).
//   Otherwise byte_hidden_in supplies only its first num_hfinal_rows; rows
//   beyond that get D_0 from the configured policy. Extra rows are causally
//   masked against nothing on their right and condition on the last patch's
//   latent sub-tokens via cross-attention.
// Numerics for rows [0, num_hfinal_rows) are bit-identical to running the
// legacy call on that prefix alone (causality guarantee).
void blt_local_decoder_forward_ext(
    const blt_local_decoder *model,
    const blt_tensor *byte_hidden_in, // [seq_len, embed_dim]; only first num_hfinal_rows read in non-HFINAL modes
    const blt_tensor *patch_in,       // [num_patches, patch_dim]
    const blt_patch_info *patches,    // must tile [0, num_hfinal_rows) exactly
    size_t num_patches,
    const blt_tensor *bytes_in,   // [seq_len] UINT8 or NULL (iff loss_out == NULL)
    const blt_tensor *targets,    // [seq_len] UINT8 — CE targets; NULL = use bytes_in
    const size_t *doc_boundaries, // mask-builder convention; see above
    size_t num_docs, const blt_local_decoder_d0_opts *d0_opts,
    int trailing_closed,    // 1 if a patch starts at seq_len; makes the last byte final (own latent)
    blt_tensor *logits_out, // [seq_len, vocab_size]
    blt_tensor *loss_out,   // scalar or NULL
    blt_arena *arena);

void blt_local_decoder_backward(const blt_local_decoder *model, const blt_tensor *byte_hidden_in,
                                const blt_tensor *patch_in, const blt_patch_info *patches, size_t num_patches,
                                const blt_tensor *bytes_in, const blt_tensor *targets, const size_t *doc_boundaries,
                                size_t num_docs,
                                blt_tensor *grad_byte_hidden_in, // [seq_len, embed_dim]
                                blt_tensor *grad_patch_in,       // [num_patches, embed_dim]
                                blt_local_decoder_grad *grad, blt_arena *arena);
#ifdef __cplusplus
}
#endif

#endif // BLT_LOCAL_DECODER_H
