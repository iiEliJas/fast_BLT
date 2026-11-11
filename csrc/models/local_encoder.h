#ifndef BLT_LOCAL_ENCODER_H
#define BLT_LOCAL_ENCODER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>

#include "core/tensor.h"
#include "core/allocator.h"
#include "ops/patch_pool.h"
#include "models/patcher.h"
#include "models/hash_ngram.h"
#include "core/dtype.h"
#include "models/local_common.h"

// Local Encoder: bytes -> embeddings -> P_0 = pool(e_0), then l_E
// transformer layers. Patch reps update via block-diagonal cross-attention:
//   P_l = P_{l-1} + cross_attn_l(query = RMSNorm(P_{l-1}), kv = h_l)

typedef struct {
    size_t embed_dim;                         // h_E
    size_t num_layers;                        // l_E, default: 1
    size_t patch_dim;                         // h_G, default: 0 = same as embed_dim
    size_t hidden_dim;                        // FFN intermediate width
    size_t num_heads;                         // byte self-attention heads
    size_t cross_attn_heads;                  // U_E
    size_t local_window;                      // w_E for byte self-attn (0 = full causal)
    bool cross_attn_all_layers;               // false = cross-attn only after final layer
    blt_xattn_placement cross_attn_placement; // sweep option; DEFAULT = use the bool above
    blt_patch_pool_type pool_type;            // Init strategy for P_0 (default: MEAN)
    blt_hash_ngram_config ngram_config;       // embed_dim must match config.embed_dim
    float rope_theta;                         // RoPE base frequency
    size_t max_seq_len;                       // Maximum sequence length for RoPE cache
    blt_dtype mixed_dtype;                    // shadow dtype for matmul weights (FP32 = none)
} blt_local_encoder_config;

typedef blt_local_layer_storage blt_local_encoder_layer_storage;

typedef struct {
    blt_local_encoder_config config;
    blt_tensor byte_embedding_weight;        // [256, embed_dim]
    blt_hash_ngram_weights ngram_weights;    // Hash n-gram lookup tables
    blt_tensor rope_cos_cache;               // [max_seq_len, head_dim / 2]
    blt_tensor rope_sin_cache;               // [max_seq_len, head_dim / 2]
    blt_local_encoder_layer_storage *layers; // [num_layers]
    blt_local_layer_mixed *layers_mixed;     // [num_layers], NULL when mixed_dtype is FP32
} blt_local_encoder;

typedef blt_local_layer_grad blt_local_encoder_layer_grad;

// Embedding/ngram grads use scatter-add; layer grads are overwritten.
typedef struct {
    blt_tensor embedding_grad;                 // [256, embed_dim]
    blt_hash_ngram_weights ngram_grads;        // Scatter-added gradient tables
    blt_local_encoder_layer_grad *layer_grads; // [num_layers]
} blt_local_encoder_grad;

// Allocates zero-initialized weights, initializes hash n-gram tables, and
// precomputes shared RoPE cache. Caller fills weight data afterward.
blt_local_encoder *blt_local_encoder_create(blt_arena *arena, const blt_local_encoder_config *config);

blt_local_encoder_grad *blt_local_encoder_grad_create(blt_arena *arena, const blt_local_encoder *model);

// bytes_in [seq_len] UINT8, patches [num_patches] tiling [0, seq_len),
// doc_boundaries uses the mask-builder convention: entry[d] starts document
// d+1 and document 0 starts implicitly at byte 0; num_docs is the document
// count. The high-level model API converts its one-entry-per-document form.
// Returns patch_out [num_patches, embed_dim] (P_final) and
// byte_hidden_out [seq_len, embed_dim] (h_final, kept for decoder).
// Embeds bytes + n-grams, pools initial patches (P_0), runs through byte
// transformer layers with block-diagonal cross-attention updates.
void blt_local_encoder_forward(const blt_local_encoder *model, const blt_tensor *bytes_in,
                               const blt_patch_info *patches, size_t num_patches, const size_t *doc_boundaries,
                               size_t num_docs, blt_tensor *patch_out, blt_tensor *byte_hidden_out, blt_arena *arena);

// Copies every layer's FP32 matmul weights into its mixed-precision shadow.
// No-op when the model was not created with shadows.
void blt_local_encoder_refresh_mixed(blt_local_encoder *model);

// grad_patch_out [num_patches, embed_dim] (dL/dP_final),
// grad_byte_hidden_out [seq_len, embed_dim] (dL/dh_final, optional).
// Recomputes forward activations per layer, then walks in reverse.
void blt_local_encoder_backward(const blt_local_encoder *model, const blt_tensor *bytes_in,
                                const blt_patch_info *patches, size_t num_patches, const size_t *doc_boundaries,
                                size_t num_docs, const blt_tensor *grad_patch_out,
                                const blt_tensor *grad_byte_hidden_out, blt_local_encoder_grad *grad, blt_arena *arena);

#ifdef __cplusplus
}
#endif

#endif // BLT_LOCAL_ENCODER_H
