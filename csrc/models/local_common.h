#ifndef BLT_MODELS_LOCAL_COMMON_H
#define BLT_MODELS_LOCAL_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

#include "core/tensor.h"
#include "core/allocator.h"
#include "models/transformer.h"
#include "models/cross_attention.h"

// Shared per-layer weight layout for the local encoder and decoder.
// Both compose each layer from a cross-attention block (RMSNorm -> QKV
// cross-attn) and a byte-level transformer block (causal self-attn + FFN).
// blt_local_{encoder,decoder}_layer_storage are aliases of this type.

typedef struct {
    blt_tensor norm1_weight; // [embed_dim]
    blt_tensor attn_qkv_w;   // [embed_dim, 3 * embed_dim]
    blt_tensor attn_proj_w;  // [embed_dim, embed_dim]
    blt_tensor norm2_weight; // [embed_dim]
    blt_tensor ffn_up_w;     // [embed_dim, hidden_dim]
    blt_tensor ffn_gate_w;   // [embed_dim, hidden_dim]
    blt_tensor ffn_down_w;   // [hidden_dim, embed_dim]
    // Cross-attention block — allocated on every layer, fires per config.
    blt_tensor cross_norm_weight; // [embed_dim]
    blt_tensor cross_weight_q;    // [embed_dim, embed_dim]
    blt_tensor cross_weight_k;    // [embed_dim, embed_dim]
    blt_tensor cross_weight_v;    // [embed_dim, embed_dim]
    blt_tensor cross_weight_proj; // [embed_dim, embed_dim]
} blt_local_layer_storage;

typedef struct {
    blt_tensor norm1_weight;
    blt_tensor attn_qkv_w;
    blt_tensor attn_proj_w;
    blt_tensor norm2_weight;
    blt_tensor ffn_up_w;
    blt_tensor ffn_gate_w;
    blt_tensor ffn_down_w;
    blt_tensor cross_norm_weight;
    blt_tensor cross_weight_q;
    blt_tensor cross_weight_k;
    blt_tensor cross_weight_v;
    blt_tensor cross_weight_proj;
} blt_local_layer_grad;

void blt_local_layer_storage_alloc(blt_arena *arena, blt_local_layer_storage *s, size_t embed_dim, size_t hidden_dim);

void blt_local_layer_grad_alloc(blt_arena *arena, blt_local_layer_grad *g, size_t embed_dim, size_t hidden_dim);

// BF16 shadows of the matmul-participating weights in blt_local_layer_storage.
// Norm weights stay FP32; these copies exist only to feed mixed-precision
// matmuls and are refreshed from the FP32 masters each step.
typedef struct {
    blt_tensor attn_qkv_w;     // [embed_dim, 3*embed_dim]
    blt_tensor attn_proj_w;    // [embed_dim, embed_dim]
    blt_tensor ffn_up_w;       // [embed_dim, hidden_dim]
    blt_tensor ffn_gate_w;     // [embed_dim, hidden_dim]
    blt_tensor ffn_down_w;     // [hidden_dim, embed_dim]
    blt_tensor cross_weight_q; // [embed_dim, embed_dim]
    blt_tensor cross_weight_k;
    blt_tensor cross_weight_v;
    blt_tensor cross_weight_proj;
} blt_local_layer_bf16;

void blt_local_layer_bf16_alloc(blt_arena *arena, blt_local_layer_bf16 *b, size_t embed_dim, size_t hidden_dim);

void blt_local_layer_bf16_refresh(blt_local_layer_bf16 *b, const blt_local_layer_storage *s);

// Cross-attention placement modes for ablation sweeps.
// DEFAULT keeps legacy bool behavior of cross_attn_all_layers;
// explicit modes override it. FIRST is only meaningful for the decoder.
typedef enum {
    BLT_XATTN_DEFAULT = 0,
    BLT_XATTN_NONE,
    BLT_XATTN_LAST,
    BLT_XATTN_ALL,
    BLT_XATTN_FIRST
} blt_xattn_placement;

// Returns whether cross-attention fires at a given layer. Explicit placement
// (anything but DEFAULT) overrides the cross_attn_all_layers bool entirely.
bool blt_local_cross_attn_fires(blt_xattn_placement placement, bool cross_attn_all_layers, size_t num_layers,
                                size_t layer);

// Wrappers that view a shared layer's tensors as the op-level weight/grad
// structs expected by blt_transformer_forward/backward and blt_cross_attention_*.
blt_transformer_weights blt_local_byte_weights_view(const blt_local_layer_storage *s);
blt_cross_attention_weights blt_local_cross_weights_view(const blt_local_layer_storage *s);
blt_transformer_layer_grad blt_local_byte_grad_view(blt_local_layer_grad *lg);

// Same views, with the matmul weights overridden by their BF16 shadows when
// b is non-NULL. Norm weights always point at the FP32 masters.
blt_transformer_weights blt_local_byte_weights_view_mixed(const blt_local_layer_storage *s,
                                                          const blt_local_layer_bf16 *b);
blt_cross_attention_weights blt_local_cross_weights_view_mixed(const blt_local_layer_storage *s,
                                                               const blt_local_layer_bf16 *b);

// Builds the byte-transformer-layer config: RMSNorm + SwiGLU + causal
// RoPE'd self-attention over local_mask.
blt_transformer_config blt_local_byte_layer_config(size_t embed_dim, size_t num_heads, float rope_theta,
                                                   size_t hidden_dim, const blt_mask_config *local_mask,
                                                   const blt_tensor *rope_cos_view, const blt_tensor *rope_sin_view);

#ifdef __cplusplus
}
#endif

#endif // BLT_MODELS_LOCAL_COMMON_H
