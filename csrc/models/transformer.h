#ifndef BLT_MODELS_TRANSFORMER_H
#define BLT_MODELS_TRANSFORMER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "core/tensor.h"
#include "core/allocator.h"
#include "core/dtype.h"
#include "models/attention.h"

// normalization the block uses. RMSNorm has no bias term, so
// norm1_bias/norm2_bias in blt_transformer_weights are ignored (may be
// NULL) when BLT_NORM_RMSNORM is set.
typedef enum {
    BLT_NORM_LAYERNORM = 0, // classic pre-norm block: mean/var norm + weight + bias
    BLT_NORM_RMSNORM = 1,   // BLT-style block: RMS norm + weight only
} blt_norm_type;

// FFN activation the block uses. SwiGLU needs a gate projection in
// addition to up/down, so ffn_gate_w in blt_transformer_weights is
// required (non-NULL) when BLT_ACTIVATION_SWIGLU is set, and ignored
// otherwise.
typedef enum {
    BLT_ACTIVATION_GELU = 0,   // out = down(gelu(up(x)))
    BLT_ACTIVATION_SWIGLU = 1, // out = down(silu(gate(x)) * up(x))
} blt_activation_type;

// A classic transformer block and BLT's local encoder/decoder/patch-transformer
// blocks are the same function with a different config.
typedef struct {
    blt_attention_config attn_config;
    size_t hidden_dim;    // FFN hidden width
    float layer_norm_eps; // used only when norm_type == BLT_NORM_LAYERNORM
    blt_norm_type norm_type;
    blt_activation_type activation_type;
    // Dtype of the matmul-weight shadows the caller substitutes into
    // blt_transformer_weights. FP32 means "no shadows". BF16/FP16 shadows hold
    // the same values at lower precision; FP8 shadows carry a per-tensor scale
    // in blt_tensor.scale. Norm weights and all activations stay FP32.
    blt_dtype mixed_dtype;
} blt_transformer_config;

// All learned weights for one block. Fields whose relevance depends on
// config are documented above (norm1_bias/norm2_bias, ffn_gate_w).
typedef struct {
    const blt_tensor *norm1_weight;
    const blt_tensor *norm1_bias; // NULL when norm_type == BLT_NORM_RMSNORM
    const blt_tensor *attn_qkv_w;
    const blt_tensor *attn_proj_w;
    const blt_tensor *norm2_weight;
    const blt_tensor *norm2_bias; // NULL when norm_type == BLT_NORM_RMSNORM
    const blt_tensor *ffn_up_w;
    const blt_tensor *ffn_gate_w; // required (non-NULL) when activation_type == BLT_ACTIVATION_SWIGLU
    const blt_tensor *ffn_down_w;
} blt_transformer_weights;

// Owned storage for one transformer layer's weights. A parallel
// blt_transformer_weights entry (of const pointers into this struct) is
// what actually gets passed to blt_transformer_forward.
typedef struct {
    blt_tensor norm1_weight;
    blt_tensor attn_qkv_w;
    blt_tensor attn_proj_w;
    blt_tensor norm2_weight;
    blt_tensor ffn_up_w;
    blt_tensor ffn_gate_w;
    blt_tensor ffn_down_w;
    // Mixed-precision shadows (empty when mixed_dtype is FP32). The *_scale
    // entries are the per-tensor FP8 scales the shadows point at.
    blt_tensor attn_qkv_w_mixed;
    blt_tensor attn_proj_w_mixed;
    blt_tensor ffn_up_w_mixed;
    blt_tensor ffn_gate_w_mixed;
    blt_tensor ffn_down_w_mixed;
    blt_tensor attn_qkv_w_scale;
    blt_tensor attn_proj_w_scale;
    blt_tensor ffn_up_w_scale;
    blt_tensor ffn_gate_w_scale;
    blt_tensor ffn_down_w_scale;
} blt_transformer_layer_storage;

typedef struct {
    blt_tensor norm1_weight;
    blt_tensor attn_qkv_w;
    blt_tensor attn_proj_w;
    blt_tensor norm2_weight;
    blt_tensor ffn_up_w;
    blt_tensor ffn_gate_w;
    blt_tensor ffn_down_w;
} blt_transformer_layer_grad;

// Pre-norm attention + residual, pre-norm FFN + residual. Norm and activation
// are chosen via config. `arena` is not reset by this function — caller
// controls intermediate lifetime. `w_bf16` may be NULL (fp32-only path).
void blt_transformer_forward(const blt_tensor *input, const blt_transformer_weights *weights, blt_tensor *output,
                             const blt_transformer_config *config, blt_arena *arena);

// Refresh one shadow from its FP32 master. BF16/FP16 shadow -> cast; FP8
// shadow -> amax-derived per-tensor scale (written to shadow->scale) plus
// quantization. Must run once per step, after the optimizer update and before
// the next forward pass, or the shadow silently goes stale.
void blt_mixed_weight_refresh(blt_tensor *shadow, const blt_tensor *fp32_master);

#ifdef __cplusplus
}
#endif

#endif // BLT_MODELS_TRANSFORMER_H