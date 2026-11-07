#include "models/local_common.h"

void blt_local_layer_storage_alloc(blt_arena *arena, blt_local_layer_storage *s, size_t embed_dim, size_t hidden_dim) {
    size_t norm_shape[1] = {embed_dim};
    size_t qkv_shape[2] = {embed_dim, 3 * embed_dim};
    size_t proj_shape[2] = {embed_dim, embed_dim};
    size_t ffn_up_shape[2] = {embed_dim, hidden_dim};
    size_t ffn_down_shape[2] = {hidden_dim, embed_dim};

    s->norm1_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
    s->attn_qkv_w = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
    s->attn_proj_w = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    s->norm2_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
    s->ffn_up_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
    s->ffn_gate_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
    s->ffn_down_w = blt_tensor_create(arena, ffn_down_shape, 2, BLT_DTYPE_FP32);

    // cross-attention weights are allocated on every layer even if the
    // config never fires them on this layer
    s->cross_norm_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
    s->cross_weight_q = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    s->cross_weight_k = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    s->cross_weight_v = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    s->cross_weight_proj = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
}

void blt_local_layer_grad_alloc(blt_arena *arena, blt_local_layer_grad *g, size_t embed_dim, size_t hidden_dim) {
    size_t norm_shape[1] = {embed_dim};
    size_t qkv_shape[2] = {embed_dim, 3 * embed_dim};
    size_t proj_shape[2] = {embed_dim, embed_dim};
    size_t ffn_up_shape[2] = {embed_dim, hidden_dim};
    size_t ffn_down_shape[2] = {hidden_dim, embed_dim};

    g->norm1_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
    g->attn_qkv_w = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
    g->attn_proj_w = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    g->norm2_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
    g->ffn_up_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
    g->ffn_gate_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
    g->ffn_down_w = blt_tensor_create(arena, ffn_down_shape, 2, BLT_DTYPE_FP32);

    g->cross_norm_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
    g->cross_weight_q = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    g->cross_weight_k = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    g->cross_weight_v = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
    g->cross_weight_proj = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);
}

void blt_local_layer_bf16_alloc(blt_arena *arena, blt_local_layer_bf16 *b, size_t embed_dim, size_t hidden_dim) {
    size_t qkv_shape[2] = {embed_dim, 3 * embed_dim};
    size_t proj_shape[2] = {embed_dim, embed_dim};
    size_t ffn_up_shape[2] = {embed_dim, hidden_dim};
    size_t ffn_down_shape[2] = {hidden_dim, embed_dim};

    b->attn_qkv_w = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_BF16);
    b->attn_proj_w = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_BF16);
    b->ffn_up_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_BF16);
    b->ffn_gate_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_BF16);
    b->ffn_down_w = blt_tensor_create(arena, ffn_down_shape, 2, BLT_DTYPE_BF16);
    b->cross_weight_q = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_BF16);
    b->cross_weight_k = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_BF16);
    b->cross_weight_v = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_BF16);
    b->cross_weight_proj = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_BF16);
}

void blt_local_layer_bf16_refresh(blt_local_layer_bf16 *b, const blt_local_layer_storage *s) {
    blt_mixed_weight_refresh(&b->attn_qkv_w, &s->attn_qkv_w);
    blt_mixed_weight_refresh(&b->attn_proj_w, &s->attn_proj_w);
    blt_mixed_weight_refresh(&b->ffn_up_w, &s->ffn_up_w);
    blt_mixed_weight_refresh(&b->ffn_gate_w, &s->ffn_gate_w);
    blt_mixed_weight_refresh(&b->ffn_down_w, &s->ffn_down_w);
    blt_mixed_weight_refresh(&b->cross_weight_q, &s->cross_weight_q);
    blt_mixed_weight_refresh(&b->cross_weight_k, &s->cross_weight_k);
    blt_mixed_weight_refresh(&b->cross_weight_v, &s->cross_weight_v);
    blt_mixed_weight_refresh(&b->cross_weight_proj, &s->cross_weight_proj);
}

bool blt_local_cross_attn_fires(blt_xattn_placement placement, bool cross_attn_all_layers, size_t num_layers,
                                size_t layer) {
    switch (placement) {
    case BLT_XATTN_NONE:
        return false;
    case BLT_XATTN_LAST:
        return layer + 1 == num_layers;
    case BLT_XATTN_ALL:
        return true;
    case BLT_XATTN_FIRST:
        return layer == 0;
    case BLT_XATTN_DEFAULT:
    default:
        return cross_attn_all_layers || (layer + 1 == num_layers);
    }
}

blt_transformer_weights blt_local_byte_weights_view(const blt_local_layer_storage *s) {
    blt_transformer_weights w = {0};
    w.norm1_weight = &s->norm1_weight;
    w.attn_qkv_w = &s->attn_qkv_w;
    w.attn_proj_w = &s->attn_proj_w;
    w.norm2_weight = &s->norm2_weight;
    w.ffn_up_w = &s->ffn_up_w;
    w.ffn_gate_w = &s->ffn_gate_w;
    w.ffn_down_w = &s->ffn_down_w;
    return w;
}

blt_cross_attention_weights blt_local_cross_weights_view(const blt_local_layer_storage *s) {
    blt_cross_attention_weights w = {0};
    w.weight_q = &s->cross_weight_q;
    w.weight_k = &s->cross_weight_k;
    w.weight_v = &s->cross_weight_v;
    w.weight_proj = &s->cross_weight_proj;
    return w;
}

blt_transformer_weights blt_local_byte_weights_view_mixed(const blt_local_layer_storage *s,
                                                          const blt_local_layer_bf16 *b) {
    blt_transformer_weights w = blt_local_byte_weights_view(s);
    if (b == NULL) {
        return w;
    }
    w.attn_qkv_w = &b->attn_qkv_w;
    w.attn_proj_w = &b->attn_proj_w;
    w.ffn_up_w = &b->ffn_up_w;
    w.ffn_gate_w = &b->ffn_gate_w;
    w.ffn_down_w = &b->ffn_down_w;
    return w;
}

blt_cross_attention_weights blt_local_cross_weights_view_mixed(const blt_local_layer_storage *s,
                                                               const blt_local_layer_bf16 *b) {
    blt_cross_attention_weights w = blt_local_cross_weights_view(s);
    if (b == NULL) {
        return w;
    }
    w.weight_q = &b->cross_weight_q;
    w.weight_k = &b->cross_weight_k;
    w.weight_v = &b->cross_weight_v;
    w.weight_proj = &b->cross_weight_proj;
    return w;
}

blt_transformer_layer_grad blt_local_byte_grad_view(blt_local_layer_grad *lg) {
    blt_transformer_layer_grad g;
    g.norm1_weight = lg->norm1_weight;
    g.attn_qkv_w = lg->attn_qkv_w;
    g.attn_proj_w = lg->attn_proj_w;
    g.norm2_weight = lg->norm2_weight;
    g.ffn_up_w = lg->ffn_up_w;
    g.ffn_gate_w = lg->ffn_gate_w;
    g.ffn_down_w = lg->ffn_down_w;
    return g;
}

blt_transformer_config blt_local_byte_layer_config(size_t embed_dim, size_t num_heads, float rope_theta,
                                                   size_t hidden_dim, const blt_mask_config *local_mask,
                                                   const blt_tensor *rope_cos_view, const blt_tensor *rope_sin_view) {
    blt_transformer_config t = {0};
    t.attn_config.embed_dim = embed_dim;
    t.attn_config.num_heads = num_heads;
    t.attn_config.is_causal = true;
    t.attn_config.use_rope = true;
    t.attn_config.rope_theta = rope_theta;
    t.attn_config.rope_cos_cache = rope_cos_view;
    t.attn_config.rope_sin_cache = rope_sin_view;
    t.attn_config.mask_config = local_mask;
    t.hidden_dim = hidden_dim;
    t.norm_type = BLT_NORM_RMSNORM;
    t.activation_type = BLT_ACTIVATION_SWIGLU;
    return t;
}
