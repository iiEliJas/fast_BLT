#include "core/backend.h"
#include "core/allocator.h"
#include "models/transformer_stack.h"
#include "models/attention.h"
#include "ops/matmul.h"
#include "ops/elementwise.h"
#include "ops/rmsnorm.h"
#include "ops/swiglu.h"
#include "ops/rope.h"
#include "ops/cast.h"

blt_transformer_weights_bf16 blt_transformer_layer_bf16_view(const blt_transformer_layer_storage *s, int use_bf16) {
    blt_transformer_weights_bf16 w = {0};
    if (!use_bf16) return w;
    w.attn_qkv_w = &s->attn_qkv_w_bf16;
    w.attn_proj_w = &s->attn_proj_w_bf16;
    w.ffn_up_w = &s->ffn_up_w_bf16;
    w.ffn_gate_w = &s->ffn_gate_w_bf16;
    w.ffn_down_w = &s->ffn_down_w_bf16;
    return w;
}

void blt_mixed_weight_refresh(blt_tensor *bf16_copy, const blt_tensor *fp32_master) {
    BLT_REQUIRE(bf16_copy != NULL && fp32_master != NULL,
                "blt_mixed_weight_refresh: bf16_copy and fp32_master must not be NULL");
    BLT_REQUIRE(bf16_copy->numel == fp32_master->numel,
                "blt_mixed_weight_refresh: bf16_copy and fp32_master must have same numel");
    blt_cast(fp32_master, bf16_copy);
}

void blt_transformer_stack_init(blt_arena *arena, blt_transformer_stack *stack,
                                const blt_transformer_stack_config *config) {
    BLT_REQUIRE(arena != NULL && stack != NULL && config != NULL,
                "blt_transformer_stack_init: arena, stack, and config cannot be NULL");
    BLT_REQUIRE(config->num_layers > 0, "blt_transformer_stack_init: num_layers must be > 0");
    BLT_REQUIRE(config->num_heads > 0 && config->embed_dim % config->num_heads == 0,
                "blt_transformer_stack_init: embed_dim must be divisible by num_heads");
    BLT_REQUIRE(config->max_seq_len > 0, "blt_transformer_stack_init: max_seq_len must be > 0");

    size_t embed_dim = config->embed_dim;
    size_t hidden_dim = config->hidden_dim;
    size_t num_heads = config->num_heads;
    size_t head_dim = embed_dim / num_heads;
    BLT_REQUIRE(head_dim % 2 == 0, "blt_transformer_stack_init: RoPE requires an even head_dim");

    stack->num_layers = config->num_layers;
    stack->embed_dim = embed_dim;
    stack->hidden_dim = hidden_dim;
    stack->num_heads = num_heads;
    stack->head_dim = head_dim;
    stack->max_seq_len = config->max_seq_len;

    // Shared RoPE cache, computed once (sized to max_seq_len).
    size_t half = head_dim / 2;
    size_t rope_shape[2] = {config->max_seq_len, half};
    stack->rope_cos_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    stack->rope_sin_cache = blt_tensor_create(arena, rope_shape, 2, BLT_DTYPE_FP32);
    blt_rope_config rope_cfg = {.theta = config->rope_theta, .head_dim = head_dim};
    blt_rope_precompute(config->max_seq_len, &rope_cfg, &stack->rope_cos_cache, &stack->rope_sin_cache);

    // Shared transformer config template for every layer/call. Per-call
    // sites overlay their own seq_len-sized RoPE view (and, for callers
    // needing block-causal masking, a mask_config) via
    // blt_transformer_stack_call_config().
    stack->layer_config = (blt_transformer_config){0}; // zero out
    stack->layer_config.attn_config.embed_dim = embed_dim;
    stack->layer_config.attn_config.num_heads = num_heads;
    stack->layer_config.attn_config.head_dim = head_dim;
    stack->layer_config.attn_config.is_causal = true;
    stack->layer_config.attn_config.use_rope = true;
    stack->layer_config.attn_config.rope_theta = config->rope_theta;
    stack->layer_config.attn_config.rope_cos_cache = &stack->rope_cos_cache;
    stack->layer_config.attn_config.rope_sin_cache = &stack->rope_sin_cache;
    stack->layer_config.hidden_dim = hidden_dim;
    stack->layer_config.layer_norm_eps = 1e-6f; // not used by RMSNorm
    stack->layer_config.norm_type = BLT_NORM_RMSNORM;
    stack->layer_config.activation_type = BLT_ACTIVATION_SWIGLU;
    stack->layer_config.use_bf16 = config->use_bf16;
    stack->use_bf16 = config->use_bf16;
    stack->use_fp8 = config->use_fp8;

    stack->layer_storage = (blt_transformer_layer_storage *)blt_container_alloc(
        arena, config->num_layers * sizeof(blt_transformer_layer_storage));
    stack->layer_weights =
        (blt_transformer_weights *)blt_container_alloc(arena, config->num_layers * sizeof(blt_transformer_weights));

    stack->layer_weights_bf16 = NULL;
    if (config->use_bf16) {
        stack->layer_weights_bf16 = (blt_transformer_weights_bf16 *)blt_container_alloc(
            arena, config->num_layers * sizeof(blt_transformer_weights_bf16));
    }
    stack->layer_weights_fp8 = NULL;
    if (config->use_fp8) {
        stack->layer_weights_fp8 =
            (blt_transformer_weights *)blt_container_alloc(arena, config->num_layers * sizeof(blt_transformer_weights));
    }

    for (size_t l = 0; l < config->num_layers; ++l) {
        blt_transformer_layer_storage *s = &stack->layer_storage[l];

        size_t norm_shape[1] = {embed_dim};
        s->norm1_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
        s->norm2_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);

        size_t qkv_shape[2] = {embed_dim, 3 * embed_dim};
        s->attn_qkv_w = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);

        size_t proj_shape[2] = {embed_dim, embed_dim};
        s->attn_proj_w = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);

        size_t ffn_up_shape[2] = {embed_dim, hidden_dim};
        s->ffn_up_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
        s->ffn_gate_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);

        size_t ffn_down_shape[2] = {hidden_dim, embed_dim};
        s->ffn_down_w = blt_tensor_create(arena, ffn_down_shape, 2, BLT_DTYPE_FP32);

        if (config->use_bf16) {
            s->attn_qkv_w_bf16 = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_BF16);
            s->attn_proj_w_bf16 = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_BF16);
            s->ffn_up_w_bf16 = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_BF16);
            s->ffn_gate_w_bf16 = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_BF16);
            s->ffn_down_w_bf16 = blt_tensor_create(arena, ffn_down_shape, 2, BLT_DTYPE_BF16);
        }
        if (config->use_fp8) {
            s->attn_qkv_w_fp8 = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP8_E4M3);
            s->attn_proj_w_fp8 = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP8_E4M3);
            s->ffn_up_w_fp8 = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP8_E4M3);
            s->ffn_gate_w_fp8 = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP8_E4M3);
            s->ffn_down_w_fp8 = blt_tensor_create(arena, ffn_down_shape, 2, BLT_DTYPE_FP8_E4M3);
        }

        blt_transformer_weights *w = &stack->layer_weights[l];
        w->norm1_weight = &s->norm1_weight;
        w->norm1_bias = NULL; // no bias for RMSNorm
        w->attn_qkv_w = &s->attn_qkv_w;
        w->attn_proj_w = &s->attn_proj_w;
        w->norm2_weight = &s->norm2_weight;
        w->norm2_bias = NULL;
        w->ffn_up_w = &s->ffn_up_w;
        w->ffn_gate_w = &s->ffn_gate_w;
        w->ffn_down_w = &s->ffn_down_w;

        if (config->use_bf16) {
            stack->layer_weights_bf16[l] = blt_transformer_layer_bf16_view(s, 1);
        }
        if (config->use_fp8) {
            blt_transformer_weights *wfp = &stack->layer_weights_fp8[l];
            wfp->norm1_weight = &s->norm1_weight;
            wfp->norm1_bias = NULL;
            wfp->attn_qkv_w = &s->attn_qkv_w_fp8;
            wfp->attn_proj_w = &s->attn_proj_w_fp8;
            wfp->norm2_weight = &s->norm2_weight;
            wfp->norm2_bias = NULL;
            wfp->ffn_up_w = &s->ffn_up_w_fp8;
            wfp->ffn_gate_w = &s->ffn_gate_w_fp8;
            wfp->ffn_down_w = &s->ffn_down_w_fp8;
        }
    }
}

blt_transformer_stack_grad *blt_transformer_stack_grad_create(blt_arena *arena, const blt_transformer_stack *stack) {
    BLT_REQUIRE(arena != NULL && stack != NULL, "blt_transformer_stack_grad_create: arena and stack cannot be NULL");

    blt_transformer_stack_grad *grad =
        (blt_transformer_stack_grad *)blt_container_alloc(arena, sizeof(blt_transformer_stack_grad));

    size_t embed_dim = stack->embed_dim;
    size_t hidden_dim = stack->hidden_dim;
    size_t num_layers = stack->num_layers;

    grad->layer_grads =
        (blt_transformer_layer_grad *)blt_container_alloc(arena, num_layers * sizeof(blt_transformer_layer_grad));

    for (size_t l = 0; l < num_layers; ++l) {
        blt_transformer_layer_grad *lg = &grad->layer_grads[l];

        size_t norm_shape[1] = {embed_dim};
        lg->norm1_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);
        lg->norm2_weight = blt_tensor_create(arena, norm_shape, 1, BLT_DTYPE_FP32);

        size_t qkv_shape[2] = {embed_dim, 3 * embed_dim};
        lg->attn_qkv_w = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);

        size_t proj_shape[2] = {embed_dim, embed_dim};
        lg->attn_proj_w = blt_tensor_create(arena, proj_shape, 2, BLT_DTYPE_FP32);

        size_t ffn_up_shape[2] = {embed_dim, hidden_dim};
        lg->ffn_up_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);
        lg->ffn_gate_w = blt_tensor_create(arena, ffn_up_shape, 2, BLT_DTYPE_FP32);

        size_t ffn_down_shape[2] = {hidden_dim, embed_dim};
        lg->ffn_down_w = blt_tensor_create(arena, ffn_down_shape, 2, BLT_DTYPE_FP32);
    }

    return grad;
}

blt_transformer_config blt_transformer_stack_call_config(const blt_transformer_stack *stack, size_t seq_len,
                                                         blt_tensor *cos_view, blt_tensor *sin_view) {
    BLT_REQUIRE(stack != NULL && cos_view != NULL && sin_view != NULL,
                "blt_transformer_stack_call_config: stack, cos_view, and sin_view cannot be NULL");
    BLT_REQUIRE(seq_len >= 1 && seq_len <= stack->max_seq_len,
                "blt_transformer_stack_call_config: seq_len must be in [1, max_seq_len] (RoPE cache too small)");

    size_t half = stack->head_dim / 2;
    blt_tensor_view_2d(cos_view, stack->rope_cos_cache.data, seq_len, half, stack->rope_cos_cache.backend);
    blt_tensor_view_2d(sin_view, stack->rope_sin_cache.data, seq_len, half, stack->rope_sin_cache.backend);

    blt_transformer_config cfg = stack->layer_config; // copy shared template
    cfg.attn_config.rope_cos_cache = cos_view;
    cfg.attn_config.rope_sin_cache = sin_view;
    return cfg;
}

// norm weights stay fp32 — bf16 only accelerates matmul
static blt_transformer_weights bf16_weights_view(const blt_transformer_weights *w,
                                                 const blt_transformer_weights_bf16 *w_bf16) {
    blt_transformer_weights bw = *w;
    if (w_bf16->attn_qkv_w) bw.attn_qkv_w = w_bf16->attn_qkv_w;
    if (w_bf16->attn_proj_w) bw.attn_proj_w = w_bf16->attn_proj_w;
    if (w_bf16->ffn_up_w) bw.ffn_up_w = w_bf16->ffn_up_w;
    if (w_bf16->ffn_gate_w) bw.ffn_gate_w = w_bf16->ffn_gate_w;
    if (w_bf16->ffn_down_w) bw.ffn_down_w = w_bf16->ffn_down_w;
    return bw;
}

void blt_transformer_stack_forward(const blt_transformer_stack *stack, const blt_tensor *x,
                                   const blt_transformer_config *call_cfg, size_t seq_len, blt_tensor *out,
                                   blt_arena *arena) {
    BLT_REQUIRE(stack != NULL && x != NULL && call_cfg != NULL && out != NULL && arena != NULL,
                "blt_transformer_stack_forward: arguments cannot be NULL");

    size_t embed_shape[2] = {seq_len, stack->embed_dim};
    blt_tensor cur = *x;
    for (size_t l = 0; l < stack->num_layers; ++l) {
        blt_tensor next = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
        const blt_transformer_weights *w = &stack->layer_weights[l];
        blt_transformer_weights bf16_w;
        if (stack->use_bf16 && stack->layer_weights_bf16) {
            bf16_w = bf16_weights_view(w, &stack->layer_weights_bf16[l]);
            w = &bf16_w;
        }
        blt_transformer_forward(&cur, w, &next, call_cfg, arena);
        cur = next;
    }
    *out = cur;
}

// Forward recomputes every layer's intermediates here since
// blt_transformer_forward doesn't save them.
struct blt_transformer_layer_cache {
    blt_tensor x_in; // layer input
    blt_tensor norm1_out;
    blt_tensor attn_out;
    blt_tensor attn_residual;
    blt_tensor norm2_out;
    blt_tensor ffn_up;
    blt_tensor ffn_gate;
    blt_tensor ffn_activated;
};

struct blt_transformer_stack_cache {
    blt_transformer_layer_cache **layers; // [num_layers], each arena-allocated
    size_t num_layers;
};

blt_transformer_layer_cache *blt_transformer_layer_forward_cached(const blt_tensor *x, const blt_transformer_weights *w,
                                                                  const blt_transformer_config *cfg, blt_arena *arena,
                                                                  size_t seq_len, size_t embed_dim, size_t hidden_dim,
                                                                  blt_tensor *out) {
    BLT_REQUIRE(x != NULL && w != NULL && cfg != NULL && arena != NULL && out != NULL,
                "blt_transformer_layer_forward_cached: arguments cannot be NULL");

    blt_transformer_layer_cache *cache =
        (blt_transformer_layer_cache *)blt_container_alloc(arena, sizeof(blt_transformer_layer_cache));

    size_t embed_shape[2] = {seq_len, embed_dim};
    size_t hidden_shape[2] = {seq_len, hidden_dim};

    cache->x_in = *x;

    cache->norm1_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_rmsnorm_forward(x, w->norm1_weight, &cache->norm1_out);

    cache->attn_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_multihead_attention(&cache->norm1_out, w->attn_qkv_w, w->attn_proj_w, &cache->attn_out, &cfg->attn_config,
                            arena);

    cache->attn_residual = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_add(x, &cache->attn_out, &cache->attn_residual);

    cache->norm2_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_rmsnorm_forward(&cache->attn_residual, w->norm2_weight, &cache->norm2_out);

    cache->ffn_up = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&cache->norm2_out, w->ffn_up_w, &cache->ffn_up);

    cache->ffn_gate = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&cache->norm2_out, w->ffn_gate_w, &cache->ffn_gate);

    cache->ffn_activated = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_swiglu_forward(&cache->ffn_gate, &cache->ffn_up, &cache->ffn_activated);

    blt_tensor ffn_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(&cache->ffn_activated, w->ffn_down_w, &ffn_out);

    *out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_add(&cache->attn_residual, &ffn_out, out);

    return cache;
}

void blt_transformer_layer_backward(const blt_transformer_layer_cache *cache, const blt_transformer_weights *w,
                                    const blt_transformer_config *cfg, blt_arena *arena, size_t seq_len,
                                    size_t embed_dim, size_t hidden_dim, const blt_tensor *grad_out,
                                    blt_transformer_layer_grad *lg, blt_tensor *grad_x) {
    BLT_REQUIRE(cache != NULL && w != NULL && cfg != NULL && arena != NULL && grad_out != NULL && lg != NULL &&
                    grad_x != NULL,
                "blt_transformer_layer_backward: arguments cannot be NULL");

    size_t embed_shape[2] = {seq_len, embed_dim};
    size_t hidden_shape[2] = {seq_len, hidden_dim};

    blt_tensor grad_ffn_activated = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_matmul_backward(&cache->ffn_activated, w->ffn_down_w, grad_out, &grad_ffn_activated, &lg->ffn_down_w);

    blt_tensor grad_gate = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_tensor grad_up = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
    blt_swiglu_backward(&grad_ffn_activated, &cache->ffn_gate, &cache->ffn_up, &grad_gate, &grad_up);

    blt_tensor grad_norm2_a = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_matmul_backward(&cache->norm2_out, w->ffn_up_w, &grad_up, &grad_norm2_a, &lg->ffn_up_w);

    blt_tensor grad_norm2_b = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_matmul_backward(&cache->norm2_out, w->ffn_gate_w, &grad_gate, &grad_norm2_b, &lg->ffn_gate_w);

    blt_tensor grad_attn_residual_from_norm2 = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_rmsnorm_backward_add(&grad_norm2_a, &grad_norm2_b, &cache->attn_residual, w->norm2_weight,
                             &grad_attn_residual_from_norm2, &lg->norm2_weight);

    blt_tensor grad_attn_residual = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_add(grad_out, &grad_attn_residual_from_norm2, &grad_attn_residual);

    blt_tensor grad_norm1_out = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_multihead_attention_backward(&cache->norm1_out, w->attn_qkv_w, w->attn_proj_w, &grad_attn_residual,
                                     &grad_norm1_out, &lg->attn_qkv_w, &lg->attn_proj_w, &cfg->attn_config, arena);

    blt_tensor grad_x_from_norm1 = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_rmsnorm_backward(&grad_norm1_out, &cache->x_in, w->norm1_weight, &grad_x_from_norm1, &lg->norm1_weight);

    *grad_x = blt_tensor_create(arena, embed_shape, 2, BLT_DTYPE_FP32);
    blt_add(&grad_attn_residual, &grad_x_from_norm1, grad_x);
}

blt_transformer_stack_cache *blt_transformer_stack_forward_cached(const blt_transformer_stack *stack,
                                                                  const blt_tensor *x,
                                                                  const blt_transformer_config *call_cfg,
                                                                  size_t seq_len, blt_arena *arena, blt_tensor *out) {
    BLT_REQUIRE(stack != NULL && x != NULL && call_cfg != NULL && out != NULL && arena != NULL,
                "blt_transformer_stack_forward_cached: arguments cannot be NULL");

    blt_transformer_stack_cache *cache =
        (blt_transformer_stack_cache *)blt_container_alloc(arena, sizeof(blt_transformer_stack_cache));

    cache->num_layers = stack->num_layers;
    cache->layers = (blt_transformer_layer_cache **)blt_container_alloc(
        arena, stack->num_layers * sizeof(blt_transformer_layer_cache *));

    blt_tensor cur = *x;
    for (size_t l = 0; l < stack->num_layers; ++l) {
        blt_tensor next;
        const blt_transformer_weights *w = &stack->layer_weights[l];
        blt_transformer_weights bf16_w;
        if (stack->use_bf16 && stack->layer_weights_bf16) {
            bf16_w = bf16_weights_view(w, &stack->layer_weights_bf16[l]);
            w = &bf16_w;
        }
        cache->layers[l] = blt_transformer_layer_forward_cached(&cur, w, call_cfg, arena, seq_len, stack->embed_dim,
                                                                stack->hidden_dim, &next);
        cur = next;
    }
    *out = cur;
    return cache;
}

void blt_transformer_stack_backward(const blt_transformer_stack *stack, const blt_transformer_stack_cache *cache,
                                    const blt_transformer_config *call_cfg, size_t seq_len, const blt_tensor *grad_out,
                                    blt_transformer_stack_grad *grad, blt_tensor *grad_x, blt_arena *arena) {
    BLT_REQUIRE(stack != NULL && cache != NULL && call_cfg != NULL && grad_out != NULL && grad != NULL &&
                    grad_x != NULL && arena != NULL,
                "blt_transformer_stack_backward: arguments cannot be NULL");
    BLT_REQUIRE(cache->num_layers == stack->num_layers,
                "blt_transformer_stack_backward: cache was not produced by this stack");

    blt_tensor g = *grad_out;
    for (size_t i = 0; i < stack->num_layers; ++i) {
        size_t l = stack->num_layers - 1 - i; // walk layers in reverse
        blt_tensor grad_prev;
        const blt_transformer_weights *w = &stack->layer_weights[l];
        blt_transformer_weights bf16_w;
        if (stack->use_bf16 && stack->layer_weights_bf16) {
            bf16_w = bf16_weights_view(w, &stack->layer_weights_bf16[l]);
            w = &bf16_w;
        }
        blt_transformer_layer_backward(cache->layers[l], w, call_cfg, arena, seq_len, stack->embed_dim,
                                       stack->hidden_dim, &g, &grad->layer_grads[l], &grad_prev);
        g = grad_prev;
    }
    *grad_x = g;
}

void blt_transformer_stack_refresh_bf16(blt_transformer_stack *stack) {
    if (!stack || !stack->use_bf16) return;
    for (size_t l = 0; l < stack->num_layers; l++) {
        blt_transformer_layer_storage *s = &stack->layer_storage[l];
        blt_mixed_weight_refresh(&s->attn_qkv_w_bf16, &s->attn_qkv_w);
        blt_mixed_weight_refresh(&s->attn_proj_w_bf16, &s->attn_proj_w);
        blt_mixed_weight_refresh(&s->ffn_up_w_bf16, &s->ffn_up_w);
        blt_mixed_weight_refresh(&s->ffn_gate_w_bf16, &s->ffn_gate_w);
        blt_mixed_weight_refresh(&s->ffn_down_w_bf16, &s->ffn_down_w);
    }
}

void blt_transformer_stack_refresh_fp8(blt_transformer_stack *stack) {
    if (!stack || !stack->use_fp8 || !stack->layer_weights_fp8) return;
    for (size_t l = 0; l < stack->num_layers; l++) {
        blt_transformer_layer_storage *s = &stack->layer_storage[l];
        blt_cast(&s->attn_qkv_w, &s->attn_qkv_w_fp8);
        blt_cast(&s->attn_proj_w, &s->attn_proj_w_fp8);
        blt_cast(&s->ffn_up_w, &s->ffn_up_w_fp8);
        blt_cast(&s->ffn_gate_w, &s->ffn_gate_w_fp8);
        blt_cast(&s->ffn_down_w, &s->ffn_down_w_fp8);
    }
}
