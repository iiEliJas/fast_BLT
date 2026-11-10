#include "core/backend.h"
#include "ops/elementwise.h"
#include "ops/matmul.h"
#include "ops/softmax.h"
#include "ops/rope.h"
#include "ops/layernorm.h"
#include "ops/rmsnorm.h"
#include "ops/vecmath.h"
#include "ops/gelu.h"
#include "ops/swiglu.h"
#include "ops/cross_entropy.h"
#include "ops/optim.h"
#include "ops/mask_builder.h"
#include "ops/patch_pool.h"
#include "ops/attn_core.h"
#include "ops/gather_scatter.h"
#include "ops/row_stats.h"
#include "ops/cast.h"
#include "ops/quant.h"
#include "core/tensor.h"
#include "core/cuda_shim.h"

int g_blt_deterministic = 0;

// CPU forward declarations
void blt_add_cpu(const blt_tensor *a, const blt_tensor *b, blt_tensor *out);
void blt_mul_cpu(const blt_tensor *a, const blt_tensor *b, blt_tensor *out);
void blt_gelu_forward_cpu(const blt_tensor *x, blt_tensor *out);
void blt_gelu_backward_cpu(const blt_tensor *grad_out, const blt_tensor *x, blt_tensor *grad_x);
void blt_swiglu_forward_cpu(const blt_tensor *gate, const blt_tensor *up, blt_tensor *out);
void blt_swiglu_backward_cpu(const blt_tensor *grad_out, const blt_tensor *gate, const blt_tensor *up,
                             blt_tensor *grad_gate, blt_tensor *grad_up);

void blt_matmul_cpu(const blt_tensor *a, const blt_tensor *b, blt_tensor *out);
void blt_matmul_backward_cpu(const blt_tensor *a, const blt_tensor *b, const blt_tensor *grad_out, blt_tensor *grad_a,
                             blt_tensor *grad_b);

float blt_vec_dot_cpu(blt_backend backend, const float *a, const float *b, size_t n);
void blt_vec_dot_into_cpu(blt_backend backend, const float *a, const float *b, size_t n, float *out);
void blt_softmax_masked_row_inplace_cpu(blt_backend backend, float *row, size_t row_len, size_t row_idx, bool is_causal,
                                        const float *mask_row, float scale);
void blt_strided_copy_cpu(blt_backend backend, float *dst, size_t dst_stride, const float *src, size_t src_stride,
                          size_t rows, size_t cols);
void blt_fill_uniform_cpu(blt_backend backend, float *data, size_t n, uint64_t *rng_state);
void blt_fill_constant_cpu(blt_backend backend, float *data, size_t n, float v);

void *blt_container_alloc_cpu(blt_arena *arena, size_t bytes);
void *blt_container_alloc_cuda(blt_arena *arena, size_t bytes);

void blt_scale_cpu(blt_tensor *t, float scalar);
void blt_scaled_copy_cpu(blt_tensor *dst, const blt_tensor *src, float scalar);

void blt_sgd_step_cpu(blt_tensor *param, const blt_tensor *grad, float lr);
void blt_adamw_step_cpu(blt_tensor *param, const blt_tensor *grad, blt_tensor *exp_avg, blt_tensor *exp_avg_sq,
                        const blt_adamw_config *config);

void blt_build_attention_mask_cpu(const blt_mask_config *config, blt_tensor *out_mask, blt_arena *arena);
void blt_build_block_diffusion_mask_cpu(const blt_block_diffusion_config *config, blt_tensor *out_mask,
                                        blt_arena *arena);

void blt_patch_pool_forward_cpu(const blt_tensor *byte_hidden, const blt_patch_info *patches, size_t num_patches,
                                blt_patch_pool_type pool_type, blt_tensor *out);
void blt_patch_pool_backward_cpu(const blt_tensor *grad_out, const blt_tensor *byte_hidden,
                                 const blt_patch_info *patches, size_t num_patches, blt_patch_pool_type pool_type,
                                 blt_tensor *grad_byte_hidden);

void blt_attention_head_core_cpu(blt_backend backend, const blt_attention_head_args *args);
void blt_attention_head_core_backward_cpu(blt_backend backend, const blt_attention_head_bwd_args *args);

void blt_embedding_lookup_cpu(const blt_tensor *table, const uint32_t *ids_host, blt_tensor *out);
void blt_embedding_scatter_add_cpu(const blt_tensor *grad_table, const uint32_t *ids_host, const blt_tensor *grad_out);
void blt_indexed_row_accumulate_cpu(const blt_tensor *table, const uint32_t *idx_host, blt_tensor *io);
void blt_indexed_row_scatter_add_cpu(const blt_tensor *grad_table, const uint32_t *idx_host, const blt_tensor *grad_out,
                                     float scale);
void blt_indexed_row_scatter_add_normalized_cpu(const blt_tensor *grad_table, const uint32_t *idx_host,
                                                const blt_tensor *grad_out, float scale);
void blt_rows_gather_cpu(const blt_tensor *src, const size_t *pos_host, blt_tensor *dst);

void blt_entropy_rows_cpu(const blt_tensor *probs, blt_tensor *entropy_out, int use_log2);
void blt_argmax_rows_cpu(const blt_tensor *logits, uint32_t *out_ids_host);

void blt_cast_cpu(const blt_tensor *in, blt_tensor *out);

void blt_amax_cpu(const blt_tensor *x, blt_tensor *out);
void blt_quant_scale_cpu(const blt_tensor *amax_in, float max_value, float min_scale, blt_tensor *scale_out);
void blt_fp8_quantize_cpu(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose);
void blt_fp8_dequantize_cpu(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose);
void blt_fp8_roundtrip_cpu(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out);

void blt_softmax_cpu(const blt_tensor *in, blt_tensor *out);
void blt_softmax_backward_cpu(const blt_tensor *grad_out, const blt_tensor *softmax_out, blt_tensor *grad_in);
void blt_cross_entropy_forward_cpu(const blt_tensor *logits, const blt_tensor *targets, blt_tensor *loss_out);
void blt_cross_entropy_backward_cpu(const blt_tensor *logits, const blt_tensor *targets, blt_tensor *grad_logits);
void blt_rope_precompute_cpu(size_t max_seq_len, const blt_rope_config *config, blt_tensor *cos_out,
                             blt_tensor *sin_out);
void blt_rope_apply_cpu(const blt_tensor *x, const blt_tensor *cos, const blt_tensor *sin, blt_tensor *out);
void blt_rope_apply_backward_cpu(const blt_tensor *grad_out, const blt_tensor *cos, const blt_tensor *sin,
                                 blt_tensor *grad_in);
void blt_layernorm_forward_cpu(const blt_tensor *x, const blt_tensor *weight, const blt_tensor *bias, blt_tensor *out,
                               float eps);
void blt_layernorm_backward_cpu(const blt_tensor *grad_out, const blt_tensor *x, const blt_tensor *weight,
                                blt_tensor *grad_x, blt_tensor *grad_weight, blt_tensor *grad_bias, float eps);
void blt_rmsnorm_forward_cpu(const blt_tensor *x, const blt_tensor *weight, blt_tensor *out);
void blt_rmsnorm_backward_cpu(const blt_tensor *grad_out, const blt_tensor *x, const blt_tensor *weight,
                              blt_tensor *grad_x, blt_tensor *grad_weight);
void blt_rmsnorm_backward_add_cpu(const blt_tensor *grad_a, const blt_tensor *grad_b, const blt_tensor *x,
                                  const blt_tensor *weight, blt_tensor *grad_x, blt_tensor *grad_weight);

#define BLT_CUDA_NOT_IMPLEMENTED(op_name) BLT_FATAL(op_name " is not yet implemented for CUDA")

#ifdef BLT_WITH_CUDA
#define BLT_DISPATCH(tensor, cpu_call, cuda_call)                                                                      \
    do {                                                                                                               \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {                                                                   \
            cuda_call;                                                                                                 \
            return;                                                                                                    \
        }                                                                                                              \
        cpu_call;                                                                                                      \
    } while (0)

#define BLT_DISPATCH_RET(tensor, cpu_call, cuda_call)                                                                  \
    do {                                                                                                               \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {                                                                   \
            return cuda_call;                                                                                          \
        }                                                                                                              \
        return cpu_call;                                                                                               \
    } while (0)

// Backend-keyed variant for raw-pointer utilities that have no tensor handle.
#define BLT_DISPATCH_BACKEND(backend, cpu_call, cuda_call)                                                             \
    do {                                                                                                               \
        if ((backend) == BLT_BACKEND_CUDA) {                                                                           \
            cuda_call;                                                                                                 \
            return;                                                                                                    \
        }                                                                                                              \
        cpu_call;                                                                                                      \
    } while (0)

#define BLT_DISPATCH_BACKEND_RET(backend, cpu_call, cuda_call)                                                         \
    do {                                                                                                               \
        if ((backend) == BLT_BACKEND_CUDA) {                                                                           \
            return cuda_call;                                                                                          \
        }                                                                                                              \
        return cpu_call;                                                                                               \
    } while (0)

float blt_vec_dot_cuda(blt_backend backend, const float *a, const float *b, size_t n);
void blt_vec_dot_into_cuda(blt_backend backend, const float *a, const float *b, size_t n, float *out);
void blt_softmax_masked_row_inplace_cuda(blt_backend backend, float *row, size_t row_len, size_t row_idx, int is_causal,
                                         const float *mask_row, float scale);
void blt_strided_copy_cuda(blt_backend backend, float *dst, size_t dst_stride, const float *src, size_t src_stride,
                           size_t rows, size_t cols);
void blt_fill_uniform_cuda(blt_backend backend, float *data, size_t n, uint64_t *rng_state);
void blt_fill_constant_cuda(blt_backend backend, float *data, size_t n, float v);

void blt_add_cuda(const blt_tensor *a, const blt_tensor *b, blt_tensor *out);
void blt_mul_cuda(const blt_tensor *a, const blt_tensor *b, blt_tensor *out);
void blt_scale_cuda(blt_tensor *t, float scalar);
void blt_scaled_copy_cuda(blt_tensor *dst, const blt_tensor *src, float scalar);
void blt_gelu_forward_cuda(const blt_tensor *x, blt_tensor *out);
void blt_gelu_backward_cuda(const blt_tensor *grad_out, const blt_tensor *x, blt_tensor *grad_x);
void blt_swiglu_forward_cuda(const blt_tensor *gate, const blt_tensor *up, blt_tensor *out);
void blt_swiglu_backward_cuda(const blt_tensor *grad_out, const blt_tensor *gate, const blt_tensor *up,
                              blt_tensor *grad_gate, blt_tensor *grad_up);

void blt_softmax_cuda(const blt_tensor *in, blt_tensor *out);
void blt_softmax_backward_cuda(const blt_tensor *grad_out, const blt_tensor *softmax_out, blt_tensor *grad_in);
void blt_cross_entropy_forward_cuda(const blt_tensor *logits, const blt_tensor *targets, blt_tensor *loss_out);
void blt_cross_entropy_backward_cuda(const blt_tensor *logits, const blt_tensor *targets, blt_tensor *grad_logits);
void blt_rope_precompute_cuda(size_t max_seq_len, const blt_rope_config *config, blt_tensor *cos_out,
                              blt_tensor *sin_out);
void blt_rope_apply_cuda(const blt_tensor *x, const blt_tensor *cos, const blt_tensor *sin, blt_tensor *out);
void blt_rope_apply_backward_cuda(const blt_tensor *grad_out, const blt_tensor *cos, const blt_tensor *sin,
                                  blt_tensor *grad_in);

void blt_rope_apply_packed_cuda(float *qkv_data, size_t qkv_stride, size_t head_offset, size_t seq_len, size_t head_dim,
                                const float *cos, const float *sin);
void blt_rope_apply_packed_backward_cuda(float *qkv_data, size_t qkv_stride, size_t head_offset, size_t seq_len,
                                         size_t head_dim, const float *cos, const float *sin);

void blt_layernorm_forward_cuda(const blt_tensor *x, const blt_tensor *weight, const blt_tensor *bias, blt_tensor *out,
                                float eps);
void blt_rmsnorm_forward_cuda(const blt_tensor *x, const blt_tensor *weight, blt_tensor *out);
void blt_rmsnorm_backward_cuda(const blt_tensor *grad_out, const blt_tensor *x, const blt_tensor *weight,
                               blt_tensor *grad_x, blt_tensor *grad_weight);
void blt_rmsnorm_backward_add_cuda(const blt_tensor *grad_a, const blt_tensor *grad_b, const blt_tensor *x,
                                   const blt_tensor *weight, blt_tensor *grad_x, blt_tensor *grad_weight);

void blt_matmul_cuda(const blt_tensor *a, const blt_tensor *b, blt_tensor *out);
void blt_matmul_backward_cuda(const blt_tensor *a, const blt_tensor *b, const blt_tensor *grad_out, blt_tensor *grad_a,
                              blt_tensor *grad_b);

void blt_sgd_step_cuda(blt_tensor *param, const blt_tensor *grad, float lr);
void blt_adamw_step_cuda(blt_tensor *param, const blt_tensor *grad, blt_tensor *exp_avg, blt_tensor *exp_avg_sq,
                         const blt_adamw_config *config);

void blt_build_attention_mask_cuda(const blt_mask_config *config, blt_tensor *out_mask, blt_arena *arena);
void blt_build_block_diffusion_mask_cuda(const blt_block_diffusion_config *config, blt_tensor *out_mask,
                                         blt_arena *arena);

void blt_patch_pool_forward_cuda(const blt_tensor *byte_hidden, const blt_patch_info *patches, size_t num_patches,
                                 blt_patch_pool_type pool_type, blt_tensor *out);
void blt_patch_pool_backward_cuda(const blt_tensor *grad_out, const blt_tensor *byte_hidden,
                                  const blt_patch_info *patches, size_t num_patches, blt_patch_pool_type pool_type,
                                  blt_tensor *grad_byte_hidden);

void blt_attention_head_core_cuda(blt_backend backend, const blt_attention_head_args *args);
void blt_attention_head_core_backward_cuda(blt_backend backend, const blt_attention_head_bwd_args *args);

void blt_embedding_lookup_cuda(const blt_tensor *table, const uint32_t *ids_host, blt_tensor *out);
void blt_embedding_scatter_add_cuda(const blt_tensor *grad_table, const uint32_t *ids_host, const blt_tensor *grad_out);
void blt_indexed_row_accumulate_cuda(const blt_tensor *table, const uint32_t *idx_host, blt_tensor *io);
void blt_indexed_row_scatter_add_cuda(const blt_tensor *grad_table, const uint32_t *idx_host,
                                      const blt_tensor *grad_out, float scale);
void blt_indexed_row_scatter_add_normalized_cuda(const blt_tensor *grad_table, const uint32_t *idx_host,
                                                 const blt_tensor *grad_out, float scale);
void blt_rows_gather_cuda(const blt_tensor *src, const size_t *pos_host, blt_tensor *dst);

void blt_entropy_rows_cuda(const blt_tensor *probs, blt_tensor *entropy_out, int use_log2);
void blt_argmax_rows_cuda(const blt_tensor *logits, uint32_t *out_ids_host);

void blt_cast_cuda(const blt_tensor *in, blt_tensor *out);

void blt_amax_cuda(const blt_tensor *x, blt_tensor *out);
void blt_quant_scale_cuda(const blt_tensor *amax_in, float max_value, float min_scale, blt_tensor *scale_out);
void blt_fp8_quantize_cuda(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose);
void blt_fp8_dequantize_cuda(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose);
void blt_fp8_roundtrip_cuda(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out);
#else
#define BLT_DISPATCH(tensor, cpu_call, cuda_call)                                                                      \
    do {                                                                                                               \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {                                                                   \
            BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled");                                      \
        }                                                                                                              \
        cpu_call;                                                                                                      \
    } while (0)

#define BLT_DISPATCH_RET(tensor, cpu_call, cuda_call)                                                                  \
    do {                                                                                                               \
        if ((tensor)->backend == BLT_BACKEND_CUDA) {                                                                   \
            BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled");                                      \
        }                                                                                                              \
        return cpu_call;                                                                                               \
    } while (0)

#define BLT_DISPATCH_BACKEND(backend, cpu_call, cuda_call)                                                             \
    do {                                                                                                               \
        if ((backend) == BLT_BACKEND_CUDA) {                                                                           \
            BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled");                                      \
        }                                                                                                              \
        cpu_call;                                                                                                      \
    } while (0)

#define BLT_DISPATCH_BACKEND_RET(backend, cpu_call, cuda_call)                                                         \
    do {                                                                                                               \
        if ((backend) == BLT_BACKEND_CUDA) {                                                                           \
            BLT_FATAL("CUDA backend requested but BLT_WITH_CUDA is not enabled");                                      \
        }                                                                                                              \
        return cpu_call;                                                                                               \
    } while (0)
#endif

// BLT Backend Operations

// ELEMENTWISE

void blt_add(const blt_tensor *a, const blt_tensor *b, blt_tensor *out) {
    BLT_DISPATCH(a, blt_add_cpu(a, b, out), blt_add_cuda(a, b, out));
}

void blt_mul(const blt_tensor *a, const blt_tensor *b, blt_tensor *out) {
    BLT_DISPATCH(a, blt_mul_cpu(a, b, out), blt_mul_cuda(a, b, out));
}

void blt_scale(blt_tensor *t, float scalar) { BLT_DISPATCH(t, blt_scale_cpu(t, scalar), blt_scale_cuda(t, scalar)); }

void blt_scaled_copy(blt_tensor *dst, const blt_tensor *src, float scalar) {
    BLT_DISPATCH(src, blt_scaled_copy_cpu(dst, src, scalar), blt_scaled_copy_cuda(dst, src, scalar));
}

void blt_gelu_forward(const blt_tensor *x, blt_tensor *out) {
    BLT_DISPATCH(x, blt_gelu_forward_cpu(x, out), blt_gelu_forward_cuda(x, out));
}

void blt_gelu_backward(const blt_tensor *grad_out, const blt_tensor *x, blt_tensor *grad_x) {
    BLT_DISPATCH(x, blt_gelu_backward_cpu(grad_out, x, grad_x), blt_gelu_backward_cuda(grad_out, x, grad_x));
}

void blt_swiglu_forward(const blt_tensor *gate, const blt_tensor *up, blt_tensor *out) {
    BLT_DISPATCH(gate, blt_swiglu_forward_cpu(gate, up, out), blt_swiglu_forward_cuda(gate, up, out));
}

void blt_swiglu_backward(const blt_tensor *grad_out, const blt_tensor *gate, const blt_tensor *up,
                         blt_tensor *grad_gate, blt_tensor *grad_up) {
    BLT_DISPATCH(gate, blt_swiglu_backward_cpu(grad_out, gate, up, grad_gate, grad_up),
                 blt_swiglu_backward_cuda(grad_out, gate, up, grad_gate, grad_up));
}

// REDUCTIONS

void blt_softmax(const blt_tensor *in, blt_tensor *out) {
    BLT_DISPATCH(in, blt_softmax_cpu(in, out), blt_softmax_cuda(in, out));
}

void blt_softmax_backward(const blt_tensor *grad_out, const blt_tensor *softmax_out, blt_tensor *grad_in) {
    BLT_DISPATCH(grad_out, blt_softmax_backward_cpu(grad_out, softmax_out, grad_in),
                 blt_softmax_backward_cuda(grad_out, softmax_out, grad_in));
}

void blt_cross_entropy_forward(const blt_tensor *logits, const blt_tensor *targets, blt_tensor *loss_out) {
    BLT_DISPATCH(logits, blt_cross_entropy_forward_cpu(logits, targets, loss_out),
                 blt_cross_entropy_forward_cuda(logits, targets, loss_out));
}

void blt_cross_entropy_backward(const blt_tensor *logits, const blt_tensor *targets, blt_tensor *grad_logits) {
    BLT_DISPATCH(logits, blt_cross_entropy_backward_cpu(logits, targets, grad_logits),
                 blt_cross_entropy_backward_cuda(logits, targets, grad_logits));
}

void blt_rope_precompute(size_t max_seq_len, const blt_rope_config *config, blt_tensor *cos_out, blt_tensor *sin_out) {
    BLT_DISPATCH(cos_out, blt_rope_precompute_cpu(max_seq_len, config, cos_out, sin_out),
                 blt_rope_precompute_cuda(max_seq_len, config, cos_out, sin_out));
}

void blt_rope_apply(const blt_tensor *x, const blt_tensor *cos, const blt_tensor *sin, blt_tensor *out) {
    BLT_DISPATCH(x, blt_rope_apply_cpu(x, cos, sin, out), blt_rope_apply_cuda(x, cos, sin, out));
}

void blt_rope_apply_backward(const blt_tensor *grad_out, const blt_tensor *cos, const blt_tensor *sin,
                             blt_tensor *grad_in) {
    BLT_DISPATCH(grad_out, blt_rope_apply_backward_cpu(grad_out, cos, sin, grad_in),
                 blt_rope_apply_backward_cuda(grad_out, cos, sin, grad_in));
}

// Fused RoPE on packed QKV layout
void blt_rope_apply_packed(float *qkv_data, size_t qkv_stride, size_t head_offset, size_t seq_len, size_t head_dim,
                           const float *cos, const float *sin, blt_backend backend) {
    if (backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        blt_rope_apply_packed_cuda(qkv_data, qkv_stride, head_offset, seq_len, head_dim, (const float *)cos,
                                   (const float *)sin);
#endif
    } else {
        size_t half = head_dim / 2;
        for (size_t t = 0; t < seq_len; t++) {
            float *base = qkv_data + t * qkv_stride + head_offset;
            const float *c = cos + t * half;
            const float *s = sin + t * half;
            for (size_t i = 0; i < half; i++) {
                float x0 = base[2 * i];
                float x1 = base[2 * i + 1];
                base[2 * i] = x0 * c[i] - x1 * s[i];
                base[2 * i + 1] = x1 * c[i] + x0 * s[i];
            }
        }
    }
}

void blt_rope_apply_packed_backward(float *qkv_data, size_t qkv_stride, size_t head_offset, size_t seq_len,
                                    size_t head_dim, const float *cos, const float *sin, blt_backend backend) {
    if (backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        blt_rope_apply_packed_backward_cuda(qkv_data, qkv_stride, head_offset, seq_len, head_dim, (const float *)cos,
                                            (const float *)sin);
#endif
    } else {
        size_t half = head_dim / 2;
        for (size_t t = 0; t < seq_len; t++) {
            float *base = qkv_data + t * qkv_stride + head_offset;
            const float *c = cos + t * half;
            const float *s = sin + t * half;
            for (size_t i = 0; i < half; i++) {
                float g0 = base[2 * i];
                float g1 = base[2 * i + 1];
                base[2 * i] = g0 * c[i] + g1 * s[i];
                base[2 * i + 1] = g1 * c[i] - g0 * s[i];
            }
        }
    }
}

void blt_layernorm_forward(const blt_tensor *x, const blt_tensor *weight, const blt_tensor *bias, blt_tensor *out,
                           float eps) {
    BLT_DISPATCH(x, blt_layernorm_forward_cpu(x, weight, bias, out, eps),
                 blt_layernorm_forward_cuda(x, weight, bias, out, eps));
}
void blt_layernorm_backward(const blt_tensor *grad_out, const blt_tensor *x, const blt_tensor *weight,
                            blt_tensor *grad_x, blt_tensor *grad_weight, blt_tensor *grad_bias, float eps) {
    BLT_DISPATCH(x, blt_layernorm_backward_cpu(grad_out, x, weight, grad_x, grad_weight, grad_bias, eps),
                 BLT_CUDA_NOT_IMPLEMENTED("blt_layernorm_backward"));
}

void blt_rmsnorm_forward(const blt_tensor *x, const blt_tensor *weight, blt_tensor *out) {
    BLT_DISPATCH(x, blt_rmsnorm_forward_cpu(x, weight, out), blt_rmsnorm_forward_cuda(x, weight, out));
}

void blt_rmsnorm_backward(const blt_tensor *grad_out, const blt_tensor *x, const blt_tensor *weight, blt_tensor *grad_x,
                          blt_tensor *grad_weight) {
    BLT_DISPATCH(x, blt_rmsnorm_backward_cpu(grad_out, x, weight, grad_x, grad_weight),
                 blt_rmsnorm_backward_cuda(grad_out, x, weight, grad_x, grad_weight));
}
void blt_rmsnorm_backward_add(const blt_tensor *grad_a, const blt_tensor *grad_b, const blt_tensor *x,
                              const blt_tensor *weight, blt_tensor *grad_x, blt_tensor *grad_weight) {
    BLT_DISPATCH(x, blt_rmsnorm_backward_add_cpu(grad_a, grad_b, x, weight, grad_x, grad_weight),
                 blt_rmsnorm_backward_add_cuda(grad_a, grad_b, x, weight, grad_x, grad_weight));
}

// LINALG

void blt_matmul(const blt_tensor *a, const blt_tensor *b, blt_tensor *out) {
    BLT_DISPATCH(a, blt_matmul_cpu(a, b, out), blt_matmul_cuda(a, b, out));
}

void blt_matmul_backward(const blt_tensor *a, const blt_tensor *b, const blt_tensor *grad_out, blt_tensor *grad_a,
                         blt_tensor *grad_b) {
    BLT_DISPATCH(a, blt_matmul_backward_cpu(a, b, grad_out, grad_a, grad_b),
                 blt_matmul_backward_cuda(a, b, grad_out, grad_a, grad_b));
}

// VECMATH

float blt_vec_dot(blt_backend backend, const float *a, const float *b, size_t n) {
    BLT_DISPATCH_BACKEND_RET(backend, blt_vec_dot_cpu(backend, a, b, n), blt_vec_dot_cuda(backend, a, b, n));
}

void blt_vec_dot_into(blt_backend backend, const float *a, const float *b, size_t n, float *out) {
    BLT_DISPATCH_BACKEND(backend, blt_vec_dot_into_cpu(backend, a, b, n, out),
                         blt_vec_dot_into_cuda(backend, a, b, n, out));
}

void blt_softmax_masked_row_inplace(blt_backend backend, float *row, size_t row_len, size_t row_idx, bool is_causal,
                                    const float *mask_row, float scale) {
    BLT_DISPATCH_BACKEND(
        backend, blt_softmax_masked_row_inplace_cpu(backend, row, row_len, row_idx, is_causal, mask_row, scale),
        blt_softmax_masked_row_inplace_cuda(backend, row, row_len, row_idx, is_causal ? 1 : 0, mask_row, scale));
}

void blt_strided_copy(blt_backend backend, float *dst, size_t dst_stride, const float *src, size_t src_stride,
                      size_t rows, size_t cols) {
    BLT_DISPATCH_BACKEND(backend, blt_strided_copy_cpu(backend, dst, dst_stride, src, src_stride, rows, cols),
                         blt_strided_copy_cuda(backend, dst, dst_stride, src, src_stride, rows, cols));
}

void blt_fill_uniform(blt_backend backend, float *data, size_t n, uint64_t *rng_state) {
    BLT_DISPATCH_BACKEND(backend, blt_fill_uniform_cpu(backend, data, n, rng_state),
                         blt_fill_uniform_cuda(backend, data, n, rng_state));
}

void blt_fill_constant(blt_backend backend, float *data, size_t n, float v) {
    BLT_DISPATCH_BACKEND(backend, blt_fill_constant_cpu(backend, data, n, v),
                         blt_fill_constant_cuda(backend, data, n, v));
}

// OPTIMIZERS

void blt_sgd_step(blt_tensor *param, const blt_tensor *grad, float lr) {
    BLT_DISPATCH(param, blt_sgd_step_cpu(param, grad, lr), blt_sgd_step_cuda(param, grad, lr));
}

void blt_adamw_step(blt_tensor *param, const blt_tensor *grad, blt_tensor *exp_avg, blt_tensor *exp_avg_sq,
                    const blt_adamw_config *config) {
    BLT_DISPATCH(param, blt_adamw_step_cpu(param, grad, exp_avg, exp_avg_sq, config),
                 blt_adamw_step_cuda(param, grad, exp_avg, exp_avg_sq, config));
}

// MASK BUILDER (dispatch keyed on arena backend)

void blt_build_attention_mask(const blt_mask_config *config, blt_tensor *out_mask, blt_arena *arena) {
    BLT_DISPATCH_BACKEND((blt_backend)(arena ? arena->backend : BLT_BACKEND_CPU),
                         blt_build_attention_mask_cpu(config, out_mask, arena),
                         blt_build_attention_mask_cuda(config, out_mask, arena));
}

void blt_build_block_diffusion_mask(const blt_block_diffusion_config *config, blt_tensor *out_mask, blt_arena *arena) {
    BLT_DISPATCH_BACKEND((blt_backend)(arena ? arena->backend : BLT_BACKEND_CPU),
                         blt_build_block_diffusion_mask_cpu(config, out_mask, arena),
                         blt_build_block_diffusion_mask_cuda(config, out_mask, arena));
}

// PATCH POOL

void blt_patch_pool_forward(const blt_tensor *byte_hidden, const blt_patch_info *patches, size_t num_patches,
                            blt_patch_pool_type pool_type, blt_tensor *out) {
    BLT_DISPATCH(byte_hidden, blt_patch_pool_forward_cpu(byte_hidden, patches, num_patches, pool_type, out),
                 blt_patch_pool_forward_cuda(byte_hidden, patches, num_patches, pool_type, out));
}

void blt_patch_pool_backward(const blt_tensor *grad_out, const blt_tensor *byte_hidden, const blt_patch_info *patches,
                             size_t num_patches, blt_patch_pool_type pool_type, blt_tensor *grad_byte_hidden) {
    BLT_DISPATCH(
        grad_out, blt_patch_pool_backward_cpu(grad_out, byte_hidden, patches, num_patches, pool_type, grad_byte_hidden),
        blt_patch_pool_backward_cuda(grad_out, byte_hidden, patches, num_patches, pool_type, grad_byte_hidden));
}

// ATTENTION CORE

void blt_attention_head_core(blt_backend backend, const blt_attention_head_args *args) {
    BLT_DISPATCH_BACKEND(backend, blt_attention_head_core_cpu(backend, args),
                         blt_attention_head_core_cuda(backend, args));
}

void blt_attention_head_core_backward(blt_backend backend, const blt_attention_head_bwd_args *args) {
    BLT_DISPATCH_BACKEND(backend, blt_attention_head_core_backward_cpu(backend, args),
                         blt_attention_head_core_backward_cuda(backend, args));
}

void *blt_container_alloc(blt_arena *arena, size_t bytes) {
#ifdef BLT_WITH_CUDA
    if (arena->backend == BLT_BACKEND_CUDA) {
        return blt_container_alloc_cuda(arena, bytes);
    }
#endif
    return blt_container_alloc_cpu(arena, bytes);
}

// GATHER / SCATTER

void blt_embedding_lookup(const blt_tensor *table, const uint32_t *ids_host, blt_tensor *out) {
    BLT_DISPATCH(table, blt_embedding_lookup_cpu(table, ids_host, out),
                 blt_embedding_lookup_cuda(table, ids_host, out));
}

void blt_embedding_scatter_add(const blt_tensor *grad_table, const uint32_t *ids_host, const blt_tensor *grad_out) {
    BLT_DISPATCH(grad_table, blt_embedding_scatter_add_cpu(grad_table, ids_host, grad_out),
                 blt_embedding_scatter_add_cuda(grad_table, ids_host, grad_out));
}

void blt_indexed_row_accumulate(const blt_tensor *table, const uint32_t *idx_host, blt_tensor *io) {
    BLT_DISPATCH(table, blt_indexed_row_accumulate_cpu(table, idx_host, io),
                 blt_indexed_row_accumulate_cuda(table, idx_host, io));
}

void blt_indexed_row_scatter_add(const blt_tensor *grad_table, const uint32_t *idx_host, const blt_tensor *grad_out,
                                 float scale) {
    BLT_DISPATCH(grad_table, blt_indexed_row_scatter_add_cpu(grad_table, idx_host, grad_out, scale),
                 blt_indexed_row_scatter_add_cuda(grad_table, idx_host, grad_out, scale));
}

void blt_indexed_row_scatter_add_normalized(blt_backend backend, const blt_tensor *grad_table, const uint32_t *idx_host,
                                            const blt_tensor *grad_out, float scale) {
    BLT_DISPATCH_BACKEND(backend, blt_indexed_row_scatter_add_normalized_cpu(grad_table, idx_host, grad_out, scale),
                         blt_indexed_row_scatter_add_normalized_cuda(grad_table, idx_host, grad_out, scale));
}

void blt_rows_gather(const blt_tensor *src, const size_t *pos_host, blt_tensor *dst) {
    BLT_DISPATCH(src, blt_rows_gather_cpu(src, pos_host, dst), blt_rows_gather_cuda(src, pos_host, dst));
}

// ROW STATISTICS

void blt_entropy_rows(const blt_tensor *probs, blt_tensor *entropy_out, int use_log2) {
    BLT_DISPATCH(probs, blt_entropy_rows_cpu(probs, entropy_out, use_log2),
                 blt_entropy_rows_cuda(probs, entropy_out, use_log2));
}

void blt_argmax_rows(const blt_tensor *logits, uint32_t *out_ids_host) {
    BLT_DISPATCH(logits, blt_argmax_rows_cpu(logits, out_ids_host), blt_argmax_rows_cuda(logits, out_ids_host));
}

// CAST

void blt_cast(const blt_tensor *in, blt_tensor *out) {
    BLT_DISPATCH(in, blt_cast_cpu(in, out), blt_cast_cuda(in, out));
}

// QUANTIZATION

void blt_amax(const blt_tensor *x, blt_tensor *out) { BLT_DISPATCH(x, blt_amax_cpu(x, out), blt_amax_cuda(x, out)); }

void blt_quant_scale(const blt_tensor *amax_in, float max_value, float min_scale, blt_tensor *scale_out) {
    BLT_DISPATCH(amax_in, blt_quant_scale_cpu(amax_in, max_value, min_scale, scale_out),
                 blt_quant_scale_cuda(amax_in, max_value, min_scale, scale_out));
}

void blt_fp8_quantize(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose) {
    BLT_DISPATCH(x, blt_fp8_quantize_cpu(x, scale, out, transpose), blt_fp8_quantize_cuda(x, scale, out, transpose));
}

void blt_fp8_dequantize(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out, int transpose) {
    BLT_DISPATCH(x, blt_fp8_dequantize_cpu(x, scale, out, transpose),
                 blt_fp8_dequantize_cuda(x, scale, out, transpose));
}

void blt_fp8_roundtrip(const blt_tensor *x, const blt_tensor *scale, blt_tensor *out) {
    BLT_DISPATCH(x, blt_fp8_roundtrip_cpu(x, scale, out), blt_fp8_roundtrip_cuda(x, scale, out));
}

// PASS SYNC

void blt_backend_pass_sync(blt_backend backend) {
#ifdef BLT_WITH_CUDA
    if (backend == BLT_BACKEND_CUDA) {
        blt_backend_pass_sync_cuda();
        return;
    }
#endif
    // CPU: no-op
}
