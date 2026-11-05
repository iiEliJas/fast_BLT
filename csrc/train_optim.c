#include "train_optim.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ops/vecmath.h"
#include "core/cuda_shim.h"

struct sgd_ctx {
    float lr;
    float scale;
};

static void sgd_pair_fn(float *w, float *g, const blt_param_info *info, void *ctx) {
    const struct sgd_ctx *c = (const struct sgd_ctx *)ctx;
    if (info->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        blt_cuda_sgd_step(w, (const float *)g, c->lr, c->scale, info->numel);
#endif
    } else {
        for (size_t i = 0; i < info->numel; i++) {
            const float gi = g[i] * c->scale;
            w[i] -= c->lr * gi;
        }
    }
}

void sgd_all(blt_model *m, blt_model_grad *g, float lr, float grad_scale) {
    struct sgd_ctx ctx = {lr, grad_scale};
    blt_model_visit_param_pairs(m, g, sgd_pair_fn, &ctx);
}

// AdamW state: one exp_avg + one exp_avg_sq per parameter tensor.
// Allocated once before training; zero-initialized by arena.

static adamw_pair mk_pair(blt_arena *arena, const blt_tensor *ref) {
    adamw_pair p;
    p.em = blt_tensor_create(arena, ref->shape, ref->ndim, BLT_DTYPE_FP32);
    p.esq = blt_tensor_create(arena, ref->shape, ref->ndim, BLT_DTYPE_FP32);
    return p;
}

adamw_state *adamw_state_create(blt_arena *arena, blt_model *m) {
    adamw_state *s = (adamw_state *)malloc(sizeof(adamw_state));
    memset(s, 0, sizeof(*s));
    s->step = 0;
    int idx = 0;

    s->flat[idx++] = mk_pair(arena, &m->encoder->byte_embedding_weight);
    for (size_t i = 0; i < m->encoder->ngram_weights.num_tables; i++)
        s->flat[idx++] = mk_pair(arena, &m->encoder->ngram_weights.tables[i]);
    for (size_t i = 0; i < m->encoder->config.num_layers; i++) {
        blt_local_layer_storage *w = &m->encoder->layers[i];
        s->flat[idx++] = mk_pair(arena, &w->norm1_weight);
        s->flat[idx++] = mk_pair(arena, &w->attn_qkv_w);
        s->flat[idx++] = mk_pair(arena, &w->attn_proj_w);
        s->flat[idx++] = mk_pair(arena, &w->norm2_weight);
        s->flat[idx++] = mk_pair(arena, &w->ffn_up_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_gate_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_down_w);
        s->flat[idx++] = mk_pair(arena, &w->cross_norm_weight);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_q);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_k);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_v);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_proj);
    }
    for (size_t i = 0; i < m->global->stack.num_layers; i++) {
        blt_transformer_layer_storage *w = &m->global->stack.layer_storage[i];
        s->flat[idx++] = mk_pair(arena, &w->norm1_weight);
        s->flat[idx++] = mk_pair(arena, &w->attn_qkv_w);
        s->flat[idx++] = mk_pair(arena, &w->attn_proj_w);
        s->flat[idx++] = mk_pair(arena, &w->norm2_weight);
        s->flat[idx++] = mk_pair(arena, &w->ffn_up_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_gate_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_down_w);
    }
    for (size_t i = 0; i < m->decoder->config.num_layers; i++) {
        blt_local_layer_storage *w = &m->decoder->layers[i];
        s->flat[idx++] = mk_pair(arena, &w->norm1_weight);
        s->flat[idx++] = mk_pair(arena, &w->attn_qkv_w);
        s->flat[idx++] = mk_pair(arena, &w->attn_proj_w);
        s->flat[idx++] = mk_pair(arena, &w->norm2_weight);
        s->flat[idx++] = mk_pair(arena, &w->ffn_up_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_gate_w);
        s->flat[idx++] = mk_pair(arena, &w->ffn_down_w);
        s->flat[idx++] = mk_pair(arena, &w->cross_norm_weight);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_q);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_k);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_v);
        s->flat[idx++] = mk_pair(arena, &w->cross_weight_proj);
    }
    s->flat[idx++] = mk_pair(arena, &m->decoder->lm_head_weight);
    s->flat[idx++] = mk_pair(arena, &m->decoder->d0_embed_weight);
    s->n = (size_t)idx;
    return s;
}

struct adamw_pair_ctx {
    adamw_state *s;
    blt_adamw_config cfg;
    float scale;
};

static void adamw_pair_fn(float *w, float *g, const blt_param_info *info, void *ctx) {
    struct adamw_pair_ctx *c = (struct adamw_pair_ctx *)ctx;
    adamw_pair *p = &c->s->flat[info->idx];
    const float lr = c->cfg.lr;
    const float b1 = c->cfg.beta1;
    const float b2 = c->cfg.beta2;
    const float eps = c->cfg.eps;
    const float wd = (c->cfg.weight_decay_norms || info->decay) ? c->cfg.weight_decay : 0.0f;
    const float bc1 = 1.0f - powf(b1, (float)c->cfg.step);
    const float bc2 = 1.0f - powf(b2, (float)c->cfg.step);
    if (info->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        blt_cuda_adamw_step(w, (const float *)g, (float *)p->em.data, (float *)p->esq.data, lr, b1, b2, eps, wd, bc1,
                            bc2, c->scale, info->numel);
#endif
    } else {
        float *m = (float *)p->em.data;
        float *v = (float *)p->esq.data;
        for (size_t i = 0; i < info->numel; i++) {
            const float gi = g[i] * c->scale;
            m[i] = b1 * m[i] + (1.0f - b1) * gi;
            v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
            const float mhat = m[i] / bc1;
            const float vhat = v[i] / bc2;
            w[i] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * w[i]);
        }
    }
}

void adamw_all(blt_model *m, blt_model_grad *g, adamw_state *s, const blt_adamw_config *cfg, float grad_scale) {
    s->step++;
    struct adamw_pair_ctx ctx = {.s = s, .cfg = *cfg, .scale = grad_scale};
    ctx.cfg.step = s->step;
    blt_model_visit_param_pairs(m, g, adamw_pair_fn, &ctx);
}

struct adamw_unorm_ctx {
    adamw_state *s;
    float b1, b2, ep, wd;
    int weight_decay_norms;
    size_t step;
    float scale;
    float sq;
};

static void adamw_unorm_fn(float *w, float *g, const blt_param_info *info, void *ctx) {
    struct adamw_unorm_ctx *c = (struct adamw_unorm_ctx *)ctx;
    adamw_pair *p = &c->s->flat[info->idx];
    const float wd = (c->weight_decay_norms || info->decay) ? c->wd : 0.0f;
    const float bc1 = 1.0f - powf(c->b1, (float)c->step);
    const float bc2 = 1.0f - powf(c->b2, (float)c->step);
    if (info->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        c->sq +=
            blt_cuda_adamw_sqnorm((const float *)w, (const float *)g, (const float *)p->em.data,
                                  (const float *)p->esq.data, c->b1, c->b2, c->ep, wd, bc1, bc2, c->scale, info->numel);
#endif
    } else {
        float *m = (float *)p->em.data;
        float *v = (float *)p->esq.data;
        for (size_t i = 0; i < info->numel; i++) {
            const float gi = g[i] * c->scale;
            float m_new = c->b1 * m[i] + (1.0f - c->b1) * gi;
            float v_new = c->b2 * v[i] + (1.0f - c->b2) * gi * gi;
            float m_hat = m_new / bc1;
            float v_hat = v_new / bc2;
            float wd_i = (c->weight_decay_norms || info->decay) ? c->wd : 0.0f;
            float u = m_hat / (sqrtf(v_hat) + c->ep) + wd_i * w[i];
            c->sq += u * u;
        }
    }
}

float adamw_update_norm(blt_model *m, blt_model_grad *g, adamw_state *s, const blt_adamw_config *cfg,
                        float grad_scale) {
    struct adamw_unorm_ctx ctx = {
        .s = s,
        .b1 = cfg->beta1,
        .b2 = cfg->beta2,
        .ep = cfg->eps,
        .wd = cfg->weight_decay,
        .weight_decay_norms = cfg->weight_decay_norms,
        .step = s->step + 1,
        .scale = grad_scale,
        .sq = 0.0f,
    };
    blt_model_visit_param_pairs(m, g, adamw_unorm_fn, &ctx);
    return sqrtf(ctx.sq) * cfg->lr;
}

// AdamW state serialization. Written next to the weight checkpoint so a run can
// be extended without losing the moment estimates.
//
//   magic  "FBOP" (4 bytes)
//   u32    version (1)
//   u64    global_step       (training step at the time of the write)
//   u64    adamw_step        (number of adamw_all calls; bias-correction counter)
//   u64    num_pairs
//   per pair:
//     u32  ndim
//     u64  dims[ndim]
//     f32  exp_avg[numel]
//     f32  exp_avg_sq[numel]
//
// Pair order is adamw_state_create's, which matches blt_model_tensor_at and
// blt_model_visit_params, so pair i always corresponds to parameter i.

static void optim_tensor_write(const blt_tensor *t, FILE *f, const char *path) {
    const uint32_t ndim = (uint32_t)t->ndim;
    if (fwrite(&ndim, sizeof(uint32_t), 1, f) != 1 || fwrite(t->shape, sizeof(size_t), ndim, f) != ndim)
        BLT_FATAL("optim save: header write failed for pair in '%s'", path);
    if (t->backend == BLT_BACKEND_CPU) {
        if (fwrite(t->data, sizeof(float), t->numel, f) != t->numel)
            BLT_FATAL("optim save: payload write failed for pair in '%s'", path);
        return;
    }
    float *stage = (float *)malloc(t->numel * sizeof(float));
    BLT_REQUIRE(stage != NULL, "optim save: staging alloc failed");
    blt_tensor_download(t, stage, blt_tensor_bytes(t));
    const size_t wrote = fwrite(stage, sizeof(float), t->numel, f);
    free(stage);
    if (wrote != t->numel) BLT_FATAL("optim save: payload write failed for pair in '%s'", path);
}

static void optim_tensor_read(blt_tensor *t, FILE *f, const char *path) {
    uint32_t ndim = 0;
    if (fread(&ndim, sizeof(uint32_t), 1, f) != 1 || ndim > BLT_MAX_NDIM)
        BLT_FATAL("optim load: truncated or invalid ndim in '%s'", path);
    if (ndim != t->ndim) BLT_FATAL("optim load: ndim %u != %u (model shape mismatch?)", ndim, (uint32_t)t->ndim);
    for (uint32_t d = 0; d < ndim; d++) {
        size_t dim = 0;
        if (fread(&dim, sizeof(size_t), 1, f) != 1) BLT_FATAL("optim load: truncated dims in '%s'", path);
        if (dim != t->shape[d])
            BLT_FATAL("optim load: dim[%u] %zu != %zu (model shape mismatch?)", d, dim, t->shape[d]);
    }
    if (t->backend == BLT_BACKEND_CPU) {
        if (fread(t->data, sizeof(float), t->numel, f) != t->numel)
            BLT_FATAL("optim load: truncated payload in '%s'", path);
        return;
    }
    float *stage = (float *)malloc(t->numel * sizeof(float));
    BLT_REQUIRE(stage != NULL, "optim load: staging alloc failed");
    const size_t got = fread(stage, sizeof(float), t->numel, f);
    if (got != t->numel) {
        free(stage);
        BLT_FATAL("optim load: truncated payload in '%s'", path);
    }
    blt_tensor_upload(t, stage, blt_tensor_bytes(t));
    free(stage);
}

void adamw_state_save(const adamw_state *s, size_t global_step, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) BLT_FATAL("optim save: cannot open '%s'", path);

    const uint32_t version = 1;
    const uint64_t n_pairs = (uint64_t)s->n;
    const uint64_t gstep = (uint64_t)global_step;
    const uint64_t astep = (uint64_t)s->step;
    if (fwrite("FBOP", 1, 4, f) != 4 || fwrite(&version, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&gstep, sizeof(uint64_t), 1, f) != 1 || fwrite(&astep, sizeof(uint64_t), 1, f) != 1 ||
        fwrite(&n_pairs, sizeof(uint64_t), 1, f) != 1)
        BLT_FATAL("optim save: header write failed ('%s')", path);

    for (uint64_t i = 0; i < n_pairs; i++) {
        optim_tensor_write(&s->flat[i].em, f, path);
        optim_tensor_write(&s->flat[i].esq, f, path);
    }
    fclose(f);
}

void adamw_state_load(adamw_state *s, size_t *global_step_out, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) BLT_FATAL("optim load: cannot open '%s'", path);

    char magic[4];
    uint32_t version = 0;
    uint64_t gstep = 0, astep = 0, n_pairs = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "FBOP", 4) != 0) BLT_FATAL("optim load: bad magic in '%s'", path);
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version != 1)
        BLT_FATAL("optim load: unsupported version %u in '%s'", version, path);
    if (fread(&gstep, sizeof(uint64_t), 1, f) != 1 || fread(&astep, sizeof(uint64_t), 1, f) != 1 ||
        fread(&n_pairs, sizeof(uint64_t), 1, f) != 1)
        BLT_FATAL("optim load: truncated header in '%s'", path);
    if (n_pairs != (uint64_t)s->n)
        BLT_FATAL("optim load: pair count %llu != expected %zu (model shape mismatch?)", (unsigned long long)n_pairs,
                  s->n);

    for (uint64_t i = 0; i < n_pairs; i++) {
        optim_tensor_read(&s->flat[i].em, f, path);
        optim_tensor_read(&s->flat[i].esq, f, path);
    }
    fclose(f);

    s->step = (size_t)astep;
    if (global_step_out) *global_step_out = (size_t)gstep;
}
