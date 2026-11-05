// Diagnostic and logging functions extracted from train_blt_d.c.
// Gradient clipping, component-norm logging, activation/gradient anomaly
// detection, and batch-property recording.

#include "train_diag.h"

#include <math.h>
#include <stdlib.h>
#include "ops/vecmath.h"
#include "core/cuda_shim.h"

//----------------------------------------------------------------------
// Gradient norm for clipping

// Counting pass: the per-tensor dot buffer is sized from the actual parameter
// count before any dots launch. Growing it mid-visit would discard the sums
// already written into the old buffer, which is what corrupted the norm.
struct clip_count_ctx {
    size_t count;
    blt_backend backend;
};

static void clip_count_fn(float *p, const blt_param_info *info, void *ctx) {
    (void)p;
    struct clip_count_ctx *c = (struct clip_count_ctx *)ctx;
    if (c->count == 0) c->backend = (blt_backend)info->backend;
    c->count++;
}

struct clip_sq_ctx {
    float *sums;
    size_t count;
    size_t cap;
    blt_backend backend;
};

static void clip_sq_fn(float *p, const blt_param_info *info, void *ctx) {
    struct clip_sq_ctx *c = (struct clip_sq_ctx *)ctx;
    BLT_REQUIRE(c->count < c->cap, "grad clip: parameter count changed between visitor passes");
    BLT_REQUIRE((blt_backend)info->backend == c->backend, "grad clip: mixed gradient backends");
    blt_vec_dot_into(c->backend, (const float *)p, (const float *)p, info->numel, &c->sums[c->count]);
    c->count++;
}

float blt_grad_clip_norm(const blt_model *m, blt_model_grad *g, float max_norm, float *scale_out) {
    BLT_REQUIRE(m != NULL && g != NULL, "blt_grad_clip_norm: model and grad cannot be NULL");

    struct clip_count_ctx counted = {0, BLT_BACKEND_CPU};
    blt_model_visit_params(m, g, clip_count_fn, &counted, 1);
    BLT_REQUIRE(counted.count > 0, "blt_grad_clip_norm: model has no gradient tensors");

    const size_t bytes = counted.count * sizeof(float);
    float *sums = NULL;
    if (counted.backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        sums = (float *)blt_cuda_malloc(bytes);
#else
        BLT_FATAL("blt_grad_clip_norm: CUDA gradients in a CPU-only build");
#endif
    } else {
        sums = (float *)malloc(bytes);
        BLT_REQUIRE(sums != NULL, "blt_grad_clip_norm: staging alloc failed");
    }

    struct clip_sq_ctx ctx = {sums, 0, counted.count, counted.backend};
    blt_model_visit_params(m, g, clip_sq_fn, &ctx, 1);
    BLT_REQUIRE(ctx.count == counted.count, "grad clip: parameter count changed between visitor passes");

    float sq = 0.0f;
    if (counted.backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
        float *host = (float *)malloc(bytes);
        BLT_REQUIRE(host != NULL, "blt_grad_clip_norm: host staging alloc failed");
        blt_cuda_memcpy_d2h(host, sums, bytes);
        for (size_t i = 0; i < counted.count; i++) sq += host[i];
        free(host);
        blt_cuda_free(sums);
#endif
    } else {
        for (size_t i = 0; i < counted.count; i++) sq += sums[i];
        free(sums);
    }

    const float norm = sqrtf(sq);
    if (scale_out) *scale_out = (norm > max_norm) ? max_norm / norm : 1.0f;
    return norm;
}

// Log per-component L2 gradient norms to the component-norm-log file.
// Encoder split into embedding / ngram / per-layer to pinpoint
// which sub-tensor is responsible for encoder-dominated gradient spikes.
// Decoder layers split into self_attn / cross_attn / ffn for the same reason.
struct comp_norm_ctx {
    float enc_embed_sq;
    float enc_ngram_sq;
    float enc_layers_sq;
    float glob_sq;
    float dec_self_sq;
    float dec_cross_sq;
    float dec_ffn_sq;
    float head_sq;
    int embed_seen;
};

static void comp_norm_fn(float *p, const blt_param_info *info, void *ctx) {
    struct comp_norm_ctx *c = (struct comp_norm_ctx *)ctx;
    float sq = blt_vec_dot(info->backend, (const float *)p, (const float *)p, info->numel);
    switch (info->component_group) {
    case BLT_GROUP_EMBED:
        if (!c->embed_seen) {
            c->enc_embed_sq += sq;
            c->embed_seen = 1;
        } else {
            c->enc_ngram_sq += sq;
        }
        break;
    case BLT_GROUP_ENC_SELF:
    case BLT_GROUP_ENC_CROSS:
        c->enc_layers_sq += sq;
        break;
    case BLT_GROUP_GLOB:
        c->glob_sq += sq;
        break;
    case BLT_GROUP_DEC_SELF:
        if (info->param_in_layer <= 2) c->dec_self_sq += sq;
        else c->dec_ffn_sq += sq;
        break;
    case BLT_GROUP_DEC_CROSS:
        c->dec_cross_sq += sq;
        break;
    case BLT_GROUP_HEAD:
        c->head_sq += sq;
        break;
    }
}

void log_component_norms(FILE *fp, size_t step, blt_model *m, blt_model_grad *g) {
    struct comp_norm_ctx ctx = {0};
    blt_model_visit_params(m, g, comp_norm_fn, &ctx, 1);

    const float total_norm = sqrtf(ctx.enc_embed_sq + ctx.enc_ngram_sq + ctx.enc_layers_sq + ctx.glob_sq +
                                   ctx.dec_self_sq + ctx.dec_cross_sq + ctx.dec_ffn_sq + ctx.head_sq);

    fprintf(fp, "%zu %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", step, sqrtf(ctx.enc_embed_sq),
            sqrtf(ctx.enc_ngram_sq), sqrtf(ctx.enc_layers_sq), sqrtf(ctx.glob_sq), sqrtf(ctx.dec_self_sq),
            sqrtf(ctx.dec_cross_sq), sqrtf(ctx.dec_ffn_sq), sqrtf(ctx.head_sq), total_norm);
}

//----------------------------------------------------------------------
// Activation dump: scan tensors for numerical anomalies (NaN/Inf/saturation)
// to correlate with gradient-norm spikes.

static void tensor_stats(const blt_tensor *t, float *max_abs, size_t *nan_count, size_t *inf_count) {
    *max_abs = 0.0f;
    *nan_count = 0;
    *inf_count = 0;
    if (t->numel == 0) return;

    float *host_buf = NULL;
    const float *data;
    if (t->backend == BLT_BACKEND_CPU) {
        data = (const float *)t->data;
    } else {
        host_buf = (float *)malloc(t->numel * sizeof(float));
        if (!host_buf) {
            *nan_count = t->numel;
            return;
        }
        blt_tensor_copy_to_host(t, host_buf, t->numel * sizeof(float));
        data = host_buf;
    }

    for (size_t i = 0; i < t->numel; i++) {
        float v = data[i];
        float a = fabsf(v);
        if (a > *max_abs) *max_abs = a;
        if (isnan(v)) (*nan_count)++;
        else if (isinf(v)) (*inf_count)++;
    }
    free(host_buf);
}

static void scan_log_tensor(FILE *fp, size_t step, const char *name, const blt_tensor *t) {
    float max_abs;
    size_t nan_count, inf_count;
    tensor_stats(t, &max_abs, &nan_count, &inf_count);
    fprintf(fp, "%zu %s %.6f %zu %zu\n", step, name, max_abs, nan_count, inf_count);
}

// Log forward activation stats (before backward). NULL tensors are skipped.
void log_forward_activation_dump(FILE *fp, size_t step, const blt_tensor *P, const blt_tensor *h, const blt_tensor *O,
                                 const blt_tensor *logits) {
    if (P) scan_log_tensor(fp, step, "fwd/P", P);
    if (h) scan_log_tensor(fp, step, "fwd/h", h);
    if (O) scan_log_tensor(fp, step, "fwd/O", O);
    if (logits) scan_log_tensor(fp, step, "fwd/logits", logits);
}

struct grad_dump_ctx {
    FILE *fp;
    size_t step;
    float lm_head_max;
    size_t lm_head_nan, lm_head_inf;
    float d0_embed_max;
    size_t d0_embed_nan, d0_embed_inf;
    float dec_max;
    size_t dec_nan, dec_inf;
    float glob_max;
    size_t glob_nan, glob_inf;
    float enc_max;
    size_t enc_nan, enc_inf;
    int head_seen;
};

static void scan_floats(const float *data, size_t numel, float *max_abs, size_t *nan_count, size_t *inf_count) {
    *max_abs = 0.0f;
    *nan_count = 0;
    *inf_count = 0;
    for (size_t i = 0; i < numel; i++) {
        float v = data[i];
        float a = fabsf(v);
        if (a > *max_abs) *max_abs = a;
        if (isnan(v)) (*nan_count)++;
        else if (isinf(v)) (*inf_count)++;
    }
}

static void grad_dump_fn(float *p, const blt_param_info *info, void *ctx) {
    struct grad_dump_ctx *c = (struct grad_dump_ctx *)ctx;
    if (info->backend != BLT_BACKEND_CPU) return;
    float mx;
    size_t nn, ni;
    scan_floats(p, info->numel, &mx, &nn, &ni);
    float *max_p = NULL;
    size_t *nan_p = NULL, *inf_p = NULL;
    if (info->component_group == BLT_GROUP_HEAD) {
        if (c->head_seen == 0) {
            max_p = &c->lm_head_max;
            nan_p = &c->lm_head_nan;
            inf_p = &c->lm_head_inf;
        } else {
            max_p = &c->d0_embed_max;
            nan_p = &c->d0_embed_nan;
            inf_p = &c->d0_embed_inf;
        }
        c->head_seen++;
    } else if (info->component_group == BLT_GROUP_DEC_SELF || info->component_group == BLT_GROUP_DEC_CROSS) {
        max_p = &c->dec_max;
        nan_p = &c->dec_nan;
        inf_p = &c->dec_inf;
    } else if (info->component_group == BLT_GROUP_GLOB) {
        max_p = &c->glob_max;
        nan_p = &c->glob_nan;
        inf_p = &c->glob_inf;
    } else {
        max_p = &c->enc_max;
        nan_p = &c->enc_nan;
        inf_p = &c->enc_inf;
    }
    if (mx > *max_p) *max_p = mx;
    *nan_p += nn;
    *inf_p += ni;
}

void log_gradient_activation_dump(FILE *fp, size_t step, blt_model *m, blt_model_grad *g) {
    struct grad_dump_ctx c = {0};
    c.fp = fp;
    c.step = step;
    blt_model_visit_params(m, g, grad_dump_fn, &c, 1);
    fprintf(fp, "%zu grad/dec_lm_head %.6f %zu %zu\n", step, c.lm_head_max, c.lm_head_nan, c.lm_head_inf);
    fprintf(fp, "%zu grad/dec_d0_embed %.6f %zu %zu\n", step, c.d0_embed_max, c.d0_embed_nan, c.d0_embed_inf);
    fprintf(fp, "%zu grad/dec_layers %.6f %zu %zu\n", step, c.dec_max, c.dec_nan, c.dec_inf);
    fprintf(fp, "%zu grad/glob_layers %.6f %zu %zu\n", step, c.glob_max, c.glob_nan, c.glob_inf);
    fprintf(fp, "%zu grad/encoder %.6f %zu %zu\n", step, c.enc_max, c.enc_nan, c.enc_inf);
}

//----------------------------------------------------------------------
// Batch property logging: record per-step input characteristics so we
// can check whether gradient-norm spike steps share unusual inputs.

static uint64_t fnv1a_hash(const uint8_t *data, size_t len) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Log per-step batch properties: window identity, patch stats, diffusion
// corruption stats, and the effective loss weight. Space-separated, one
// line per step.
void log_batch_properties(FILE *fp, size_t step, size_t window_offset, const uint8_t *text, size_t window_size,
                          const blt_patch_info *patches, size_t num_patches, const blt_block_batch *batch) {

    const uint64_t whash = fnv1a_hash(text, window_size);

    // average patch length
    float avg_patch_len = 0.0f;
    if (num_patches > 0) {
        size_t total = 0;
        for (size_t i = 0; i < num_patches; i++) total += patches[i].length;
        avg_patch_len = (float)total / (float)num_patches;
    }

    // count valid and masked cells in the diffusion batch
    size_t n_valid = 0, n_masked = 0;
    if (batch != NULL) {
        for (size_t r = 0; r < batch->n_block_rows; r++) {
            if (batch->cell_valid[r]) n_valid++;
            if (batch->cell_masked[r]) n_masked++;
        }
    }
    const float masked_frac = n_valid > 0 ? (float)n_masked / (float)n_valid : 0.0f;

    const float t = batch != NULL ? batch->t : 0.0f;
    const float loss_scale = batch != NULL ? batch->loss_scale : 0.0f;
    const float eff_weight = (t > 0.0f) ? loss_scale / t : 0.0f;

    fprintf(fp, "%zu %zu %016llx %zu %.3f %zu %zu %.4f %.6f %.6f %.6f\n", step, window_offset,
            (unsigned long long)whash, num_patches, avg_patch_len, n_valid, n_masked, masked_frac, t, loss_scale,
            eff_weight);
}

void zero_norm_fn(float *p, const blt_param_info *info, void *ctx) {
    (void)ctx;
    if (info->is_norm) {
        if (info->backend == BLT_BACKEND_CUDA) {
#ifdef BLT_WITH_CUDA
            blt_cuda_memset(p, 0, info->numel * sizeof(float));
#endif
        } else {
            for (size_t i = 0; i < info->numel; i++) p[i] = 0.0f;
        }
    }
}

static uint64_t train_mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static size_t train_gcd(size_t a, size_t b) {
    while (b != 0) {
        const size_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// Affine bijection on [0, num_windows): w = (a*j + b) mod num_windows is
// injective in j exactly when gcd(a, num_windows) == 1, which gives every window
// exactly one visit per epoch with no shuffling array and no memory scaling with
// corpus size. a and b are reseeded per epoch, so epoch k is ordered differently
// from epoch k-1.
size_t train_window_index(size_t step, size_t num_windows, uint64_t seed) {
    if (num_windows <= 1) return 0;
    const size_t epoch = step / num_windows;
    const size_t j = step % num_windows;
    const uint64_t s = train_mix64(seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(epoch + 1));
    const size_t b = (size_t)(train_mix64(s) % (uint64_t)num_windows);

    size_t a = (size_t)(train_mix64(s ^ 0xD1B54A32D192ED03ULL) % (uint64_t)num_windows);
    if (a == 0) a = 1;
    // Walk down to the nearest multiplier coprime with num_windows. Bounded so a
    // pathological num_windows cannot stall the loop; 1 is coprime with every n.
    for (int tries = 0; tries < 64 && train_gcd(a, num_windows) != 1; tries++) {
        a = (a <= 1) ? (num_windows - 1) : (a - 1);
    }
    if (train_gcd(a, num_windows) != 1) a = 1;

    return (size_t)((((uint64_t)a * (uint64_t)j) + (uint64_t)b) % (uint64_t)num_windows);
}

#ifdef _WIN32
#include <direct.h>
#define blt_mkdir_one(p) _mkdir((p))
#else
#include <sys/stat.h>
#include <unistd.h>
#define blt_mkdir_one(p) mkdir((p), 0777)
#endif

static int train_dir_writable(const char *dir) {
#ifdef _WIN32
    return _access(dir, 0) == 0;
#else
    return access(dir, W_OK) == 0;
#endif
}

int train_ensure_parent_dir(const char *path) {
    if (!path || !*path) return 0;

    char dir[4096];
    if (snprintf(dir, sizeof(dir), "%s", path) >= (int)sizeof(dir)) {
        fprintf(stderr, "train: path too long: %s\n", path);
        return -1;
    }

    char *slash = strrchr(dir, '/');
    if (!slash) return 0; // bare filename, the cwd already exists
    *slash = '\0';
    if (dir[0] == '\0') return 0; // "/file" -> root

    // Create each component in turn, ignoring "already exists".
    for (char *p = dir + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        blt_mkdir_one(dir);
        *p = '/';
    }
    blt_mkdir_one(dir);

    if (!train_dir_writable(dir)) {
        fprintf(stderr, "train: directory '%s' is not writable (needed to write %s)\n", dir, path);
        return -1;
    }
    return 0;
}

void blt_eta_init(blt_eta_state *st, size_t start_step) {
    st->ema_sec_per_step = 0.0;
    st->last_step = start_step;
}

double blt_eta_update(blt_eta_state *st, size_t step, size_t end_step, double elapsed_s, double excluded_s) {
    const size_t done = step + 1;
    const size_t ran = (done > st->last_step) ? (done - st->last_step) : 0;

    double sec_per_step = (ran > 0) ? ((elapsed_s - excluded_s) / (double)ran) : 0.0;
    if (sec_per_step <= 0.0 || sec_per_step != sec_per_step) sec_per_step = 1e-9;

    st->ema_sec_per_step =
        (st->ema_sec_per_step == 0.0) ? sec_per_step : 0.9 * st->ema_sec_per_step + 0.1 * sec_per_step;
    st->last_step = done;

    if (end_step <= done) return 0.0;
    return (double)(end_step - done) * st->ema_sec_per_step;
}
