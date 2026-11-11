// BLT-D block-wise diffusion training (Fast-BLT 3.2).
//
// Composition notes:
//   - Cross-attention reuses blt_cross_attention_forward/backward verbatim;
//     the corrupted-block rows are expressible through query group ids, so
//     no bespoke cross-attention code exists here.
//   - Self-attention is bespoke (Figure 5 mask is row-type dependent) but
//     mirrors attention.c op-for-op; its backward mirrors attention.c's
//     backward with the post-softmax weights saved during recomputation.
#include "models/block_diffusion.h"

#include "core/backend.h"
#include "models/local_common.h"
#include "models/cross_attention.h"
#include "ops/matmul.h"
#include "ops/rmsnorm.h"
#include "ops/swiglu.h"
#include "ops/rope.h"
#include "ops/vecmath.h"
#include "ops/attn_core.h"
#include "ops/elementwise.h"
#include "ops/mask_builder.h"
#include "ops/cross_entropy.h"
#include "infer/rope_gather.h"
#include "ops/gather_scatter.h"
#include <stdlib.h>

#include <math.h>
#include <string.h>

// RNG (splitmix64; deterministic, libc-independent)
static uint64_t rng_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float rng_uniform01(uint64_t *state) { return (float)(rng_next(state) >> 40) / (float)(1u << 24); }

// Preprocessing

void blt_block_batch_build(blt_block_batch *out, blt_arena *arena, const uint8_t *bytes, size_t N,
                           const blt_patch_info *patches, size_t num_patches, size_t B, uint64_t rng_seed) {
    blt_block_batch_build_t(out, arena, bytes, N, patches, num_patches, B, rng_seed, 0.0f);
}

void blt_block_batch_build_t(blt_block_batch *out, blt_arena *arena, const uint8_t *bytes, size_t N,
                             const blt_patch_info *patches, size_t num_patches, size_t B, uint64_t rng_seed,
                             float t_in) {
    BLT_REQUIRE(out != NULL && arena != NULL && bytes != NULL && patches != NULL,
                "blt_block_batch_build: arguments cannot be NULL");
    BLT_REQUIRE(N >= 2, "blt_block_batch_build: N must be >= 2");
    BLT_REQUIRE(num_patches >= 2, "blt_block_batch_build: need at least 2 patches to form blocks");
    BLT_REQUIRE(B >= 1, "blt_block_batch_build: block_size must be >= 1");

    const size_t NB = num_patches - 1; // blocks (first patch excluded)
    const size_t R = B * NB;

    memset(out, 0, sizeof(*out));
    out->num_clean = N;
    out->block_size = B;
    out->num_blocks = NB;
    out->n_block_rows = R;
    out->loss_scale = 1.0f;     // paper objective by default
    out->last_row_scale = 1.0f; // off: legacy loss by default

    out->tokens = (uint32_t *)blt_container_alloc(arena, R * sizeof(uint32_t));
    out->positions = (size_t *)blt_container_alloc(arena, R * sizeof(size_t));
    out->targets = (uint8_t *)blt_container_alloc(arena, R);
    out->cell_valid = (uint8_t *)blt_container_alloc(arena, R);
    out->cell_masked = (uint8_t *)blt_container_alloc(arena, R);
    out->groups = (size_t *)blt_container_alloc(arena, R * sizeof(size_t));

    uint64_t rng = rng_seed;
    out->t = (t_in > 0.0f) ? t_in : rng_uniform01(&rng);
    if (out->t <= 1e-6f) {
        out->t = 1e-6f; // guard the 1/t loss scaling
    }

    size_t r = 0;
    for (size_t i = 1; i < num_patches; i++) { // block j = i-1
        const size_t s = patches[i].start_idx;
        for (size_t k = 0; k < B; k++, r++) {
            const size_t pos = s + k;
            const bool valid = pos < N;
            out->cell_valid[r] = valid ? 1 : 0;
            out->positions[r] = valid ? pos : (N - 1); // PAD rows keep an in-range position
            out->targets[r] = valid ? bytes[pos] : 0;
            // PAD cells reuse the MASK embedding id; excluded from L_mask.
            out->tokens[r] = (!valid || rng_uniform01(&rng) < out->t) ? BLT_MASK_TOKEN_ID : (uint32_t)bytes[pos];
            out->cell_masked[r] = (out->cell_valid[r] && out->tokens[r] == BLT_MASK_TOKEN_ID) ? 1 : 0;
            out->groups[r] = i - 1; // attends latent o_j (paper's o_{i-1})
        }
    }
}

// Shared step context

typedef struct {
    size_t N, R, S, E, V, H, XH, hd, xhd, half, L, k, pdim;
    blt_tensor d0;                 // [S, E]
    blt_tensor rope_cos, rope_sin; // [S, half], gathered by position
    blt_tensor self_mask;          // [S, S] additive 0/-inf (Figure 5)
    size_t *q_group_ids;           // [S]
    size_t *kv_group_ids;          // [M*k]
    blt_tensor patch_in_split;     // [M*k, E]
} diff_ctx;

static void diff_ctx_build(diff_ctx *c, const blt_local_decoder *model, const blt_tensor *byte_hidden_in,
                           const blt_tensor *patch_in, const blt_patch_info *patches, size_t num_patches,
                           const blt_block_batch *batch, blt_d0_mode d0_mode, blt_block_diffusion_mode mask_mode,
                           blt_arena *arena) {
    const blt_local_decoder_config *cfg = &model->config;
    c->N = batch->num_clean;
    c->R = batch->n_block_rows;
    c->S = c->N + c->R;
    c->E = cfg->embed_dim;
    c->V = cfg->vocab_size;
    c->H = cfg->num_heads;
    c->XH = cfg->cross_attn_heads;
    c->hd = c->E / c->H;
    c->xhd = c->E / c->XH;
    c->half = c->hd / 2;
    c->L = cfg->num_layers;
    c->pdim = cfg->patch_dim ? cfg->patch_dim : c->E;
    c->k = c->pdim / c->E;

    BLT_REQUIRE(c->hd > 0 && c->hd % 2 == 0, "diffusion: RoPE requires an even head_dim");
    BLT_REQUIRE(byte_hidden_in != NULL && patch_in != NULL && batch != NULL && arena != NULL,
                "diffusion: arguments cannot be NULL");
    BLT_REQUIRE(byte_hidden_in->shape[0] == c->N && byte_hidden_in->shape[1] == c->E,
                "diffusion: byte_hidden_in must be [N, embed_dim]");
    BLT_REQUIRE(patch_in->shape[0] == num_patches && patch_in->shape[1] == c->pdim,
                "diffusion: patch_in must be [num_patches, patch_dim]");
    BLT_REQUIRE(c->S >= 2 && c->S <= cfg->max_seq_len,
                "diffusion: total sequence length outside [2, decoder max_seq_len]");

    // D_0: clean rows carry h_final content; block rows follow the policy.
    size_t d0_shape[2] = {c->S, c->E};
    c->d0 = blt_tensor_create(arena, d0_shape, 2, BLT_DTYPE_FP32); // zero-filled
    blt_strided_copy(byte_hidden_in->backend, (float *)c->d0.data, c->E, (const float *)byte_hidden_in->data, c->E,
                     c->N, c->E);
    if (d0_mode == BLT_D0_LEARNED) {
        size_t tail_shape[2] = {c->R, c->E};
        blt_tensor tail = blt_tensor_create(arena, tail_shape, 2, BLT_DTYPE_FP32);
        blt_embedding_lookup(&model->d0_embed_weight, batch->tokens, &tail);
        blt_strided_copy(tail.backend, (float *)c->d0.data + c->N * c->E, c->E, (const float *)tail.data, c->E, c->R,
                         c->E);
    }
    // ZEROS (default): keep the zero-fill.

    // Positions: identity for clean rows, recorded originals for block rows.
    size_t *positions = (size_t *)blt_container_alloc(arena, c->S * sizeof(size_t));
    for (size_t i = 0; i < c->N; i++) {
        positions[i] = i;
    }
    for (size_t r = 0; r < c->R; r++) {
        positions[c->N + r] = batch->positions[r];
    }
    blt_rope_position_gather(&model->rope_cos_cache, &model->rope_sin_cache, positions, c->S, &c->rope_cos,
                             &c->rope_sin, arena);

    // Query group ids: clean rows use the paper-rule decoder latent index
    // (Fast-BLT 3.1.1); block rows take their assigned latent group
    // (paper o_{i-1} rule).
    c->q_group_ids = (size_t *)blt_container_alloc(arena, c->S * sizeof(size_t));
    for (size_t i = 0; i < c->N; i++) {
        c->q_group_ids[i] = blt_patch_decoder_latent_at(patches, num_patches, i);
    }
    if (batch->trailing_closed) {
        // blt_patch_decoder_latent_at cannot see past patches[], so it reports
        // the open-patch rule for the final byte. When the caller knows a
        // boundary fires right after it, that byte is final and reads o_M.
        c->q_group_ids[c->N - 1] = num_patches - 1;
    }
    for (size_t r = 0; r < c->R; r++) {
        c->q_group_ids[c->N + r] = batch->groups[r];
    }

    c->kv_group_ids = (size_t *)blt_container_alloc(arena, num_patches * c->k * sizeof(size_t));
    for (size_t pi = 0; pi < num_patches; pi++) {
        for (size_t s2 = 0; s2 < c->k; s2++) {
            c->kv_group_ids[pi * c->k + s2] = pi;
        }
    }

    // Per-block start positions s_i (Fast-BLT 3.2.1): block j is the paper's
    // b_{i-1} for patch p_i with i = j+1, so it begins at patch j+1's start.
    // The mask needs these because the clean cutoff is a byte position, not N.
    size_t *block_starts = (size_t *)blt_container_alloc(arena, batch->num_blocks * sizeof(size_t));
    for (size_t j = 0; j < batch->num_blocks; j++) {
        block_starts[j] = patches[j + 1].start_idx;
    }

    blt_block_diffusion_config mc = {
        .mode = mask_mode,
        .seq_len = c->S,
        .num_clean = c->N,
        .block_size = batch->block_size,
        .block_starts = block_starts,
        .num_blocks = batch->num_blocks,
    };
    blt_build_block_diffusion_mask(&mc, &c->self_mask, arena);

    blt_tensor_view_2d(&c->patch_in_split, patch_in->data, num_patches * c->k, c->E, patch_in->backend);
}

static void diff_cross_config(const diff_ctx *c, size_t num_patches, blt_cross_attention_config *xc,
                              blt_mask_config *xm) {
    memset(xc, 0, sizeof(*xc));
    memset(xm, 0, sizeof(*xm));
    xm->seq_len_q = c->S;
    xm->seq_len_kv = num_patches * c->k;
    xm->query_group_ids = c->q_group_ids;
    xm->kv_group_ids = c->kv_group_ids;
    xm->bidirectional_within_group = true;
    xm->is_causal = false;

    xc->embed_dim = c->E;
    xc->patch_dim = c->pdim;
    xc->split_mode = (c->k == 1) ? BLT_CROSS_ATTN_NO_SPLIT : BLT_CROSS_ATTN_SPLIT_KV;
    xc->num_heads = c->XH;
    xc->mask_config = xm;
}

// Forward machinery

typedef struct {
    bool has_cross;
    blt_tensor d_in;          // [S, E] entering this layer
    blt_tensor normed_q;      // cross rmsnorm output
    blt_tensor b_in;          // input to the byte block (post-cross residual)
    blt_tensor normed1;       // self-attn rmsnorm output
    blt_tensor qkv;           // post-RoPE [S, 3E]
    blt_tensor weights_all;   // [H, S, S] post-softmax self-attn weights
    blt_tensor combined;      // [S, E] view over the combined-heads buffer
    blt_tensor resid1;        // [S, E]
    blt_tensor normed2;       // ffn rmsnorm output
    blt_tensor gate, up, act; // [S, hidden]
} diff_layer_cache;

// One bespoke self-attention over the full sequence against the Figure 5
// mask. Rotates Q/K per head (gathered tables), saves post-softmax weights
// for backward, writes the projected output into attn_out.
static void diffusion_self_attention(const diff_ctx *c, const blt_transformer_weights *w, const blt_tensor *normed1,
                                     blt_tensor *attn_out, blt_tensor *qkv_out, blt_tensor *weights_all,
                                     blt_tensor *combined_out, blt_arena *arena) {
    const size_t S = c->S, E = c->E, H = c->H, hd = c->hd;
    const float scale = 1.0f / sqrtf((float)hd);

    size_t qkv_shape[2] = {S, 3 * E};
    *qkv_out = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(normed1, w->attn_qkv_w, qkv_out);

    // Rotate Q/K per head in place (mirror attention.c).
    float *qkv_data = (float *)qkv_out->data;
    const size_t stride3 = 3 * E;
    float *q_head = (float *)blt_arena_alloc(arena, S * hd * sizeof(float), sizeof(float));
    float *k_head = (float *)blt_arena_alloc(arena, S * hd * sizeof(float), sizeof(float));
    float *q_rot = (float *)blt_arena_alloc(arena, S * hd * sizeof(float), sizeof(float));
    float *k_rot = (float *)blt_arena_alloc(arena, S * hd * sizeof(float), sizeof(float));

    blt_tensor q_head_t, q_rot_t, k_head_t, k_rot_t;
    for (size_t h = 0; h < H; h++) {
        const size_t q_off = h * hd;
        const size_t k_off = E + h * hd;
        blt_strided_copy(normed1->backend, q_head, hd, qkv_data + q_off, stride3, S, hd);
        blt_strided_copy(normed1->backend, k_head, hd, qkv_data + k_off, stride3, S, hd);
        blt_tensor_view_3d(&q_head_t, q_head, S, 1, hd, normed1->backend);
        blt_tensor_view_3d(&q_rot_t, q_rot, S, 1, hd, normed1->backend);
        blt_tensor_view_3d(&k_head_t, k_head, S, 1, hd, normed1->backend);
        blt_tensor_view_3d(&k_rot_t, k_rot, S, 1, hd, normed1->backend);
        blt_rope_apply(&q_head_t, &c->rope_cos, &c->rope_sin, &q_rot_t);
        blt_rope_apply(&k_head_t, &c->rope_cos, &c->rope_sin, &k_rot_t);
        blt_strided_copy(normed1->backend, qkv_data + q_off, stride3, q_rot, hd, S, hd);
        blt_strided_copy(normed1->backend, qkv_data + k_off, stride3, k_rot, hd, S, hd);
    }

    // Per-head masked attention against the Figure 5 mask; save post-softmax
    // weights for the backward pass.
    size_t wa_shape[3] = {H, S, S};
    *weights_all = blt_tensor_create(arena, wa_shape, 3, BLT_DTYPE_FP32);
    float *combined = (float *)blt_arena_alloc(arena, S * E * sizeof(float), sizeof(float));
    {
        blt_tensor zt;
        blt_tensor_view_2d(&zt, combined, S, E, normed1->backend);
        zero_tensor(&zt);
    }

    const float *mask_base = (const float *)c->self_mask.data;

    for (size_t h = 0; h < H; h++) {
        const size_t q_off = h * hd;
        const size_t k_off = E + h * hd;
        const size_t v_off = 2 * E + h * hd;
        float *W = (float *)weights_all->data + h * S * S;

        blt_attention_head_args a;
        memset(&a, 0, sizeof(a));
        a.q = qkv_data + q_off;
        a.q_stride = stride3;
        a.k = qkv_data + k_off;
        a.k_stride = stride3;
        a.v = qkv_data + v_off;
        a.v_stride = stride3;
        a.combined = combined;
        a.combined_stride = E;
        a.combined_col_offset = q_off;
        a.weights_out = W;  // saved post-softmax weights for backward
        a.mask = mask_base; // [S * S] Figure-5 additive mask
        a.nq = S;
        a.nk = S;
        a.head_dim = hd;
        a.is_causal = false; // dense mask defines visibility
        a.scale = scale;

        blt_attention_head_core(normed1->backend, &a);
    }

    blt_tensor_view_2d(combined_out, combined, S, E, normed1->backend);
    blt_matmul(combined_out, w->attn_proj_w, attn_out);
}

// Recomputes the forward pass, caching per-layer intermediates plus the
// final states (needed by the LM-head backward).
static void forward_with_caches(const blt_local_decoder *model, const diff_ctx *c, size_t num_patches,
                                blt_tensor *logits, blt_tensor *final_states, diff_layer_cache **caches_out,
                                blt_arena *arena) {
    *caches_out = (diff_layer_cache *)blt_container_alloc(arena, c->L * sizeof(diff_layer_cache));
    memset(*caches_out, 0, c->L * sizeof(diff_layer_cache));
    diff_layer_cache *caches = *caches_out;

    blt_tensor cur = c->d0;

    for (size_t l = 0; l < c->L; l++) {
        const blt_local_decoder_layer_storage *s = &model->layers[l];
        const blt_local_layer_mixed *bf = model->config.mixed_dtype != BLT_DTYPE_FP32 ? &model->layers_mixed[l] : NULL;
        blt_transformer_weights w = blt_local_byte_weights_view_mixed(s, bf);
        diff_layer_cache *ca = &caches[l];
        ca->d_in = cur;
        size_t shape2[2] = {c->S, c->E};

        blt_tensor b;
        if (blt_local_cross_attn_fires(model->config.cross_attn_placement, model->config.cross_attn_all_layers, c->L,
                                       l)) {
            ca->has_cross = true;
            blt_cross_attention_weights xw = blt_local_cross_weights_view_mixed(s, bf);

            ca->normed_q = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&cur, &s->cross_norm_weight, &ca->normed_q);

            blt_cross_attention_config xc;
            blt_mask_config xm;
            diff_cross_config(c, num_patches, &xc, &xm);

            blt_tensor cross_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_cross_attention_forward(&ca->normed_q, &c->patch_in_split, &xw, &cross_out, &xc, arena);

            ca->b_in = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_add(&cur, &cross_out, &ca->b_in);
            b = ca->b_in;
        } else {
            ca->b_in = cur;
            b = cur;
        }

        blt_tensor attn_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        ca->normed1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_forward(&b, w.norm1_weight, &ca->normed1);
        diffusion_self_attention(c, &w, &ca->normed1, &attn_out, &ca->qkv, &ca->weights_all, &ca->combined, arena);

        ca->resid1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&b, &attn_out, &ca->resid1);

        ca->normed2 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_forward(&ca->resid1, w.norm2_weight, &ca->normed2);

        const size_t hidden_dim = w.ffn_up_w->shape[1];
        size_t hidden_shape[2] = {c->S, hidden_dim};
        ca->up = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&ca->normed2, w.ffn_up_w, &ca->up);
        ca->gate = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&ca->normed2, w.ffn_gate_w, &ca->gate);
        ca->act = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_swiglu_forward(&ca->gate, &ca->up, &ca->act);
        blt_tensor ffn_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul(&ca->act, w.ffn_down_w, &ffn_out);

        blt_tensor next = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&ca->resid1, &ffn_out, &next);
        cur = next;
    }

    *final_states = cur;
    size_t logits_shape[2] = {c->S, c->V};
    *logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    blt_matmul(final_states, blt_local_decoder_lm_head(model), logits);
}

static void make_bytes_tensor(blt_arena *arena, const uint8_t *bytes, size_t len, blt_tensor *out) {
    size_t shape[1] = {len};
    *out = blt_tensor_create(arena, shape, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(out, bytes, len * sizeof(uint8_t));
}

// Public forward

void blt_local_decoder_forward_diffusion(const blt_local_decoder *model, const blt_tensor *byte_hidden_in,
                                         const blt_tensor *patch_in, const blt_patch_info *patches, size_t num_patches,
                                         const uint8_t *clean_bytes, const blt_tensor *targets,
                                         const blt_block_batch *batch, blt_d0_mode d0_mode, blt_tensor *logits_out,
                                         blt_tensor *loss_out, blt_arena *arena) {
    BLT_REQUIRE(model != NULL && clean_bytes != NULL && logits_out != NULL && loss_out != NULL && arena != NULL,
                "forward_diffusion: arguments cannot be NULL");

    diff_ctx c;
    diff_ctx_build(&c, model, byte_hidden_in, patch_in, patches, num_patches, batch, d0_mode, BLT_BDM_TRAIN, arena);

    size_t out_shape[2] = {c.S, c.V};
    blt_check_nd_fp32(logits_out, 2, out_shape,
                      "forward_diffusion: logits_out must be [N + n_block_rows, vocab_size] FP32");

    diff_layer_cache *caches;
    blt_tensor logits, final_states;
    forward_with_caches(model, &c, num_patches, &logits, &final_states, &caches, arena);
    blt_strided_copy(logits.backend, (float *)logits_out->data, c.S * c.V, (const float *)logits.data, c.S * c.V, 1,
                     c.S * c.V);

    // Losses (Eq. 7): L_clean via the legacy CE path over clean rows +
    // L_mask/t accumulated over masked block cells.
    float loss = 0.0f;

    if (c.N >= 2) {
        blt_tensor logits_view;
        // When targets is provided (N+1 bytes), use c.N rows (all clean positions)
        // When targets is NULL, use c.N - 1 rows (legacy behavior)
        const size_t clean_rows = (targets != NULL) ? c.N : c.N - 1;
        blt_tensor_view_2d(&logits_view, logits.data, clean_rows, c.V, logits.backend);

        blt_tensor targets_t;
        if (targets != NULL) {
            // targets has c.N + 1 bytes: targets[1..c.N] covers all clean prediction positions
            view_1d_offset(&targets_t, targets, 1, clean_rows);
        } else {
            blt_tensor bytes_t;
            make_bytes_tensor(arena, clean_bytes, c.N, &bytes_t);
            view_1d_offset(&targets_t, &bytes_t, 1, c.N - 1);
        }

        blt_cross_entropy_forward(&logits_view, &targets_t, loss_out);
        blt_tensor_download(loss_out, &loss, sizeof(float));
        // cross_entropy_forward yields a mean; paper Eq. 5 is a sum over i=1..N.
        if (batch->clean_sum) loss *= (float)clean_rows;
        if (targets != NULL && batch->last_row_scale != 1.0f) {
            blt_tensor last_row_view;
            blt_tensor_view_2d(&last_row_view, (float *)logits.data + (c.N - 1) * c.V, 1, c.V, logits.backend);
            blt_tensor last_target;
            view_1d_offset(&last_target, targets, c.N, 1);
            size_t one_shape[1] = {1};
            blt_tensor ce_last = blt_tensor_create(arena, one_shape, 1, BLT_DTYPE_FP32);
            blt_cross_entropy_forward(&last_row_view, &last_target, &ce_last);
            float ce_last_val = 0.0f;
            blt_tensor_download(&ce_last, &ce_last_val, sizeof(float));
            loss += (batch->last_row_scale - 1.0f) * ce_last_val / (float)c.N;
        }
    }

    if (batch->t > 0.0f) {
        // Masked-cell loss math runs on a host copy of the logits.
        float *logits_host = (float *)malloc(logits.numel * sizeof(float));
        BLT_REQUIRE(logits_host != NULL, "forward_diffusion: staging alloc failed");
        blt_tensor_download(&logits, logits_host, logits.numel * sizeof(float));
        float l_mask = 0.0f;
        size_t n_masked_cells = 0;
        for (size_t r = 0; r < c.R; r++) {
            if (!batch->cell_masked[r]) continue;
            n_masked_cells++;
            const float *row = logits_host + (c.N + r) * c.V;
            // -log p[target], stable
            float max_val = row[0];
            for (size_t v = 1; v < c.V; v++) {
                if (row[v] > max_val) max_val = row[v];
            }
            float sum = 0.0f, pt = 0.0f;
            for (size_t v = 0; v < c.V; v++) {
                float e = expf(row[v] - max_val);
                sum += e;
                if (v == (size_t)batch->targets[r]) pt = e;
            }
            // floor guards -log(0) when the target probability underflows
            l_mask += -logf(fmaxf(pt / sum, 1e-9f));
        }
        free(logits_host);
        if (batch->mask_norm && n_masked_cells > 0) l_mask /= (float)n_masked_cells;
        loss += batch->loss_scale * l_mask / batch->t;
    }

    blt_tensor_upload(loss_out, &loss, sizeof(float));
}

// Backward

void blt_local_decoder_backward_diffusion(const blt_local_decoder *model, const blt_tensor *byte_hidden_in,
                                          const blt_tensor *patch_in, const blt_patch_info *patches, size_t num_patches,
                                          const uint8_t *clean_bytes, const blt_tensor *targets,
                                          const blt_block_batch *batch, blt_d0_mode d0_mode,
                                          blt_tensor *grad_byte_hidden_in, blt_tensor *grad_patch_in,
                                          blt_local_decoder_grad *grad, blt_arena *arena) {
    BLT_REQUIRE(model != NULL && clean_bytes != NULL && grad_byte_hidden_in != NULL && grad_patch_in != NULL &&
                    grad != NULL && arena != NULL,
                "backward_diffusion: arguments cannot be NULL");

    diff_ctx c;
    diff_ctx_build(&c, model, byte_hidden_in, patch_in, patches, num_patches, batch, d0_mode, BLT_BDM_TRAIN, arena);

    blt_check_nd_fp32(grad_byte_hidden_in, 2, (const size_t[]){c.N, c.E},
                      "backward_diffusion: grad_byte_hidden_in must be [N, embed_dim] FP32");
    blt_check_nd_fp32(grad_patch_in, 2, (const size_t[]){num_patches, c.pdim},
                      "backward_diffusion: grad_patch_in must be [num_patches, patch_dim] FP32");
    blt_check_nd_fp32(&grad->lm_head_grad, 2, (const size_t[]){c.E, c.V},
                      "backward_diffusion: lm_head_grad must be [embed_dim, vocab_size] FP32");

    // Recompute the forward, caching intermediates.
    diff_layer_cache *caches;
    blt_tensor logits, final_states;
    forward_with_caches(model, &c, num_patches, &logits, &final_states, &caches, arena);

    // Terminal gradient dL/d(logits).
    blt_tensor grad_logits = blt_tensor_create(arena, (size_t[2]){c.S, c.V}, 2, BLT_DTYPE_FP32); // zero-init

    if (c.N >= 2) {
        blt_tensor logits_view;
        const size_t clean_rows = (targets != NULL) ? c.N : c.N - 1;
        blt_tensor_view_2d(&logits_view, logits.data, clean_rows, c.V, logits.backend);

        blt_tensor targets_t;
        if (targets != NULL) {
            view_1d_offset(&targets_t, targets, 1, clean_rows);
        } else {
            blt_tensor bytes_t;
            make_bytes_tensor(arena, clean_bytes, c.N, &bytes_t);
            view_1d_offset(&targets_t, &bytes_t, 1, c.N - 1);
        }

        size_t gv_shape[2] = {clean_rows, c.V};
        blt_tensor grad_view = blt_tensor_create(arena, gv_shape, 2, BLT_DTYPE_FP32);
        blt_cross_entropy_backward(&logits_view, &targets_t, &grad_view);
        if (batch->clean_sum) blt_scale(&grad_view, (float)clean_rows);
        if (targets != NULL && batch->last_row_scale != 1.0f) {
            blt_tensor last_grad_view;
            blt_tensor_view_2d(&last_grad_view, (float *)grad_view.data + (c.N - 1) * c.V, 1, c.V, grad_view.backend);
            blt_scale(&last_grad_view, batch->last_row_scale);
        }
        blt_strided_copy(grad_view.backend, (float *)grad_logits.data, c.V, (const float *)grad_view.data, c.V,
                         clean_rows, c.V);
    }

    if (batch->t > 0.0f && batch->loss_scale != 0.0f) {
        float inv_t = batch->loss_scale / batch->t;
        if (batch->mask_norm) {
            size_t n_masked_cells = 0;
            for (size_t r = 0; r < c.R; r++)
                if (batch->cell_masked[r]) n_masked_cells++;
            if (n_masked_cells > 0) inv_t /= (float)n_masked_cells;
        }
        float *gl = (float *)grad_logits.data;
        // Masked-cell gradient math runs on a host copy of the logits; the
        // finished rows are written back one at a time.
        float *logits_host = (float *)malloc(logits.numel * sizeof(float));
        BLT_REQUIRE(logits_host != NULL, "backward_diffusion: staging alloc failed");
        blt_tensor_download(&logits, logits_host, logits.numel * sizeof(float));
        float *grow_host = (float *)malloc(c.V * sizeof(float));
        BLT_REQUIRE(grow_host != NULL, "backward_diffusion: staging alloc failed");
        for (size_t r = 0; r < c.R; r++) {
            if (!batch->cell_masked[r]) continue;
            const size_t row = c.N + r;
            const float *lr = logits_host + row * c.V;
            const uint8_t tgt = batch->targets[r];

            float max_val = lr[0];
            for (size_t v = 1; v < c.V; v++) {
                if (lr[v] > max_val) max_val = lr[v];
            }
            float sum = 0.0f;
            for (size_t v = 0; v < c.V; v++) {
                sum += expf(lr[v] - max_val);
            }
            for (size_t v = 0; v < c.V; v++) {
                const float p = expf(lr[v] - max_val) / sum;
                grow_host[v] = inv_t * (p - (v == (size_t)tgt ? 1.0f : 0.0f));
            }
            blt_tensor row_view;
            blt_tensor_view_2d(&row_view, gl + row * c.V, 1, c.V, logits.backend);
            blt_tensor_upload(&row_view, grow_host, c.V * sizeof(float));
        }
        free(logits_host);
        free(grow_host);
    }

    // LM head backward -> dL/d(final states).
    blt_tensor dh = blt_tensor_create(arena, (size_t[2]){c.S, c.E}, 2, BLT_DTYPE_FP32);
    blt_matmul_backward(&final_states, blt_local_decoder_lm_head(model), &grad_logits, &dh, &grad->lm_head_grad);

    // Reverse layer loop.
    blt_tensor dP_split = blt_tensor_create(arena, (size_t[2]){num_patches * c.k, c.E}, 2, BLT_DTYPE_FP32); // zero-init

    const size_t stride3 = 3 * c.E;
    const float self_scale = 1.0f / sqrtf((float)c.hd);

    for (size_t li = c.L; li-- > 0;) {
        const blt_local_decoder_layer_storage *s = &model->layers[li];
        const blt_local_layer_mixed *bf = model->config.mixed_dtype != BLT_DTYPE_FP32 ? &model->layers_mixed[li] : NULL;
        blt_transformer_weights w = blt_local_byte_weights_view_mixed(s, bf);
        blt_local_decoder_layer_grad *lg = &grad->layer_grads[li];
        const diff_layer_cache *ca = &caches[li];

        size_t shape2[2] = {c.S, c.E};
        const size_t hidden_dim = w.ffn_up_w->shape[1];
        size_t hidden_shape[2] = {c.S, hidden_dim};

        // ---------------- FFN backward ----------------
        // next = resid1 + ffn_out ; ffn_out = act @ ffn_down_w
        blt_tensor g_act = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&ca->act, w.ffn_down_w, &dh, &g_act, &lg->ffn_down_w);

        blt_tensor g_gate = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_tensor g_up = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_swiglu_backward(&g_act, &ca->gate, &ca->up, &g_gate, &g_up);

        blt_tensor g_n2g = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&ca->normed2, w.ffn_gate_w, &g_gate, &g_n2g, &lg->ffn_gate_w);
        blt_tensor g_n2u = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&ca->normed2, w.ffn_up_w, &g_up, &g_n2u, &lg->ffn_up_w);

        blt_tensor g_normed2 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&g_n2g, &g_n2u, &g_normed2);

        blt_tensor g_r1_norm = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_backward(&g_normed2, &ca->resid1, w.norm2_weight, &g_r1_norm, &lg->norm2_weight);

        // resid1 receives the direct path AND the norm path
        blt_tensor g_resid1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&dh, &g_r1_norm, &g_resid1);

        // ---------------- self-attention backward ----------------
        // resid1 = b_in + attn_out ; attn_out = combined @ proj
        blt_tensor g_combined = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&ca->combined, w.attn_proj_w, &g_resid1, &g_combined, &lg->attn_proj_w);

        // mirror attention.c backward using the cached post-RoPE qkv and
        // post-softmax weights
        blt_tensor g_qkv = blt_tensor_create(arena, (size_t[2]){c.S, stride3}, 2, BLT_DTYPE_FP32); // zero-init
        float *gqkv = (float *)g_qkv.data;
        const float *qkv = (const float *)ca->qkv.data;
        const float *gcomb = (const float *)g_combined.data;

        float *gs_buf = (float *)blt_arena_alloc(arena, c.S * c.S * sizeof(float), sizeof(float));

        for (size_t h = 0; h < c.H; h++) {
            const size_t q_off = h * c.hd;
            const size_t k_off = c.E + h * c.hd;
            const size_t v_off = 2 * c.E + h * c.hd;

            blt_attention_head_bwd_args a;
            memset(&a, 0, sizeof(a));
            a.q = qkv + q_off;
            a.q_stride = stride3;
            a.k = qkv + k_off;
            a.k_stride = stride3;
            a.v = qkv + v_off;
            a.v_stride = stride3;
            a.weights = (const float *)ca->weights_all.data + h * c.S * c.S;
            a.grad_combined = gcomb;
            a.gc_stride = c.E;
            a.gc_col_offset = q_off;
            a.scores_scratch = gs_buf;
            a.grad_q = gqkv + q_off;
            a.gq_stride = stride3;
            a.grad_k = gqkv + k_off;
            a.gk_stride = stride3;
            a.grad_v = gqkv + v_off;
            a.gv_stride = stride3;
            a.nq = c.S;
            a.nk = c.S;
            a.head_dim = c.hd;
            a.scale = self_scale;

            blt_attention_head_core_backward(ca->qkv.backend, &a);
        }

        // RoPE backward on the Q/K gradients (same gathered tables)
        {
            // Fused RoPE backward on packed layout - no intermediate buffers needed
            const size_t stride3 = 3 * c.E;
            for (size_t h = 0; h < c.H; h++) {
                const size_t q_off = h * c.hd;
                const size_t k_off = c.E + h * c.hd;
                blt_rope_apply_packed_backward(gqkv, stride3, q_off, c.S, c.hd, (const float *)c.rope_cos.data,
                                               (const float *)c.rope_sin.data, dh.backend);
                blt_rope_apply_packed_backward(gqkv, stride3, k_off, c.S, c.hd, (const float *)c.rope_cos.data,
                                               (const float *)c.rope_sin.data, dh.backend);
            }
        }

        blt_tensor g_normed1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul_backward(&ca->normed1, w.attn_qkv_w, &g_qkv, &g_normed1, &lg->attn_qkv_w);

        blt_tensor g_b_norm = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_backward(&g_normed1, &ca->b_in, w.norm1_weight, &g_b_norm, &lg->norm1_weight);

        blt_tensor g_b = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&g_resid1, &g_b_norm, &g_b);

        // ---------------- cross-attention backward ----------------
        if (ca->has_cross) {
            blt_cross_attention_weights xw = blt_local_cross_weights_view_mixed(s, bf);
            blt_cross_attention_grad xg = {0};
            xg.grad_weight_q = lg->cross_weight_q;
            xg.grad_weight_k = lg->cross_weight_k;
            xg.grad_weight_v = lg->cross_weight_v;
            xg.grad_weight_proj = lg->cross_weight_proj;

            blt_cross_attention_config xc;
            blt_mask_config xm;
            diff_cross_config(&c, num_patches, &xc, &xm);

            blt_tensor grad_normed_q = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_tensor grad_kv_tmp = blt_tensor_create(arena, (size_t[2]){num_patches * c.k, c.E}, 2, BLT_DTYPE_FP32);
            blt_cross_attention_backward(&ca->normed_q, &c.patch_in_split, &xw, &g_b, &grad_normed_q, &grad_kv_tmp, &xg,
                                         &xc, arena);

            blt_tensor dP_new = blt_tensor_create(arena, (size_t[2]){num_patches * c.k, c.E}, 2, BLT_DTYPE_FP32);
            blt_add(&dP_split, &grad_kv_tmp, &dP_new);
            dP_split = dP_new;

            blt_tensor g_din_norm = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_backward(&grad_normed_q, &ca->d_in, &s->cross_norm_weight, &g_din_norm, &lg->cross_norm_weight);

            blt_tensor dh_prev = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_add(&g_b, &g_din_norm, &dh_prev);
            dh = dh_prev;
        } else {
            dh = g_b;
        }
    }

    // Terminal gradients.
    blt_strided_copy(dh.backend, (float *)grad_byte_hidden_in->data, c.N * c.E, (const float *)dh.data, c.N * c.E, 1,
                     c.N * c.E);

    if (d0_mode == BLT_D0_LEARNED) {
        // Scatter block-row D0 gradients into the token table (dispatched).
        blt_tensor tailg;
        blt_tensor_view_2d(&tailg, (float *)dh.data + c.N * c.E, c.R, c.E, dh.backend);
        blt_indexed_row_scatter_add(&grad->d0_embed_grad, batch->tokens, &tailg, 1.0f);
    }

    // Reinterpret dP_split back as [num_patches, patch_dim]
    blt_tensor dP;
    blt_tensor_view_2d(&dP, dP_split.data, num_patches, c.pdim, dP_split.backend);
    blt_strided_copy(dP.backend, (float *)grad_patch_in->data, num_patches * c.pdim, (const float *)dP.data,
                     num_patches * c.pdim, 1, num_patches * c.pdim);
}

// Public inference forward (Fast-BLT section 3.1.1 / Algorithm 1)
//
// Same decoder pass as blt_local_decoder_forward_diffusion but with the
// INFERENCE self-attention pattern: clean rows causal; every block row
// attends all clean positions plus the whole block section
// bidirectionally. No loss is computed -- the caller drives iterative
// unmasking over batch->tokens / cell_masked between passes.
//
// The caller owns the batch: fill tokens (byte ids or BLT_MASK_TOKEN_ID),
// positions (true future positions for RoPE), groups (latent index each
// block row cross-attends; the paper's rule is o_M for ALL block rows),
// cell_masked, num_clean = N, n_block_rows = B, t = 0.
void blt_local_decoder_forward_diffusion_infer(const blt_local_decoder *model, const blt_tensor *byte_hidden_in,
                                               const blt_tensor *patch_in, const blt_patch_info *patches,
                                               size_t num_patches, const blt_block_batch *batch, blt_d0_mode d0_mode,
                                               blt_tensor *logits_out, blt_arena *arena) {
    BLT_REQUIRE(model != NULL && logits_out != NULL && arena != NULL && batch != NULL,
                "forward_diffusion_infer: arguments cannot be NULL");

    diff_ctx c;
    diff_ctx_build(&c, model, byte_hidden_in, patch_in, patches, num_patches, batch, d0_mode, BLT_BDM_INFER, arena);

    size_t out_shape[2] = {c.S, c.V};
    blt_check_nd_fp32(logits_out, 2, out_shape,
                      "forward_diffusion_infer: logits_out must be [N + n_block_rows, vocab_size] FP32");

    diff_layer_cache *caches;
    blt_tensor logits, final_states;
    forward_with_caches(model, &c, num_patches, &logits, &final_states, &caches, arena);
    blt_strided_copy(logits.backend, (float *)logits_out->data, c.S * c.V, (const float *)logits.data, c.S * c.V, 1,
                     c.S * c.V);
}
