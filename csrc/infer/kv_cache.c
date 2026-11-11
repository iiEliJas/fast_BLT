// Two-tier KV cache + cache-aware incremental decoder.
// See kv_cache.h for the design contract. Every computation here mirrors
// the dense composition in local_decoder.c / attention.c / cross_attention.c
// op-for-op so results are bit-identical. Host/device split: the container
// struct and handle arrays live on the heap (host); tensor payloads come
// from the caller's arena, which may be CUDA device memory.
#include "infer/kv_cache.h"

#include "core/backend.h"
#include "models/local_common.h"
#include "ops/matmul.h"
#include "ops/rmsnorm.h"
#include "ops/swiglu.h"
#include "ops/rope.h"
#include "ops/vecmath.h"
#include "ops/attn_core.h"
#include "ops/mask_builder.h"
#include "ops/gather_scatter.h"
#include "ops/elementwise.h"
#include "infer/rope_gather.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

blt_kv_cache *blt_kv_cache_create_batched(blt_arena *arena, const blt_local_decoder *decoder, size_t max_seq_len,
                                          size_t batch_size) {
    if (batch_size == 0) batch_size = 1;
    BLT_REQUIRE(arena != NULL && decoder != NULL, "blt_kv_cache_create: arena and decoder cannot be NULL");
    BLT_REQUIRE(max_seq_len >= 2 && max_seq_len <= decoder->config.max_seq_len,
                "blt_kv_cache_create: max_seq_len must be in [2, decoder max_seq_len]");

    const size_t E = decoder->config.embed_dim;
    const size_t patch_dim = decoder->config.patch_dim ? decoder->config.patch_dim : E;
    BLT_REQUIRE(patch_dim % E == 0, "blt_kv_cache_create: patch_dim must be a multiple of embed_dim");

    blt_kv_cache *c = (blt_kv_cache *)calloc(1, sizeof(*c));
    BLT_REQUIRE(c != NULL, "blt_kv_cache_create: failed to allocate cache container");
    c->decoder = decoder;
    c->max_seq_len = max_seq_len;
    c->batch_size = batch_size;
    c->max_subtokens = max_seq_len; // patches have >= 1 byte each
    c->embed_dim = E;
    c->patch_dim = patch_dim;
    c->k = patch_dim / E;

    const size_t L = decoder->config.num_layers;
    c->self_k = (blt_tensor *)calloc(L, sizeof(blt_tensor));
    c->self_v = (blt_tensor *)calloc(L, sizeof(blt_tensor));
    c->cross_k = (blt_tensor *)calloc(L, sizeof(blt_tensor));
    c->cross_v = (blt_tensor *)calloc(L, sizeof(blt_tensor));
    BLT_REQUIRE(c->self_k && c->self_v && c->cross_k && c->cross_v,
                "blt_kv_cache_create: failed to allocate layer handle arrays");

    size_t self_shape[2] = {batch_size * max_seq_len, E};
    size_t cross_shape[2] = {batch_size * c->max_subtokens, E};
    for (size_t l = 0; l < L; l++) {
        c->self_k[l] = blt_tensor_create(arena, self_shape, 2, BLT_DTYPE_FP32);
        c->self_v[l] = blt_tensor_create(arena, self_shape, 2, BLT_DTYPE_FP32);
        c->cross_k[l] = blt_tensor_create(arena, cross_shape, 2, BLT_DTYPE_FP32);
        c->cross_v[l] = blt_tensor_create(arena, cross_shape, 2, BLT_DTYPE_FP32);
    }

    c->cached_patches = (blt_patch_info *)calloc(batch_size * max_seq_len, sizeof(blt_patch_info));
    BLT_REQUIRE(c->cached_patches != NULL, "blt_kv_cache_create: failed to allocate patch table");

    c->self_len = 0;
    c->cross_num_patches = 0;
    return c;
}

blt_kv_cache *blt_kv_cache_create(blt_arena *arena, const blt_local_decoder *decoder, size_t max_seq_len) {
    return blt_kv_cache_create_batched(arena, decoder, max_seq_len, 1);
}

void blt_kv_cache_destroy(blt_kv_cache *cache) {
    if (!cache) {
        return;
    }
    free(cache->self_k);
    free(cache->self_v);
    free(cache->cross_k);
    free(cache->cross_v);
    free(cache->cached_patches);
    free(cache);
}

void blt_kv_cache_reset(blt_kv_cache *cache) {
    cache->self_len = 0;
    cache->cross_num_patches = 0;
}

void blt_kv_cache_truncate(blt_kv_cache *cache, size_t self_len, size_t num_patches) {
    BLT_REQUIRE(self_len <= cache->max_seq_len, "blt_kv_cache_truncate: self_len out of range");
    BLT_REQUIRE(num_patches <= cache->max_subtokens, "blt_kv_cache_truncate: num_patches out of range");
    cache->self_len = self_len;
    cache->cross_num_patches = num_patches;
}

size_t blt_kv_cache_common_patches(const blt_kv_cache *cache, const blt_patch_info *patches, size_t num_patches) {
    size_t common = 0;
    while (common < cache->cross_num_patches && common < num_patches &&
           cache->cached_patches[common].start_idx == patches[common].start_idx &&
           cache->cached_patches[common].length == patches[common].length) {
        common++;
    }
    return common;
}

static void refresh_cross_layer(blt_kv_cache *cache, size_t layer, const blt_cross_attention_weights *xw,
                                const blt_tensor *patch_in_split, size_t from_subtoken, size_t num_new_subtokens) {
    // Contiguous sub-token block — view as [num_new_subtokens, E] and write
    // straight into cache at the matching offset.
    const blt_backend backend = patch_in_split->backend;

    blt_tensor block;
    blt_tensor_view_2d(&block, (float *)patch_in_split->data + from_subtoken * cache->embed_dim, num_new_subtokens,
                       cache->embed_dim, backend);

    blt_tensor k_out, v_out;
    blt_tensor_view_2d(&k_out, (float *)cache->cross_k[layer].data + from_subtoken * cache->embed_dim,
                       num_new_subtokens, cache->embed_dim, backend);
    blt_tensor_view_2d(&v_out, (float *)cache->cross_v[layer].data + from_subtoken * cache->embed_dim,
                       num_new_subtokens, cache->embed_dim, backend);

    blt_matmul(&block, xw->weight_k, &k_out);
    blt_matmul(&block, xw->weight_v, &v_out);
}

void blt_kv_cache_refresh_cross(blt_kv_cache *cache, const blt_tensor *patch_in, const blt_patch_info *patches,
                                size_t num_patches, size_t from_patch, blt_arena *arena) {
    const blt_local_decoder *dec = cache->decoder;
    BLT_REQUIRE(patch_in != NULL && patches != NULL && arena != NULL,
                "blt_kv_cache_refresh_cross: arguments cannot be NULL");
    BLT_REQUIRE(num_patches >= 1 && from_patch < num_patches,
                "blt_kv_cache_refresh_cross: need from_patch < num_patches");
    BLT_REQUIRE(from_patch <= cache->cross_num_patches,
                "blt_kv_cache_refresh_cross: from_patch must not skip uncached patches "
                "(truncate to the LCP first)");
    blt_check_nd_fp32(patch_in, 2, (const size_t[]){num_patches, cache->patch_dim},
                      "blt_kv_cache_refresh_cross: patch_in must be [num_patches, patch_dim] FP32");
    BLT_REQUIRE(num_patches * cache->k <= cache->max_subtokens,
                "blt_kv_cache_refresh_cross: sub-token count exceeds cache capacity");

    blt_tensor patch_in_split;
    blt_tensor_view_2d(&patch_in_split, patch_in->data, num_patches * cache->k, cache->embed_dim, patch_in->backend);

    const size_t L = dec->config.num_layers;
    for (size_t l = 0; l < L; l++) {
        if (!blt_local_cross_attn_fires(dec->config.cross_attn_placement, dec->config.cross_attn_all_layers,
                                        dec->config.num_layers, l)) {
            continue;
        }
        const blt_local_decoder_layer_storage *s = &dec->layers[l];
        const blt_local_layer_mixed *bf = dec->config.mixed_dtype != BLT_DTYPE_FP32 ? &dec->layers_mixed[l] : NULL;
        blt_cross_attention_weights xw = blt_local_cross_weights_view_mixed(s, bf);
        refresh_cross_layer(cache, l, &xw, &patch_in_split, from_patch * cache->k,
                            (num_patches - from_patch) * cache->k);
    }

    // Snapshot patched regions.
    memcpy(&cache->cached_patches[from_patch], &patches[from_patch],
           (num_patches - from_patch) * sizeof(blt_patch_info));
    cache->cross_num_patches = num_patches;
}

void blt_kv_decode_step(blt_kv_cache *cache, const blt_tensor *patch_in, const blt_patch_info *patches,
                        size_t num_patches, const blt_tensor *d0_rows, int trailing_closed, blt_tensor *logits_out,
                        blt_arena *arena) {
    const blt_local_decoder *dec = cache->decoder;
    const blt_local_decoder_config *cfg = &dec->config;
    const size_t E = cache->embed_dim;
    const size_t V = cfg->vocab_size;
    const size_t H = cfg->num_heads;
    const size_t XH = cfg->cross_attn_heads;
    const size_t hd = E / H;
    const size_t xhd = E / XH;
    BLT_REQUIRE(hd > 0 && hd % 2 == 0, "blt_kv_decode_step: RoPE requires an even head_dim");

    BLT_REQUIRE(cache != NULL && patch_in != NULL && patches != NULL && d0_rows != NULL && logits_out != NULL &&
                    arena != NULL,
                "blt_kv_decode_step: arguments cannot be NULL");
    BLT_REQUIRE(num_patches >= 1 && num_patches == cache->cross_num_patches,
                "blt_kv_decode_step: tier-2 cache must cover all patches (refresh first)");
    blt_check_nd_fp32(d0_rows, 2, (const size_t[]){0, E}, "blt_kv_decode_step: d0_rows must be [n, embed_dim] FP32");

    const size_t n = d0_rows->shape[0];
    BLT_REQUIRE(n >= 1, "blt_kv_decode_step: need at least one row");
    const size_t base = cache->self_len;
    const size_t total = base + n;
    BLT_REQUIRE(total <= cache->max_seq_len, "blt_kv_decode_step: exceeds cache capacity");
    blt_check_nd_fp32(logits_out, 2, (const size_t[]){n, V},
                      "blt_kv_decode_step: logits_out must be [n, vocab_size] FP32");
    BLT_REQUIRE(num_patches * cache->k <= cache->max_subtokens,
                "blt_kv_decode_step: sub-token count exceeds cache capacity");

    const blt_backend backend = d0_rows->backend;
    const size_t total_sub = num_patches * cache->k;

    // Group bookkeeping: each new row takes its paper-rule decoder latent
    // index (Fast-BLT 3.1.1); rows past the last patch end fall through to
    // the last patch.
    size_t *q_group_ids = (size_t *)malloc(n * sizeof(size_t));
    size_t *kv_group_ids = (size_t *)malloc(total_sub * sizeof(size_t));
    BLT_REQUIRE(q_group_ids != NULL && kv_group_ids != NULL, "blt_kv_decode_step: failed to allocate group id tables");
    for (size_t r = 0; r < n; r++) {
        q_group_ids[r] = blt_patch_decoder_latent_at(patches, num_patches, base + r);
    }
    // A boundary at the sequence end makes the last decoded byte final (own
    // latent), matching the full forward's trailing_closed handling.
    if (trailing_closed && num_patches >= 1) {
        q_group_ids[n - 1] = num_patches - 1;
    }
    for (size_t pi = 0; pi < num_patches; pi++) {
        for (size_t s = 0; s < cache->k; s++) {
            kv_group_ids[pi * cache->k + s] = pi;
        }
    }

    // Dense additive masks — host-computed, uploaded (arena may be device memory).
    blt_mask_config mcfg;
    memset(&mcfg, 0, sizeof(mcfg));

    size_t self_shape[2] = {n, total};
    blt_tensor self_mask = blt_tensor_create(arena, self_shape, 2, BLT_DTYPE_FP32);
    float *smd = (float *)malloc(n * total * sizeof(float));
    BLT_REQUIRE(smd != NULL, "blt_kv_decode_step: failed to allocate mask staging");
    const size_t window = cfg->local_window;
    for (size_t r = 0; r < n; r++) {
        const size_t p = base + r;
        for (size_t j = 0; j < total; j++) {
            bool allowed = (j <= p);
            if (allowed && window > 0 && p - j >= window) allowed = false;
            smd[r * total + j] = allowed ? 0.0f : -INFINITY;
        }
    }
    blt_tensor_upload(&self_mask, smd, n * total * sizeof(float));
    free(smd);

    blt_tensor cross_mask;
    size_t cross_shape[2] = {n, total_sub};
    cross_mask = blt_tensor_create(arena, cross_shape, 2, BLT_DTYPE_FP32);
    float *cmd = (float *)malloc(n * total_sub * sizeof(float));
    BLT_REQUIRE(cmd != NULL, "blt_kv_decode_step: failed to allocate mask staging");
    for (size_t r = 0; r < n; r++) {
        for (size_t j = 0; j < total_sub; j++) {
            cmd[r * total_sub + j] = (kv_group_ids[j] == q_group_ids[r]) ? 0.0f : -INFINITY;
        }
    }
    blt_tensor_upload(&cross_mask, cmd, n * total_sub * sizeof(float));
    free(cmd);

    free(q_group_ids);
    free(kv_group_ids);

    float *self_scratch = (float *)blt_arena_alloc(arena, n * total * sizeof(float), sizeof(float));
    float *cross_scratch = (float *)blt_arena_alloc(arena, n * total_sub * sizeof(float), sizeof(float));

    const float self_scale = 1.0f / sqrtf((float)hd);
    const float cross_scale = 1.0f / sqrtf((float)xhd);

    // RoPE gather for new rows' positions.
    size_t *positions = (size_t *)malloc(n * sizeof(size_t));
    BLT_REQUIRE(positions != NULL, "blt_kv_decode_step: failed to allocate position buffer");
    for (size_t r = 0; r < n; r++) {
        positions[r] = base + r;
    }
    blt_tensor rope_cos, rope_sin;
    blt_rope_position_gather(&dec->rope_cos_cache, &dec->rope_sin_cache, positions, n, &rope_cos, &rope_sin, arena);
    free(positions);

    // Split view of patch_in: [num_patches*k, E]
    blt_tensor patch_in_split;
    blt_tensor_view_2d(&patch_in_split, patch_in->data, total_sub, E, backend);

    float *q_full = (float *)blt_arena_alloc(arena, n * E * sizeof(float), sizeof(float));
    blt_tensor combined = blt_tensor_create(arena, (size_t[2]){n, E}, 2, BLT_DTYPE_FP32);

    float *q_head = (float *)blt_arena_alloc(arena, n * hd * sizeof(float), sizeof(float));
    float *k_head = (float *)blt_arena_alloc(arena, n * hd * sizeof(float), sizeof(float));
    float *q_rot = (float *)blt_arena_alloc(arena, n * hd * sizeof(float), sizeof(float));
    float *k_rot = (float *)blt_arena_alloc(arena, n * hd * sizeof(float), sizeof(float));

    blt_tensor cur = *d0_rows;

    const size_t L = cfg->num_layers;
    for (size_t l = 0; l < L; l++) {
        const blt_local_decoder_layer_storage *s = &dec->layers[l];
        const blt_local_layer_mixed *bf = cfg->mixed_dtype != BLT_DTYPE_FP32 ? &dec->layers_mixed[l] : NULL;
        blt_transformer_weights w = blt_local_byte_weights_view_mixed(s, bf);
        size_t shape2[2] = {n, E};
        blt_tensor b;

        if (blt_local_cross_attn_fires(cfg->cross_attn_placement, cfg->cross_attn_all_layers, L, l)) {
            blt_cross_attention_weights xw = blt_local_cross_weights_view_mixed(s, bf);

            blt_tensor normed_q = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_rmsnorm_forward(&cur, &s->cross_norm_weight, &normed_q);

            blt_tensor q_rows = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_matmul(&normed_q, xw.weight_q, &q_rows);

            zero_tensor(&combined);
            for (size_t h = 0; h < XH; h++) {
                blt_attention_head_args ha;
                memset(&ha, 0, sizeof(ha));
                ha.q = (const float *)q_rows.data + h * xhd;
                ha.q_stride = E;
                ha.k = (const float *)cache->cross_k[l].data + h * xhd;
                ha.k_stride = E;
                ha.v = (const float *)cache->cross_v[l].data + h * xhd;
                ha.v_stride = E;
                ha.combined = (float *)combined.data;
                ha.combined_stride = E;
                ha.combined_col_offset = h * xhd;
                ha.weights_out = NULL;
                ha.scores_scratch = cross_scratch;
                ha.mask = (const float *)cross_mask.data;
                ha.nq = n;
                ha.nk = total_sub;
                ha.head_dim = xhd;
                ha.is_causal = false;
                ha.scale = cross_scale;
                blt_attention_head_core(backend, &ha);
            }

            blt_tensor cross_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_matmul(&combined, xw.weight_proj, &cross_out);

            b = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
            blt_add(&cur, &cross_out, &b);
        } else {
            b = cur;
        }

        blt_tensor normed1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_forward(&b, w.norm1_weight, &normed1);

        size_t qkv_shape[2] = {n, 3 * E};
        blt_tensor qkv = blt_tensor_create(arena, qkv_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&normed1, w.attn_qkv_w, &qkv);

        // Rotate Q/K, write K/V through to tier 1, collect rotated Q.
        float *kd = (float *)cache->self_k[l].data;
        float *vd = (float *)cache->self_v[l].data;
        const size_t stride3 = 3 * E;

        for (size_t h = 0; h < H; h++) {
            const size_t q_off = h * hd;
            const size_t k_off = E + h * hd;
            const size_t v_off = 2 * E + h * hd;

            blt_strided_copy(backend, q_head, hd, (const float *)qkv.data + q_off, stride3, n, hd);
            blt_strided_copy(backend, k_head, hd, (const float *)qkv.data + k_off, stride3, n, hd);

            blt_tensor q_head_t, q_rot_t, k_head_t, k_rot_t;
            blt_tensor_view_3d(&q_head_t, q_head, n, 1, hd, backend);
            blt_tensor_view_3d(&q_rot_t, q_rot, n, 1, hd, backend);
            blt_tensor_view_3d(&k_head_t, k_head, n, 1, hd, backend);
            blt_tensor_view_3d(&k_rot_t, k_rot, n, 1, hd, backend);
            blt_rope_apply(&q_head_t, &rope_cos, &rope_sin, &q_rot_t);
            blt_rope_apply(&k_head_t, &rope_cos, &rope_sin, &k_rot_t);

            blt_strided_copy(backend, q_full + q_off, E, q_rot, hd, n, hd);
            blt_strided_copy(backend, kd + base * E + q_off, E, k_rot, hd, n, hd);
            blt_strided_copy(backend, vd + base * E + q_off, E, (const float *)qkv.data + v_off, stride3, n, hd);
        }

        zero_tensor(&combined);
        for (size_t h = 0; h < H; h++) {
            blt_attention_head_args ha;
            memset(&ha, 0, sizeof(ha));
            ha.q = q_full + h * hd;
            ha.q_stride = E;
            ha.k = (const float *)cache->self_k[l].data + h * hd;
            ha.k_stride = E;
            ha.v = (const float *)cache->self_v[l].data + h * hd;
            ha.v_stride = E;
            ha.combined = (float *)combined.data;
            ha.combined_stride = E;
            ha.combined_col_offset = h * hd;
            ha.weights_out = NULL;
            ha.scores_scratch = self_scratch;
            ha.mask = (const float *)self_mask.data;
            ha.nq = n;
            ha.nk = total;
            ha.head_dim = hd;
            ha.is_causal = false; // dense mask defines visibility
            ha.scale = self_scale;
            blt_attention_head_core(backend, &ha);
        }

        blt_tensor attn_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul(&combined, w.attn_proj_w, &attn_out);

        blt_tensor resid1 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&b, &attn_out, &resid1);

        // FFN: up/gate -> SwiGLU -> down.
        blt_tensor normed2 = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_rmsnorm_forward(&resid1, w.norm2_weight, &normed2);

        const size_t hidden_dim = w.ffn_up_w->shape[1];
        size_t hidden_shape[2] = {n, hidden_dim};
        blt_tensor up_proj = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&normed2, w.ffn_up_w, &up_proj);
        blt_tensor gate_proj = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_matmul(&normed2, w.ffn_gate_w, &gate_proj);
        blt_tensor activated = blt_tensor_create(arena, hidden_shape, 2, BLT_DTYPE_FP32);
        blt_swiglu_forward(&gate_proj, &up_proj, &activated);
        blt_tensor ffn_out = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_matmul(&activated, w.ffn_down_w, &ffn_out);

        blt_tensor next = blt_tensor_create(arena, shape2, 2, BLT_DTYPE_FP32);
        blt_add(&resid1, &ffn_out, &next);
        cur = next;
    }

    blt_matmul(&cur, blt_local_decoder_lm_head(dec), logits_out);

    cache->self_len = total;
}
