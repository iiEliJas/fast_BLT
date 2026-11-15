// Eval and patching helper functions extracted from train_blt_d.c.
// window_metrics / window_causal_ce: per-window causal and masked metrics for eval.
// window_novel_masked: aligned-novel block metrics for the leak tripwire.
// fixed_stride: fixed-stride patching helper.
// entropy_segment: entropy-LM-based patching for --entropy-patches.

#include "train_eval.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "core/backend.h"
#include "core/tensor.h"
#include "ops/softmax.h"
#include "models/entropy.h"
#include "models/local_encoder.h"
#include "models/global_transformer.h"
#include "models/model_builder.h"
#include "models/local_decoder.h"
#include "models/patcher.h"
#include "models/block_diffusion.h"

// First-max-wins argmax over a 256-way logits row.
static size_t row_argmax_256(const float *row) {
    size_t best = 0;
    for (size_t v = 1; v < 256; v++) {
        if (row[v] > row[best]) best = v;
    }
    return best;
}

// Cross-entropy (nats) of one target byte under the 256-way softmax of row.
static double row_ce_256(const float *row, uint8_t target) {
    float mx = row[0];
    for (size_t v = 1; v < 256; v++) {
        if (row[v] > mx) mx = row[v];
    }
    double sum = 0.0;
    double pt = 0.0;
    for (size_t v = 0; v < 256; v++) {
        const double e = exp((double)row[v] - (double)mx);
        sum += e;
        if (v == (size_t)target) pt = e;
    }
    return -log(fmax(pt / sum, 1e-12));
}

void window_metrics(blt_arena *arena, const blt_model *model, const uint8_t *text, size_t N,
                    const blt_block_batch *batch_or_null, int diffusion, blt_d0_mode d0m, const blt_patch_info *patches,
                    size_t M, blt_window_metrics *out) {
    memset(out, 0, sizeof(*out));
    size_t bshape[1] = {N};
    blt_tensor bytes_in = blt_tensor_create(arena, bshape, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&bytes_in, text, N);

    size_t p_shape[2] = {M, model->config.encoder_config.embed_dim};
    blt_tensor P = blt_tensor_create(arena, p_shape, 2, BLT_DTYPE_FP32);
    size_t h_shape[2] = {N, model->config.encoder_config.embed_dim};
    blt_tensor h = blt_tensor_create(arena, h_shape, 2, BLT_DTYPE_FP32);
    blt_local_encoder_forward(model->encoder, &bytes_in, patches, M, NULL, 0, &P, &h, arena);

    blt_tensor O = blt_tensor_create(arena, p_shape, 2, BLT_DTYPE_FP32);
    blt_global_transformer_forward(model->global, &P, NULL, 0, &O, arena);

    const size_t V = 256;
    double ce_sum = 0.0;
    size_t count = 0;

    if (diffusion && batch_or_null != NULL) {
        size_t S = N + batch_or_null->n_block_rows;
        size_t lg[2] = {S, V};
        blt_tensor logits = blt_tensor_create(arena, lg, 2, BLT_DTYPE_FP32);
        size_t sc[1] = {1};
        blt_tensor loss = blt_tensor_create(arena, sc, 1, BLT_DTYPE_FP32);
        blt_local_decoder_forward_diffusion(model->decoder, &h, &O, patches, M, text, NULL, batch_or_null, d0m, &logits,
                                            &loss, arena);
        float *L = (float *)malloc(logits.numel * sizeof(float));
        BLT_REQUIRE(L != NULL, "window_metrics: logits staging alloc failed");
        blt_tensor_download(&logits, L, logits.numel * sizeof(float));

        // clean rows [0..N-2] predict bytes [1..N-1]
        for (size_t i = 0; i + 1 < N; i++) {
            ce_sum += row_ce_256(L + i * V, text[i + 1]);
            count++;
        }

        // masked-block cell accuracy and CE (the capability block drafting relies on)
        double masked_ce_sum = 0.0;
        for (size_t r = 0; r < batch_or_null->n_block_rows; r++) {
            if (!batch_or_null->cell_masked[r]) continue;
            const float *row = L + (N + r) * V;
            out->masked_cells++;
            if (row_argmax_256(row) == (size_t)batch_or_null->targets[r]) out->masked_hits++;
            masked_ce_sum += row_ce_256(row, batch_or_null->targets[r]);
        }
        if (out->masked_cells > 0) out->masked_ce = masked_ce_sum / (double)out->masked_cells;
        free(L);
    } else {
        size_t lg[2] = {N, V};
        blt_tensor logits = blt_tensor_create(arena, lg, 2, BLT_DTYPE_FP32);
        size_t sc[1] = {1};
        blt_tensor loss = blt_tensor_create(arena, sc, 1, BLT_DTYPE_FP32);
        blt_local_decoder_forward(model->decoder, &h, &O, patches, M, &bytes_in, NULL, NULL, 0, &logits, &loss, arena);
        float *L = (float *)malloc(logits.numel * sizeof(float));
        BLT_REQUIRE(L != NULL, "window_metrics: logits staging alloc failed");
        blt_tensor_download(&logits, L, logits.numel * sizeof(float));
        for (size_t i = 0; i + 1 < N; i++) {
            ce_sum += row_ce_256(L + i * V, text[i + 1]);
            count++;
        }
        free(L);
    }
    BLT_REQUIRE(count > 0, "eval window produced no predictions");
    out->causal_ce = ce_sum / (double)count;
}

double window_causal_ce(blt_arena *arena, const blt_model *model, const uint8_t *text, size_t N,
                        const blt_block_batch *batch_or_null, int diffusion, blt_d0_mode d0m,
                        const blt_patch_info *patches, size_t M, size_t *masked_hits, size_t *masked_total) {
    blt_window_metrics m;
    window_metrics(arena, model, text, N, batch_or_null, diffusion, d0m, patches, M, &m);
    if (masked_hits != NULL) *masked_hits += m.masked_hits;
    if (masked_total != NULL) *masked_total += m.masked_cells;
    return m.causal_ce;
}

int window_novel_masked(blt_arena *scratch, blt_arena *seg_arena, const blt_model *model, blt_entropy_lm *lm,
                        const uint8_t *text, size_t N, const blt_patch_info *patches, size_t M, size_t B,
                        int use_entropy, blt_d0_mode d0m, blt_novel_metrics *out) {
    memset(out, 0, sizeof(*out));

    // The first interior patch start clears the minimum prefix and still leaves
    // room for the block.
    size_t P = 0;
    int found = 0;
    for (size_t j = 1; j + 1 < M; j++) {
        const size_t s = patches[j].start_idx;
        if (s < 8 || s + B > N) continue;
        P = s;
        found = 1;
        break;
    }
    if (!found) return 0;

    blt_patch_info pp[BLT_MAX_PATCHES];
    blt_arena_reset(seg_arena);
    size_t Mp = use_entropy ? entropy_segment(seg_arena, lm, text, P, pp, BLT_MAX_PATCHES) : fixed_stride(P, 4, pp);
    if (Mp < 2 || Mp >= BLT_MAX_PATCHES) return 0;
    // The caller only ever passes a P that lands on a patch start, so the
    // final prefix patch is closed and the last closed latent is Mp-1. (masked_acc
    // handles the mid-patch case too, where it falls back to Mp-2.)
    const size_t group = (Mp >= 1) ? (Mp - 1) : 0;

    blt_arena_reset(scratch);
    size_t bshape[1] = {P};
    blt_tensor bytes_in = blt_tensor_create(scratch, bshape, 1, BLT_DTYPE_UINT8);
    // scratch may be a CUDA arena, so bytes_in.data is device memory; upload
    // does the H2D copy. A raw memcpy would fault.
    blt_tensor_upload(&bytes_in, text, P);
    blt_model_enc_out enc;
    blt_model_encode(model, &bytes_in, pp, Mp, NULL, 0, &enc, scratch);

    blt_block_batch batch;
    memset(&batch, 0, sizeof(batch));
    batch.num_clean = P;
    batch.block_size = B;
    batch.num_blocks = 1;
    batch.n_block_rows = B;
    batch.t = 1.0f;
    batch.loss_scale = 0.0f;
    batch.last_row_scale = 1.0f;
    batch.trailing_closed = 1;
    batch.tokens = (uint32_t *)blt_container_alloc(scratch, B * sizeof(uint32_t));
    batch.positions = (size_t *)blt_container_alloc(scratch, B * sizeof(size_t));
    batch.targets = (uint8_t *)blt_container_alloc(scratch, B);
    batch.cell_valid = (uint8_t *)blt_container_alloc(scratch, B);
    batch.cell_masked = (uint8_t *)blt_container_alloc(scratch, B);
    batch.groups = (size_t *)blt_container_alloc(scratch, B * sizeof(size_t));
    for (size_t r = 0; r < B; r++) {
        batch.positions[r] = P + r;
        batch.targets[r] = text[P + r];
        batch.tokens[r] = BLT_MASK_TOKEN_ID;
        batch.cell_valid[r] = 1;
        batch.cell_masked[r] = 1;
        batch.groups[r] = group;
    }

    size_t S = P + B;
    size_t lg[2] = {S, 256};
    blt_tensor logits = blt_tensor_create(scratch, lg, 2, BLT_DTYPE_FP32);
    blt_local_decoder_forward_diffusion_infer(model->decoder, &enc.byte_hidden_out, &enc.global_out, pp, Mp, &batch,
                                              d0m, &logits, scratch);
    float *L = (float *)malloc(logits.numel * sizeof(float));
    if (!L) return 0;
    blt_tensor_download(&logits, L, logits.numel * sizeof(float));
    double ce_sum = 0.0;
    for (size_t r = 0; r < B; r++) {
        const float *row = L + (P + r) * 256;
        out->cells++;
        if (row_argmax_256(row) == (size_t)batch.targets[r]) out->hits++;
        ce_sum += row_ce_256(row, batch.targets[r]);
    }
    if (out->cells > 0) out->ce = ce_sum / (double)out->cells;
    free(L);
    return 1;
}

size_t fixed_stride(size_t seq_len, size_t patch_len, blt_patch_info *out) {
    size_t np = 0, start = 0;
    while (start < seq_len) {
        size_t len = (start + patch_len <= seq_len) ? patch_len : (seq_len - start);
        out[np].start_idx = start;
        out[np].length = len;
        out[np].peak_entropy = 0.0f;
        np++;
        start += len;
    }
    return np;
}

size_t entropy_segment_ex(blt_arena *arena, blt_entropy_lm *lm, const uint8_t *bytes, size_t len, blt_patch_info *out,
                          size_t max_patches, float threshold_global) {
    size_t shape1[1] = {len};
    blt_tensor bytes_in = blt_tensor_create(arena, shape1, 1, BLT_DTYPE_UINT8);
    blt_tensor_upload(&bytes_in, bytes, len);

    size_t logits_shape[2] = {len, 256};
    blt_tensor logits = blt_tensor_create(arena, logits_shape, 2, BLT_DTYPE_FP32);
    size_t scalar_shape[1] = {1};
    blt_tensor discard = blt_tensor_create(arena, scalar_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_lm_forward(lm, &bytes_in, &logits, &discard, arena);

    size_t probs_shape[2] = {len, 256};
    blt_tensor probs = blt_tensor_create(arena, probs_shape, 2, BLT_DTYPE_FP32);
    blt_softmax(&logits, &probs);

    size_t vals_shape[1] = {len};
    blt_tensor vals = blt_tensor_create(arena, vals_shape, 1, BLT_DTYPE_FP32);
    blt_entropy_config ecfg = {.vocab_size = 256, .use_log2 = false};
    blt_compute_entropy(&probs, &vals, &ecfg);

    blt_patcher_config pcfg;
    blt_make_patcher_cfg(&pcfg, 0, threshold_global, 1.0f, 16);

    if (vals.backend == BLT_BACKEND_CPU) {
        return blt_segment_patches(&vals, bytes, out, max_patches, &pcfg);
    }
    float *vals_host = (float *)malloc(len * sizeof(float));
    BLT_REQUIRE(vals_host != NULL, "entropy_segment_ex: staging alloc failed");
    blt_tensor_download(&vals, vals_host, len * sizeof(float));
    blt_tensor vals_view;
    view_1d(&vals_view, vals_host, len, BLT_DTYPE_FP32, BLT_BACKEND_CPU);
    const size_t n = blt_segment_patches(&vals_view, bytes, out, max_patches, &pcfg);
    free(vals_host);
    return n;
}

size_t entropy_segment(blt_arena *arena, blt_entropy_lm *lm, const uint8_t *bytes, size_t len, blt_patch_info *out,
                       size_t max_patches) {
    return entropy_segment_ex(arena, lm, bytes, len, out, max_patches, 3.0f);
}

void topk_from_logits(const float *row, int n, int k, int *idx_out) {
    if (!idx_out || k <= 0) return;
    for (int i = 0; i < k; i++) idx_out[i] = -1;
    if (!row || n <= 0) return;

    const int m = (k < n) ? k : n;
    for (int rank = 0; rank < m; rank++) {
        int best = -1;
        for (int v = 0; v < n; v++) {
            int taken = 0;
            for (int t = 0; t < rank; t++) {
                if (idx_out[t] == v) {
                    taken = 1;
                    break;
                }
            }
            if (taken) continue;
            if (best < 0 || row[v] > row[best]) best = v;
        }
        idx_out[rank] = best;
    }
}
