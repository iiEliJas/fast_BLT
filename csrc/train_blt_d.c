// BLT-D training launcher
//
// Trains a full BLT-D pipeline (local encoder -> global transformer ->
// diffusion decoder) on a raw byte corpus with fixed-stride patches and
// per-window freshly sampled corruption.
//
// Usage:
//   bin/train_blt_d --corpus FILE [options]
// Options:
//   --steps N          optimizer steps (default 2000)
//   --lr F             learning rate (default 0.05)
//   --block-size B     diffusion block size B (default 4)
//   --window W         clean-sequence length N per example (default 48)
//   --embed E          model width (default 64)
//   --hidden H         FFN width (default 128)
//   --layers L         layers per submodule (default 2)
//   --d0-mode MODE     zeros | learned (default learned)
//   --seed S           RNG seed (default 7)
//   --report-every K   print every K steps (default 25)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/platform.h"

#ifdef BLT_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "core/allocator.h"
#include "core/backend.h"
#include "models/model.h"
#include "models/checkpoint.h"
#include "models/entropy_lm.h"
#include "models/patcher.h"
#include "models/model_builder.h"
#include "models/entropy.h"
#include "ops/softmax.h"
#include "models/local_encoder.h"
#include "models/global_transformer.h"
#include "models/transformer_stack.h"
#include "models/transformer.h"
#include "models/local_common.h"
#include "models/block_diffusion.h"
#include "ops/optim.h"
#include "ops/vecmath.h"
#include "core/cuda_shim.h"
#include "models/param_visitor.h"
#include "train_args.h"
#include "train_optim.h"
#include "train_diag.h"
#include "train_eval.h"
#include "train_schedule.h"
#include "core/flops.h"
#include "core/device_info.h"

// Substitutes the first %d in path with step when the path is a template;
// otherwise copies path verbatim into buf.
static void blt_format_save_path(char *buf, size_t buflen, const char *path, size_t step) {
    const char *tag = strstr(path, "%d");
    if (!tag) {
        snprintf(buf, buflen, "%s", path);
        return;
    }
    snprintf(buf, buflen, "%.*s%d%s", (int)(tag - path), path, (int)step, tag + 2);
}

//----------------------------------------------------------------------
// Gradient clipping + SGD/AdamW over every parameter in the model

// splitmix64 stream for the trainer-side timestep draw (host-side so
// curriculum schedules can reshape t before it reaches masking).
static uint64_t t_rng_next(uint64_t *state) {
    *state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Training-time leak tripwire. On held-out windows at t=1.0
// it measures the two numbers that a reintroduced label leak would break:
// train-layout masked accuracy (a block row able to read the clean twin at its
// own target drives this toward 1.0) and aligned-novel masked accuracy (a
// prefix ending exactly at a patch start, with no clean twin to copy from).
// A healthy leak-free run keeps them close. It also reports L_clean and L_mask
// so the diffusion objective can be read off the log.
typedef struct {
    double l_clean;
    double l_mask;
    // Per-masked-token CE, normalised by the masked-cell count and converted
    // to bits/byte so it is directly comparable to causal_bpb. The raw l_mask
    // above is a sum over masked cells scaled by 1/t, so it tracks t and the
    // per-window patch count rather than model quality.
    double clean_bits;
    double mask_bits;
    size_t train_hits, train_total;
    size_t novel_hits, novel_total;
    size_t windows;
} blt_monitor_result;

// One aligned-novel block: prefix bytes [0, P) re-segmented on its own, one
// fully masked block at positions P..P+B-1, cross-attention to the last closed
// latent. Mirrors inference, so there is no clean row to copy from.
static void monitor_novel_block(const blt_model *model, blt_arena *scratch, blt_arena *seg_arena, blt_entropy_lm *lm,
                                const uint8_t *text, size_t P, size_t B, int use_entropy, blt_d0_mode d0m, size_t *hits,
                                size_t *total) {
    if (P < 8) return;
    blt_patch_info pp[BLT_MAX_PATCHES];
    blt_arena_reset(seg_arena);
    size_t Mp = use_entropy ? entropy_segment(seg_arena, lm, text, P, pp, BLT_MAX_PATCHES) : fixed_stride(P, 4, pp);
    if (Mp < 2 || Mp >= BLT_MAX_PATCHES) return;
    // The caller only ever passes a P that lands on a patch start, so the
    // final prefix patch is closed and the last closed latent is Mp-1. (masked_acc
    // handles the mid-patch case too, where it falls back to Mp-2.)
    const size_t group = (Mp >= 1) ? (Mp - 1) : 0;

    blt_arena_reset(scratch);
    size_t bshape[1] = {P};
    blt_tensor bytes_in = blt_tensor_create(scratch, bshape, 1, BLT_DTYPE_UINT8);
    // scratch is a CUDA arena during training, so bytes_in.data is device
    // memory; upload does the H2D copy. A raw memcpy would fault.
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
    if (!L) return;
    blt_tensor_download(&logits, L, logits.numel * sizeof(float));
    for (size_t r = 0; r < B; r++) {
        const float *row = L + (P + r) * 256;
        size_t best = 0;
        for (size_t v = 1; v < 256; v++)
            if (row[v] > row[best]) best = v;
        (*total)++;
        if (best == (size_t)batch.targets[r]) (*hits)++;
    }
    free(L);
}

static void blt_monitor(const blt_model *model, blt_arena *scratch, blt_arena *seg_arena, blt_entropy_lm *lm,
                        const uint8_t *bytes, size_t data_len, const args_t *a, size_t step, blt_monitor_result *out) {
    memset(out, 0, sizeof(*out));
    const int use_entropy = (lm != NULL);
    const size_t B = a->block_size;
    size_t wins = a->monitor_windows;
    const size_t max_wins = (data_len > a->window + B) ? (data_len - a->window - B) / a->window : 0;
    if (wins > max_wins) wins = max_wins;
    if (wins == 0) return;

    double lcl = 0.0, lmk = 0.0;
    double mask_ce_sum = 0.0;
    size_t mask_cells = 0;
    for (size_t wi = 0; wi < wins; wi++) {
        const uint8_t *text = bytes + wi * a->window;
        const size_t N = a->window;
        blt_patch_info patches[BLT_MAX_PATCHES];
        blt_arena_reset(seg_arena);
        size_t M = use_entropy ? entropy_segment(seg_arena, lm, text, N, patches, BLT_MAX_PATCHES)
                               : fixed_stride(N, 4, patches);
        if (M < 2 || M >= BLT_MAX_PATCHES) continue;
        if (N + B * (M - 1) > model->config.decoder_config.max_seq_len) continue;

        blt_arena_reset(scratch);
        size_t bshape[1] = {N};
        blt_tensor bytes_in = blt_tensor_create_uninit(scratch, bshape, 1, BLT_DTYPE_UINT8);
        blt_tensor_upload(&bytes_in, text, N);
        blt_model_enc_out enc;
        blt_model_encode(model, &bytes_in, patches, M, NULL, 0, &enc, scratch);

        blt_block_batch batch;
        blt_block_batch_build_t(&batch, scratch, text, N, patches, M, B, (uint64_t)step + wi, 1.0f);
        size_t S = N + batch.n_block_rows;
        size_t lg[2] = {S, 256};
        blt_tensor logits = blt_tensor_create(scratch, lg, 2, BLT_DTYPE_FP32);
        size_t sc[1] = {1};
        blt_tensor loss = blt_tensor_create(scratch, sc, 1, BLT_DTYPE_FP32);
        blt_local_decoder_forward_diffusion(model->decoder, &enc.byte_hidden_out, &enc.global_out, patches, M, text,
                                            NULL, &batch, a->d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, &logits, &loss,
                                            scratch);
        float *L = (float *)malloc(logits.numel * sizeof(float));
        if (!L) continue;
        blt_tensor_download(&logits, L, logits.numel * sizeof(float));

        double cl = 0.0;
        for (size_t i = 0; i + 1 < N; i++) {
            const float *row = L + i * 256;
            float mx = row[0];
            for (size_t v = 1; v < 256; v++)
                if (row[v] > mx) mx = row[v];
            double sum = 0.0, pt = 0.0;
            for (size_t v = 0; v < 256; v++) {
                const double e = exp((double)row[v] - (double)mx);
                sum += e;
                if (v == (size_t)text[i + 1]) pt = e;
            }
            cl += -log(fmax(pt / sum, 1e-9));
        }
        lcl += cl / (double)(N - 1);

        double mk = 0.0;
        for (size_t r = 0; r < batch.n_block_rows; r++) {
            if (!batch.cell_masked[r]) continue;
            const float *row = L + (N + r) * 256;
            size_t best = 0;
            for (size_t v = 1; v < 256; v++)
                if (row[v] > row[best]) best = v;
            out->train_total++;
            if (best == (size_t)batch.targets[r]) out->train_hits++;
            float mx = row[0];
            for (size_t v = 1; v < 256; v++)
                if (row[v] > mx) mx = row[v];
            double sum = 0.0, pt = 0.0;
            for (size_t v = 0; v < 256; v++) {
                const double e = exp((double)row[v] - (double)mx);
                sum += e;
                if (v == (size_t)batch.targets[r]) pt = e;
            }
            mk += -log(fmax(pt / sum, 1e-9));
        }
        if (batch.n_block_rows) lmk += mk / (double)batch.n_block_rows;
        mask_ce_sum += mk;
        for (size_t r = 0; r < batch.n_block_rows; r++) {
            if (batch.cell_masked[r]) mask_cells++;
        }
        free(L);

        // Aligned novel: prefix ends at an interior patch start, so every prefix
        // patch is closed and the block has no clean twin. Take the first start
        // that clears the minimum prefix and still leaves room for the block.
        for (size_t j = 1; j + 1 < M; j++) {
            const size_t P = patches[j].start_idx;
            if (P < 8 || P + B > N) continue;
            monitor_novel_block(model, scratch, seg_arena, lm, text, P, B, use_entropy,
                                a->d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, &out->novel_hits, &out->novel_total);
            break;
        }
        out->windows++;
    }
    if (out->windows) {
        out->l_clean = lcl / (double)out->windows;
        out->l_mask = lmk / (double)out->windows;
    }
    const double ln2 = 0.6931471805599453;
    out->clean_bits = out->l_clean / ln2;
    out->mask_bits = mask_cells ? (mask_ce_sum / (double)mask_cells) / ln2 : 0.0;
}

// AdamW keeps two float32 moments per parameter, so its state arena has to be
// sized from the model rather than fixed: a 32 MB constant is fine for the toy
// configs and silently overflows on any real-sized model.
static void count_params_fn(float *tensor, const blt_param_info *info, void *ctx) {
    (void)tensor;
    (void)info;
    size_t *n = (size_t *)ctx;
    *n += info->numel;
}

static size_t blt_count_params(const blt_model *model) {
    size_t n = 0;
    blt_model_visit_params(model, NULL, count_params_fn, &n, 0);
    return n;
}

// Weight bytes (fp32 storage) used by the training MBU model. Optimizer state
// is counted separately by the per-step multiplier, not here.
static void param_bytes_fn(float *tensor, const blt_param_info *info, void *ctx) {
    (void)tensor;
    size_t *n = (size_t *)ctx;
    *n += info->numel * sizeof(float);
}

static size_t blt_param_bytes(const blt_model *model) {
    size_t n = 0;
    blt_model_visit_params(model, NULL, param_bytes_fn, &n, 0);
    return n;
}

int main(int argc, char **argv) {
    args_t a = parse_args(argc, argv);
    if (a.grad_accum < 1) {
        fprintf(stderr, "--grad-accum must be >= 1\n");
        return 1;
    }

    // Load corpus
    FILE *fp = fopen(a.corpus_path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open corpus '%s'\n", a.corpus_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *corpus = (uint8_t *)malloc((size_t)fsize);
    if (!corpus || fread(corpus, 1, (size_t)fsize, fp) != (size_t)fsize) {
        fprintf(stderr, "failed to read corpus\n");
        return 1;
    }
    fclose(fp);

    // Diffusion scores every patch in the window at once, so the decoder sees
    // S = window + num_patches * block_size rows, not just window + block_size.
    // num_patches is bounded by the window (a patch is at least one byte), hence
    // the (1 + block_size) factor. The old fixed 1024 happened to cover the
    // 512-byte window; deriving it from the window extends that to longer runs.
    const size_t BLT_WINDOW_MAX = 2048;
    size_t MS = a.window * (1 + a.block_size) + 64;
    if (MS < 1024) MS = 1024;
    if (a.window < 8 || a.window > BLT_WINDOW_MAX) {
        fprintf(stderr, "--window must be in [8, %zu]\n", BLT_WINDOW_MAX);
        return 1;
    }
#if !defined(BLT_WITH_CUDA)
    if (a.use_cuda) {
        fprintf(stderr, "--backend cuda requires a CUDA=1 build\n");
        return 1;
    }
#endif
    // The standalone entropy-LM trainer stays CPU-only: it is a tiny
    // side model whose save path and patcher interplay are host-bound,
    // and it never benefits from the device at this size.
    if (a.train_entlm && a.use_cuda) {
        printf("[TRAIN] note: --train-entropy-lm runs on CPU regardless of "
               "--backend\n");
        a.use_cuda = 0;
    }
    // Resolve the requested matmul dtype once: --fp8-format only refines an
    // FP8 request, and every later use reads mixed_dtype.
    const blt_dtype mixed_dtype =
        (a.mixed_dtype == BLT_DTYPE_FP8_E4M3 && a.fp8_e5m2) ? BLT_DTYPE_FP8_E5M2 : a.mixed_dtype;
    if (mixed_dtype != BLT_DTYPE_FP32 && !a.use_cuda) {
        fprintf(stderr, "train_blt_d: --dtype %s requires --backend cuda (the CPU path has no low-precision GEMM)\n",
                blt_dtype_name(mixed_dtype));
        return 1;
    }
    if (blt_dtype_is_fp8(mixed_dtype) && !blt_cuda_supports_fp8()) {
        fprintf(stderr, "train_blt_d: --dtype %s requires a device with compute capability >= 8.9 (Ada or newer)\n",
                blt_dtype_name(mixed_dtype));
        return 1;
    }
    if (mixed_dtype == BLT_DTYPE_BF16 && !blt_cuda_supports_bf16()) {
        fprintf(stderr, "train_blt_d: --dtype bf16 requires a device with compute capability >= 8.0\n");
        return 1;
    }
    if (mixed_dtype == BLT_DTYPE_FP16 && !blt_cuda_supports_fp16()) {
        fprintf(stderr, "train_blt_d: --dtype fp16 requires a device with compute capability >= 7.0\n");
        return 1;
    }
    const blt_backend dev = a.use_cuda ? BLT_BACKEND_CUDA : BLT_BACKEND_CPU;
    // Resolve reference peaks once at startup: explicit flags win, then the
    // table entry for the running card, then the run's math dtype column.
#ifdef BLT_WITH_CUDA
    char device_name_buf[256] = {0};
#endif
    const char *device_name = NULL;
    const double tflops_override = (double)a.peak_tflops;
    const double bw_override = (double)a.peak_bandwidth;
    const blt_peak_dtype peak_dtype =
        blt_dtype_is_fp8(mixed_dtype) ? BLT_PEAK_FP8 : (mixed_dtype == BLT_DTYPE_FP32 ? BLT_PEAK_FP32 : BLT_PEAK_BF16);
#ifdef BLT_WITH_CUDA
    if (a.use_cuda) {
        struct cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            snprintf(device_name_buf, sizeof(device_name_buf), "%s", prop.name);
            device_name = device_name_buf;
        } else {
            fprintf(stderr,
                    "warning: cudaGetDeviceProperties failed; pass --peak-tflops/--peak-bandwidth to report MFU/MBU\n");
        }
    }
#endif
    blt_device_peaks peaks = blt_device_resolve(device_name, peak_dtype, tflops_override, bw_override);
    a.peak_tflops = (float)peaks.tflops;
    a.peak_bandwidth = (float)peaks.bandwidth_gbps;
    if (device_name && peaks.spec == NULL) {
        fprintf(stderr, "warning: unknown device '%s'; %s\n", device_name,
                (peaks.tflops > 0.0 && peaks.bandwidth_gbps > 0.0)
                    ? "using explicit --peak-tflops/--peak-bandwidth overrides"
                    : "pass --peak-tflops/--peak-bandwidth to report MFU/MBU");
    }
    const char *const peak_dtype_name =
        blt_dtype_is_fp8(mixed_dtype) ? "fp8" : (mixed_dtype == BLT_DTYPE_FP32 ? "fp32" : "bf16");
    const char *const dtype_name = blt_dtype_name(mixed_dtype);
    g_blt_deterministic = a.deterministic;
#ifdef BLT_WITH_CUDA
    if (a.use_cuda && a.cuda_scratch_mb > 0) blt_cuda_set_scratch_size(a.cuda_scratch_mb * 1024ULL * 1024);
#endif

    blt_arena *model_arena = blt_arena_create(a.use_cuda ? 1024ULL * 1024 * 1024 : 64 * 1024 * 1024, dev);
    // Activation and segmentation arenas scale with the window. At the original
    // 512-byte window these are exactly the old constants, so that path is
    // unchanged; a 2048 window needs roughly 4x the scratch and segmentation
    // staging because the byte and patch tensors all grow with it.
    // Scratch grows faster than the window because diffusion scores every patch
    // at once: the decoder sees S = window * (1 + block_size) rows, and the
    // attention scores over S are quadratic in S. Measured on the TinyStories
    // 10M shape: window 512 needs 512 MB, window 2048 needs 4096 MB, and that
    // is ~96% of an 8 GB card. Scaling linearly under-provisioned it and
    // overflowed two steps in, which is what this fixes.
    const size_t arena_scale = (a.window + 511) / 512;
    size_t scratch_mb = a.scratch_mb;
    if (scratch_mb == 0) {
        scratch_mb = 512 * arena_scale * arena_scale / 2;
        if (scratch_mb < 512) scratch_mb = 512;
    }
    blt_arena *scratch = blt_arena_create(scratch_mb * 1024ULL * 1024, dev);
    if (a.use_cuda) {
        fprintf(stderr, "[TRAIN] scratch arena %zu MB (window %zu), cuda kernel scratch %zu MB\n", scratch_mb, a.window,
                a.cuda_scratch_mb);
    }
    blt_arena *adamw_state_arena = NULL;
    // sized after the model is built, below
    // Host arena for the segmentation LM + patcher (host-only by design),
    // valid in both backends. Segmentation buffers get their own resettable
    // arena so per-step calls never overwrite the LM weights.
    blt_arena *host_lm_arena = blt_arena_create(32 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *host_seg_arena = blt_arena_create(16ULL * 1024 * 1024 * arena_scale, BLT_BACKEND_CPU);

    // Model config (paper-ish defaults; ngram tables as in the repo baseline).
    // Per-submodule layer counts default to --layers; the decoder can be
    // scaled independently for depth sweeps (paper: decoder scaling matters
    // most for BLT-D/DV).
    if (a.enc_layers == 0) a.enc_layers = a.layers;
    if (a.glob_layers == 0) a.glob_layers = a.layers;
    if (a.dec_layers == 0) a.dec_layers = a.layers;

    blt_model_config cfg;
    blt_model_config_defaults(&cfg, a.embed, a.hidden, a.enc_layers, a.glob_layers, a.dec_layers, MS, a.cross_last);
    cfg.encoder_config.mixed_dtype = mixed_dtype;
    cfg.decoder_config.mixed_dtype = mixed_dtype;
    cfg.global_config.mixed_dtype = mixed_dtype;

    blt_model *model = blt_model_create(model_arena, &cfg);

    if (a.optimizer == 1) {
        // Two float32 AdamW moments per parameter, plus 1 MB of headroom. A
        // fixed 32 MB constant fits the toy configs and overflows on any
        // real-sized model.
        const size_t n_params = blt_count_params(model);
        const size_t adamw_bytes = 2 * n_params * sizeof(float) + (1u << 20);
        adamw_state_arena = blt_arena_create(adamw_bytes, dev);
    }
    if (a.load_path) {
        blt_model_load(model, a.load_path);
        printf("[CKPT] loaded weights from %s\n", a.load_path);
    }
    // Create and validate the checkpoint destinations now. Waiting for the first
    // periodic save meant a missing directory killed the run at step
    // --save-every, discarding everything trained so far.
    const char *save_paths[4];
    save_paths[0] = a.save_path;
    save_paths[1] = a.save_optim_path;
    save_paths[2] = a.grad_norm_log;
    save_paths[3] = a.update_norm_log;
    for (size_t i = 0; i < 4; i++) {
        if (save_paths[i]) {
            BLT_REQUIRE(train_ensure_parent_dir(save_paths[i]) == 0, "train_blt_d: unusable output path %s",
                        save_paths[i]);
        }
    }
    const char *more_paths[4];
    more_paths[0] = a.component_norm_log;
    more_paths[1] = a.activation_dump_log;
    more_paths[2] = a.load_path;
    more_paths[3] = a.load_optim_path;
    for (size_t i = 0; i < 4; i++) {
        if (more_paths[i]) {
            BLT_REQUIRE(train_ensure_parent_dir(more_paths[i]) == 0, "train_blt_d: unusable path %s", more_paths[i]);
        }
    }

    blt_model_grad *grad = blt_model_grad_create(model_arena, model);
    // The accumulation buffer doubles gradient memory; only the multi-micro
    // path needs it. Preflight the arena so an oversized second buffer names
    // the flag and the shortfall instead of failing inside the allocator.
    blt_model_grad *grad_accum = NULL;
    if (a.grad_accum > 1) {
        const size_t grad_bytes = blt_param_bytes(model);
        const size_t free_bytes = model_arena->capacity - model_arena->offset;
        BLT_REQUIRE(free_bytes >= grad_bytes,
                    "train_blt_d: --grad-accum %zu needs a second gradient buffer (%.1f MB) but only "
                    "%.1f MB remains in the model arena; raise --model-mb",
                    a.grad_accum, (double)grad_bytes / (1024.0 * 1024.0), (double)free_bytes / (1024.0 * 1024.0));
        grad_accum = blt_model_grad_create(model_arena, model);
    }
    blt_model_grad *step_grad = (a.grad_accum > 1) ? grad_accum : grad;
    blt_entropy_lm *train_lm = a.entropy_patches ? blt_make_entropy_lm(host_lm_arena, MS, NULL, 11) : NULL;
    if (train_lm && a.entropy_lm) {
        blt_entropy_lm_load(train_lm, a.entropy_lm);
        printf("[TRAIN] entropy-patch segmentation (trained LM: %s)\n", a.entropy_lm);
    } else if (train_lm) {
        printf("[TRAIN] entropy-patch segmentation (random LM)\n");
    }

    // Standalone entropy-LM training mode: plain byte CE on corpus
    // windows, SGD+clip, then save. Skips the main model entirely.
    if (a.train_entlm) {
        const size_t num_windows_ = (size_t)fsize / a.window;
        blt_entropy_lm *lm0 = blt_make_entropy_lm(model_arena, MS, NULL, 11);
        if (a.entropy_lm) blt_entropy_lm_load(lm0, a.entropy_lm);
        blt_entropy_lm_grad *lm_grad = blt_entropy_lm_grad_create(model_arena, lm0);
        printf("[ENTLM] training entropy LM -> %s\n", a.train_entlm);
        blt_tensor *lm_ts[9];
        blt_tensor *lm_gs[9];
        {
            blt_transformer_layer_storage *l = &lm0->stack.layer_storage[0];
            blt_transformer_layer_grad *g = &lm_grad->stack_grad->layer_grads[0];
            blt_tensor *wtmp[] = {&l->norm1_weight, &l->attn_qkv_w, &l->attn_proj_w, &l->norm2_weight,
                                  &l->ffn_up_w,     &l->ffn_gate_w, &l->ffn_down_w};
            blt_tensor *gtmp[] = {&g->norm1_weight, &g->attn_qkv_w, &g->attn_proj_w, &g->norm2_weight,
                                  &g->ffn_up_w,     &g->ffn_gate_w, &g->ffn_down_w};
            lm_ts[2] = wtmp[0];
            lm_gs[2] = gtmp[0];
            lm_ts[3] = wtmp[1];
            lm_gs[3] = gtmp[1];
            lm_ts[4] = wtmp[2];
            lm_gs[4] = gtmp[2];
            lm_ts[5] = wtmp[3];
            lm_gs[5] = gtmp[3];
            lm_ts[6] = wtmp[4];
            lm_gs[6] = gtmp[4];
            lm_ts[7] = wtmp[5];
            lm_gs[7] = gtmp[5];
            lm_ts[8] = wtmp[6];
            lm_gs[8] = gtmp[6];
        }
        lm_ts[0] = &lm0->embedding_weight;
        lm_gs[0] = &lm_grad->embedding_grad;
        lm_ts[1] = &lm0->lm_head_weight;
        lm_gs[1] = &lm_grad->lm_head_grad;

        for (size_t step = 0; step < a.steps; step++) {
            blt_arena_reset(scratch);
            const size_t w = step % num_windows_;
            const uint8_t *text = corpus + w * a.window;
            const size_t N = a.window;
            size_t sh1[1] = {N};
            blt_tensor bytes_in = blt_tensor_create(scratch, sh1, 1, BLT_DTYPE_UINT8);
            memcpy(bytes_in.data, text, N);
            size_t sh2[2] = {N, 256};
            blt_tensor logits = blt_tensor_create(scratch, sh2, 2, BLT_DTYPE_FP32);
            size_t shs[1] = {1};
            blt_tensor loss = blt_tensor_create(scratch, shs, 1, BLT_DTYPE_FP32);
            blt_entropy_lm_forward(lm0, &bytes_in, &logits, &loss, scratch);
            for (size_t ti = 0; ti < 9; ti++) zero_tensor(lm_gs[ti]);
            blt_entropy_lm_backward(lm0, &bytes_in, lm_grad, scratch);
            // global grad clip
            float sq = 0.0f;
            for (size_t ti = 0; ti < 9; ti++) {
                const float *d = (const float *)lm_gs[ti]->data;
                for (size_t j = 0; j < lm_gs[ti]->numel; j++) sq += d[j] * d[j];
            }
            const float nrm = sqrtf(sq);
            if (nrm > 5.0f && nrm > 0.0f) {
                const float scl = 5.0f / nrm;
                for (size_t ti = 0; ti < 9; ti++) blt_scale(lm_gs[ti], scl);
            }
            for (size_t ti = 0; ti < 9; ti++) blt_sgd_step(lm_ts[ti], lm_gs[ti], 0.1f);
            if (a.report_every && step % a.report_every == 0) {
                printf("[ENTLM] step %zu/%zu loss %.4f\n", step, a.steps, ((const float *)loss.data)[0]);
                fflush(stdout);
            }
        }
        train_ensure_parent_dir(a.train_entlm);
        blt_entropy_lm_save(lm0, a.train_entlm);
        printf("[ENTLM] saved %s\n", a.train_entlm);
        fflush(stdout);
        return 0;
    }

    // Deterministic random init (libc-independent LCG). Skipped entirely
    // when --load-weights supplied the full parameter set. Values are drawn
    // in a fixed order; under a device backend they are generated host-side
    // and uploaded per tensor.
    uint64_t rng_state = a.seed ? a.seed : 1;
    const int need_init = (a.load_path == NULL);
    if (need_init) {
#define RAND01() ((float)((rng_state += 0x9E3779B97F4A7C15ULL) >> 40) / (float)(1u << 24))
#define FILL_RAND(t, s)                                                                                                \
    do {                                                                                                               \
        blt_tensor *t_ = &(t);                                                                                         \
        const size_t n_ = t_->numel;                                                                                   \
        float *d_ = (float *)(t_->backend == BLT_BACKEND_CPU ? t_->data : malloc(n_ * sizeof(float)));                 \
        BLT_REQUIRE(d_ != NULL, "init: staging alloc failed");                                                         \
        for (size_t i_ = 0; i_ < n_; i_++) d_[i_] = (RAND01() * 2.0f - 1.0f) * (s);                                    \
        if (t_->backend != BLT_BACKEND_CPU) {                                                                          \
            blt_tensor_upload(t_, d_, n_ * sizeof(float));                                                             \
            free(d_);                                                                                                  \
        }                                                                                                              \
    } while (0)
#define FILL_ONES(t)                                                                                                   \
    do {                                                                                                               \
        blt_tensor *t_ = &(t);                                                                                         \
        const size_t n_ = t_->numel;                                                                                   \
        float *d_ = (float *)(t_->backend == BLT_BACKEND_CPU ? t_->data : malloc(n_ * sizeof(float)));                 \
        BLT_REQUIRE(d_ != NULL, "init: staging alloc failed");                                                         \
        for (size_t i2_ = 0; i2_ < n_; i2_++) d_[i2_] = 1.0f;                                                          \
        if (t_->backend != BLT_BACKEND_CPU) {                                                                          \
            blt_tensor_upload(t_, d_, n_ * sizeof(float));                                                             \
            free(d_);                                                                                                  \
        }                                                                                                              \
    } while (0)

        FILL_RAND(model->encoder->byte_embedding_weight, 0.1f);
        for (size_t l = 0; l < cfg.encoder_config.num_layers; l++) {
            blt_local_layer_storage *ly = &model->encoder->layers[l];
            blt_tensor *ws[12] = {&ly->norm1_weight,   &ly->attn_qkv_w,        &ly->attn_proj_w,
                                  &ly->norm2_weight,   &ly->ffn_up_w,          &ly->ffn_gate_w,
                                  &ly->ffn_down_w,     &ly->cross_norm_weight, &ly->cross_weight_q,
                                  &ly->cross_weight_k, &ly->cross_weight_v,    &ly->cross_weight_proj};
            for (size_t j = 0; j < 12; j++) {
                const int is_norm = (j == 0 || j == 3 || j == 7);
                if (is_norm) FILL_ONES(*ws[j]);
                else FILL_RAND(*ws[j], 0.1f);
            }
        }
        for (size_t l = 0; l < cfg.global_config.num_layers; l++) {
            blt_transformer_layer_storage *ly = &model->global->stack.layer_storage[l];
            FILL_ONES(ly->norm1_weight);
            FILL_ONES(ly->norm2_weight);
            FILL_RAND(ly->attn_qkv_w, 0.1f);
            FILL_RAND(ly->attn_proj_w, 0.1f);
            FILL_RAND(ly->ffn_up_w, 0.1f);
            FILL_RAND(ly->ffn_gate_w, 0.1f);
            FILL_RAND(ly->ffn_down_w, 0.1f);
        }
        for (size_t l = 0; l < cfg.decoder_config.num_layers; l++) {
            blt_local_layer_storage *ly = &model->decoder->layers[l];
            blt_tensor *ws[12] = {&ly->norm1_weight,   &ly->attn_qkv_w,        &ly->attn_proj_w,
                                  &ly->norm2_weight,   &ly->ffn_up_w,          &ly->ffn_gate_w,
                                  &ly->ffn_down_w,     &ly->cross_norm_weight, &ly->cross_weight_q,
                                  &ly->cross_weight_k, &ly->cross_weight_v,    &ly->cross_weight_proj};
            for (size_t j = 0; j < 12; j++) {
                const int is_norm = (j == 0 || j == 3 || j == 7);
                if (is_norm) FILL_ONES(*ws[j]);
                else FILL_RAND(*ws[j], 0.1f);
            }
        }
        FILL_RAND(model->decoder->lm_head_weight, 0.1f);
        FILL_RAND(model->decoder->d0_embed_weight, 0.05f);
#undef FILL_ONES
#undef FILL_RAND
#undef RAND01
    }

    blt_model_refresh_mixed(model);

    const size_t n_params = blt_count_params(model);
    printf("train_blt_d: mode=%s corpus=%s (%ld bytes) steps=%zu grad_accum=%zu lr=%.4g B=%zu W=%zu "
           "embed=%zu hidden=%zu enc/glob/dec layers=%zu/%zu/%zu xattn=%s backend=%s dtype=%s d0=%s"
           " optimizer=%s params=%zu (%.3fM)\n",
           a.diffusion ? "BLT-D" : "PLAIN-BLT", a.corpus_path, fsize, a.steps, a.grad_accum, (double)a.lr, a.block_size,
           a.window, a.embed, a.hidden, a.enc_layers, a.glob_layers, a.dec_layers, a.cross_last ? "last" : "all",
           a.use_cuda ? "cuda" : "cpu", dtype_name, a.d0_learned ? "learned" : "zeros", a.optimizer ? "adamw" : "sgd",
           n_params, (double)n_params / 1e6);

    // Telemetry: name the matched table entry and the source of each peak so a
    // disabled MFU/MBU is never confused with a measured 0 TFLOP/s.
    printf("[TRAIN] device=%s peak_tflops=", device_name ? device_name : "unknown");
    if (peaks.tflops > 0.0) printf("%.2f (%s)", peaks.tflops, tflops_override > 0.0 ? "override" : peak_dtype_name);
    else printf("n/a");
    printf(" peak_bw=");
    if (peaks.bandwidth_gbps > 0.0)
        printf("%.1f GB/s (%s)", peaks.bandwidth_gbps, bw_override > 0.0 ? "override" : "hbm");
    else printf("n/a");
    if (peaks.tflops <= 0.0 || peaks.bandwidth_gbps <= 0.0)
        printf(" (pass --peak-tflops/--peak-bandwidth for MFU/MBU)");
    printf("\n");

    // Training loop: non-overlapping windows over the corpus, fixed-stride
    // patches, freshly sampled corruption per visit.

    size_t num_windows = (size_t)(fsize - a.window) / a.window;
    BLT_REQUIRE(num_windows >= 1, "corpus smaller than one training window");
    BLT_REQUIRE(!(a.lr_schedule == 1 && (a.lr_decay || a.lr_decay_count > 0)),
                "train_blt_d: --lr-schedule cosine cannot be combined with --lr-decay");
    BLT_REQUIRE(a.save_best_path == NULL || a.eval_path != NULL, "train_blt_d: --save-best requires --eval-corpus");
    BLT_REQUIRE(a.save_best_path == NULL || a.eval_every > 0, "train_blt_d: --save-best requires --eval-every > 0");

    const size_t total_windows = (a.start_step + a.steps) * a.grad_accum;
    size_t total_epochs = (total_windows + num_windows - 1) / num_windows;
    printf("train_blt_d: %zu windows (~%zu epochs)\n", num_windows, total_epochs);

    adamw_state *aw = NULL;
    if (a.optimizer == 1) aw = adamw_state_create(adamw_state_arena, model);

    // Without this the optimizer restarts cold on every invocation: m = v = 0 and
    // the bias-correction counter at 0, so the first steps behave like
    // from-scratch AdamW rather than a continuation.
    if (a.load_optim_path) {
        BLT_REQUIRE(a.load_path != NULL, "train_blt_d: --load-optim requires --load-weights");
        BLT_REQUIRE(a.optimizer == 1, "train_blt_d: --load-optim only applies to --optimizer adamw");
        BLT_REQUIRE(aw != NULL, "train_blt_d: --load-optim given but the AdamW state was not created");
        size_t stored_step = 0;
        adamw_state_load(aw, &stored_step, a.load_optim_path);
        printf("[CKPT] loaded optimizer state from %s (global_step=%zu adamw_step=%zu)\n", a.load_optim_path,
               stored_step, aw->step);
        if (!a.start_step_given) a.start_step = stored_step;
    }
    const size_t end_step = a.start_step + a.steps;

    double running = 0.0f;
    size_t running_n = 0;
    double last_loss = 0.0;

    double best_metric_value = (a.best_metric == 0) ? INFINITY : -INFINITY;
    size_t best_step = 0;

    double report_t0 = blt_time_sec();
    size_t report_step0 = a.start_step;
    size_t patch_sum = 0, patch_n = 0;

    FILE *gnorm_fp = NULL;
    if (a.grad_norm_log) gnorm_fp = fopen(a.grad_norm_log, "w");
    FILE *unorm_fp = NULL;
    if (a.update_norm_log) unorm_fp = fopen(a.update_norm_log, "w");
    FILE *cnorm_fp = NULL;
    if (a.component_norm_log) cnorm_fp = fopen(a.component_norm_log, "w");
    FILE *adump_fp = NULL;
    if (a.activation_dump_log) adump_fp = fopen(a.activation_dump_log, "w");
    FILE *batch_fp = NULL;
    if (a.batch_log) {
        if (train_ensure_parent_dir(a.batch_log) == 0) batch_fp = fopen(a.batch_log, "w");
        BLT_REQUIRE(batch_fp != NULL, "train_blt_d: cannot open --batch-log %s", a.batch_log);
    }
    FILE *loss_fp = NULL;
    if (a.loss_log) {
        if (train_ensure_parent_dir(a.loss_log) == 0) loss_fp = fopen(a.loss_log, "w");
        BLT_REQUIRE(loss_fp != NULL, "train_blt_d: cannot open --loss-log %s", a.loss_log);
    }
    FILE *metrics_fp = NULL;
    if (a.metrics_log) {
        if (train_ensure_parent_dir(a.metrics_log) == 0) metrics_fp = fopen(a.metrics_log, "w");
        BLT_REQUIRE(metrics_fp != NULL, "train_blt_d: cannot open --metrics-log %s", a.metrics_log);
        setvbuf(metrics_fp, NULL, _IOLBF, 0);
    }

    double t_fwd = 0.0, t_bwd = 0.0, t_opt = 0.0, t_step = 0.0;
    double ts0, tsA, tsB, tsC, tsD;
    const int timing = (getenv("BLT_TRAIN_TIMING") != NULL);

    double train_start = blt_time_sec();
    double last_tflops = 0.0;
    int have_last_tflops = 0;
    double last_t = train_start, excl_s = 0.0;
    blt_eta_state eta;
    blt_eta_init(&eta, a.start_step);

    for (size_t step = a.start_step; step < end_step; step++) {
        if (timing) ts0 = blt_time_sec();
        if (a.grad_accum > 1) blt_model_grad_zero(model, grad_accum);
        double step_loss = 0.0;
        size_t micros_done = 0;
        for (size_t micro = 0; micro < a.grad_accum; micro++) {
            const size_t data_step = step * a.grad_accum + micro;
            blt_arena_reset(scratch);
#ifdef BLT_WITH_CUDA
            if (a.use_cuda) blt_cuda_scratch_reset();
#endif

            const size_t w = train_window_index(data_step, num_windows, a.seed);
            const uint8_t *text = corpus + w * a.window;
            const size_t N = a.window;

            blt_patch_info patches[BLT_MAX_PATCHES];
            const size_t M = train_lm ? (blt_arena_reset(host_seg_arena),
                                         entropy_segment(host_seg_arena, train_lm, text, N, patches, BLT_MAX_PATCHES))
                                      : fixed_stride(N, 4, patches);
            BLT_REQUIRE(M >= 2, "training window produced < 2 patches");
            if (M >= BLT_MAX_PATCHES) continue;

            blt_block_batch batch;
            if (a.diffusion) {
                uint64_t trng = a.seed * 0x9E3779B97F4A7C15ULL + data_step + 1;
                float t_draw = (float)((double)(t_rng_next(&trng) >> 11) / 9007199254740992.0);
                if (t_draw <= 1e-6f) t_draw = 1e-6f;
                // High-t curriculum: hold the draw above a floor that decays
                // linearly from --t-hi-start to t_min over the first
                // --t-warmup-hi fraction of steps, so early training emphasizes
                // heavily-masked blocks (the regime block drafting relies on)
                // before annealing into the full U(0,1) distribution.
                if (a.t_warmup_frac > 0.0f && end_step > 0 && a.t_hi_start > a.t_min) {
                    const size_t warm = (size_t)((double)end_step * (double)a.t_warmup_frac);
                    if (warm > 0 && step < warm) {
                        const float thr = a.t_hi_start + (a.t_min - a.t_hi_start) * ((float)step / (float)warm);
                        if (t_draw < thr) t_draw = thr;
                    }
                }
                blt_block_batch_build_t(&batch, scratch, text, N, patches, M, a.block_size, a.seed + data_step, t_draw);
                // --paper-loss is Eq. 5 + Eq. 7 exactly: both terms are sums.
                batch.mask_norm = a.paper_loss ? 0 : a.mask_loss_norm;
                batch.clean_sum = a.paper_loss;
                batch.last_row_scale = a.last_row_scale;
                // Floor the timestep: without it, rare tiny-t draws give 1/t
                // weights up to ~1e6 that dominate gradients and starve
                // L_clean (observed as wild loss swings). Standard masked-
                // diffusion stabilization; caps the weight at 1/t_min.
                if (batch.t < a.t_min) batch.t = a.t_min;
                // Optional warmup: ramp the mask-loss weight 0->1 (paper §6
                // explicitly sanctions reweighting toward next-byte prediction).
                if (a.mask_warmup > 0) {
                    batch.loss_scale = (float)step / (float)a.mask_warmup;
                }
                float cap = a.mask_scale;
                if (a.mask_late_step > 0 && step >= a.mask_late_step && end_step > a.mask_late_step) {
                    const float frac = (float)(step - a.mask_late_step) / (float)(end_step - a.mask_late_step);
                    cap = a.mask_scale + (a.mask_late_scale - a.mask_scale) * frac;
                }
                if (batch.loss_scale > cap) batch.loss_scale = cap;
                // Log batch properties for spike investigation
                if (batch_fp) log_batch_properties(batch_fp, step, w * a.window, text, N, patches, M, &batch);
            }

            size_t bshape[1] = {N};
            blt_tensor bytes_in = blt_tensor_create_uninit(scratch, bshape, 1, BLT_DTYPE_UINT8);
            blt_tensor_upload(&bytes_in, text, N);

            // Targets for L_clean: window+1 bytes so the last row has a prediction target
            size_t tshape[1] = {N + 1};
            blt_tensor targets = blt_tensor_create_uninit(scratch, tshape, 1, BLT_DTYPE_UINT8);
            blt_tensor_upload(&targets, text, N + 1);

            size_t h_shape[2] = {N, a.embed};
            size_t sc_shape[1] = {1};

            blt_model_grad_zero(model, grad);

            if (a.diffusion) {
                size_t p_shape2[2] = {M, a.embed};
                blt_tensor P = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
                blt_tensor h = blt_tensor_create(scratch, h_shape, 2, BLT_DTYPE_FP32);
                if (timing) tsA = blt_time_sec();
                blt_local_encoder_forward(model->encoder, &bytes_in, patches, M, NULL, 0, &P, &h, scratch);

                blt_tensor O = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
                blt_global_transformer_forward(model->global, &P, NULL, 0, &O, scratch);

                size_t S = N + batch.n_block_rows;
                size_t lg_shape[2] = {S, 256};
                blt_tensor logits = blt_tensor_create(scratch, lg_shape, 2, BLT_DTYPE_FP32);
                blt_tensor loss = blt_tensor_create(scratch, sc_shape, 1, BLT_DTYPE_FP32);
                blt_local_decoder_forward_diffusion(model->decoder, &h, &O, patches, M, text, &targets, &batch,
                                                    a.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, &logits, &loss,
                                                    scratch);

                if (adump_fp && ((step + 1) % a.report_every == 0 || step + 1 == end_step))
                    log_forward_activation_dump(adump_fp, step + 1, &P, &h, &O, &logits);

                float lv;
                blt_tensor_download(&loss, &lv, sizeof(float));
                if (timing) {
                    tsB = blt_time_sec();
                    t_fwd += tsB - tsA;
                }
                step_loss += lv;
                patch_sum += M;
                patch_n++;

                blt_tensor grad_h = blt_tensor_create(scratch, h_shape, 2, BLT_DTYPE_FP32);
                blt_tensor grad_O = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
                blt_local_decoder_backward_diffusion(model->decoder, &h, &O, patches, M, text, &targets, &batch,
                                                     a.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, &grad_h, &grad_O,
                                                     grad->decoder_grad, scratch);

                blt_tensor grad_P = blt_tensor_create(scratch, p_shape2, 2, BLT_DTYPE_FP32);
                blt_global_transformer_backward(model->global, &P, NULL, 0, &grad_O, &grad_P, grad->global_grad,
                                                scratch);

                blt_local_encoder_backward(model->encoder, &bytes_in, patches, M, NULL, 0, &grad_P, &grad_h,
                                           grad->encoder_grad, scratch);
                if (adump_fp && ((step + 1) % a.report_every == 0 || step + 1 == end_step))
                    log_gradient_activation_dump(adump_fp, step + 1, model, grad);
                if (timing) {
                    tsC = blt_time_sec();
                    t_bwd += tsC - tsB;
                }
            } else {
                // plain causal BLT baseline: identical pipeline, standard
                // shifted-CE objective through the composed model API
                size_t lg_shape[2] = {N, 256};
                blt_tensor logits = blt_tensor_create(scratch, lg_shape, 2, BLT_DTYPE_FP32);
                blt_tensor loss = blt_tensor_create(scratch, sc_shape, 1, BLT_DTYPE_FP32);
                blt_model_forward(model, &bytes_in, &targets, patches, M, NULL, 0, &logits, &loss, scratch);

                if (adump_fp && ((step + 1) % a.report_every == 0 || step + 1 == end_step))
                    log_forward_activation_dump(adump_fp, step + 1, NULL, NULL, NULL, &logits);

                float lv;
                blt_tensor_download(&loss, &lv, sizeof(float));
                step_loss += lv;
                patch_sum += M;
                patch_n++;

                blt_model_backward(model, &bytes_in, &targets, patches, M, NULL, 0, grad, scratch);

                if (adump_fp && ((step + 1) % a.report_every == 0 || step + 1 == end_step))
                    log_gradient_activation_dump(adump_fp, step + 1, model, grad);
            }

            if (a.grad_accum > 1) blt_model_grad_add(model, grad_accum, grad);
            micros_done++;
        }
        BLT_REQUIRE(micros_done > 0, "train_blt_d: no usable micro-batch in gradient accumulation");
        if (a.grad_accum > 1) blt_model_grad_scale(model, grad_accum, 1.0f / (float)micros_done);

        last_loss = step_loss / (double)micros_done;
        if (loss_fp) {
            fprintf(loss_fp, "%zu %.6f\n", step + 1, last_loss);
            fflush(loss_fp);
        }
        running += last_loss;
        running_n++;

        float lr = train_lr_at_step(&a, step, end_step);

        if (timing) tsC = blt_time_sec();
        if (cnorm_fp && ((step + 1) % a.report_every == 0 || step + 1 == end_step))
            log_component_norms(cnorm_fp, step + 1, model, step_grad);
        float clip_scale = 1.0f;
        float pre_clip_norm = blt_grad_clip_norm(model, step_grad, a.max_norm, &clip_scale);
        if (gnorm_fp && ((step + 1) % a.report_every == 0 || step + 1 == end_step))
            fprintf(gnorm_fp, "%zu %.6f\n", step + 1, pre_clip_norm);
        if (unorm_fp && pre_clip_norm > 100.0f) {
            float post_clip = pre_clip_norm <= a.max_norm ? pre_clip_norm : a.max_norm;
            float u_norm;
            if (a.optimizer == 1) {
                blt_adamw_config ucfg = {.lr = lr,
                                         .beta1 = a.beta1,
                                         .beta2 = a.beta2,
                                         .eps = a.eps,
                                         .weight_decay = a.weight_decay,
                                         .weight_decay_norms = a.weight_decay_norms,
                                         .step = 0};
                u_norm = adamw_update_norm(model, step_grad, aw, &ucfg, clip_scale);
            } else {
                u_norm = lr * post_clip;
            }
            if ((step + 1) % a.report_every == 0 || step + 1 == end_step)
                fprintf(unorm_fp, "%zu %.6f %.6f\n", step + 1, post_clip, u_norm);
        }
        if (a.optimizer == 1) {
            blt_adamw_config cfg = {.lr = lr,
                                    .beta1 = a.beta1,
                                    .beta2 = a.beta2,
                                    .eps = a.eps,
                                    .weight_decay = a.weight_decay,
                                    .weight_decay_norms = a.weight_decay_norms,
                                    .step = 0};
            adamw_all(model, step_grad, aw, &cfg, clip_scale);
        } else {
            sgd_all(model, step_grad, lr, clip_scale);
        }
        // The optimizer mutated the FP32 masters; rebuild every low-precision
        // shadow so the next forward (report, monitor, eval) reads current weights.
        blt_model_refresh_mixed(model);
        if (metrics_fp) {
            fprintf(metrics_fp, "{\"event\":\"train\",\"step\":%zu,\"loss\":%.6f,\"lr\":%.6g,\"grad_norm\":%.6f}\n",
                    step + 1, last_loss, (double)lr, (double)pre_clip_norm);
            fflush(metrics_fp);
        }
        if (timing) {
            tsD = blt_time_sec();
            t_opt += tsD - tsC;
            t_step += tsD - ts0;
            if ((step % 50) == 49) {
                printf("[TIMING] step %zu fwd %.1fms bwd %.1fms opt %.1fms step %.1fms\n", step + 1,
                       1000.0 * t_fwd / 50.0, 1000.0 * t_bwd / 50.0, 1000.0 * t_opt / 50.0, 1000.0 * t_step / 50.0);
                fflush(stdout);
                t_fwd = t_bwd = t_opt = t_step = 0.0;
            }
        }

        if (a.monitor_every > 0 && (step + 1) % a.monitor_every == 0 && a.eval_path != NULL) {
            FILE *mf = fopen(a.eval_path, "rb");
            if (mf) {
                fseek(mf, 0, SEEK_END);
                long msz = ftell(mf);
                fseek(mf, 0, SEEK_SET);
                uint8_t *mbytes = (msz > 0) ? (uint8_t *)malloc((size_t)msz) : NULL;
                if (mbytes && fread(mbytes, 1, (size_t)msz, mf) == (size_t)msz) {
                    blt_monitor_result mr;
                    blt_monitor(model, scratch, host_seg_arena, train_lm, mbytes + a.eval_skip,
                                (size_t)msz - a.eval_skip, &a, step, &mr);
                    const double tacc = mr.train_total ? (double)mr.train_hits / (double)mr.train_total : 0.0;
                    const double nacc = mr.novel_total ? (double)mr.novel_hits / (double)mr.novel_total : 0.0;
                    const double gap = tacc - nacc;
                    printf("[MONITOR] step %zu  L_clean %.4f (%.3f bpb)  L_mask %.4f (raw sum/t)  masked_bpb %.3f  "
                           "train_acc %.4f (n=%zu)  aligned_novel_acc %.4f (n=%zu)  gap %+.4f\n",
                           step + 1, mr.l_clean, mr.clean_bits, mr.l_mask, mr.mask_bits, tacc, mr.train_total, nacc,
                           mr.novel_total, gap);
                    fflush(stdout);
                    if (metrics_fp) {
                        fprintf(
                            metrics_fp,
                            "{\"event\":\"monitor\",\"step\":%zu,\"l_clean\":%.6f,\"clean_bpb\":%.6f,\"mask_bpb\":%.6f,"
                            "\"train_acc\":%.6f,\"novel_acc\":%.6f,\"gap\":%.6f}\n",
                            step + 1, mr.l_clean, mr.clean_bits, mr.mask_bits, tacc, nacc, gap);
                        fflush(metrics_fp);
                    }
                    // Tripwires: a reintroduced clean-twin leak sends train_acc to
                    // ~1.0; a train/novel split after 5k steps means the layouts
                    // disagree, which the leak-free mask should prevent.
                    if (mr.train_total > 0 && tacc > 0.95) {
                        printf("[MONITOR] ABORT: train-layout masked acc %.4f > 0.95 at step %zu -- label leak\n", tacc,
                               step + 1);
                        fflush(stdout);
                        free(mbytes);
                        fclose(mf);
                        return 3;
                    }
                    if (step + 1 > 5000 && mr.train_total > 0 && mr.novel_total > 0 && gap > 0.10) {
                        printf("[MONITOR] ABORT: train - aligned_novel = %.4f > 0.10 at step %zu -- mask leak\n", gap,
                               step + 1);
                        fflush(stdout);
                        free(mbytes);
                        fclose(mf);
                        return 3;
                    }
                }
                free(mbytes);
                fclose(mf);
            }
        }

        if ((step + 1) % a.report_every == 0 || step + 1 == end_step) {
            size_t epoch = (step + 1) * a.grad_accum / num_windows;
            double now = blt_time_sec();
            const double remaining = blt_eta_update(&eta, step, end_step, now - last_t, excl_s);
            last_t = now;
            excl_s = 0.0;
            size_t rem_h = (size_t)remaining / 3600;
            size_t rem_m = ((size_t)remaining % 3600) / 60;
            const double steps_done = (double)((step + 1) - report_step0);
            const double windows_done = steps_done * (double)a.grad_accum;
            const double dt = now - report_t0;
            const double ms_per_step = (steps_done > 0) ? 1000.0 * dt / steps_done : 0.0;
            const double bytes_per_s = (dt > 0) ? windows_done * (double)a.window / dt : 0.0;
            const double avg_patch = patch_n ? (double)patch_sum / (double)patch_n : 4.0;
            double tflops = 0.0;
            int have_tflops = 0;
            if (dt > 0 && steps_done > 0) {
                blt_flops_config fc;
                memset(&fc, 0, sizeof(fc));
                fc.h_G = (int)cfg.global_config.embed_dim;
                fc.l_G = (int)cfg.global_config.num_layers;
                fc.n_ctx = a.window;
                fc.global_heads = (int)cfg.global_config.num_heads;
                fc.global_head_dim = fc.h_G / (fc.global_heads ? fc.global_heads : 1);
                fc.d_ff_G = (int)(cfg.global_config.hidden_dim /
                                  (cfg.global_config.embed_dim ? cfg.global_config.embed_dim : 1));
                fc.h_E = (int)cfg.encoder_config.embed_dim;
                fc.l_E = (int)cfg.encoder_config.num_layers;
                fc.w_E = a.window;
                fc.enc_heads = (int)cfg.encoder_config.num_heads;
                fc.enc_head_dim = fc.h_E / (fc.enc_heads ? fc.enc_heads : 1);
                fc.d_ff_E = (int)(cfg.encoder_config.hidden_dim /
                                  (cfg.encoder_config.embed_dim ? cfg.encoder_config.embed_dim : 1));
                fc.h_D = (int)cfg.decoder_config.embed_dim;
                fc.l_D = (int)cfg.decoder_config.num_layers;
                fc.w_D = a.window;
                fc.dec_heads = (int)cfg.decoder_config.num_heads;
                fc.dec_head_dim = fc.h_D / (fc.dec_heads ? fc.dec_heads : 1);
                fc.d_ff_D = (int)(cfg.decoder_config.hidden_dim /
                                  (cfg.decoder_config.embed_dim ? cfg.decoder_config.embed_dim : 1));
                fc.n_p = (size_t)llround(avg_patch > 1 ? avg_patch : 1);
                fc.k = 1;
                fc.vocab_size = 256;
                double fpb = blt_flops_per_byte(&fc);
                if (isfinite(fpb) && fpb > 0) {
                    const double flops_per_step = 3.0 * fpb * (double)a.window * (double)a.grad_accum;
                    tflops = flops_per_step / (dt / steps_done) / 1e12;
                    have_tflops = isfinite(tflops);
                }
            }
            printf("step %6zu/%zu  epoch %zu/%zu  avg_loss %.4f  lr %.3g  ms/step %.1f  B/s %.1f  ETA %zuh%02zum",
                   step + 1, end_step, epoch, total_epochs, running / running_n, (double)lr, ms_per_step, bytes_per_s,
                   rem_h, rem_m);
            if (have_tflops) printf("  ~%.2f TFLOP/s", tflops);
            if (a.peak_tflops > 0 && have_tflops) printf("  MFU %.1f%%", 100.0 * tflops / a.peak_tflops);
            if (have_tflops) {
                last_tflops = tflops;
                have_last_tflops = 1;
            }
            printf("\n");
            fflush(stdout);
            running = 0.0;
            running_n = 0;
            report_t0 = blt_time_sec();
            report_step0 = step + 1;
            patch_sum = 0;
            patch_n = 0;
        }

        const double post0 = blt_time_sec();
        if (a.save_path && a.save_every > 0 && (step + 1) % a.save_every == 0) {
            char path_buf[4096];
            blt_format_save_path(path_buf, sizeof(path_buf), a.save_path, step + 1);
            char meta_buf[4096];
            snprintf(
                meta_buf, sizeof(meta_buf),
                "{\"format\":\"fblt-checkpoint\",\"format_version\":2,\"config\":{\"embed\":%zu,\"hidden\":%zu,"
                "\"enc_layers\":%zu,\"glob_layers\":%zu,\"dec_layers\":%zu,\"cross_attn\":\"%s\",\"d0_mode\":\"%s\","
                "\"diffusion\":%d,\"block_size\":%zu,\"window\":%zu},\"training\":{\"step\":%zu,\"optimizer\":\"%s\","
                "\"seed\":%llu}}",
                a.embed, a.hidden, a.enc_layers, a.glob_layers, a.dec_layers, a.cross_last ? "last" : "all",
                a.d0_learned ? "learned" : "zeros", a.diffusion, a.block_size, a.window, step + 1,
                a.optimizer ? "adamw" : "sgd", (unsigned long long)a.seed);
            blt_model_save_ex(model, path_buf, meta_buf);
            if (a.save_optim_path && aw) {
                char opath_buf[4096];
                blt_format_save_path(opath_buf, sizeof(opath_buf), a.save_optim_path, step + 1);
                adamw_state_save(aw, step + 1, opath_buf);
            }
            printf("[CKPT] periodic save step %zu -> %s\n", step + 1, path_buf);
            fflush(stdout);
        }

        // Periodic eval: run causal BPB on the held-out corpus every N steps
        if (a.eval_every > 0 && (step + 1) % a.eval_every == 0 && a.eval_path != NULL) {
            FILE *ef = fopen(a.eval_path, "rb");
            if (ef) {
                fseek(ef, 0, SEEK_END);
                long esz = ftell(ef);
                fseek(ef, 0, SEEK_SET);
                uint8_t *eval_bytes = (uint8_t *)malloc((size_t)esz);
                if (eval_bytes && fread(eval_bytes, 1, (size_t)esz, ef) == (size_t)esz) {
                    size_t avail = (size_t)esz - a.eval_skip;
                    size_t eval_count = a.eval_windows;
                    size_t max_windows = avail / a.window;
                    if (eval_count > max_windows) eval_count = max_windows;
                    uint8_t *base = eval_bytes + a.eval_skip;
                    double ce_sum = 0.0;
                    size_t hits = 0, total = 0;
                    for (size_t wi = 0; wi < eval_count; wi++) {
                        blt_arena_reset(scratch);
                        const uint8_t *text = base + wi * a.window;
                        blt_patch_info ep[BLT_MAX_PATCHES];
                        size_t eM = 0;
                        if (train_lm) {
                            blt_arena_reset(host_seg_arena);
                            eM = entropy_segment(host_seg_arena, train_lm, text, a.window, ep, BLT_MAX_PATCHES);
                        } else {
                            eM = fixed_stride(a.window, 4, ep);
                        }
                        blt_block_batch ebatch;
                        if (a.diffusion) {
                            blt_block_batch_build(&ebatch, scratch, text, a.window, ep, eM, a.block_size,
                                                  a.seed + 999999);
                        }
                        ce_sum +=
                            window_causal_ce(scratch, model, text, a.window, a.diffusion ? &ebatch : NULL, a.diffusion,
                                             a.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, ep, eM, &hits, &total);
                    }
                    double bpb = (ce_sum / (double)eval_count) / log(2.0);
                    double masked_acc = total ? (double)hits / (double)total : 0.0;
                    printf("EVAL step=%zu causal_bpb=%.4f masked_acc=%.3f\n", step + 1, bpb, masked_acc);
                    fflush(stdout);
                    if (a.save_best_path) {
                        const double metric = (a.best_metric == 0) ? bpb : masked_acc;
                        const int valid = (a.best_metric == 0) ? isfinite(bpb) : (total > 0);
                        const int improved = valid && ((a.best_metric == 0) ? (metric < best_metric_value)
                                                                            : (metric > best_metric_value));
                        if (improved) {
                            best_metric_value = metric;
                            best_step = step + 1;
                            char path_buf[4096];
                            blt_format_save_path(path_buf, sizeof(path_buf), a.save_best_path, step + 1);
                            blt_model_save_ex(model, path_buf, NULL);
                            if (a.save_best_optim_path && aw) {
                                char opath_buf[4096];
                                blt_format_save_path(opath_buf, sizeof(opath_buf), a.save_best_optim_path, step + 1);
                                adamw_state_save(aw, step + 1, opath_buf);
                            }
                            printf("[BEST] step=%zu %s=%.6f -> %s\n", step + 1,
                                   a.best_metric == 0 ? "causal_bpb" : "masked_acc", metric, path_buf);
                            fflush(stdout);
                        }
                    }
                    if (metrics_fp) {
                        fprintf(metrics_fp,
                                "{\"event\":\"eval\",\"step\":%zu,\"causal_bpb\":%.6f,\"masked_acc\":%.6f}\n", step + 1,
                                bpb, total ? (double)hits / (double)total : 0.0);
                        fflush(metrics_fp);
                    }
                }
                free(eval_bytes);
                fclose(ef);
            }
        }
        excl_s += blt_time_sec() - post0;
    }

    if (a.save_path) {
        char path_buf[4096];
        blt_format_save_path(path_buf, sizeof(path_buf), a.save_path, end_step);
        char meta_buf[4096];
        snprintf(meta_buf, sizeof(meta_buf),
                 "{\"format\":\"fblt-checkpoint\",\"format_version\":2,\"config\":{\"embed\":%zu,\"hidden\":%zu,"
                 "\"enc_layers\":%zu,\"glob_layers\":%zu,\"dec_layers\":%zu,\"cross_attn\":\"%s\",\"d0_mode\":\"%s\","
                 "\"diffusion\":%d,\"block_size\":%zu,\"window\":%zu},\"training\":{\"step\":%zu,\"optimizer\":\"%s\","
                 "\"seed\":%llu}}",
                 a.embed, a.hidden, a.enc_layers, a.glob_layers, a.dec_layers, a.cross_last ? "last" : "all",
                 a.d0_learned ? "learned" : "zeros", a.diffusion, a.block_size, a.window, end_step,
                 a.optimizer ? "adamw" : "sgd", (unsigned long long)a.seed);
        blt_model_save_ex(model, path_buf, meta_buf);
        if (a.save_optim_path && aw) {
            char opath_buf[4096];
            blt_format_save_path(opath_buf, sizeof(opath_buf), a.save_optim_path, end_step);
            adamw_state_save(aw, end_step, opath_buf);
        }
        printf("[CKPT] saved weights to %s\n", path_buf);
    }

    // Held-out causal BPB evaluation
    // Same windowing/patches as training

    double bpb = -1.0;
    double final_masked_acc = 0.0;
    size_t eval_count = 0, hits = 0, total = 0;
    if (a.eval_path != NULL) {
        FILE *ef = fopen(a.eval_path, "rb");
        if (!ef) {
            fprintf(stderr, "cannot open eval corpus '%s'\n", a.eval_path);
            return 1;
        }
        fseek(ef, 0, SEEK_END);
        long esz = ftell(ef);
        fseek(ef, 0, SEEK_SET);
        uint8_t *eval_bytes = (uint8_t *)malloc((size_t)esz);
        if (!eval_bytes || fread(eval_bytes, 1, (size_t)esz, ef) != (size_t)esz) {
            fprintf(stderr, "failed to read eval corpus\n");
            return 1;
        }
        fclose(ef);

        if ((size_t)esz <= a.eval_skip + a.window) {
            fprintf(stderr, "eval corpus too small for --eval-skip\n");
            return 1;
        }
        size_t avail = (size_t)esz - a.eval_skip;
        eval_count = a.eval_windows;
        size_t max_windows = avail / a.window;
        if (eval_count > max_windows) eval_count = max_windows;
        uint8_t *base = eval_bytes + a.eval_skip;

        double ce_sum = 0.0;
        hits = 0;
        total = 0;
        for (size_t wi = 0; wi < eval_count; wi++) {
            blt_arena_reset(scratch);
            const uint8_t *text = base + wi * a.window;

            blt_patch_info ep[BLT_MAX_PATCHES];
            size_t eM = 0;
            if (train_lm) {
                blt_arena_reset(host_seg_arena);
                eM = entropy_segment(host_seg_arena, train_lm, text, a.window, ep, BLT_MAX_PATCHES);
            } else {
                eM = fixed_stride(a.window, 4, ep);
            }

            blt_block_batch ebatch;
            if (a.diffusion) {
                // deterministic eval corruption (never used by the CE above)
                blt_block_batch_build(&ebatch, scratch, text, a.window, ep, eM, a.block_size, a.seed + 999999);
            }

            ce_sum += window_causal_ce(scratch, model, text, a.window, a.diffusion ? &ebatch : NULL, a.diffusion,
                                       a.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS, ep, eM, &hits, &total);
        }
        bpb = (ce_sum / (double)eval_count) / log(2.0);
        final_masked_acc = total ? (double)hits / (double)total : 0.0;
        if (a.save_best_path) {
            const double metric = (a.best_metric == 0) ? bpb : final_masked_acc;
            const int valid = (a.best_metric == 0) ? isfinite(bpb) : (total > 0);
            const int improved =
                valid && ((a.best_metric == 0) ? (metric < best_metric_value) : (metric > best_metric_value));
            if (improved) {
                best_metric_value = metric;
                best_step = end_step;
                char path_buf[4096];
                blt_format_save_path(path_buf, sizeof(path_buf), a.save_best_path, end_step);
                blt_model_save_ex(model, path_buf, NULL);
                if (a.save_best_optim_path && aw) {
                    char opath_buf[4096];
                    blt_format_save_path(opath_buf, sizeof(opath_buf), a.save_best_optim_path, end_step);
                    adamw_state_save(aw, end_step, opath_buf);
                }
                printf("[BEST] step=%zu %s=%.6f -> %s\n", end_step, a.best_metric == 0 ? "causal_bpb" : "masked_acc",
                       metric, path_buf);
                fflush(stdout);
            }
        }
        free(eval_bytes);
    }

    // Training MBU model: per optimizer step the weights are read by the
    // forward and backward passes and touched again by the update; gradients
    // are written then read; AdamW additionally reads and writes its two
    // moments. That is 6x weight bytes for SGD and 10x for AdamW. Activations
    // (the dominant traffic at long windows) are excluded, so MBU is a lower
    // bound on the bytes the hardware actually moves.
    const size_t param_bytes = blt_param_bytes(model);
    const double per_step_weight_mult = (a.optimizer == 1) ? 10.0 : 6.0;
    const double train_elapsed = blt_time_sec() - train_start;
    const double bytes_moved_gb = (double)param_bytes * per_step_weight_mult * (double)(end_step - a.start_step) / 1e9;
    char mfu_buf[32];
    char mbu_buf[32];
    if (a.peak_tflops > 0.0f && have_last_tflops) {
        snprintf(mfu_buf, sizeof(mfu_buf), "%.2f%%", 100.0 * last_tflops / (double)a.peak_tflops);
    } else {
        snprintf(mfu_buf, sizeof(mfu_buf), "n/a");
    }
    if (a.peak_bandwidth > 0.0f && train_elapsed > 0.0 && param_bytes > 0) {
        snprintf(mbu_buf, sizeof(mbu_buf), "%.2f%%",
                 100.0 * blt_mbu(bytes_moved_gb, train_elapsed, (double)a.peak_bandwidth));
    } else {
        snprintf(mbu_buf, sizeof(mbu_buf), "n/a");
    }
    printf("RESULT mode=%s steps=%zu embed=%zu window=%zu B=%zu "
           "enc/glob/dec=%zu/%zu/%zu xattn=%s backend=%s dtype=%s "
           "eval_windows=%zu causal_bpb=%.4f masked_acc=%.3f (%zu/%zu) params=%zu "
           "device=%s peak_tflops=%.2f mfu=%s mbu=%s\n",
           a.diffusion ? "bltd" : "plain", a.steps, a.embed, a.window, a.block_size, a.enc_layers, a.glob_layers,
           a.dec_layers, a.cross_last ? "last" : "all", a.use_cuda ? "cuda" : "cpu", dtype_name, eval_count, bpb,
           total ? (double)hits / (double)total : 0.0, hits, total, n_params, device_name ? device_name : "unknown",
           (double)a.peak_tflops, mfu_buf, mbu_buf);

    if (metrics_fp) {
        const size_t n_params = blt_count_params(model);
        fprintf(metrics_fp,
                "{\"event\":\"result\",\"steps\":%zu,\"dtype\":\"%s\",\"causal_bpb\":%.6f,\"masked_acc\":%.6f,"
                "\"params\":%zu}\n",
                end_step, dtype_name, bpb, final_masked_acc, n_params);
        fflush(metrics_fp);
        fclose(metrics_fp);
    }

    blt_arena_destroy(host_lm_arena);
    blt_arena_destroy(host_seg_arena);
    blt_arena_destroy(model_arena);
    blt_arena_destroy(scratch);
    if (adamw_state_arena) blt_arena_destroy(adamw_state_arena);
    if (aw) free(aw);
    if (gnorm_fp) fclose(gnorm_fp);
    if (unorm_fp) fclose(unorm_fp);
    if (cnorm_fp) fclose(cnorm_fp);
    if (adump_fp) fclose(adump_fp);
    if (batch_fp) fclose(batch_fp);
    if (loss_fp) fclose(loss_fp);
    free(corpus);
    return 0;
}
