// Benchmarks for CUDA backend: micro, meso, macro
//   micro   raw kernel throughput: matmul TFLOP/s, elementwise GB/s
//   meso    full-pipeline fwd / bwd wall time at swept sequence lengths,
//           with model MFU against the paper FLOPs model in tools/flops.c
//   macro   training-step core throughput (fwd+bwd tokens/s) and greedy
//           generation NFE/s on the real checkpoint
//
// Every dispatched CUDA op synchronizes internally, so wall-clock timing is
// valid as-is and honestly includes launch + sync overhead. Results append
// to bench/results.jsonl under phase "7_cuda_bench" (tags carry the backend
// suffix), rendered by tools/bench_report.py like any other phase.
//
// Build + run:  make CUDA=1 bench-cuda [-- backend cpu]

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h"

#include "core/allocator.h"
#include "core/backend.h"
#include "models/model.h"
#include "models/checkpoint.h"
#include "ops/elementwise.h"
#include "ops/gelu.h"
#include "ops/matmul.h"
#include "ops/cast.h"
#include "models/transformer_stack.h"
#include "core/flops.h"

// Desktop RTX 4060 (AD107, CC 8.9): 3072 fp32 cores × 2460 MHz boost × 2
// (mad) = 15.11 TFLOP/s per NVIDIA spec. The earlier 11.2 value was a
// conservative estimate that understated the desktop peak by ~26%.
// Verified via nvidia-smi: 115 W power limit, 3105 MHz max SM clock,
// 8188 MiB GDDR6 — unambiguously the desktop (not laptop) variant.
static const double PEAK_TFLOPS_CUDA = 15.11;

typedef struct {
    int use_cuda;
    int macro_only;
    size_t iters;
    const char *results_path;
    const char *ckpt;
} bench_args;

static uint32_t prng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void fill_random_host(float *d, size_t n, float scale, uint32_t *seed) {
    for (size_t i = 0; i < n; i++) {
        d[i] = ((float)(prng_next(seed) & 0xFFFF) / 32768.0f - 1.0f) * scale;
    }
}

// Forces completion of any asynchronously-enqueued device work (cuBLAS
// GEMMs are not synchronized inside the op): a 4-byte D2H copy cannot
// return before all prior stream work retires.
static void backend_sync(const blt_tensor *t) {
    float sentinel;
    blt_tensor_download(t, &sentinel, sizeof(float));
}

static blt_tensor make_random_2d(blt_arena *host, blt_arena *arena, size_t rows, size_t cols, float scale,
                                 uint32_t *seed) {
    blt_tensor t = blt_tensor_create(arena, (size_t[2]){rows, cols}, 2, BLT_DTYPE_FP32);
    if (arena->backend != BLT_BACKEND_CPU) {
        float *stage = (float *)malloc(rows * cols * sizeof(float));
        fill_random_host(stage, rows * cols, scale, seed);
        blt_tensor_upload(&t, stage, rows * cols * sizeof(float));
        free(stage);
    } else {
        fill_random_host((float *)t.data, t.numel, scale, seed);
    }
    (void)host;
    return t;
}

//----------------------------------------------------------------------
// Micro: GEMM and elementwise kernel rates
//----------------------------------------------------------------------

static void bench_matmul_cell(blt_arena *arena, size_t n, size_t iters, const char *backend_tag,
                              const char *results_path) {
    uint32_t seed = 0xBEEF0000u ^ (uint32_t)n;
    blt_tensor a = make_random_2d(NULL, arena, n, n, 0.05f, &seed);
    blt_tensor b = make_random_2d(NULL, arena, n, n, 0.05f, &seed);
    blt_tensor c = blt_tensor_create(arena, (size_t[2]){n, n}, 2, BLT_DTYPE_FP32);

    // Warmup (context init, cuBLAS handle/workspace).
    for (size_t i = 0; i < 3; i++) blt_matmul(&a, &b, &c);

    bench_timer_start();
    for (size_t i = 0; i < iters; i++) blt_matmul(&a, &b, &c);
    backend_sync(&c);
    const double sec = bench_timer_stop_sec();

    const double tflops = 2.0 * (double)n * n * n * iters / sec / 1e12;
    const double mfu = 100.0 * tflops / PEAK_TFLOPS_CUDA;

    char tag[BENCH_TAG_LEN];
    snprintf(tag, sizeof(tag), "micro_matmul_%zu_%s", n, backend_tag);
    bench_result r;
    bench_result_init(&r, "cuda_bench", tag, "7_cuda_bench");
    double lat[1] = {sec};
    bench_stats_compute(&r.latency, lat, 1);
    bench_result_add_metric(&r, "tflops", tflops);
    bench_result_add_metric(&r, "mfu_pct", mfu);
    bench_write_json(results_path, &r);
    printf("  %-28s %10.2f TFLOP/s (%5.1f%% of %.1f ref)\n", tag, tflops, mfu, PEAK_TFLOPS_CUDA);
}

static void bench_matmul_bf16_cell(blt_arena *arena, size_t n, size_t iters, const char *backend_tag,
                                   const char *results_path) {
    uint32_t seed = 0xBEEF0000u ^ (uint32_t)n;
    // Create bf16 inputs directly on the backend
    blt_tensor a = blt_tensor_create(arena, (size_t[2]){n, n}, 2, BLT_DTYPE_BF16);
    blt_tensor b = blt_tensor_create(arena, (size_t[2]){n, n}, 2, BLT_DTYPE_BF16);
    blt_tensor c = blt_tensor_create(arena, (size_t[2]){n, n}, 2, BLT_DTYPE_FP32);
    // Fill with random data via fp32→bf16 cast
    blt_tensor tmp_a = blt_tensor_create(arena, (size_t[2]){n, n}, 2, BLT_DTYPE_FP32);
    blt_tensor tmp_b = blt_tensor_create(arena, (size_t[2]){n, n}, 2, BLT_DTYPE_FP32);
    if (arena->backend != BLT_BACKEND_CPU) {
        float *stage = (float *)malloc(n * n * sizeof(float));
        fill_random_host(stage, n * n, 0.05f, &seed);
        blt_tensor_upload(&tmp_a, stage, n * n * sizeof(float));
        fill_random_host(stage, n * n, 0.05f, &seed);
        blt_tensor_upload(&tmp_b, stage, n * n * sizeof(float));
        free(stage);
    } else {
        fill_random_host((float *)tmp_a.data, tmp_a.numel, 0.05f, &seed);
        fill_random_host((float *)tmp_b.data, tmp_b.numel, 0.05f, &seed);
    }
    blt_cast(&tmp_a, &a);
    blt_cast(&tmp_b, &b);

    for (size_t i = 0; i < 3; i++) blt_matmul(&a, &b, &c);

    bench_timer_start();
    for (size_t i = 0; i < iters; i++) blt_matmul(&a, &b, &c);
    backend_sync(&c);
    const double sec = bench_timer_stop_sec();

    const double tflops = 2.0 * (double)n * n * n * iters / sec / 1e12;
    const double mfu = 100.0 * tflops / PEAK_TFLOPS_CUDA;

    char tag[BENCH_TAG_LEN];
    snprintf(tag, sizeof(tag), "micro_matmul_bf16_%zu_%s", n, backend_tag);
    bench_result r;
    bench_result_init(&r, "cuda_bench", tag, "7_cuda_bench");
    double lat[1] = {sec};
    bench_stats_compute(&r.latency, lat, 1);
    bench_result_add_metric(&r, "tflops", tflops);
    bench_result_add_metric(&r, "mfu_pct", mfu);
    bench_write_json(results_path, &r);
    printf("  %-28s %10.2f TFLOP/s (%5.1f%% of %.1f ref)\n", tag, tflops, mfu, PEAK_TFLOPS_CUDA);
}

static void bench_elementwise_cell(blt_arena *arena, size_t n_floats, size_t iters, const char *backend_tag,
                                   const char *results_path) {
    uint32_t seed = 0xF00D0000u;
    blt_tensor a = make_random_2d(NULL, arena, n_floats, 1, 1.0f, &seed);
    blt_tensor b = make_random_2d(NULL, arena, n_floats, 1, 1.0f, &seed);
    blt_tensor c = blt_tensor_create(arena, (size_t[2]){n_floats, 1}, 2, BLT_DTYPE_FP32);

    for (size_t i = 0; i < 3; i++) blt_add(&a, &b, &c);

    bench_timer_start();
    for (size_t i = 0; i < iters; i++) blt_add(&a, &b, &c);
    backend_sync(&c);
    const double sec_add = bench_timer_stop_sec();

    for (size_t i = 0; i < 3; i++) blt_gelu_forward(&a, &c);

    bench_timer_start();
    for (size_t i = 0; i < iters; i++) blt_gelu_forward(&a, &c);
    backend_sync(&c);
    const double sec_gelu = bench_timer_stop_sec();

    // add moves 3 arrays (2R+1W), gelu moves 2 (1R+1W).
    const double gbps_add = 3.0 * 4.0 * (double)n_floats * iters / sec_add / 1e9;
    const double gbps_gelu = 2.0 * 4.0 * (double)n_floats * iters / sec_gelu / 1e9;

    char tag[BENCH_TAG_LEN];
    bench_result r;
    snprintf(tag, sizeof(tag), "micro_add_%s", backend_tag);
    bench_result_init(&r, "cuda_bench", tag, "7_cuda_bench");
    double lat[1] = {sec_add};
    bench_stats_compute(&r.latency, lat, 1);
    bench_result_add_metric(&r, "gbps", gbps_add);
    bench_write_json(results_path, &r);
    printf("  %-28s %10.1f GB/s\n", tag, gbps_add);

    snprintf(tag, sizeof(tag), "micro_gelu_%s", backend_tag);
    bench_result_init(&r, "cuda_bench", tag, "7_cuda_bench");
    lat[0] = sec_gelu;
    bench_stats_compute(&r.latency, lat, 1);
    bench_result_add_metric(&r, "gbps", gbps_gelu);
    bench_write_json(results_path, &r);
    printf("  %-28s %10.1f GB/s\n", tag, gbps_gelu);
}

//----------------------------------------------------------------------
// Meso + macro model setup
//
// One synthetic config shared by every model-level cell so numbers are
// directly comparable; dims sit near the trained checkpoints' shape but
// large enough that device kernels have real work per pass.
//----------------------------------------------------------------------

#define BENCH_E 64
#define BENCH_HID 128
#define BENCH_L 2

static void build_model_config(blt_model_config *cfg, size_t MS) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->encoder_config.embed_dim = BENCH_E;
    cfg->encoder_config.patch_dim = 0;
    cfg->encoder_config.num_layers = BENCH_L;
    cfg->encoder_config.hidden_dim = BENCH_HID;
    cfg->encoder_config.num_heads = 4;
    cfg->encoder_config.cross_attn_heads = 4;
    cfg->encoder_config.cross_attn_all_layers = true;
    cfg->encoder_config.pool_type = BLT_POOL_MEAN;
    cfg->encoder_config.rope_theta = 500000.0f;
    cfg->encoder_config.max_seq_len = MS;
    // Hash n-gram module disabled for the synthetic bench model; embed_dim
    // still must match (validated even when disabled).
    cfg->encoder_config.ngram_config.num_ngram_sizes = 0;
    cfg->encoder_config.ngram_config.embed_dim = BENCH_E;
    cfg->global_config.embed_dim = BENCH_E;
    cfg->global_config.num_layers = BENCH_L;
    cfg->global_config.hidden_dim = BENCH_HID;
    cfg->global_config.num_heads = 4;
    cfg->global_config.rope_theta = 500000.0f;
    cfg->global_config.max_seq_len = MS;
    cfg->decoder_config.embed_dim = BENCH_E;
    cfg->decoder_config.patch_dim = 0;
    cfg->decoder_config.num_layers = BENCH_L;
    cfg->decoder_config.hidden_dim = BENCH_HID;
    cfg->decoder_config.num_heads = 4;
    cfg->decoder_config.cross_attn_heads = 4;
    cfg->decoder_config.cross_attn_all_layers = true;
    cfg->decoder_config.rope_theta = 500000.0f;
    cfg->decoder_config.max_seq_len = MS;
    cfg->decoder_config.vocab_size = 256;
}

// Randomizes every weight through the same stable enumeration checkpoints
// use, so encoder/global/decoder all get live values.
static void randomize_model(blt_model *m, blt_arena *host, uint32_t *seed) {
    blt_model *twin_view = (blt_model *)m;
    const size_t n = blt_model_num_tensors(m);
    float *stage = NULL;
    size_t cap = 0;
    for (size_t i = 0; i < n; i++) {
        const char *name;
        blt_tensor *t;
        blt_model_tensor_at(twin_view, i, &name, &t);
        if (cap < t->numel) {
            free(stage);
            cap = t->numel;
            stage = (float *)malloc(cap * sizeof(float));
            BLT_REQUIRE(stage != NULL, "randomize_model: staging alloc failed");
        }
        fill_random_host(stage, t->numel, 0.05f, seed);
        blt_tensor_upload(t, stage, blt_tensor_bytes(t));
    }
    free(stage);
    (void)host;
}

// Paper-FLOPs estimate for one full pipeline forward over `seq` bytes with
// fixed-stride-4 patching (n_p = 4), using the Table-11/Eq-19..22 formulas.
static double pipeline_flops_per_byte(size_t seq) {
    blt_flops_config fc;
    memset(&fc, 0, sizeof(fc));
    fc.h_G = BENCH_E;
    fc.l_G = BENCH_L;
    fc.n_ctx = seq / 4;
    fc.global_heads = 4;
    fc.global_head_dim = BENCH_E / 4;
    fc.d_ff_G = 4;
    fc.h_E = BENCH_E;
    fc.l_E = BENCH_L;
    fc.w_E = 0;
    fc.enc_heads = 4;
    fc.enc_head_dim = BENCH_E / 4;
    fc.d_ff_E = 4;
    fc.h_D = BENCH_E;
    fc.l_D = BENCH_L;
    fc.w_D = 0;
    fc.dec_heads = 4;
    fc.dec_head_dim = BENCH_E / 4;
    fc.d_ff_D = 4;
    fc.n_p = 4;
    fc.k = 1;
    fc.vocab_size = 256;
    return blt_flops_per_byte(&fc);
}

typedef struct {
    blt_model *model;
    blt_model_grad *grad;
    blt_tensor bytes_in;
    blt_patch_info patches[4096];
    size_t num_patches;
} scenario;

// Model weights/grads land in `persist` and survive resets; all forward/
// backward intermediates go through `scratch`, which callers reset per
// iteration (ops allocate without freeing).
static void scenario_build(scenario *sc, blt_arena *host, blt_arena *persist, blt_arena *scratch, size_t seq,
                           int use_bf16) {
    (void)scratch;
    blt_model_config cfg;
    build_model_config(&cfg, 2048);
    cfg.encoder_config.use_bf16 = use_bf16;
    cfg.decoder_config.use_bf16 = use_bf16;
    cfg.global_config.use_bf16 = use_bf16;

    sc->model = blt_model_create(persist, &cfg);
    uint32_t seed = 0x5EED0000u ^ (uint32_t)seq;
    randomize_model(sc->model, host, &seed);
    blt_model_refresh_bf16(sc->model);
    sc->grad = blt_model_grad_create(persist, sc->model);

    sc->bytes_in = blt_tensor_create(persist, (size_t[1]){seq}, 1, BLT_DTYPE_UINT8);
    uint8_t *hb = (uint8_t *)malloc(seq);
    for (size_t i = 0; i < seq; i++) hb[i] = (uint8_t)(prng_next(&seed) & 0xFF);
    blt_tensor_upload(&sc->bytes_in, hb, seq);
    free(hb);

    sc->num_patches = 0;
    for (size_t s = 0; s < seq; s += 4) {
        sc->patches[sc->num_patches].start_idx = s;
        sc->patches[sc->num_patches].length = (s + 4 <= seq) ? 4 : (seq - s);
        sc->patches[sc->num_patches].peak_entropy = 0.5f;
        sc->num_patches++;
    }
}

static void meso_cell(blt_arena *host, blt_arena *arena, size_t seq, size_t iters, const char *backend_tag,
                      const char *results_path) {
    // Cells own their whole scenario; resetting keeps repeated cells from
    // accumulating dead models.
    blt_arena_reset(host);
    blt_arena_reset(arena);
    blt_arena *scratch = blt_arena_create(768 * 1024 * 1024, arena->backend);
    scenario sc;
    scenario_build(&sc, host, arena, scratch, seq, 0);

    bench_timer_start();
    for (size_t i = 0; i < iters + 2; i++) {
        blt_arena_reset(scratch);
        blt_tensor logits = blt_tensor_create(scratch, (size_t[2]){seq, 256}, 2, BLT_DTYPE_FP32);
        blt_tensor loss = blt_tensor_create(scratch, (size_t[1]){1}, 1, BLT_DTYPE_FP32);
        if (i < 2) continue;
        blt_model_forward(sc.model, &sc.bytes_in, NULL, sc.patches, sc.num_patches, NULL, 0, &logits, &loss, scratch);
        backend_sync(&logits);
        if (i == 2) bench_timer_start(); // restart after warmup
    }
    const double sec_fwd = bench_timer_stop_sec() / (double)iters;

    bench_timer_start();
    for (size_t i = 0; i < iters; i++) {
        blt_arena_reset(scratch);
        blt_model_backward(sc.model, &sc.bytes_in, NULL, sc.patches, sc.num_patches, NULL, 0, sc.grad, scratch);
    }
    backend_sync((const blt_tensor *)&sc.model->decoder->lm_head_weight);
    const double sec_bwd = bench_timer_stop_sec() / (double)iters;

    // MFU from the paper FLOPs model: forward counts once, backward ~2x.
    const double fl_per_byte = pipeline_flops_per_byte(seq);
    const double tflops_fwd = fl_per_byte * (double)seq / sec_fwd / 1e12;
    const double tflops_step = 3.0 * fl_per_byte * (double)seq / (sec_fwd + sec_bwd) / 1e12;

    char tag[BENCH_TAG_LEN];
    bench_result r;
    snprintf(tag, sizeof(tag), "meso_pipeline_%zu_%s", seq, backend_tag);
    bench_result_init(&r, "cuda_bench", tag, "7_cuda_bench");
    double lat[1] = {sec_fwd};
    bench_stats_compute(&r.latency, lat, 1);
    bench_result_add_metric(&r, "fwd_ms", 1000.0 * sec_fwd);
    bench_result_add_metric(&r, "bwd_ms", 1000.0 * sec_bwd);
    bench_result_add_metric(&r, "fwd_tflops", tflops_fwd);
    bench_result_add_metric(&r, "steptflops", tflops_step);
    bench_result_add_metric(&r, "mfu_pct", 100.0 * tflops_step / PEAK_TFLOPS_CUDA);
    bench_write_json(results_path, &r);
    printf("  %-28s fwd %8.2f ms  bwd %8.2f ms  step %6.2f TFLOP/s (%5.1f%% MFU)\n", tag, 1000.0 * sec_fwd,
           1000.0 * sec_bwd, tflops_step, 100.0 * tflops_step / PEAK_TFLOPS_CUDA);

    blt_arena_destroy(scratch);
}

static void meso_cell_bf16(blt_arena *host, blt_arena *arena, size_t seq, size_t iters, const char *backend_tag,
                           const char *results_path) {
    blt_arena_reset(host);
    blt_arena_reset(arena);
    blt_arena *scratch = blt_arena_create(768 * 1024 * 1024, arena->backend);
    scenario sc;
    scenario_build(&sc, host, arena, scratch, seq, 1);

    bench_timer_start();
    for (size_t i = 0; i < iters + 2; i++) {
        blt_arena_reset(scratch);
        blt_tensor logits = blt_tensor_create(scratch, (size_t[2]){seq, 256}, 2, BLT_DTYPE_FP32);
        blt_tensor loss = blt_tensor_create(scratch, (size_t[1]){1}, 1, BLT_DTYPE_FP32);
        if (i < 2) continue;
        blt_model_forward(sc.model, &sc.bytes_in, NULL, sc.patches, sc.num_patches, NULL, 0, &logits, &loss, scratch);
        backend_sync(&logits);
        if (i == 2) bench_timer_start();
    }
    const double sec_fwd = bench_timer_stop_sec() / (double)iters;

    bench_timer_start();
    for (size_t i = 0; i < iters; i++) {
        blt_arena_reset(scratch);
        blt_model_backward(sc.model, &sc.bytes_in, NULL, sc.patches, sc.num_patches, NULL, 0, sc.grad, scratch);
    }
    backend_sync((const blt_tensor *)&sc.model->decoder->lm_head_weight);
    const double sec_bwd = bench_timer_stop_sec() / (double)iters;

    const double fl_per_byte = pipeline_flops_per_byte(seq);
    const double tflops_fwd = fl_per_byte * (double)seq / sec_fwd / 1e12;
    const double tflops_step = 3.0 * fl_per_byte * (double)seq / (sec_fwd + sec_bwd) / 1e12;

    char tag[BENCH_TAG_LEN];
    bench_result r;
    snprintf(tag, sizeof(tag), "meso_pipeline_bf16_%zu_%s", seq, backend_tag);
    bench_result_init(&r, "cuda_bench", tag, "7_cuda_bench");
    double lat[1] = {sec_fwd};
    bench_stats_compute(&r.latency, lat, 1);
    bench_result_add_metric(&r, "fwd_ms", 1000.0 * sec_fwd);
    bench_result_add_metric(&r, "bwd_ms", 1000.0 * sec_bwd);
    bench_result_add_metric(&r, "fwd_tflops", tflops_fwd);
    bench_result_add_metric(&r, "steptflops", tflops_step);
    bench_result_add_metric(&r, "mfu_pct", 100.0 * tflops_step / PEAK_TFLOPS_CUDA);
    bench_write_json(results_path, &r);
    printf("  %-28s fwd %8.2f ms  bwd %8.2f ms  step %6.2f TFLOP/s (%5.1f%% MFU)\n", tag, 1000.0 * sec_fwd,
           1000.0 * sec_bwd, tflops_step, 100.0 * tflops_step / PEAK_TFLOPS_CUDA);

    blt_arena_destroy(scratch);
}

static void scenario_build_production(scenario *sc, blt_arena *host, blt_arena *persist, blt_arena *scratch,
                                      size_t seq) {
    (void)scratch;
    blt_model_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.encoder_config.embed_dim = 192;
    cfg.encoder_config.patch_dim = 0;
    cfg.encoder_config.num_layers = 2;
    cfg.encoder_config.hidden_dim = 384;
    cfg.encoder_config.num_heads = 4;
    cfg.encoder_config.cross_attn_heads = 4;
    cfg.encoder_config.cross_attn_all_layers = true;
    cfg.encoder_config.pool_type = BLT_POOL_MEAN;
    cfg.encoder_config.rope_theta = 500000.0f;
    cfg.encoder_config.max_seq_len = seq;
    cfg.encoder_config.ngram_config.ngram_sizes[0] = 3;
    cfg.encoder_config.ngram_config.ngram_sizes[1] = 4;
    cfg.encoder_config.ngram_config.num_ngram_sizes = 2;
    cfg.encoder_config.ngram_config.per_ngram_vocab = 50;
    cfg.encoder_config.ngram_config.hash_prime = 1000000007ULL;
    cfg.encoder_config.ngram_config.normalize = true;
    cfg.encoder_config.ngram_config.embed_dim = 192;
    cfg.global_config.embed_dim = 192;
    cfg.global_config.num_layers = 2;
    cfg.global_config.hidden_dim = 384;
    cfg.global_config.num_heads = 4;
    cfg.global_config.rope_theta = 500000.0f;
    cfg.global_config.max_seq_len = seq;
    cfg.decoder_config.embed_dim = 192;
    cfg.decoder_config.patch_dim = 0;
    cfg.decoder_config.num_layers = 2;
    cfg.decoder_config.hidden_dim = 384;
    cfg.decoder_config.num_heads = 4;
    cfg.decoder_config.cross_attn_heads = 4;
    cfg.decoder_config.cross_attn_all_layers = true;
    cfg.decoder_config.rope_theta = 500000.0f;
    cfg.decoder_config.max_seq_len = seq;
    cfg.decoder_config.vocab_size = 256;

    sc->model = blt_model_create(persist, &cfg);
    uint32_t seed = 0x5EED0000u ^ (uint32_t)seq;
    randomize_model(sc->model, host, &seed);
    sc->grad = blt_model_grad_create(persist, sc->model);

    sc->bytes_in = blt_tensor_create(persist, (size_t[1]){seq}, 1, BLT_DTYPE_UINT8);
    uint8_t *hb = (uint8_t *)malloc(seq);
    for (size_t i = 0; i < seq; i++) hb[i] = (uint8_t)(prng_next(&seed) & 0xFF);
    blt_tensor_upload(&sc->bytes_in, hb, seq);
    free(hb);

    sc->num_patches = 0;
    for (size_t s = 0; s < seq; s += 4) {
        sc->patches[sc->num_patches].start_idx = s;
        sc->patches[sc->num_patches].length = (s + 4 <= seq) ? 4 : (seq - s);
        sc->patches[sc->num_patches].peak_entropy = 0.5f;
        sc->num_patches++;
    }
}

static void macro_training_cell(blt_arena *host, blt_arena *arena, size_t seq, size_t steps, const char *backend_tag,
                                const char *results_path) {
    blt_arena_reset(host);
    blt_arena_reset(arena);
    blt_arena *scratch = blt_arena_create(768 * 1024 * 1024, arena->backend);
    scenario sc;
    scenario_build_production(&sc, host, arena, scratch, seq);

    bench_timer_start();
    for (size_t i = 0; i < steps + 2; i++) {
        blt_arena_reset(scratch);
        blt_tensor logits = blt_tensor_create(scratch, (size_t[2]){seq, 256}, 2, BLT_DTYPE_FP32);
        blt_tensor loss = blt_tensor_create(scratch, (size_t[1]){1}, 1, BLT_DTYPE_FP32);
        if (i < 2) continue; // warmup, unmeasured
        blt_model_forward(sc.model, &sc.bytes_in, NULL, sc.patches, sc.num_patches, NULL, 0, &logits, &loss, scratch);
        blt_model_backward(sc.model, &sc.bytes_in, NULL, sc.patches, sc.num_patches, NULL, 0, sc.grad, scratch);
        if (i == 2) bench_timer_start(); // restart after warmup
    }
    backend_sync((const blt_tensor *)&sc.model->decoder->lm_head_weight);
    const double sec_total = bench_timer_stop_sec();

    const double tokens_per_s = (double)(seq * steps) / sec_total;

    char tag[BENCH_TAG_LEN];
    snprintf(tag, sizeof(tag), "macro_train_%zu_%s", seq, backend_tag);
    bench_result r;
    bench_result_init(&r, "cuda_bench", tag, "7_cuda_bench");
    double lat[1] = {sec_total};
    bench_stats_compute(&r.latency, lat, 1);
    bench_result_add_metric(&r, "tokens_per_s", tokens_per_s);
    bench_result_add_metric(&r, "steps", (double)steps);
    bench_write_json(results_path, &r);
    printf("  %-28s %10.0f tokens/s (fwd+bwd core)\n", tag, tokens_per_s);

    blt_arena_destroy(scratch);
}

//----------------------------------------------------------------------
// Macro: greedy generation on the real trained checkpoint
//----------------------------------------------------------------------

static void macro_generation_cell(blt_arena *arena, const bench_args *a, const char *backend_tag,
                                  const char *results_path) {
    blt_arena_reset(arena);
    FILE *f = fopen(a->ckpt, "rb");
    if (!f) {
        printf("  macro_gen skipped: %s not found\n", a->ckpt);
        return;
    }
    fclose(f);

    blt_model_config cfg;
    build_model_config(&cfg, 512);
    cfg.encoder_config.embed_dim = 192;
    cfg.encoder_config.hidden_dim = 384;
    cfg.encoder_config.ngram_config.embed_dim = 192;
    cfg.global_config.embed_dim = 192;
    cfg.global_config.hidden_dim = 384;
    cfg.decoder_config.embed_dim = 192;
    cfg.decoder_config.hidden_dim = 384;
    cfg.decoder_config.vocab_size = 256;
    cfg.encoder_config.ngram_config.ngram_sizes[0] = 3;
    cfg.encoder_config.ngram_config.ngram_sizes[1] = 4;
    cfg.encoder_config.ngram_config.num_ngram_sizes = 2;
    cfg.encoder_config.ngram_config.per_ngram_vocab = 50;
    cfg.encoder_config.ngram_config.hash_prime = 1000000007ULL;
    cfg.encoder_config.ngram_config.normalize = true;

    blt_arena *load_arena = blt_arena_create(64 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_model *model;
    if (arena->backend == BLT_BACKEND_CPU) {
        model = blt_model_create(arena, &cfg);
        blt_model_load(model, a->ckpt);
    } else {
        blt_model *twin = blt_model_create(load_arena, &cfg);
        blt_model_load(twin, a->ckpt);
        model = blt_model_create(arena, &cfg);
        const size_t n = blt_model_num_tensors(twin);
        float *stage = NULL;
        size_t cap = 0;
        for (size_t i = 0; i < n; i++) {
            const char *name;
            blt_tensor *ts;
            const char *nd;
            blt_tensor *td;
            blt_model_tensor_at(twin, i, &name, &ts);
            blt_model_tensor_at(model, i, &nd, &td);
            if (cap < ts->numel) {
                free(stage);
                cap = ts->numel;
                stage = (float *)malloc(cap * sizeof(float));
            }
            blt_tensor_download(ts, stage, blt_tensor_bytes(ts));
            blt_tensor_upload(td, stage, blt_tensor_bytes(td));
        }
        free(stage);
    }
    blt_arena_destroy(load_arena);

    // Synthetic prompt; only timing matters here, not text quality.
    const size_t prompt_len = 64;
    const size_t new_bytes = 48;
    uint8_t prompt[128];
    for (size_t i = 0; i < prompt_len; i++) prompt[i] = (uint8_t)(i * 31 + 7);

    blt_patch_info dummy[4096];
    blt_arena *scratch = blt_arena_create(256 * 1024 * 1024, arena->backend);
    uint8_t out[128];
    memcpy(out, prompt, prompt_len);

    bench_timer_start();
    for (size_t b = 0; b < new_bytes; b++) {
        const size_t total = prompt_len + b;
        size_t nd = 0;
        for (size_t s = 0; s < total; s += 4) {
            size_t len = 4;
            if (s + len > total) len = total - s;
            dummy[nd].start_idx = s;
            dummy[nd].length = len;
            dummy[nd].peak_entropy = 0.5f;
            nd++;
        }
        blt_tensor bytes_in = blt_tensor_create(scratch, (size_t[1]){total}, 1, BLT_DTYPE_UINT8);
        blt_tensor_upload(&bytes_in, out, total);
        blt_tensor logits = blt_tensor_create(scratch, (size_t[2]){total, 256}, 2, BLT_DTYPE_FP32);
        blt_tensor loss = blt_tensor_create(scratch, (size_t[1]){1}, 1, BLT_DTYPE_FP32);
        blt_model_forward(model, &bytes_in, NULL, dummy, nd, NULL, 0, &logits, &loss, scratch);
        float last[256];
        blt_tensor row_view;
        view_1d(&row_view, (float *)logits.data + b * 256, 256, BLT_DTYPE_FP32, logits.backend);
        blt_tensor_download(&row_view, last, 256 * sizeof(float));
        // argmax on host
        size_t best = 0;
        for (size_t v = 1; v < 256; v++)
            if (last[v] > last[best]) best = v;
        out[prompt_len + b] = (uint8_t)best;
        scratch->offset = 0;
    }
    const double sec = bench_timer_stop_sec();
    const double nfe_per_s = (double)new_bytes / sec;

    char tag[BENCH_TAG_LEN];
    snprintf(tag, sizeof(tag), "macro_gen_greedy_%s", backend_tag);
    bench_result r;
    bench_result_init(&r, "cuda_bench", tag, "7_cuda_bench");
    double lat[1] = {sec};
    bench_stats_compute(&r.latency, lat, 1);
    bench_result_add_metric(&r, "nfe_per_s", nfe_per_s);
    bench_result_add_metric(&r, "new_bytes", (double)new_bytes);
    bench_write_json(results_path, &r);
    printf("  %-28s %10.1f decoder passes/s (naive greedy)\n", tag, nfe_per_s);

    blt_arena_destroy(scratch);
}

int main(int argc, char **argv) {
    bench_args a;
    memset(&a, 0, sizeof(a));
    a.iters = 10;
    a.results_path = "bench/results.jsonl";
    a.ckpt = "runs/followup1_base_s104.fblt";
#ifdef BLT_WITH_CUDA
    a.use_cuda = 1;
#endif
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            ++i;
            if (!strcmp(argv[i], "cuda")) a.use_cuda = 1;
            else if (!strcmp(argv[i], "cpu")) a.use_cuda = 0;
            else BLT_FATAL("cuda_bench: unknown --backend '%s'", argv[i]);
        } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) a.iters = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--results") && i + 1 < argc) a.results_path = argv[++i];
        else if (!strcmp(argv[i], "--ckpt") && i + 1 < argc) a.ckpt = argv[++i];
        else if (!strcmp(argv[i], "--macro-only")) a.macro_only = 1;
    }
#ifndef BLT_WITH_CUDA
    if (a.use_cuda) BLT_FATAL("cuda_bench: --backend cuda requires a CUDA=1 build");
#endif

    const char *bt = a.use_cuda ? "cuda" : "cpu";
    printf("[CUDA-BENCH] backend %s\n", bt);

    blt_arena *host = blt_arena_create(256 * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *arena = blt_arena_create(a.use_cuda ? 512 * 1024 * 1024 : 512 * 1024 * 1024,
                                        a.use_cuda ? BLT_BACKEND_CUDA : BLT_BACKEND_CPU);

    if (!a.macro_only) {
        printf(" micro\n");
        bench_elementwise_cell(arena, 1u << 24, a.iters > 20 ? 20 : a.iters, bt, a.results_path);
        bench_matmul_cell(arena, 512, a.iters * 4, bt, a.results_path);
        bench_matmul_cell(arena, 1024, a.iters * 2, bt, a.results_path);
        bench_matmul_cell(arena, 2048, a.iters, bt, a.results_path);
        bench_matmul_bf16_cell(arena, 512, a.iters * 4, bt, a.results_path);
        bench_matmul_bf16_cell(arena, 1024, a.iters * 2, bt, a.results_path);
        bench_matmul_bf16_cell(arena, 2048, a.iters, bt, a.results_path);

        printf(" meso (pipeline fwd/bwd, E=%d L=%d)\n", BENCH_E, BENCH_L);
        meso_cell(host, arena, 256, a.iters, bt, a.results_path);
        meso_cell(host, arena, 1024, a.iters, bt, a.results_path);
        meso_cell_bf16(host, arena, 256, a.iters, bt, a.results_path);
        meso_cell_bf16(host, arena, 1024, a.iters, bt, a.results_path);
    }

    printf(" macro\n");
    macro_training_cell(host, arena, 1024, a.iters, bt, a.results_path);
    macro_generation_cell(arena, &a, bt, a.results_path);

    printf("[CUDA-BENCH] results appended to %s\n", a.results_path);
    blt_arena_destroy(arena);
    blt_arena_destroy(host);
    return 0;
}
