// Corpus BPB evaluator: scores held-out windows with the causal decoder and,
// when diffusion is enabled, the masked block cells. Emits one reproducible
// JSON record so runs can be compared across checkpoints and protocols.
//
// Usage: eval_bpb --checkpoint FILE --corpus FILE [options]

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/allocator.h"
#include "core/backend.h"
#include "core/json.h"
#include "core/tensor.h"
#include "models/block_diffusion.h"
#include "models/checkpoint.h"
#include "models/entropy_lm.h"
#include "models/model.h"
#include "models/model_builder.h"
#include "models/param_visitor.h"
#include "models/patcher.h"
#include "train_eval.h"

typedef struct {
    size_t embed;
    size_t hidden;
    size_t enc_layers;
    size_t glob_layers;
    size_t dec_layers;
    size_t window;
    size_t block_size;
    int cross_last; // 1 = cross-attention only after the final local layer
    int d0_learned; // 1 = learned D_0 table
    int diffusion;  // 1 = score masked block cells
} model_shape;

static long file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

// Quotes and escapes a string for a JSON value.
static void json_write_string(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p != '\0'; p++) {
        switch (*p) {
        case '"':
            fputs("\\\"", f);
            break;
        case '\\':
            fputs("\\\\", f);
            break;
        case '\n':
            fputs("\\n", f);
            break;
        case '\r':
            fputs("\\r", f);
            break;
        case '\t':
            fputs("\\t", f);
            break;
        case '\b':
            fputs("\\b", f);
            break;
        case '\f':
            fputs("\\f", f);
            break;
        default:
            if (*p < 0x20) {
                fprintf(f, "\\u%04x", (unsigned)*p);
            } else {
                fputc((int)*p, f);
            }
            break;
        }
    }
    fputc('"', f);
}

static void count_params_fn(float *tensor, const blt_param_info *info, void *ctx) {
    (void)tensor;
    size_t *n = (size_t *)ctx;
    *n += info->numel;
}

static size_t count_params(const blt_model *model) {
    size_t n = 0;
    blt_model_visit_params(model, NULL, count_params_fn, &n, 0);
    return n;
}

// Reads the shape fields the checkpoint records, leaving anything absent at
// its sentinel. v1 checkpoints carry no metadata at all.
static void resolve_from_metadata(const char *checkpoint, model_shape *shape, int *cross_opt, int *d0_opt,
                                  int *diffusion_opt) {
    char meta[1 << 20];
    meta[0] = '\0';
    if (!blt_model_read_meta(checkpoint, meta, sizeof(meta))) return;

    blt_arena *ma = blt_arena_create(1 << 20, BLT_BACKEND_CPU);
    blt_json_value *root = blt_json_parse(ma, meta);
    const blt_json_value *cfg = blt_json_get(root, "config");
    if (shape->embed == 0) shape->embed = (size_t)blt_json_get_int(cfg, "embed", 0);
    if (shape->hidden == 0) shape->hidden = (size_t)blt_json_get_int(cfg, "hidden", 0);
    if (shape->enc_layers == 0) shape->enc_layers = (size_t)blt_json_get_int(cfg, "enc_layers", 0);
    if (shape->glob_layers == 0) shape->glob_layers = (size_t)blt_json_get_int(cfg, "glob_layers", 0);
    if (shape->dec_layers == 0) shape->dec_layers = (size_t)blt_json_get_int(cfg, "dec_layers", 0);
    if (shape->window == 0) shape->window = (size_t)blt_json_get_int(cfg, "window", 0);
    if (shape->block_size == 0) shape->block_size = (size_t)blt_json_get_int(cfg, "block_size", 0);
    if (*cross_opt < 0) {
        const char *s = blt_json_get_string(cfg, "cross_attn", NULL);
        if (s != NULL) *cross_opt = strcmp(s, "last") == 0 ? 1 : 0;
    }
    if (*d0_opt < 0) {
        const char *s = blt_json_get_string(cfg, "d0_mode", NULL);
        if (s != NULL) *d0_opt = strcmp(s, "zeros") == 0 ? 0 : 1;
    }
    if (*diffusion_opt < 0) {
        const int64_t v = blt_json_get_int(cfg, "diffusion", -1);
        if (v >= 0) *diffusion_opt = (int)v;
    }
    blt_arena_destroy(ma);
}

static int parse_size(const char *s, size_t *out) {
    char *end = NULL;
    const unsigned long long v = strtoull(s, &end, 10);
    if (end == s || *end != '\0') return 0;
    *out = (size_t)v;
    return 1;
}

static void print_usage(void) {
    fprintf(stderr, "Usage: eval_bpb --checkpoint FILE --corpus FILE [options]\n"
                    "  --output-json FILE        write the JSON record here (default stdout)\n"
                    "  --backend cpu|cuda        execution backend (default cpu)\n"
                    "  --embed N --hidden N --enc-layers N --glob-layers N --dec-layers N\n"
                    "                            model shape; defaults come from checkpoint metadata\n"
                    "  --cross-attn all|last     cross-attention placement (default metadata, then all)\n"
                    "  --d0 learned|zeros        block-row D_0 policy (default metadata, then learned)\n"
                    "  --diffusion 0|1           score masked block cells (default metadata, then 1)\n"
                    "  --window W                bytes per window (default metadata, then 512)\n"
                    "  --windows N               windows requested (default 20)\n"
                    "  --skip S                  windows to skip before the first one (default 0)\n"
                    "  --patcher fixed|entropy   segmentation (default fixed)\n"
                    "  --patch-len L             fixed-stride patch length (default 4)\n"
                    "  --entropy-lm FILE         entropy LM weights; required for --patcher entropy\n"
                    "  --entropy-threshold F     global entropy threshold (default 3.0)\n"
                    "  --block-size B            diffusion block size (default metadata, then 4)\n"
                    "  --layout paper|novel      masked layout (default paper)\n"
                    "  --t F                     diffusion timestep for batch masking (default 1.0)\n"
                    "  --seed N                  base RNG seed (default 7)\n"
                    "  --git-commit STR          commit recorded in the JSON record (default unknown)\n");
}

int main(int argc, char **argv) {
    const char *checkpoint = NULL;
    const char *corpus = NULL;
    const char *output_json = NULL;
    const char *entropy_lm_path = NULL;
    const char *git_commit = "unknown";

    int backend_cuda = 0;
    model_shape shape;
    memset(&shape, 0, sizeof(shape));
    int cross_opt = -1;
    int d0_opt = -1;
    int diffusion_opt = -1;
    size_t windows = 20;
    size_t skip = 0;
    int patcher_entropy = 0;
    size_t patch_len = 4;
    double threshold = 3.0;
    int layout_novel = 0;
    double t_value = 1.0;
    unsigned long long seed = 7;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--checkpoint") && i + 1 < argc) {
            checkpoint = argv[++i];
        } else if (!strcmp(arg, "--corpus") && i + 1 < argc) {
            corpus = argv[++i];
        } else if (!strcmp(arg, "--output-json") && i + 1 < argc) {
            output_json = argv[++i];
        } else if (!strcmp(arg, "--backend") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) backend_cuda = 1;
            else if (!strcmp(argv[i], "cpu")) backend_cuda = 0;
            else {
                fprintf(stderr, "eval_bpb: --backend must be cpu|cuda\n");
                return 1;
            }
        } else if (!strcmp(arg, "--embed") && i + 1 < argc) {
            if (!parse_size(argv[++i], &shape.embed)) goto bad_number;
        } else if (!strcmp(arg, "--hidden") && i + 1 < argc) {
            if (!parse_size(argv[++i], &shape.hidden)) goto bad_number;
        } else if (!strcmp(arg, "--enc-layers") && i + 1 < argc) {
            if (!parse_size(argv[++i], &shape.enc_layers)) goto bad_number;
        } else if (!strcmp(arg, "--glob-layers") && i + 1 < argc) {
            if (!parse_size(argv[++i], &shape.glob_layers)) goto bad_number;
        } else if (!strcmp(arg, "--dec-layers") && i + 1 < argc) {
            if (!parse_size(argv[++i], &shape.dec_layers)) goto bad_number;
        } else if (!strcmp(arg, "--cross-attn") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "last")) cross_opt = 1;
            else if (!strcmp(argv[i], "all")) cross_opt = 0;
            else {
                fprintf(stderr, "eval_bpb: --cross-attn must be all|last\n");
                return 1;
            }
        } else if (!strcmp(arg, "--d0") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "learned")) d0_opt = 1;
            else if (!strcmp(argv[i], "zeros")) d0_opt = 0;
            else {
                fprintf(stderr, "eval_bpb: --d0 must be learned|zeros\n");
                return 1;
            }
        } else if (!strcmp(arg, "--diffusion") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "0")) diffusion_opt = 0;
            else if (!strcmp(argv[i], "1")) diffusion_opt = 1;
            else {
                fprintf(stderr, "eval_bpb: --diffusion must be 0|1\n");
                return 1;
            }
        } else if (!strcmp(arg, "--window") && i + 1 < argc) {
            if (!parse_size(argv[++i], &shape.window)) goto bad_number;
        } else if (!strcmp(arg, "--windows") && i + 1 < argc) {
            if (!parse_size(argv[++i], &windows)) goto bad_number;
        } else if (!strcmp(arg, "--skip") && i + 1 < argc) {
            if (!parse_size(argv[++i], &skip)) goto bad_number;
        } else if (!strcmp(arg, "--patcher") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "entropy")) patcher_entropy = 1;
            else if (!strcmp(argv[i], "fixed")) patcher_entropy = 0;
            else {
                fprintf(stderr, "eval_bpb: --patcher must be fixed|entropy\n");
                return 1;
            }
        } else if (!strcmp(arg, "--patch-len") && i + 1 < argc) {
            if (!parse_size(argv[++i], &patch_len)) goto bad_number;
        } else if (!strcmp(arg, "--entropy-lm") && i + 1 < argc) {
            entropy_lm_path = argv[++i];
        } else if (!strcmp(arg, "--entropy-threshold") && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (!strcmp(arg, "--block-size") && i + 1 < argc) {
            if (!parse_size(argv[++i], &shape.block_size)) goto bad_number;
        } else if (!strcmp(arg, "--layout") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "novel")) layout_novel = 1;
            else if (!strcmp(argv[i], "paper")) layout_novel = 0;
            else {
                fprintf(stderr, "eval_bpb: --layout must be paper|novel\n");
                return 1;
            }
        } else if (!strcmp(arg, "--t") && i + 1 < argc) {
            t_value = atof(argv[++i]);
        } else if (!strcmp(arg, "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(arg, "--git-commit") && i + 1 < argc) {
            git_commit = argv[++i];
        } else {
            fprintf(stderr, "eval_bpb: unknown argument '%s'\n", arg);
            print_usage();
            return 1;
        }
    }

    if (!checkpoint || !corpus) {
        print_usage();
        return 1;
    }
    if (patcher_entropy && entropy_lm_path == NULL) {
        fprintf(stderr, "eval_bpb: --patcher entropy requires --entropy-lm FILE\n");
        return 1;
    }
    if (patch_len < 1) {
        fprintf(stderr, "eval_bpb: --patch-len must be >= 1\n");
        return 1;
    }
    if (windows < 1) {
        fprintf(stderr, "eval_bpb: --windows must be >= 1\n");
        return 1;
    }

#if !defined(BLT_WITH_CUDA)
    if (backend_cuda) {
        fprintf(stderr, "eval_bpb: --backend cuda requires a CUDA build\n");
        return 1;
    }
#endif

    // A missing checkpoint is a usage error, not a fatal checkpoint-format one.
    FILE *cf = fopen(checkpoint, "rb");
    if (!cf) {
        fprintf(stderr, "eval_bpb: cannot open checkpoint '%s'\n", checkpoint);
        return 1;
    }
    fclose(cf);

    resolve_from_metadata(checkpoint, &shape, &cross_opt, &d0_opt, &diffusion_opt);

    if (shape.window == 0) shape.window = 512;
    if (shape.block_size == 0) shape.block_size = 4;
    if (cross_opt < 0) cross_opt = 0;
    if (d0_opt < 0) d0_opt = 1;
    if (diffusion_opt < 0) diffusion_opt = 1;
    shape.cross_last = cross_opt;
    shape.d0_learned = d0_opt;
    shape.diffusion = diffusion_opt;

    if (shape.embed == 0 || shape.hidden == 0 || shape.enc_layers == 0 || shape.glob_layers == 0 ||
        shape.dec_layers == 0) {
        fprintf(stderr, "eval_bpb: checkpoint metadata does not define the model shape; pass --embed --hidden "
                        "--enc-layers --glob-layers --dec-layers (v1 checkpoints carry no metadata)\n");
        return 1;
    }
    if (shape.window < 2) {
        fprintf(stderr, "eval_bpb: --window must be >= 2\n");
        return 1;
    }
    if (shape.block_size < 1) {
        fprintf(stderr, "eval_bpb: --block-size must be >= 1\n");
        return 1;
    }
    if (!patcher_entropy) {
        const size_t m_fixed = (shape.window + patch_len - 1) / patch_len;
        if (m_fixed > BLT_MAX_PATCHES) {
            fprintf(stderr, "eval_bpb: --window %zu --patch-len %zu yields %zu patches > %d cap\n", shape.window,
                    patch_len, m_fixed, BLT_MAX_PATCHES);
            return 1;
        }
    }

    const long corpus_size = file_size(corpus);
    if (corpus_size <= 0) {
        fprintf(stderr, "eval_bpb: cannot read corpus '%s'\n", corpus);
        return 1;
    }
    const size_t corpus_len = (size_t)corpus_size;
    uint8_t *data = (uint8_t *)malloc(corpus_len);
    if (data == NULL) {
        fprintf(stderr, "eval_bpb: corpus staging alloc failed\n");
        return 1;
    }
    FILE *fp = fopen(corpus, "rb");
    if (!fp || fread(data, 1, corpus_len, fp) != corpus_len) {
        fprintf(stderr, "eval_bpb: failed to read corpus '%s'\n", corpus);
        if (fp) fclose(fp);
        free(data);
        return 1;
    }
    fclose(fp);

    const blt_backend dev = backend_cuda ? BLT_BACKEND_CUDA : BLT_BACKEND_CPU;
    blt_arena *model_arena = blt_arena_create(backend_cuda ? 1024ULL * 1024 * 1024 : 256ULL * 1024 * 1024, dev);
    blt_arena *scratch = blt_arena_create(512ULL * 1024 * 1024, dev);
    // The patcher is host-only: the entropy LM and its segmentation arena stay
    // on CPU even under the device backend, matching train_blt_d.
    blt_arena *seg_arena = blt_arena_create(64ULL * 1024 * 1024, BLT_BACKEND_CPU);
    blt_arena *lm_arena = blt_arena_create(64ULL * 1024 * 1024, BLT_BACKEND_CPU);

    // RoPE caches and the decoder row count are sized from max_seq_len; cover
    // the widest possible decoder pass (window + block rows) as train_blt_d does.
    size_t MS = shape.window * (1 + shape.block_size) + 64;
    if (MS < 1024) MS = 1024;

    blt_model_config cfg;
    blt_model_config_defaults(&cfg, shape.embed, shape.hidden, shape.enc_layers, shape.glob_layers, shape.dec_layers,
                              MS, shape.cross_last);
    blt_model *model = blt_model_create(model_arena, &cfg);
    blt_model_load(model, checkpoint);

    blt_entropy_lm *lm = patcher_entropy ? blt_make_entropy_lm(lm_arena, shape.window, entropy_lm_path, 11) : NULL;

    const blt_d0_mode d0m = shape.d0_learned ? BLT_D0_LEARNED : BLT_D0_ZEROS;
    const size_t window = shape.window;
    const size_t block_size = shape.block_size;

    size_t evaluated = 0;
    size_t clean_bytes = 0;
    double causal_sum = 0.0;
    double masked_ce_sum = 0.0;
    size_t masked_cells = 0;
    size_t masked_hits = 0;

    for (size_t w = 0; w < windows; w++) {
        const size_t offset = (skip + w) * window;
        if (offset + window + 1 > corpus_len) break;
        const uint8_t *text = data + offset;

        blt_patch_info patches[BLT_MAX_PATCHES];
        blt_arena_reset(seg_arena);
        const size_t M = patcher_entropy ? entropy_segment_ex(seg_arena, lm, text, window, patches, BLT_MAX_PATCHES,
                                                              (float)threshold)
                                         : fixed_stride(window, patch_len, patches);
        if (M < 2 || M >= BLT_MAX_PATCHES) continue;

        blt_arena_reset(scratch);
        blt_block_batch batch;
        blt_block_batch_build_t(&batch, scratch, text, window, patches, M, block_size, (uint64_t)seed + (uint64_t)w,
                                (float)t_value);

        blt_window_metrics m;
        window_metrics(scratch, model, text, window, &batch, shape.diffusion, d0m, patches, M, &m);
        causal_sum += m.causal_ce * (double)(window - 1);
        clean_bytes += window - 1;

        if (shape.diffusion) {
            if (!layout_novel) {
                if (m.masked_cells > 0) {
                    masked_ce_sum += m.masked_ce * (double)m.masked_cells;
                    masked_cells += m.masked_cells;
                    masked_hits += m.masked_hits;
                }
            } else {
                blt_novel_metrics nm;
                if (window_novel_masked(scratch, seg_arena, model, lm, text, window, patches, M, block_size,
                                        patcher_entropy, d0m, &nm) &&
                    nm.cells > 0) {
                    masked_ce_sum += nm.ce * (double)nm.cells;
                    masked_cells += nm.cells;
                    masked_hits += nm.hits;
                }
            }
        }
        evaluated++;
    }

    if (evaluated == 0) {
        fprintf(stderr, "eval_bpb: no windows evaluated (corpus too small for --window/--skip)\n");
        free(data);
        return 1;
    }

    const double causal_bpb = causal_sum / (double)clean_bytes / log(2.0);
    const int have_masked = shape.diffusion && masked_cells > 0;
    const double masked_bpb = have_masked ? (masked_ce_sum / (double)masked_cells) / log(2.0) : 0.0;
    const double masked_acc = have_masked ? (double)masked_hits / (double)masked_cells : 0.0;
    const size_t params = count_params(model);

    time_t now = time(NULL);
    struct tm tmv;
    gmtime_r(&now, &tmv);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &tmv);

    FILE *out = stdout;
    if (output_json != NULL) {
        out = fopen(output_json, "w");
        if (out == NULL) {
            fprintf(stderr, "eval_bpb: cannot open output '%s'\n", output_json);
            free(data);
            return 1;
        }
    }

    fputs("{\n", out);
    fputs("  \"format\": \"fblt-eval/bpb-v1\",\n", out);
    fputs("  \"task\": \"bpb\",\n", out);
    fputs("  \"checkpoint\": ", out);
    json_write_string(out, checkpoint);
    fputs(",\n  \"corpus\": ", out);
    json_write_string(out, corpus);
    fputs(",\n  \"backend\": ", out);
    json_write_string(out, backend_cuda ? "cuda" : "cpu");
    fputs(",\n", out);

    fprintf(out,
            "  \"protocol\": {\n"
            "    \"window\": %zu,\n"
            "    \"windows_requested\": %zu,\n"
            "    \"windows_evaluated\": %zu,\n"
            "    \"skip\": %zu,\n"
            "    \"patcher\": \"%s\",\n"
            "    \"patch_len\": %zu,\n"
            "    \"entropy_lm\": ",
            window, windows, evaluated, skip, patcher_entropy ? "entropy" : "fixed", patch_len);
    if (patcher_entropy) json_write_string(out, entropy_lm_path);
    else fputs("null", out);
    fputs(",\n    \"entropy_threshold\": ", out);
    if (patcher_entropy) fprintf(out, "%.6f", threshold);
    else fputs("null", out);
    fprintf(out,
            ",\n"
            "    \"block_size\": %zu,\n"
            "    \"layout\": \"%s\",\n"
            "    \"t\": %.6f,\n"
            "    \"seed\": %llu\n"
            "  },\n",
            block_size, layout_novel ? "novel" : "paper", t_value, seed);

    fprintf(out,
            "  \"model\": {\n"
            "    \"embed\": %zu,\n"
            "    \"hidden\": %zu,\n"
            "    \"enc_layers\": %zu,\n"
            "    \"glob_layers\": %zu,\n"
            "    \"dec_layers\": %zu,\n"
            "    \"cross_attn\": \"%s\",\n"
            "    \"d0_mode\": \"%s\",\n"
            "    \"diffusion\": %d,\n"
            "    \"params\": %zu\n"
            "  },\n",
            shape.embed, shape.hidden, shape.enc_layers, shape.glob_layers, shape.dec_layers,
            shape.cross_last ? "last" : "all", shape.d0_learned ? "learned" : "zeros", shape.diffusion, params);

    fprintf(out, "  \"metrics\": {\n    \"causal_bpb\": %.6f,\n    \"masked_bpb\": ", causal_bpb);
    if (have_masked) fprintf(out, "%.6f", masked_bpb);
    else fputs("null", out);
    fputs(",\n    \"masked_acc\": ", out);
    if (have_masked) fprintf(out, "%.6f", masked_acc);
    else fputs("null", out);
    fprintf(out, ",\n    \"clean_bytes\": %zu,\n    \"masked_cells\": ", clean_bytes);
    if (have_masked) fprintf(out, "%zu", masked_cells);
    else fputs("null", out);
    fputs("\n  },\n", out);

    fputs("  \"git_commit\": ", out);
    json_write_string(out, git_commit);
    fputs(",\n  \"timestamp\": ", out);
    json_write_string(out, timestamp);
    fputs("\n}\n", out);

    if (output_json != NULL) fclose(out);
    else fflush(out);

    fprintf(stderr, "[EVAL_BPB] windows=%zu/%zu bytes=%zu causal_bpb=%.4f masked_bpb=", evaluated, windows, clean_bytes,
            causal_bpb);
    if (have_masked) fprintf(stderr, "%.4f", masked_bpb);
    else fputs("null", stderr);
    fputs(" masked_acc=", stderr);
    if (have_masked) fprintf(stderr, "%.4f", masked_acc);
    else fputs("null", stderr);
    fprintf(stderr, " params=%zu\n", params);

    free(data);
    blt_arena_destroy(lm_arena);
    blt_arena_destroy(seg_arena);
    blt_arena_destroy(scratch);
    blt_arena_destroy(model_arena);
    return 0;

bad_number:
    fprintf(stderr, "eval_bpb: expected a non-negative integer argument\n");
    return 1;
}
