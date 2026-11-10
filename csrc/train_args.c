#include "train_args.h"
#include "core/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr, "usage: bin/train_blt_d --corpus FILE [options]\n"
                    "\n"
                    "required:\n"
                    "  --corpus FILE           raw byte corpus\n"
                    "\n"
                    "training:\n"
                    "  --steps N               optimizer steps (default 2000)\n"
                    "  --lr F                  learning rate (default 0.0001)\n"
                    "  --optimizer sgd|adamw   optimizer (default adamw)\n"
                    "  --beta1 F               AdamW first moment decay (default 0.9)\n"
                    "  --beta2 F               AdamW second moment decay (default 0.999)\n"
                    "  --eps F                 AdamW epsilon (default 1e-8)\n"
                    "  --weight-decay F        AdamW weight decay (default 0.01)\n"
                    "  --max-norm F            gradient clip threshold (default 20000)\n"
                    "  --grad-accum N          gradient accumulation steps: N micro-batches are averaged\n"
                    "                          per optimizer step (default 1)\n"
                    "  --seed S                RNG seed (default 7)\n"
                    "  --backend cpu|cuda      backend (default cpu)\n"
                    "  --dtype fp32|bf16       weight/activation precision (default fp32); --bf16 is an alias\n"
                    "  --cuda-scratch-mb N     CUDA kernel scratch pool size in MB\n"
                    "  --scratch-mb N           trainer scratch arena size in MB; 0 derives it from --window\n"
                    "                          (see docs/CLI_REFERENCE.md for the memory cost of long windows)\n"
                    "  --model-mb N           CUDA model arena size in MB (default 1024)\n"
                    "  --deterministic         bit-reproducible training (slower)\n"
                    "\n"
                    "model:\n"
                    "  --embed E               model width (default 64)\n"
                    "  --hidden H              FFN width (default 128)\n"
                    "  --layers L              layers per submodule (default 2)\n"
                    "  --enc-layers N          encoder layers (overrides --layers)\n"
                    "  --glob-layers N         global transformer layers (overrides --layers)\n"
                    "  --dec-layers N          decoder layers (overrides --layers)\n"
                    "  --cross-attn all|last   cross-attention placement (default all)\n"
                    "  --d0-mode zeros|learned decoder d0 init (default learned)\n"
                    "\n"
                    "diffusion:\n"
                    "  --diffusion 0|1         enable BLT-D masked diffusion (default 1)\n"
                    "  --t-min F               diffusion timestep floor (default 0.1)\n"
                    "  --t-warmup-hi F         high-t curriculum fraction (default 0, disabled)\n"
                    "  --t-hi-start F          high-t curriculum start floor (default 0.8)\n"
                    "  --block-size B          diffusion block size (default 4)\n"
                    "  --window W              clean-sequence length (default 48)\n"
                    "\n"
                    "mask schedule:\n"
                    "  --mask-warmup N         ramp mask scale 0->1 over N steps (default 0)\n"
                    "  --mask-scale F          mask loss ceiling (default 1.0)\n"
                    "  --last-row-scale F      last-row L_clean upweight (default 1.0)\n"
                    "  --mask-late-step N      step to begin raising mask cap (default 0, disabled)\n"
                    "  --mask-late-scale F     final mask cap after ramp (default 1.0)\n"
                    "\n"
                    "learning rate schedule:\n"
                    "  --lr-decay 0|1          x0.3 at 60%% and 85%% of steps (default 0)\n"
                    "  --lr-decay-steps LIST   custom decay step positions (CSV)\n"
                    "  --lr-decay-factor F     factor per custom decay point (default 0.3)\n"
                    "\n"
                    "telemetry:\n"
                    "  --peak-tflops F         peak TFLOP/s for MFU; wins over the device table\n"
                    "  --peak-bandwidth F      peak HBM GB/s for MBU; wins over the device table\n"
                    "\n"
                    "entropy patching:\n"
                    "  --entropy-patches       enable entropy-LM patcher segmentation\n"
                    "  --train-entropy-lm FILE train entropy LM and save to FILE\n"
                    "  --entropy-lm FILE       load pretrained entropy-LM weights\n"
                    "\n"
                    "evaluation:\n"
                    "  --eval-corpus FILE      held-out corpus for causal BPB eval\n"
                    "  --eval-windows N        eval window count (default 200)\n"
                    "  --eval-skip N           bytes to skip before first eval window (default 0)\n"
                    "  --eval-every N          run eval every N steps (default 0, disabled)\n"
                    "\n"
                    "I/O:\n"
                    "  --save-weights PATH     write weights after training (first %%d is replaced by step)\n"
                    "  --save-every N          save checkpoint every N steps (0 = only at end)\n"
                    "  --load-weights PATH     load weights before training\n"
                    "  --save-optim PATH       write AdamW moments + step counter alongside the weights\n"
                    "  --load-optim PATH       restore AdamW moments + step counter (needs --load-weights)\n"
                    "  --start-step N          absolute step to start at; defaults to the value stored in\n"
                    "                         --load-optim, else 0\n"
                    "  --report-every K        print every K steps (default 25)\n"
                    "\n"
                    "logging (diagnostic, all optional):\n"
                    "  --grad-norm-log FILE        pre-clip gradient norms per step\n"
                    "  --update-norm-log FILE      post-clip update norms at spike steps\n"
                    "  --component-norm-log FILE   per-component gradient norms per step\n"
                    "  --activation-dump-log FILE  activation stats every report-every steps\n"
                    "  --batch-log FILE            batch properties per step\n"
                    "  --loss-log FILE             per-optimizer-step loss (micro-batch average)\n");
    exit(1);
}

args_t parse_args(int argc, char **argv) {
    args_t a = {.corpus_path = NULL,
                .steps = 2000,
                .lr = 0.0001f,
                .block_size = 4,
                .window = 48,
                .grad_accum = 1,
                .embed = 64,
                .hidden = 128,
                .layers = 2,
                .d0_learned = 1,
                .seed = 7,
                .report_every = 25,
                .diffusion = 1,
                .eval_path = NULL,
                .eval_windows = 200,
                .save_path = NULL,
                .save_every = 0,
                .save_best_path = NULL,
                .best_metric = 0,
                .save_best_optim_path = NULL,
                .load_path = NULL,
                .eval_skip = 0,
                .monitor_every = 2000,
                .monitor_windows = 24,
                .t_min = 0.1f,
                .lr_decay = 0,
                .lr_decay_count = 0,
                .lr_decay_factor = 0.3f,
                .warmup_steps = 0,
                .lr_schedule = 0,
                .min_lr = -1.0f,
                .lr_schedule_given = 0,
                .mask_warmup = 0,
                .mask_scale = 1.0f,
                .last_row_scale = 1.0f,
                .mask_loss_norm = 0,
                .paper_loss = 1,
                .mask_late_step = 0,
                .mask_late_scale = 1.0f,
                .entropy_patches = 0,
                .train_entlm = NULL,
                .entropy_lm = NULL,
                .use_cuda = 0,
                .enc_layers = 0,
                .glob_layers = 0,
                .dec_layers = 0,
                .cross_last = 0,
                .t_warmup_frac = 0.0f,
                .t_hi_start = 0.8f,
                .optimizer = 1,
                .beta1 = 0.9f,
                .beta2 = 0.999f,
                .eps = 1e-8f,
                .weight_decay = 0.01f,
                .weight_decay_norms = 0,
                .max_norm = 20000.0f,
                .grad_norm_log = NULL,
                .update_norm_log = NULL,
                .component_norm_log = NULL,
                .activation_dump_log = NULL,
                .batch_log = NULL,
                .eval_every = 0,
                .loss_log = NULL,
                .metrics_log = NULL,
                .deterministic = 0,
                .bf16 = 0,
                .flash_attn = 0,
                .cuda_scratch_mb = 0,
                .scratch_mb = 0,
                .model_mb = 0,
                .peak_tflops = 0.0f,
                .peak_bandwidth = 0.0f,
                .doc_boundaries = "off"};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--corpus") && i + 1 < argc) a.corpus_path = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) a.steps = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) a.lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--block-size") && i + 1 < argc) a.block_size = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--window") && i + 1 < argc) a.window = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--grad-accum") && i + 1 < argc) a.grad_accum = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--embed") && i + 1 < argc) a.embed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--hidden") && i + 1 < argc) a.hidden = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) a.layers = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--d0-mode") && i + 1 < argc) a.d0_learned = !strcmp(argv[++i], "learned");
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) a.seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--report-every") && i + 1 < argc) a.report_every = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--diffusion") && i + 1 < argc) a.diffusion = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save-weights") && i + 1 < argc) a.save_path = argv[++i];
        else if (!strcmp(argv[i], "--save-every") && i + 1 < argc) a.save_every = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--save-best") && i + 1 < argc) a.save_best_path = argv[++i];
        else if (!strcmp(argv[i], "--best-metric") && i + 1 < argc) {
            if (!strcmp(argv[i + 1], "bpb")) a.best_metric = 0;
            else if (!strcmp(argv[i + 1], "masked_acc")) a.best_metric = 1;
            else BLT_FATAL("train_blt_d: --best-metric must be bpb or masked_acc");
            i++;
        } else if (!strcmp(argv[i], "--save-best-optim") && i + 1 < argc) a.save_best_optim_path = argv[++i];
        else if (!strcmp(argv[i], "--load-weights") && i + 1 < argc) a.load_path = argv[++i];
        else if (!strcmp(argv[i], "--save-optim") && i + 1 < argc) a.save_optim_path = argv[++i];
        else if (!strcmp(argv[i], "--load-optim") && i + 1 < argc) a.load_optim_path = argv[++i];
        else if (!strcmp(argv[i], "--start-step") && i + 1 < argc) {
            a.start_step = strtoull(argv[++i], NULL, 10);
            a.start_step_given = 1;
        } else if (!strcmp(argv[i], "--eval-corpus") && i + 1 < argc) a.eval_path = argv[++i];
        else if (!strcmp(argv[i], "--eval-windows") && i + 1 < argc) a.eval_windows = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--eval-skip") && i + 1 < argc) a.eval_skip = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--monitor-every") && i + 1 < argc) a.monitor_every = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--monitor-windows") && i + 1 < argc)
            a.monitor_windows = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--eval-every") && i + 1 < argc) a.eval_every = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--t-min") && i + 1 < argc) a.t_min = atof(argv[++i]);
        else if (!strcmp(argv[i], "--lr-decay") && i + 1 < argc) a.lr_decay = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr-decay-steps") && i + 1 < argc) {
            char *tok = strtok(argv[++i], ",");
            while (tok && a.lr_decay_count < 16) {
                a.lr_decay_steps[a.lr_decay_count++] = strtoull(tok, NULL, 10);
                tok = strtok(NULL, ",");
            }
        } else if (!strcmp(argv[i], "--lr-decay-factor") && i + 1 < argc) a.lr_decay_factor = atof(argv[++i]);
        else if (!strcmp(argv[i], "--warmup-steps") && i + 1 < argc) a.warmup_steps = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--lr-schedule") && i + 1 < argc) {
            if (!strcmp(argv[i + 1], "cosine")) a.lr_schedule = 1;
            else if (!strcmp(argv[i + 1], "step")) a.lr_schedule = 0;
            else BLT_FATAL("train_blt_d: --lr-schedule must be step or cosine");
            a.lr_schedule_given = 1;
            i++;
        } else if (!strcmp(argv[i], "--min-lr") && i + 1 < argc) {
            a.min_lr = atof(argv[++i]);
            a.min_lr = a.min_lr; // keep as given (>=0 means specified)
        } else if (!strcmp(argv[i], "--mask-warmup") && i + 1 < argc) a.mask_warmup = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--mask-scale") && i + 1 < argc) a.mask_scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--mask-loss-norm") && i + 1 < argc) a.mask_loss_norm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--paper-loss") && i + 1 < argc) a.paper_loss = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--last-row-scale") && i + 1 < argc) a.last_row_scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--mask-late-step") && i + 1 < argc) a.mask_late_step = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--mask-late-scale") && i + 1 < argc) a.mask_late_scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--entropy-patches")) a.entropy_patches = 1;
        else if (!strcmp(argv[i], "--train-entropy-lm") && i + 1 < argc) a.train_entlm = argv[++i];
        else if (!strcmp(argv[i], "--entropy-lm") && i + 1 < argc) a.entropy_lm = argv[++i];
        else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            ++i;
            if (!strcmp(argv[i], "cuda")) a.use_cuda = 1;
            else if (!strcmp(argv[i], "cpu")) a.use_cuda = 0;
            else {
                fprintf(stderr, "unknown --backend '%s'\n", argv[i]);
                exit(1);
            }
        } else if (!strcmp(argv[i], "--enc-layers") && i + 1 < argc) a.enc_layers = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--glob-layers") && i + 1 < argc) a.glob_layers = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--dec-layers") && i + 1 < argc) a.dec_layers = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--cross-attn") && i + 1 < argc) {
            ++i;
            if (!strcmp(argv[i], "all")) a.cross_last = 0;
            else if (!strcmp(argv[i], "last")) a.cross_last = 1;
            else {
                fprintf(stderr, "unknown --cross-attn '%s'\n", argv[i]);
                exit(1);
            }
        } else if (!strcmp(argv[i], "--t-warmup-hi") && i + 1 < argc) a.t_warmup_frac = atof(argv[++i]);
        else if (!strcmp(argv[i], "--t-hi-start") && i + 1 < argc) a.t_hi_start = atof(argv[++i]);
        else if (!strcmp(argv[i], "--optimizer") && i + 1 < argc) {
            ++i;
            if (!strcmp(argv[i], "sgd")) a.optimizer = 0;
            else if (!strcmp(argv[i], "adamw")) a.optimizer = 1;
            else {
                fprintf(stderr, "unknown --optimizer '%s'\n", argv[i]);
                exit(1);
            }
        } else if (!strcmp(argv[i], "--beta1") && i + 1 < argc) a.beta1 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--beta2") && i + 1 < argc) a.beta2 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--eps") && i + 1 < argc) a.eps = atof(argv[++i]);
        else if (!strcmp(argv[i], "--weight-decay") && i + 1 < argc) a.weight_decay = atof(argv[++i]);
        else if (!strcmp(argv[i], "--weight-decay-norms")) a.weight_decay_norms = 1;
        else if (!strcmp(argv[i], "--grad-norm-log") && i + 1 < argc) a.grad_norm_log = argv[++i];
        else if (!strcmp(argv[i], "--max-norm") && i + 1 < argc) a.max_norm = atof(argv[++i]);
        else if (!strcmp(argv[i], "--update-norm-log") && i + 1 < argc) a.update_norm_log = argv[++i];
        else if (!strcmp(argv[i], "--component-norm-log") && i + 1 < argc) a.component_norm_log = argv[++i];
        else if (!strcmp(argv[i], "--activation-dump-log") && i + 1 < argc) a.activation_dump_log = argv[++i];
        else if (!strcmp(argv[i], "--batch-log") && i + 1 < argc) a.batch_log = argv[++i];
        else if (!strcmp(argv[i], "--loss-log") && i + 1 < argc) a.loss_log = argv[++i];
        else if (!strcmp(argv[i], "--metrics-log") && i + 1 < argc) a.metrics_log = argv[++i];
        else if (!strcmp(argv[i], "--scratch-mb") && i + 1 < argc) a.scratch_mb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--cuda-scratch-mb") && i + 1 < argc)
            a.cuda_scratch_mb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--model-mb") && i + 1 < argc) a.model_mb = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--peak-tflops") && i + 1 < argc) a.peak_tflops = atof(argv[++i]);
        else if (!strcmp(argv[i], "--peak-bandwidth") && i + 1 < argc) a.peak_bandwidth = atof(argv[++i]);
        else if (!strcmp(argv[i], "--doc-boundaries") && i + 1 < argc)
            snprintf(a.doc_boundaries, sizeof(a.doc_boundaries), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--deterministic")) a.deterministic = 1;
        else if (!strcmp(argv[i], "--dtype") && i + 1 < argc) {
            ++i;
            if (!strcmp(argv[i], "bf16")) a.bf16 = 1;
            else if (!strcmp(argv[i], "fp32")) a.bf16 = 0;
            else {
                fprintf(stderr, "train_blt_d: unknown --dtype '%s' (expected fp32 or bf16)\n", argv[i]);
                usage();
            }
        } else if (!strcmp(argv[i], "--bf16")) a.bf16 = 1;
        else if (!strcmp(argv[i], "--fp8")) a.fp8 = 1;
        else if (!strcmp(argv[i], "--flash-attn")) a.flash_attn = 1;
        else {
            usage();
        }
    }
    if (a.corpus_path == NULL) {
        usage();
    }

    // --deterministic guardrails
    if (a.deterministic) {
#if defined(BLT_WITH_CUDA)
        fprintf(stderr, "deterministic mode: single-threaded scatter_add + pedantic cuBLAS, expect ~3-10x slowdown, "
                        "not for production training\n");
#endif
        // CPU backend is always deterministic; flag is a no-op there.
    }

    return a;
}
