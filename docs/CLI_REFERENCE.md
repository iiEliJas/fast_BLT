# CLI Reference

All commands assume the repo root as working directory. Build output goes to `build/` (CPU) or `build-cuda/` (CUDA).

---

## Build & Test

| Command | Description |
|---------|-------------|
| `cmake --build build --target test` | Build + run full C test suite (72 tests, ~4s) |
| `cmake --build build-cuda --target test` | Same, CUDA backend |
| `cmake --build build --target main` | Build minimal main (linking stub) |
| `cmake --build build-cuda --target main` | Same, CUDA |
| `cmake --build build-cuda --target train_blt_d` | Build `train_blt_d` |
| `cmake --build build --target sandbox_run` | Build + run scratch playground |
| `cmake --build build-cuda --target cuda-smoke` | Device sanity check (H2D → kernel → D2H) |
| `cmake --build build-cuda --target cuda-sanitize` | Run tests under `compute-sanitizer` (native Linux) |
| `cmake --build build --target sanity_check` | Build `sanity_check` diagnostic tool |
| `cmake --build build --target last_row_acc` | Build `last_row_acc` diagnostic tool |
| `cmake --build build --target pos_accuracy` | Build `pos_accuracy` diagnostic tool |
| `cmake --build build --target patch_trunc_split` | Build `patch_trunc_split` diagnostic tool |
| `cmake --build build --target patch_final_split` | Build `patch_final_split` diagnostic tool (per-row patch-final accuracy, dump + buckets) |
| `cmake --build build --target eval_paper_rule_split` | Build `eval_paper_rule_split` diagnostic tool (paper-rule decoder cross-attention eval, dump + buckets) |
| `cmake --build build --target eval_isolate_redirect` | Build `eval_isolate_redirect` diagnostic tool (isolated single-patch paper-rule redirect, dump + invariant) |
| `pytest tests/` | Python test suite (config round-trips, YAML, overrides, entry points) |
| `ruff check .` | Python lint |
| `ruff format --check .` | Python format check |

CUDA builds default to architecture `89`. For other GPUs, configure with `-DCMAKE_CUDA_ARCHITECTURES=86` (or the architecture required by the target device). A standalone toolkit at `/usr/local/cuda` is selected when its `bin/nvcc` exists; use `-DCUDAToolkit_ROOT=...` or `-DCMAKE_CUDA_COMPILER=...` to override toolkit selection.

---

## Python Wrapper: `fblt-train`

Installed via `pip install -e .`. Wraps `train_blt_d` with YAML config loading and run-directory management.

```
fblt-train --backend cpu|cuda [--config FILE] [any train_blt_d flag...] [--run-name NAME]
```

| Flag | Required | Description |
|------|----------|-------------|
| `--backend {cpu,cuda}` | yes | Compute backend (never in YAML) |
| `--config FILE` | no | YAML config file (see `configs/train/`) |
| `--run-name NAME` | no | Run directory name (default: `{config}-{timestamp}`) |
| `--override key=value` | no | Escape hatch; a typed flag exists for every field |

The wrapper creates `runs/<run-name>/` with `resolved_config.yaml` containing the full effective config.

Every `train_blt_d` flag is also a flag here, named the same with underscores turned into dashes.
`--embed`, `--hidden`, `--glob-layers`, `--block-size`, `--save-optim`, `--load-optim`,
`--start-step`, `--last-row-scale` and the rest all work directly; `--override key=value` remains
supported but is no longer needed for anything.

Precedence, lowest to highest: dataclass default, then `--config` YAML, then `--override`, then an
explicit flag. `--backend` must always be passed on the command line.

```bash
fblt-train --config configs/train/production.yaml --backend cuda
fblt-train --config configs/train/debug.yaml --backend cpu --override steps=100
```

---

## Python Wrapper: `fblt-infer`

Installed via `pip install -e .`. Wraps `infer` with YAML config loading and auto shape-matching.

```
fblt-infer --backend cpu|cuda --checkpoint FILE [--config FILE] [any infer flag...]
           [--prompt TEXT | --prompt-file FILE] [--output FILE]
```

| Flag | Required | Description |
|------|----------|-------------|
| `--backend {cpu,cuda}` | yes | Compute backend (never in YAML) |
| `--checkpoint FILE` | yes | Model checkpoint (.fblt) |
| `--config FILE` | no | YAML config file (see `configs/infer/`) |
| `--prompt TEXT` | no | Prompt text (mutually exclusive with --prompt-file) |
| `--prompt-file FILE` | no | Prompt file (mutually exclusive with --prompt) |
| `--output FILE` | no | Output file (default: stdout) |
| `--override key=value` | no | Escape hatch; a typed flag exists for every field |

Every `infer` flag is also a flag here, named the same with underscores turned into dashes:
`--method`, `--block-size`, `--entropy-lm`, `--temperature`, `--top-p`, `--repeat-penalty`,
`--no-repeat-ngram`, `--k`, `--unmask`, `--threshold`, and the rest.

```bash
fblt-infer --checkpoint runs/debug_checkpoint.fblt --backend cpu --prompt "int main"
fblt-infer --checkpoint my_model.fblt --backend cpu \
  --embed 192 --hidden 384 --enc-layers 2 --glob-layers 2 --dec-layers 2
```

If no `--prompt` or `--prompt-file`, reads from stdin. Auto-detects model dimensions from `resolved_config.yaml` next to the checkpoint (when produced by `fblt-train`), so the shape flags are usually unnecessary.

---

## Training: `train_blt_d`

**Binary**: `build/train_blt_d` (CPU) / `build-cuda/train_blt_d` (CUDA)

### Core

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus FILE` | *required* | Raw byte corpus |
| `--steps N` | 2000 | Optimizer steps |
| `--lr F` | 0.0001 | Learning rate |
| `--block-size B` | 4 | Diffusion block size |
| `--window W` | 48 | Clean-sequence length per example (8 to 2048) |

A window above 512 needs roughly 400 patches instead of 98, so expect several times the activation
memory and slower steps. `configs/train/tinystories_1k_phase1.yaml` and
`configs/train/tinystories_1k_continue.yaml` are working examples at window 1024.

### Memory cost of long windows

Scratch grows **faster than the window**, because diffusion scores every patch in the window at
once: the decoder sees `S = window * (1 + block_size)` rows, and attention over `S` is quadratic in
`S`. Measured on the 10M TinyStories shape (E=256, H=512, 2/6/2, `B=4`):

| `--window` | derived scratch | peak GPU (RTX 4060, 8 GB) |
|---:|---:|---|
| 512 | 512 MB | ~1.5 GB |
| 1024 | 1 GB | 4867 MiB, about 59% |
| 2048 | 4 GB | 7851 MiB of 8188, **about 96%** |

Window 2048 fits on an 8 GB card with almost no headroom. Window 1024 is the largest size with
real margin. Anything larger, a wider model, or a bigger `--cuda-scratch-mb` will not fit.

| Flag | Default | Description |
|------|---------|-------------|
| `--scratch-mb N` | derived | Trainer scratch arena in MB; 0 derives `512 * (window/512)^2 / 2`, floor 512 |
| `--cuda-scratch-mb N` | 0 | CUDA kernel scratch pool, separate from the arena above |

The trainer prints its derived arena size at startup:

```
[TRAIN] scratch arena 4096 MB (window 2048), cuda kernel scratch 2048 MB
```

If a long-window run dies with `arena allocation exceeded capacity`, that is this. Raise
`--scratch-mb` if there is GPU room, lower `--cuda-scratch-mb` to make room for it.

Inference has no window: it segments the prompt as it grows. A model trained with a long window
works with a prompt of any length up to the sequence bound, which is derived from the prompt and
`--new-bytes` and can be overridden with `--max-seq-len`.
| `--embed E` | 64 | Model width |
| `--hidden H` | 128 | FFN width |
| `--layers L` | 2 | Layers per submodule (encoder/global/decoder) |
| `--max-norm F` | 20000 | Global gradient clip threshold |
| `--grad-accum N` | 1 | Gradient accumulation steps (micro-batches per optimizer step) |
| `--d0-mode MODE` | learned | `zeros` \| `learned` for decoder d0 embed |
| `--seed S` | 7 | RNG seed |
| `--report-every K` | 25 | Print frequency |
| `--deterministic` | off | Deterministic training (slower, no cuda atomics) |
| `--bf16` | off | BF16 mixed precision (CUDA only) |
| `--fp8` | off | FP8 (E4M3) mixed precision (CUDA only) |
| `--flash-attn` | off | LSE-based flash attention (CUDA only, experimental) |

### Window Order

Training windows are visited in a deterministic shuffled order. Each epoch walks every
window exactly once, so corpus coverage is identical to sequential striding, but the order
within an epoch is reseeded and the order is a pure function of `(step, seed)`.

This matters when a run is extended: sequential striding replays the exact byte sequence a
previous run already trained on, because `(step % num_windows)` starts over at step 0. The
shuffled order does not.

The order is reproducible from `--seed` alone. Changing `--seed` changes both the window
order and the diffusion timestep sequence.

### Optimizer

| Flag | Default | Description |
|------|---------|-------------|
| `--optimizer sgd\|adamw` | adamw | Optimizer selection |
| `--beta1 F` | 0.9 | AdamW first moment decay |
| `--beta2 F` | 0.999 | AdamW second moment decay |
| `--eps F` | 1e-8 | AdamW numerical stability |
| `--weight-decay F` | 0.01 | AdamW decoupled L2 weight decay |
| `--weight-decay-norms` | off | Also apply weight decay to RMSNorm weights (legacy; default is selective) |

SGD uses vanilla gradient descent with global-norm clip at 5.0. AdamW uses the standard PyTorch-compatible algorithm (bias-corrected m/v).

### Mode Selection

| Flag | Default | Description |
|------|---------|-------------|
| `--diffusion 0\|1` | 1 | 0 = plain causal BLT, 1 = BLT-D (masked diffusion) |

### Learning rate schedule

| Flag | Default | Description |
|------|---------|-------------|
| `--warmup-steps N` | 0 | Linear warmup steps |
| `--lr-schedule step\|cosine` | step | Schedule after warmup |
| `--min-lr F` | lr*0.1 | Cosine floor (negative means lr*0.1) |

### Telemetry

| Flag | Default | Description |
|------|---------|-------------|
| `--peak-tflops F` | 0 | Peak TFLOP/s for MFU. Overrides the device table; 0 = look up the running device |
| `--peak-bandwidth F` | 0 | Peak HBM bandwidth in GB/s for MBU. Overrides the device table; 0 = look up the running device |

### Checkpointing

| Flag | Description |
|------|-------------|
| `--save-weights PATH` | Write weights after training; if PATH contains `%d`, the first `%d` is replaced by the step number |
| `--save-dir DIR` | Directory convention: writes `DIR/checkpoints/*.fblt` (ergonomic alternative to `--save-weights`) |
| `--resume auto|latest|best` | Resume from a checkpoint under `--save-dir` (best = lowest `best_loss`) |
| `--save-every N` | Save checkpoint every N steps (0 = only at end) |
| `--load-weights PATH` | Load weights before training (steps=0 → eval only) |
| `--save-optim PATH` | Write AdamW moments and step counter alongside the weights |
| `--load-optim PATH` | Restore AdamW moments and step counter (requires `--load-weights`) |
| `--start-step N` | Absolute step to start at; defaults to the value stored in `--load-optim`, else 0 |
| `--eval-corpus FILE` | Held-out corpus for causal BPB eval |
| `--eval-windows N` | Eval window count (default 200) |
| `--eval-skip N` | Bytes to skip before first eval window |
| `--eval-every N` | Run causal BPB eval every N steps (default 0, disabled) |
| `--monitor-every N` | Leak-monitor interval in steps (default 2000, 0 = disabled) |
| `--monitor-windows N` | Held-out windows per monitor check (default 24) |
| `--save-best PATH` | Write weights when the eval metric improves (requires `--eval-corpus` and `--eval-every > 0`) |
| `--best-metric bpb\|masked_acc` | bpb | Metric to optimize for `--save-best` |
| `--save-best-optim PATH` | Write AdamW state alongside the best weights |
| `--metrics-log FILE` | (none) | Write a machine-readable JSONL event stream (train/eval/monitor/result) |

### Resuming

`--load-weights` alone restarts the optimizer cold: AdamW moment estimates and the
bias-correction counter both begin at zero, so the first steps behave like from-scratch AdamW rather
than a continuation. Pass `--load-optim` as well to carry both across.

```bash
# original run
build-cuda/train_blt_d --corpus data/train.bin --steps 300000 \
  --save-weights runs/x/model.fblt --save-optim runs/x/model.fbop

# extend it by another 300k steps
build-cuda/train_blt_d --corpus data/train.bin --steps 300000 \
  --load-weights runs/x/model.fblt --load-optim runs/x/model.fbop \
  --save-weights runs/x/model.fblt --save-optim runs/x/model.fbop
```

`--steps` is the number of steps to run in this invocation. The absolute step counter continues from
the value stored in the optimizer snapshot, so all logs, `--save-every` filenames, and any
step-indexed schedule (`--lr-decay-steps`, `--mask-late-step`) continue where they left off.
Fraction-based schedules (`--lr-decay`, `--t-warmup-hi`) use `start_step + steps` as the planned end,
so they behave as if the whole thing had been one continuous run. `--start-step N` overrides the
stored counter if you need to.

The optimizer snapshot is a separate file with its own header and per-tensor shapes, so it never
changes the weight checkpoint format and existing checkpoints keep loading.

With the window order from `--seed`, a resumed run is bit-identical to an equivalent single
continuous run, and a run split across several resumes is bit-identical too.

### Diffusion Schedule

| Flag | Default | Description |
|------|---------|-------------|
| `--t-min F` | 0.1 | Diffusion timestep floor |
| `--lr-decay 0\|1` | 0 | ×0.3 at 60% and 85% of steps (default schedule) |
| `--lr-decay-steps LIST` | — | Comma-separated step numbers for custom decay points (overrides `--lr-decay`) |
| `--lr-decay-factor F` | 0.3 | Multiplicative factor per custom decay point |
| `--mask-warmup N` | 0 | Ramp L_mask scale 0→1 over N steps |
| `--mask-scale F` | 1.0 | Ceiling for L_mask weight |
| `--mask-late-step N` + `--mask-late-scale F` | 0 | Late ramp from mask_scale to mask_late_scale |
| `--t-warmup-hi F` + `--t-hi-start F` | 0 + 0.8 | High-t curriculum (fraction + start floor) |
| `--paper-loss 0\|1` | 1 | 1 = paper-exact loss: `L_clean` a SUM (Eq. 5) and `L_mask` a SUM (Eq. 6), combined per Eq. 7. Implies `--mask-loss-norm 0` |
| `--mask-loss-norm 0\|1` | 0 | 1 = L_mask is a per-token mean over masked cells; 0 = sum over masked cells (paper Eq. 6 literally writes a sum) |
| `--last-row-scale F` | 1.0 | Scale row N-1's L_clean (corrected window+1 targets only); ~511 evens its gradient with the interior rows (1.0 = off) |

### Leak Monitor

Every `--monitor-every` steps, on `--monitor-windows` held-out windows at
`t=1.0`, training logs one line:

```
[MONITOR] step N  L_clean F (F bpb)  L_mask F (raw sum/t)  masked_bpb F
            train_acc F (n=N)  aligned_novel_acc F (n=N)  gap F
```

`train_acc` is masked-cell accuracy in the training layout; `aligned_novel_acc`
is masked-cell accuracy for a single block placed after a prefix that ends
exactly at a patch start, where no clean row exists to copy from. A mask that
lets a block row read the clean twin at its own target position drives
`train_acc` toward 1.0; a mask that leaks across blocks opens a gap between the
two.

`L_mask` is the raw training term: a **sum** over masked cells scaled by `1/t`
(Eq. 6 carries the `1/t` factor, and `mask_scale` weights it), so its magnitude
tracks the sampled `t` and the per-window patch count rather than model quality
— it is not a per-token loss and should not be compared against `L_total`
values from the paper, which use unscaled sums for both terms. For a
comparable figure use **`masked_bpb`** (per-masked-token cross-entropy in
bits/byte) and the `bpb` beside `L_clean`; both divide out `t` and the cell
count, and 8.0 bits/byte is uniform guessing over256 symbols.

The run **aborts with exit code 3** if either tripwire fires:

- `train_acc > 0.95` — the clean-twin leak is back
- `train_acc - aligned_novel_acc > 0.10` after 5k steps — the layouts disagree,
  which a leak-free block mask should prevent

Both conditions mean the run is not measuring what it claims to, so failing
loudly beats producing a checkpoint that looks fine.

### Loss Normalization

`--paper-loss 1` selects the paper's Eq. 5 + Eq. 6 + Eq. 7 exactly: `L_clean` as
a sum rather than a mean, and no `mask_scale` weighting. The `1/t` factor and the
sum over masked cells are already paper-exact, so this only changes `L_clean`.

`--mask-loss-norm 1` instead makes `L_mask` a per-token mean, which shrinks the
gradient but deviates from Eq. 6. `--paper-loss` takes precedence over it.

Both change the loss scale substantially; re-derive `--lr` and `--max-norm` after
enabling either. Measurements and rationale: `docs/experiments/E002_loss_normalization.md`.

### Patching

| Flag | Description |
|------|-------------|
| `--entropy-patches` | Segment with entropy LM + patcher |
| `--train-entropy-lm PATH` | Train standalone entropy LM, save to PATH |
| `--entropy-lm PATH` | Load pretrained entropy-LM weights |

### Backend

| Flag | Default | Description |
|------|---------|-------------|
| `--backend cpu\|cuda` | cpu | Select backend |
| `--cuda-scratch-mb N` | 512 | CUDA scratch arena size in MB (temp allocations during ops) |

### Per-Submodule Overrides

| Flag | Description |
|------|-------------|
| `--enc-layers N` | Encoder layers (default `--layers`) |
| `--glob-layers N` | Global transformer layers |
| `--dec-layers N` | Decoder layers |
| `--cross-attn all\|last` | Cross-attention placement (default all) |

### Logging (diagnostic, all optional)

| Flag | Default | Description |
|------|---------|-------------|
| `--grad-norm-log FILE` | (none) | Write pre-clip gradient norms per step |
| `--update-norm-log FILE` | (none) | Write post-clip update norms at spike steps |
| `--component-norm-log FILE` | (none) | Write per-component gradient norms per step |
| `--activation-dump-log FILE` | (none) | Log activation stats every report-every steps |
| `--batch-log FILE` | (none) | Log batch properties per step |
| `--loss-log FILE` | (none) | Write per-step loss |

---

## build/infer — Production inference

**Note:** For most use cases, prefer `fblt-infer` (Python wrapper above) which handles config loading and shape auto-detection. Use the raw `build/infer` binary directly only when you need to skip the wrapper or debug it.

Single-checkpoint, single-prompt, one-shot generation tool.

```
build/infer --checkpoint FILE --embed E --hidden H --enc-layers N --glob-layers N --dec-layers N --backend cpu|cuda [options]
```

### Required

| Flag | Description |
|------|-------------|
| `--checkpoint FILE` | Model checkpoint (.fblt) |
| `--embed E` | Model embed dim (must match checkpoint) |
| `--hidden H` | Model hidden dim (must match checkpoint) |
| `--enc-layers N` | Encoder layers (must match checkpoint) |
| `--glob-layers N` | Global transformer layers (must match checkpoint) |
| `--dec-layers N` | Decoder layers (must match checkpoint) |
| `--backend cpu\|cuda` | Compute backend |

### Prompt Source

Exactly one required.

| Flag | Description |
|------|-------------|
| `--prompt TEXT` | Raw bytes to use as prompt |
| `--prompt-file FILE` | Read prompt from file (raw bytes) |
| (stdin) | Read raw bytes from stdin until EOF |

### Generation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--new-bytes N` | 64 | Number of bytes to generate |
| `--num-samples N` | 1 | Draw N independent samples (greedy method only) |
| `--method MODE` | greedy | `greedy`, `selfspec`, `blockdiff`, `blockdv` |
| `--seed N` | 11 | RNG seed for the sampling stream and for a random-init entropy LM |

### Sampling Options

Only valid with `--method greedy`. Omit all of them for plain argmax decoding.

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature F` | 0 | Sampling temperature; 0 means no rescaling |
| `--top-p F` | 0 | Nucleus mass in `(0,1]`; 0 disables nucleus truncation |
| `--top-k N` | 0 | Keep only the top-N logits; 0 disables |
| `--repeat-penalty F` | 1.0 | Divides the logits of already-emitted bytes; 1.0 disables |
| `--no-repeat-ngram N` | 0 | Bans bytes that would repeat an N-gram already in the sequence; 0 disables |
| `--max-seq-len N` | derived | Override the decoder sequence bound (default: prompt + `--new-bytes` + 256, min 512) |

They apply in this order, which matters when several are combined:

1. `--no-repeat-ngram` bans
2. `--repeat-penalty` on already-emitted bytes
3. divide by `--temperature`
4. softmax
5. keep the top-`--top-k` logits (if `--top-k > 0`)
6. keep the smallest prefix whose mass reaches `--top-p`
7. draw

Bans come before the penalty so a banned byte stays banned regardless of how the
softmax shifts. If the history is short enough that the ban would cover the whole
alphabet, the ban is dropped rather than leaving nothing to sample from.

`--repeat-penalty` and `--no-repeat-ngram` are deterministic on their own: with
`--temperature 0` and `--top-p 0` they still run, so you get repeatable output with
repetition suppressed.

These flags are rejected for `selfspec`, `blockdiff`, and `blockdv`. All three verify a draft
against the greedy AR argmax, so a sampled byte can never equal the prediction it is checked
against and acceptance would be zero by construction rather than by model quality.

```bash
build-cuda/infer --checkpoint runs/my_model.fblt \
  --embed 256 --hidden 512 --enc-layers 2 --glob-layers 6 --dec-layers 2 \
  --backend cuda --method greedy \
  --entropy-lm runs/entropylm/entropy_lm.fblt \
  --temperature 0.8 --top-p 0.9 --seed 7 \
  --prompt "Once upon a time" --new-bytes 200
```

### Self-Speculation Options

Used with `--method selfspec`.

| Flag | Default | Description |
|------|---------|-------------|
| `--k N` | 8 | Speculative draft window size |

### Block Diffusion Options

Used with `--method blockdiff` or `blockdv`.

| Flag | Default | Description |
|------|---------|-------------|
| `--block-size B` | 4 | Diffusion block size; required for `blockdiff` and `blockdv` |
| `--unmask STRAT` | confidence | `confidence` or `eb` (entropy-bounded) |
| `--threshold F` | 0.7/1.0 | Alpha (confidence) or gamma (eb) |
| `--boundary-aligned` | off | Commit only at patch boundaries |
| `--adaptive` | off | Adaptive block size |
| `--b-min N` | 4 | Adaptive lower bound |
| `--b-max N` | 16 | Adaptive upper bound |
| `--accept-target F` | 0.5 | Rolling acceptance target |
| `--adapt-window N` | 8 | Rounds per rolling average |

`--block-size` must equal the `block_size` the checkpoint was trained with, and `blockdiff` and
`blockdv` refuse to start without it. Block cells past the trained `B` form a bidirectional block
that L_mask never supervised, so each one attends partners the model was never trained to
condition on. Acceptance collapses accordingly (measured on the 10M TinyStories BLT-D model:
91% at `B=4`, 15% at `B=8`).

The trained value is the `block_size` entry in the run's `resolved_config.yaml`, which
`fblt-infer` reads automatically.

Drafted bytes are verified against the causal AR predictions that greedy decoding would make.
The draft pass and the verifier agree on cross-attention for the final clean byte, which is the
patch-closing byte whenever a block is appended at a patch start.

### Model Options

| Flag | Default | Description |
|------|---------|-------------|
| `--cross-attn MODE` | all | `all` or `last` cross-attention placement |

### Patcher Options

| Flag | Default | Description |
|------|---------|-------------|
| `--fixed-patches` | off | Fixed-stride-4 patching (matches training) |
| `--patch-threshold-global F` | 3.0 | Global entropy threshold |
| `--patch-threshold-monotonic F` | 1.0 | Monotonic threshold |
| `--max-patch-length N` | 16 | Maximum patch size |
| `--entropy-lm FILE` | (none) | Trained entropy-LM weights; required unless `--fixed-patches` |
| `--allow-random-entropy-lm` | off | Permit a random-init entropy LM (will not match training) |

Entropy patching without `--entropy-lm` is a fatal error. A random-init entropy LM assigns
near-maximal entropy to every byte, so every byte clears the threshold and becomes its own
patch. The model then conditions on an all-1-byte-patch layout it never saw during training,
which collapses generation quality and speculative acceptance (measured on the 10M TinyStories
BLT-D model at `B=4`: 92% acceptance with the trained LM, 14% with a random-init one).

`fblt-infer` reads `entropy_lm` from the `resolved_config.yaml` next to the checkpoint when the
model was trained with entropy patching.

### I/O Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output FILE` | stdout | Write generated bytes to file |

### Examples

```bash
# Greedy generation with fixed patches (matches training)
build/infer --checkpoint runs/my_model.fblt \
  --embed 192 --hidden 384 --enc-layers 2 --glob-layers 2 --dec-layers 2 \
  --backend cpu --fixed-patches --prompt "int main()" --new-bytes 64

# BLT-DV with entropy patching
build/infer --checkpoint runs/my_model.fblt \
  --embed 192 --hidden 384 --enc-layers 2 --glob-layers 2 --dec-layers 2 \
  --backend cpu --method blockdv --block-size 4 --threshold 0.7 \
  --entropy-lm runs/entlm.bin --prompt "def " --new-bytes 128

# CUDA inference with output to file
build-cuda/infer --checkpoint runs/my_model.fblt \
  --embed 192 --hidden 384 --enc-layers 2 --glob-layers 2 --dec-layers 2 \
  --backend cuda --fixed-patches --prompt-file prompt.bin \
  --new-bytes 256 --output generated.bin
```

Generated bytes go to stdout (or `--output` file) as raw bytes. All diagnostics and stats go to stderr.

Exit codes: 0 on success, 1 on error.

---

## Inference Benchmark: `infer_bench`

> **Note:** `infer_bench` is a research/benchmarking tool for comparing methods against paired checkpoints. For generating text from a single checkpoint, use `fblt-infer` or `build/infer` instead.

**Binary**: `build/infer_bench` (CPU) / `build-cuda/infer_bench` (CUDA)

| Flag | Default | Description |
|------|---------|-------------|
| `--plain FILE` | runs/plain_40k.fblt | Plain model checkpoint |
| `--bltd FILE` | runs/bltd_l03_40k.fblt | BLT-D model checkpoint |
| `--embed N` | 64 | Embed dim (must match checkpoint) |
| `--hidden N` | 128 | Hidden dim (must match checkpoint) |
| `--dec-layers N` | 2 | Decoder layers (must match checkpoint) |
| `--backend cpu\|cuda` | cpu | Backend |
| `--new-bytes N` | 64 | Bytes to generate per prompt |
| `--dv-gate F` | 0.90 | Min agreement for verified methods (0 = disable) |
| `--fixed-patches` | off | Fixed-stride-4 patching (vs entropy) |
| `--entropy-lm FILE` | off | Trained entropy LM for patching |
| `--peak-tflops F` | 0 | Peak TFLOP/s override (device table fallback) |
| `--peak-bandwidth F` | 0 | Peak HBM GB/s override; enables the MBU column/results field |
| `--device NAME` | (auto) | Device table name to use; lets CPU runs reproduce a GPU entry for MBU |
| `--results FILE` | bench/results.jsonl | Output results file |

---

## CUDA Benchmark: `cuda_bench`

**Binary**: `build-cuda/cuda_bench`

| Flag | Description |
|------|-------------|
| `--backend cpu\|cuda` | Backend |
| `--iters N` | Iterations per micro-benchmark |
| `--ckpt FILE` | Checkpoint for macro gen (default runs/plain_40k.fblt) |
| `--results FILE` | Output results file |

---

## Diagnostic Tools

Evaluation tools for measuring prediction accuracy on held-out data. All three share common flags.

### Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint FILE` | *required* | Model checkpoint (.fblt) |
| `--corpus FILE` | *required* | Held-out byte corpus for evaluation |
| `--window N` | 512 | Sequence length (must match training) |
| `--embed N` | 256 | Embed dim (must match checkpoint) |
| `--hidden N` | 512 | Hidden dim (must match checkpoint) |
| `--enc-layers N` | 2 | Encoder layers (must match checkpoint) |
| `--glob-layers N` | 6 | Global transformer layers (must match checkpoint) |
| `--dec-layers N` | 2 | Decoder layers (must match checkpoint) |
| `--cross-attn all\|last` | all | Cross-attention placement (must match checkpoint) |
| `--diffusion 0\|1` | 1 | 0 = plain BLT, 1 = BLT-D |
| `--num-windows N` | varies | Number of windows to evaluate |
| `--skip N` | 0 | Bytes to skip before first window |
| `--entropy-lm FILE` | (none) | Trained entropy-LM weights for patching |

Build: `cmake --build build --target sanity_check`, `cmake --build build --target last_row_acc`, `cmake --build build --target pos_accuracy`

---

### `masked_acc` — Masked-cell accuracy by layout

Build: `cmake --build build --target masked_acc`

Measures masked-cell top-1 accuracy separately for each layout, so a layout
that can copy a clean twin is never compared against one that cannot.

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint FILE` | *required* | Model checkpoint (.fblt) |
| `--corpus FILE` | *required* | Held-out byte corpus |
| `--entropy-lm FILE` | (none) | Entropy-LM weights; required unless `--patcher fixed` |
| `--layout twin\|novel` | twin | `novel` adds the two inference-time layouts |
| `--t F` | 0.5, 1.0 | Mask probability; repeatable |
| `--block-size N` | 4 | Cells per block |
| `--windows N` | 32 | Held-out windows to evaluate |
| `--sites N` | 1 | Novel-layout sites sampled per window |
| `--novel-prefix-max N` | 256 | Cap on novel-layout prefix length in bytes |
| `--patcher entropy\|fixed` | entropy | Segmentation rule; `fixed` is deterministic |
| `--guard` | off | Assert the leak tripwires at `t=1.0` and exit 1 on failure |

With `--layout novel`, four layouts are reported per `t`:

| Layout | Block placement | Cross-attention |
|--------|-----------------|----------------|
| `train` | at a patch start `s_i` | `o_{i-1}` (training rule) |
| `train_lastlat` | at a patch start `s_i` | last closed latent |
| `aligned` | after a prefix ending exactly at a patch start | last closed latent |
| `midpatch` | after a prefix ending 1..max_patch_len-1 bytes into an open patch | last closed latent |

The last two have no clean row at the block's own position, so nothing can be
copied. Output gives overall accuracy, cell 0 vs cell B−1 accuracy, and AR
top-1 on clean rows, plus 95% CIs on the convention and aligned-vs-midpatch
deltas. `aligned` is the headline number.

`--guard` fails (exit 1) when train-layout accuracy at `t=1.0` reaches 0.99 or
above (clean-twin leak) or differs from `aligned` by 0.10 or more (mask leak).
It requires `--layout novel`, and a `--t 1.0` sweep point.

---

### `entropy_lm_eval` — Entropy-LM quality and patch statistics

Build: `cmake --build build --target entropy_lm_eval`

Loads a trained entropy LM, runs it on a byte corpus, and reports its
next-byte cross-entropy plus the patch counts produced by the entropy patcher.
Useful for deciding whether an entropy LM is good enough for entropy patching.

| Flag | Default | Description |
|------|---------|-------------|
| `--entropy-lm FILE` | *required* | Trained entropy-LM weights (`.fblt`) |
| `--corpus FILE` | *required* | Byte corpus to evaluate on |
| `--window N` | 512 | Bytes per evaluation window |
| `--windows N` | 100 | Number of windows to evaluate |
| `--skip N` | 0 | Bytes to skip before the first window |
| `--threshold F` | 3.0 | Global entropy threshold for patching |
| `--max-patch-len N` | 16 | Maximum patch length |

Output includes average cross-entropy in nats and bits, patch count
statistics, average bytes per patch, and patch-cap hit rate.

---

---

### `sanity_check` — Interior accuracy

Teacher-forced evaluation: feeds real held-out bytes with true history and reports argmax predictions vs actual next bytes. Measures accuracy across all positions (rows 0 through N-1) to verify the model learned the training distribution.

```
build/sanity_check --checkpoint MODEL --corpus FILE [options]
```

**Output:** Mean CE, Mean BPB, Top-1/5/10 accuracy across all positions. Prints sample predictions for first 3 windows.

```bash
build/sanity_check --checkpoint runs/my_model.fblt \
  --corpus data/tinystories/heldout.bin --window 512 --num-windows 50 \
  --entropy-lm runs/entropylm/entropy_lm.fblt
```

---

### `last_row_acc` — Last-row accuracy

Same forward pass as `sanity_check`, but only evaluates prediction accuracy at row N-1 (the byte immediately after the window). Useful for detecting whether the last position receives proper gradient during training.

```
build/last_row_acc --checkpoint MODEL --corpus FILE [options]
```

**Output:** Mean CE, Mean BPB, Top-1/5/10 accuracy at the last position only. Per-sample details for first 10 windows.

```bash
build/last_row_acc --checkpoint runs/my_model.fblt \
  --corpus data/tinystories/heldout.bin --window 512 --num-windows 100 \
  --entropy-lm runs/entropylm/entropy_lm.fblt
```

---

### `pos_accuracy` — Per-position accuracy

Reports top-1 accuracy at every row (0 through N-1) separately. Reveals whether accuracy drops at specific positions (e.g., the last-row cliff from the off-by-one training loss bug).

```
build/pos_accuracy --checkpoint MODEL --corpus FILE [options]
```

**Output:** Per-position accuracy table, plus min/max/mean summary.

```bash
build/pos_accuracy --checkpoint runs/my_model.fblt \
  --corpus data/tinystories/heldout.bin --window 512 --num-windows 10 \
  --entropy-lm runs/entropylm/entropy_lm.fblt
```

---

### `patch_trunc_split` — Patch truncation split diagnostic

Splits interior-row top-1 accuracy by how the containing patch closed: naturally (entropy trigger) vs forced (max_patch_length or buffer boundary). Used to test Hypothesis B from `docs/experiments/E001_last_row_gap.md` — whether arbitrarily-truncated patches have a representational weakness that more training alone won't fix.

```
build/patch_trunc_split --checkpoint MODEL --corpus FILE [options]
```

**Output:** Three-way accuracy split (natural / max-length capped / buffer-boundary), plus buffer-boundary sub-buckets by patch length (1–2, 3–4, 5–8, 9–16, 17+).

```bash
build/patch_trunc_split --checkpoint runs/tinystories_p7/tinystories_p7.fblt \
  --corpus data/tinystories/heldout.bin --window 512 --num-windows 50 \
  --entropy-lm runs/entropylm/entropy_lm.fblt
```

---

### `patch_final_split` — Per-row patch-final accuracy

Evaluates per-row top-1 prediction accuracy over windows of a byte corpus with entropy patching, classifying every row as final vs non-final byte of its patch. Splits accuracy by patch closure (natural / max-length capped / buffer-boundary) and patch length bucket, and dumps a per-byte TSV.

```
build/patch_final_split --checkpoint MODEL --corpus FILE --dump PATH [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint FILE` | *required* | Model checkpoint (.fblt) |
| `--corpus FILE` | *required* | Held-out byte corpus for evaluation |
| `--dump PATH` | *required* | Per-byte TSV output path |
| `--window N` | 512 | Sequence length (must match training) |
| `--embed N` | 256 | Embed dim (must match checkpoint) |
| `--hidden N` | 512 | Hidden dim (must match checkpoint) |
| `--enc-layers N` | 2 | Encoder layers (must match checkpoint) |
| `--glob-layers N` | 6 | Global transformer layers (must match checkpoint) |
| `--dec-layers N` | 2 | Decoder layers (must match checkpoint) |
| `--cross-attn all\|last` | all | Cross-attention placement (must match checkpoint) |
| `--diffusion 0\|1` | 1 | 0 = plain BLT, 1 = BLT-D |
| `--num-windows N` | 50 | Number of windows to evaluate |
| `--skip S` | 0 | Windows to skip before first evaluated window |
| `--entropy-lm FILE` | (none) | Trained entropy-LM weights for patching |
| `--max-patch-length L` | 16 | Maximum patch size |
| `--backend cpu\|cuda` | cpu | Compute backend |
| `--selftest` | off | Run internal classification self-check (no model/corpus needed) |

**Output:** Accuracy buckets by closure × patch length, split final vs non-final rows. TSV header: `window\trow\tpatch_idx\tpatch_start\tpatch_len\tclosure\tis_final\tcorrect` — `closure` in {nat, max, bnd}; `is_final` and `correct` are 0/1.

```bash
patch_final_split --checkpoint runs/tinystories_p7/tinystories_p7.fblt --corpus data/tinystories/heldout.bin --window 512 --embed 256 --hidden 512 --enc-layers 2 --glob-layers 6 --dec-layers 2 --cross-attn all --num-windows 50 --entropy-lm runs/entropylm/entropy_lm.fblt --dump docs/last_row_analysis/patch_final_50w.tsv
```

---

### `eval_paper_rule_split` — Paper-rule cross-attention eval (eval-time diagnostic)

**Historical:** written when training still used the leaky repo rule (every byte attends its own patch). Training and inference now apply the paper rule (Fast-BLT §3.1.1) in-process via `blt_patch_decoder_latent_at`; this tool remains the eval-time A/B switch against old checkpoints and for bucketed diagnostics on a frozen model.

Same evaluation pipeline as `patch_final_split`, but at eval time only the decoder's cross-attention group ids are switched from the repo rule (every byte attends its own patch) to the paper rule (Fast BLT §3.1.1: a patch's final byte attends its own patch j; a non-final byte attends the previous patch j-1; first-patch non-final bytes run on group 0 and are flagged `no_prev` for exclusion from the headline aggregate). Encoder and global transformer keep the real patches; no training, model, or masking code is changed by the tool itself.

```
build/eval_paper_rule_split --checkpoint MODEL --corpus FILE --dump PATH [options]
```

Flags are identical to `patch_final_split` (see its table above): `--checkpoint`, `--corpus`, `--dump`, `--window`, `--embed`, `--hidden`, `--enc-layers`, `--glob-layers`, `--dec-layers`, `--cross-attn`, `--diffusion`, `--num-windows`, `--skip`, `--entropy-lm`, `--max-patch-length`, `--backend`, `--selftest`.

**Output:** TSV header `window\trow\tpatch_idx\tpatch_start\tpatch_len\tclosure\tis_final\tpos_in_patch\tno_prev\tcorrect` — inserts `pos_in_patch` (row − patch_start) and `no_prev` (1 iff patch_idx==0 and is_final==0) before `correct`. stderr reports first-patch exclusion counts; stdout prints one machine line including `no_prev=`/`no_prev_c=`. `--selftest` exercises the synthetic run-array construction and group-id equivalence on hardcoded patch sets (no checkpoint/corpus needed). Companion analyzer: `python3 fblt/scripts/analyze_paper_rule.py --tsv ... --machine ... --baseline-tsv ... --out ...`.

```bash
eval_paper_rule_split --checkpoint runs/tinystories_p7/tinystories_p7.fblt --corpus data/tinystories/heldout.bin --window 512 --embed 256 --hidden 512 --enc-layers 2 --glob-layers 6 --dec-layers 2 --cross-attn all --num-windows 50 --entropy-lm runs/entropylm/entropy_lm.fblt --dump docs/last_row_analysis/paper_rule_50w.tsv
```

---

### `eval_isolate_redirect` — Isolated single-patch redirect eval (eval-time diagnostic)

**Historical:** same transition as `eval_paper_rule_split` — the single-redirect A/B was used to isolate the paper-rule effect while training still ran the leaky rule; training now applies the rule in-process.

Isolated variant of `eval_paper_rule_split`: per window, the paper rule (Fast BLT §3.1.1) is applied to exactly ONE redirect target r per forward pass — patch r's non-final bytes attend group r-1, its final byte stays on group r — while every other patch keeps the repo rule (all bytes attend their own group j). One decoder forward per (window, r); encoder and global are hoisted onto the real patches. Rows before patch r keep their group ids and are checked against the `patch_final` baseline (hard pre-flip invariant); only patch r's non-final bytes are scored. Patch 0 is never redirected (no previous patch), matching part 1's `no_prev` exclusion. Eval-only: no training, model, or masking code is changed.

```
build/eval_isolate_redirect --checkpoint MODEL --corpus FILE --dump PATH --baseline-tsv PATH [options]
```

Flags are identical to `patch_final_split` (see its table above) plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--baseline-tsv PATH` | *required* (except control) | `patch_final` baseline TSV for the pre-flip invariant and the `baseline_correct` column |
| `--redirect-patch R` | sweep | `-1` = control mode (no redirect; writes the 8-column `patch_final_split` schema for bit-level diffing); omitted = sweep r=1..M-1; `R>=1` = single fixed redirect target |

**Output:** Sweep-mode TSV header `window	row	redirect_patch_idx	patch_start	patch_len	closure	pos_in_patch	correct	baseline_correct`; also writes `<dump>.inv` with one `window redirect_patch_idx pre_rows pre_flips` line per pass and a machine line (`windows= passes= scored_rows= ...`) on stdout. stderr prints one progress line per window and `[INVARIANT]` lines on any pre-r flip (must be zero). `--selftest` exercises the synthetic single-redirect construction (r=1, interior, r=M-1, length-1 targets) with group-id round trips, no checkpoint/corpus needed. Companion analyzer: `python3 fblt/scripts/analyze_isolate_redirect.py --tsv ... --inv ... --machine ... --out ...`.

```bash
eval_isolate_redirect --checkpoint runs/tinystories_p7/tinystories_p7.fblt --corpus data/tinystories/heldout.bin --window 512 --embed 256 --hidden 512 --enc-layers 2 --glob-layers 6 --dec-layers 2 --cross-attn all --num-windows 50 --entropy-lm runs/entropylm/entropy_lm.fblt --baseline-tsv runs/tinystories_p7/analyses/patch_final_500w.tsv --dump runs/tinystories_p7/analyses/isolate_redirect_50w.tsv
```

---

## Sweep Scripts

Sweep scripts are not shipped in the repository; ablation sweeps run via the `train_sweep`
CMake target (`cmake --build build --target train_sweep`). See `docs/ABLATIONS.md`.

## Built tools

The following tools are built by CMake (`cmake --build build --target <name>`):

| Tool | Description |
|------|-------------|
| `selfspec_audit` | Self-speculation audit |
| `greedy_tf_acc` | Greedy TF accuracy |
| `ownlatent_value` | Own-latent value analysis |
| `leak_probe` | Mask-leak probe |
| `prefix_invariance` | Prefix invariance check |
| `eval_isolate_redirect` | Eval isolate redirect |
| `bench-harness` | Benchmark harness self-test |
| `e2e-dv` | End-to-end DV driver |

---

## Example Commands

### Toy-scale training (plain + BLT-D, SGD)
```bash
cmake --build build-cuda --target train_blt_d

build-cuda/train_blt_d --corpus data/train.bin --eval-corpus data/heldout.bin \
    --eval-windows 300 --steps 40000 --diffusion 0 --lr-decay 1 --seed 7 \
    --save-weights runs/plain_40k.fblt --backend cuda

build-cuda/train_blt_d --corpus data/train.bin --eval-corpus data/heldout.bin \
    --eval-windows 300 --steps 40000 --diffusion 1 --t-min 0.1 \
    --mask-warmup 10000 --mask-scale 0.3 --lr-decay 1 --seed 7 \
    --save-weights runs/bltd_l03_40k.fblt --backend cuda
```

### Toy-scale training with AdamW
```bash
build-cuda/train_blt_d --corpus data/train.bin --eval-corpus data/heldout.bin \
    --eval-windows 300 --steps 40000 --diffusion 1 --t-min 0.1 \
    --mask-warmup 10000 --mask-scale 0.3 --lr-decay 1 --seed 7 \
    --optimizer adamw --lr 0.001 --weight-decay 0.01 \
    --save-weights runs/bltd_adamw_40k.fblt --backend cuda
```

### Inference comparison
```bash
cmake --build build-cuda --target bench-infer

build-cuda/infer_bench --plain runs/plain_40k.fblt --bltd runs/bltd_l03_40k.fblt \
    --backend cuda --new-bytes 64 --results bench/results.jsonl
```

### At-scale sweep (27 runs)
```bash
cmake --build build-cuda --target train_sweep
```

### Speedrun (fblt.speedrun)

| Command | Description |
|---|---|
| `python -m fblt.speedrun --config PATH --out DIR` | Run a reproducible speedrun, writing `DIR/result.json` |

### Datasets

Cache layout: `$FBLT_CACHE` (default `~/.cache/fblt`)/datasets/<name>/ with manifest.json and shard-*.bin.
CLI: `python -m fblt.dataset --list|--download NAME|--verify NAME`.

| `--doc-boundaries MODE` | off | Document boundary handling: off, mask, sentinel |

### Evaluation (fblt.eval)

| Command | Description |
|---|---|
| `python -m fblt.eval --task bpb --checkpoint PATH --corpus FILE --out eval.json` | Run evaluation, write JSON record |
| `... --few-shot N` | Few-shot mode (0 = zero-shot) |
| `... --csv out.csv` | Also write CSV |

### Hugging Face (fblt.hf_export)

| Function | Description |
|---|---|
| `export_to_hf(fblt_path, out_dir)` | Export checkpoint to HF-style dir (config.json + model.fblt) |
| `import_from_hf(hf_dir, out_path)` | Import back to fBLT checkpoint |
| `verify_roundtrip(...)` | Verify byte-identical reload |
