# fBLT Starter Guide

This guide is the end-to-end companion to the [README](../README.md). It covers installation, building, running a first model, training, inference, checkpoints, evaluation, benchmarking, and troubleshooting. For the complete flag reference, see [CLI_REFERENCE.md](CLI_REFERENCE.md).

## 1. What fBLT does

fBLT is a byte-level language model. It reads and writes raw bytes; there is no tokenizer and no vocabulary. Long runs of predictable bytes are grouped into **patches** by an entropy-based patcher, and the main transformer operates over patches rather than bytes.

The implementation has two inference-relevant variants:

- **BLT-D**: block-diffusion decoding. The decoder proposes a block of future bytes in parallel; this is the cheapest per-byte path.
- **BLT-DV**: BLT-D proposes a block, then the model verifies it. Output is byte-identical to greedy decoding when verification succeeds.

Training supports both a plain causal BLT baseline (`--diffusion 0`) and BLT-D masked diffusion (`--diffusion 1`, the default).

## 2. Concepts

### Entropy model and patcher

The entropy model is a small byte-level language model. It estimates per-byte entropy, and the patcher uses those estimates to decide where one patch ends and the next begins. Training can use fixed-stride patching or entropy patching (`--entropy-patches`).

### Local encoder and decoder

The local encoder compresses the bytes inside a patch into one embedding. The local decoder expands a patch embedding back into bytes. Both use hash n-gram embeddings to recognize byte patterns without a vocabulary.

### Patch transformer

The global transformer is the main model. It attends over patches in block-causal order.

### Block size

`--block-size` (`B`) is the number of future bytes decoded together in BLT-D. It must match between training and inference. A model trained at `B=4` should not be asked to draft `B=8` blocks; the extra cells were never supervised.

### Window

`--window` is the clean byte sequence length per training example. Larger windows use much more memory because diffusion scores every patch in the window at once.

## 3. Prerequisites

- C compiler: `gcc` or `clang`, C99-compatible
- CMake >= 3.18
- Python >= 3.9
- Optional: CUDA toolkit with `nvcc` for GPU builds
- Optional Python extras: `ruff`, `pytest`, `torch`, `numpy`, `matplotlib`, `datasets`, `wandb`

## 4. Install and build

### Python package

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install ruff pytest
```

This installs `fblt-train` and `fblt-infer`.

### CPU build

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
cmake --build build --target test
```

### CUDA build

```bash
cmake -S . -B build-cuda -DUSE_CUDA=ON
cmake --build build-cuda -j$(nproc)
cmake --build build-cuda --target test
```

CUDA builds default to architecture `89`. For another GPU:

```bash
cmake -S . -B build-cuda -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
```

If `nvcc` is outside `/usr/local/cuda`:

```bash
cmake -S . -B build-cuda -DUSE_CUDA=ON -DCUDAToolkit_ROOT=/path/to/cuda
```

### WSL2 CUDA note

If an apt NVIDIA library shadows the WSL2 driver, put the WSL2 driver directory first on `LD_LIBRARY_PATH` and set `CUDA_VISIBLE_DEVICES=0` if it is empty:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/drivers/nvmdsi.inf_amd64_*/:/usr/lib/wsl/lib:/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=0
```

### Useful build targets

| Target | Purpose |
|--------|---------|
| `test` | Build and run the C test suite from the repo root |
| `main` | Minimal linking stub |
| `train_blt_d` | Trainer binary |
| `infer` | Inference binary |
| `sandbox_run` | Scratch playground |
| `bench-infer` | Inference benchmark |
| `cuda-smoke` | CUDA device sanity check (CUDA only) |
| `cuda-sanitize` | Tests under `compute-sanitizer` (native Linux, CUDA only) |

## 5. First run

Generate the sample corpus from the repository's own C/H source files:

```bash
python fblt/scripts/build_sample_corpus.py
```

This writes `data/train.bin` and `data/heldout.bin`. The generator is deterministic and writes `data/sample_corpus_manifest.json` with checksums.

Train the debug model:

```bash
fblt-train --config configs/train/debug.yaml --backend cpu
```

The wrapper creates `runs/<run-name>/`, writes `resolved_config.yaml`, and runs `train_blt_d`. The training log reports the checkpoint path:

```text
[CKPT] saved weights to runs/debug-20260101-120000/checkpoint.fblt
```

Run inference with that checkpoint:

```bash
fblt-infer --checkpoint runs/debug-*/checkpoint.fblt --backend cpu --prompt "int main"
```

If more than one debug run exists, use the full path from the `[CKPT]` line. The wrapper auto-detects model dimensions and patching settings from `resolved_config.yaml`.

## 6. Training

### Basic wrapper command

```bash
fblt-train --config configs/train/production.yaml --backend cuda
```

The wrapper:

- rejects `backend` in YAML (it must be a command-line flag)
- creates `runs/<run-name>/`
- writes `resolved_config.yaml`
- calls `train_blt_d`
- fills default output paths for unset log/checkpoint fields

Every `train_blt_d` flag is also a wrapper flag, with underscores turned into dashes. Precedence is: dataclass default, YAML, `--override`, explicit flag.

### Configuration files

Configs live in `configs/train/`. Important examples:

- `debug.yaml`: 10 steps, tiny model, CPU-friendly.
- `production.yaml`: 2.97M parameters, 40k steps.
- `tinystories.yaml`: 7.8M parameters, 300k steps, entropy patching.

A config is ordinary YAML. All keys correspond to `TrainConfig` fields.

### Training your own dataset

The trainer expects a flat `.bin` file of raw bytes: no header, no framing, no length prefix. Separate multiple source files with a newline.

To use your own data:

```bash
fblt-train --config configs/train/production.yaml --backend cuda \
  --corpus /path/to/train.bin
```

Or edit `corpus:` in the YAML. For held-out evaluation, set `--eval-corpus` to a separate `.bin` file that does not overlap with training data.

### Entropy patching

Training can segment with the entropy LM and patcher:

```yaml
entropy-patches: true
entropy-lm: runs/entropylm/entropy_lm.fblt
```

This matches inference better than fixed-stride patching and is required for the benchmarked BLT-DV results. Models trained without it use fixed-stride-4 patching, which inference detects from `resolved_config.yaml`.

An entropy LM can be trained separately:

```bash
fblt-train --config configs/train/debug.yaml --backend cpu \
  --train-entropy-lm runs/entropylm/entropy_lm.fblt --steps 1000 --lr 0.1
```

### Resuming

Checkpointing stores weights and, optionally, optimizer state:

```yaml
save_weights: runs/prod/checkpoint.fblt
save_optim: runs/prod/checkpoint.fbop
```

Resume by loading both:

```bash
fblt-train --config configs/train/production.yaml --backend cuda --steps 300000 \
  --load-weights runs/prod/checkpoint.fblt \
  --load-optim runs/prod/checkpoint.fbop
```

`--steps` is the number of steps in this invocation. The step counter continues from the optimizer snapshot, so logs and step-indexed schedules continue where they stopped. A resumed run is bit-identical to an equivalent continuous run when window order and seed match.

Loading weights without optimizer state restarts AdamW cold. Use `--start-step N` to override the stored counter.

For ergonomic runs, use `--save-dir DIR` (writes `DIR/checkpoints/*.fblt`) and `--resume auto|latest|best` to pick the newest or lowest-`best_loss` checkpoint. Atomic writes (`.tmp` + fsync + replace) avoid partial files. Inspect a checkpoint with `python -m fblt.model_info PATH`.

### Reproducible speedruns (item 15)

`fblt/speedrun.py` provides a thin reproducible runner and leaderboard aggregator:

```bash
python -m fblt.speedrun --config configs/train/production.yaml --out runs/speedrun/run1
python -c "from fblt.speedrun import aggregate_leaderboard; aggregate_leaderboard('runs/speedrun','runs/leaderboard.json')"
```

It records the resolved config and writes `result.json` per run; `aggregate_leaderboard` collects them into one JSON. (Training invocation integration is a follow-up; currently the runner records metadata only.)
### Best checkpoints

Save weights only when an eval metric improves:

```bash
fblt-train --config configs/train/production.yaml --backend cuda \
  --eval-corpus data/heldout.bin --eval-every 1000 \
  --save-best runs/prod/best.fblt --best-metric bpb
```

This requires `--eval-corpus` and `--eval-every > 0`.

### Direct binary usage

The C binary can be called directly:

```bash
./build-cuda/train_blt_d --backend cuda \
  --embed 192 --hidden 384 --layers 2 \
  --steps 40000 --lr 0.0001 --lr-decay 1 \
  --mask-warmup 33200 --mask-scale 0.3 \
  --t-min 0.1 --t-warmup-hi 0.25 --t-hi-start 0.8 \
  --cross-attn all --optimizer sgd \
  --diffusion 1 --block-size 4 --window 48 \
  --save-weights runs/my_checkpoint.fblt
```

The wrapper is recommended for normal use because it manages run directories and resolved configs.

### Common training options

| Option | Meaning |
|--------|---------|
| `--optimizer sgd\|adamw` | Optimizer |
| `--lr F` | Learning rate |
| `--warmup-steps N` | Linear LR warmup |
| `--lr-schedule step\|cosine` | Schedule after warmup |
| `--weight-decay F` | AdamW decoupled weight decay |
| `--max-norm F` | Global gradient-norm clip |
| `--grad-accum N` | Accumulate N windows per optimizer step (roadmap item) |
| `--bf16` | BF16 mixed precision on CUDA (roadmap item) |
| `--diffusion 0\|1` | Plain causal BLT or BLT-D |
| `--window N` | Clean sequence length |
| `--block-size B` | Diffusion block size |
| `--cross-attn all\|last` | Cross-attention placement |
| `--enc-layers`, `--glob-layers`, `--dec-layers` | Per-module depth |
| `--monitor-every N` | Leak-monitor interval |
| `--metrics-log FILE` | JSONL event stream |

See [CLI_REFERENCE.md](CLI_REFERENCE.md) for the full list, defaults, and interactions.

### Memory and long windows

Scratch grows faster than the window because diffusion scores every patch in the window at once. The decoder sees `S = window * (1 + block_size)` rows, and attention over `S` is quadratic. A window above 512 needs roughly 400 patches instead of 98 and uses several times the activation memory. The CUDA scratch arena defaults to 512 MB and can be raised with `--cuda-scratch-mb`.

### Telemetry and monitoring

The startup line reports parameter count. Periodic reports include loss, LR, ms/step, bytes/s, and an approximate TFLOP/s. `--peak-tflops F` enables MFU reporting. `RESULT` reports params, BPB, masked accuracy, and best-checkpoint state when enabled.

The leak monitor runs on held-out windows at `t=1.0` and aborts with exit code 3 if a clean-twin leak or layout disagreement is detected. `L_mask` is a raw sum; use `masked_bpb` for a comparable per-token figure.

## 7. Inference

### Basic usage

```bash
fblt-infer --checkpoint runs/debug-*/checkpoint.fblt --backend cpu \
  --prompt "int main" --new-bytes 64
```

If the checkpoint was produced by `fblt-train`, the wrapper auto-detects dimensions, block size, and patching strategy from `resolved_config.yaml` or checkpoint metadata. For checkpoints without metadata, pass shape flags:

```bash
fblt-infer --checkpoint my_model.fblt --backend cpu \
  --embed 192 --hidden 384 --enc-layers 2 --glob-layers 2 --dec-layers 2 \
  --prompt "def "
```

Inference methods:

- `greedy` (default)
- `selfspec` (BLT-S)
- `blockdiff` (BLT-D)
- `blockdv` (BLT-DV)

Example configs are in `configs/infer/`.

### Sampling

Sampling flags apply to `--method greedy` only:

```bash
fblt-infer --checkpoint runs/prod/checkpoint.fblt --backend cuda \
  --prompt "Once upon a time" --new-bytes 200 \
  --method greedy --temperature 0.8 --top-p 0.9 --top-k 40 --seed 7
```

`--repeat-penalty F` and `--no-repeat-ngram N` are also available. Speculative and block-diffusion methods verify against greedy argmax, so they cannot use sampling. Greedy decoding can loop at this model scale; sampling is the fix.

`--num-samples N` draws multiple samples with per-sample seeds `seed`, `seed+1`, ... and is supported only with greedy.

### Block-diffusion options

Important flags for `blockdiff` and `blockdv`:

- `--block-size B`: must equal training block size.
- `--unmask confidence|random|left-to-right`
- `--threshold F`: confidence threshold for unmasking.
- `--adaptive`: start at a larger block and back off.
- `--b-min`, `--b-max`: adaptive block-size range.
- `--k`: draft length for self-speculation.

### Matching training at inference

Block-diffusion inference only works if it segments bytes the same way training did:

1. The entropy LM used for patching must match.
2. The block size must equal the training block size.

`fblt-infer` reads both from `resolved_config.yaml` next to the checkpoint. The raw `build/infer` binary cannot infer them and requires explicit flags:

```bash
build-cuda/infer --checkpoint runs/prod/checkpoint.fblt \
  --embed 256 --hidden 512 --enc-layers 2 --glob-layers 6 --dec-layers 2 \
  --backend cuda --method blockdv --block-size 4 \
  --entropy-lm runs/entropylm/entropy_lm.fblt \
  --prompt "Once upon a time" --new-bytes 200
```

If training used `--entropy-patches`, keep `--entropy-lm` in the training config so the snapshot path is carried over. Models trained without entropy patching use fixed-stride-4 patching, which inference selects automatically.

### Direct binary usage

Use `build/infer` directly for debugging or when bypassing the wrapper. It requires `--backend`, `--checkpoint`, and shape flags for checkpoints without metadata.

## 8. Checkpoints and resumption

Checkpoint v2 stores architecture config, training step, and best metric in the `.fblt` file. The wrapper can auto-detect dimensions from metadata or `resolved_config.yaml`.

Files:

- `.fblt`: weights and metadata.
- `.fbop`: AdamW moments and step counter.

Resuming best practice:

```bash
# original run
fblt-train --config configs/train/production.yaml --backend cuda --steps 300000 \
  --save-weights runs/prod/checkpoint.fblt --save-optim runs/prod/checkpoint.fbop

# continue
fblt-train --config configs/train/production.yaml --backend cuda --steps 300000 \
  --load-weights runs/prod/checkpoint.fblt --load-optim runs/prod/checkpoint.fbop \
  --save-weights runs/prod/checkpoint.fblt --save-optim runs/prod/checkpoint.fbop
```

`--steps` is steps in this invocation. Fraction-based schedules use `start_step + steps` as the planned end. `--save-every N` writes step-numbered checkpoints when PATH contains `%d`.

## 9. Evaluation and diagnostics

Held-out BPB during training:

```bash
fblt-train --config configs/train/production.yaml --backend cuda \
  --eval-corpus data/heldout.bin --eval-windows 200 --eval-every 1000
```

Quality metrics:

- causal BPB on held-out bytes
- masked-cell accuracy
- monitor leak checks

Diagnostic binaries:

- `sanity_check`
- `last_row_acc`
- `pos_accuracy`
- `patch_trunc_split`, `patch_final_split`
- `eval_paper_rule_split`, `eval_isolate_redirect`
- `selfspec_audit`
- `greedy_tf_acc`
- `ownlatent_value`
- `leak_probe`
- `prefix_invariance`

Build any with `cmake --build build --target <name>`. See [CLI_REFERENCE.md](CLI_REFERENCE.md) for their flags.

## 10. Benchmarks

Build and run the inference benchmark:

```bash
cmake --build build-cuda --target infer_bench
./build-cuda/infer_bench --backend cuda \
  --plain runs/tinystories_600k/tinystories_600k.fblt \
  --bltd runs/tinystories_600k/tinystories_600k.fblt \
  --entropy-lm runs/entropylm/entropy_lm.fblt \
  --heldout data/tinystories/heldout.bin \
  --embed 256 --hidden 512 --enc-layers 2 --glob-layers 6 --dec-layers 2 \
  --prompts 8 --new-bytes 64 \
  --results bench/results_tinystories_10m.jsonl
```

`--entropy-lm` is mandatory for entropy-patched checkpoints. `--block-size` must match training. Both tools abort rather than measure a mismatch.

Key results at 7.8M parameters:

- BLT-DV at `B=4`: 65% draft acceptance, output byte-identical to greedy, 0.85x memory bandwidth, 0.89x wall-clock.
- Raw BLT-D is cheapest (0.41-0.59x bandwidth) but diverges from greedy.
- Entropy patching more than doubles BLT-DV acceptance over fixed-stride (65.0% vs 28.7%).
- At `B=8` and above, BLT-DV loses badly.

CUDA training/inference speedups and hardware numbers are in [BENCHMARKS.md](BENCHMARKS.md). The 7.8M checkpoint and entropy LM are release assets.

## 11. Ablations

Ablations at 2.97M parameters (embed=192, hidden=384, 2/2/2 layers) on ~67 MB of C source code, 3-6 seeds, 40k steps:

| Setting | Value | Finding |
|---|---|---|
| Step budget | 40k | Safe convergence floor |
| Mask-warmup | 83% of steps | 95% is a minor improvement |
| Decoder depth | 2 layers | Deeper decoder does not help |
| Encoder depth | 2 layers | Deeper encoder is harmful |
| Cross-attn | all or last | Tied; `last` ~15% faster |
| High-t warmup | on | Improves BPB by 0.33 |
| Mask-late | off | No benefit found |

Full tables and raw numbers: [ABLATIONS.md](ABLATIONS.md).

## 12. Profiling

CPU:

```bash
BLT_TRAIN_TIMING=1 ./build/train_blt_d ...
```

CUDA:

```bash
scripts/profile_train.sh build-cuda --corpus data/train.bin --steps 50 ...
```

See [PROFILING.md](PROFILING.md) for nsys, ncu, perf, and sanitizer recipes.

## 13. Known limitations

- Patch count is capped at 128 per window (`BLT_MAX_PATCHES`).
- Demonstrated scale is 7.8M parameters; larger configs are untested.
- CUDA runs are non-deterministic unless `--deterministic` is passed.
- Sampling works only with `--method greedy`.
- Single-GPU / single-process only.
- BF16 mixed precision is not implemented (FP32 only).

## 14. Troubleshooting

### `Binary not found`

Build first, or set `FBLT_BIN_DIR`:

```bash
cmake --build build --target train_blt_d
export FBLT_BIN_DIR=/path/to/build
```

### `--entropy-lm` required

For entropy-patched checkpoints, pass the trained entropy LM. Without it, every byte becomes its own patch and results are meaningless. `infer` and `infer_bench` fail rather than measure this.

### Block-size mismatch

`--block-size` must equal the value used in training. The wrapper carries it over from `resolved_config.yaml`.

### CUDA device or driver issues

Use a standalone CUDA toolkit, not the apt `nvcc` package. On WSL2, fix `LD_LIBRARY_PATH` and `CUDA_VISIBLE_DEVICES` as described above.

### Checkpoint moved without `resolved_config.yaml`

Checkpoint v2 metadata can supply dimensions. For v1 checkpoints, pass shape flags explicitly or keep the sidecar config next to the checkpoint.

### Tests must run from repo root

Parity tests load golden files via relative paths. Use `cmake --build build --target test`, not `build/test_main` from another directory.

## 15. Development and tests

C tests:

```bash
cmake -S . -B build
cmake --build build --target test
```

Python tests and lint:

```bash
pytest tests/
ruff check .
ruff format --check .
```

Parity golden data:

```bash
pip install torch numpy
python tests/py/parity_generate.py
cmake --build build --target test
```

The C suite has 80 tests. Suite selection is hardcoded in `tests/test_main.c` as `do_test[4]`. Flip a suite flag, rebuild, and revert before committing. The `tools` suite ships disabled.

## 16. Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md). Open an issue before starting scoped work; create a feature branch; do not commit directly to `main`. Include tests and docs for user-facing changes.

## 17. Research and experiments

Significant experiments are indexed in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). The process for
formulating, running, and reporting experiments is described in
[RESEARCH_GUIDE.md](RESEARCH_GUIDE.md). New reports should start from
[experiments/TEMPLATE.md](experiments/TEMPLATE.md).

To reproduce a documented result, use the report's commands, configuration paths, checkpoint
names, and environment notes. Keep training and evaluation data disjoint, and record the Git
commit, seeds, hardware, and measurement conditions. Negative and inconclusive results are
worth reporting.

## 18. Documentation map

- [README.md](../README.md) - project landing page.
- [CLI_REFERENCE.md](CLI_REFERENCE.md) - all flags.
- [BENCHMARKS.md](BENCHMARKS.md) - performance results.
- [ABLATIONS.md](ABLATIONS.md) - ablation results.
- [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) - research workflow.
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) - experiment index.
- [experiments/](experiments/) - investigation notes and report template.
- [PROFILING.md](PROFILING.md) - profiling recipes.
- [todo.md](../todo.md) - roadmap.
- [fast_bltPAPER.md](fast_bltPAPER.md), [bltransformerPAPER.md](bltransformerPAPER.md) - papers and notes.
