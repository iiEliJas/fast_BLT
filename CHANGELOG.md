# Changelog

All notable changes to fBLT are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

Changes since 1.0.1. Everything below landed on `main`; the next tagged release is expected to
be a minor version.

### Added

- **Mixed precision.** `--dtype fp32|bf16|fp16|fp8` narrows GEMM operands only: master weights,
  gradients, optimizer state, norms and activations stay FP32. `--fp8-format e4m3|e5m2` picks
  the FP8 encoding, and `--bf16` / `--fp8` remain aliases. CUDA needs compute capability 8.0
  (BF16), 7.0 (FP16) or 8.9 (FP8); the gates are checked before the first step. The CPU backend
  rejects narrow dtypes and remains the numerical reference. See `docs/MIXED_PRECISION.md` and
  the E010/E011 experiment reports. (#68, #74, #75, #90, #92)
- **Gradient accumulation.** `--grad-accum N` averages N micro-batches per optimizer step and
  does not allocate the accumulator for `N = 1`. A ready-to-run four-way config ships as
  `configs/train/production_grad_accum.yaml`. (#66, #91)
- **Document-aware windows.** `--doc-boundaries mask` builds the training-window table from
  `--corpus-manifest` (`--manifest-split`, default `train`), so neither a window nor its
  next-byte target crosses a document boundary. Both the `index.<split>` and `<split>_files`
  manifest layouts are accepted, with bounds, ordering and declared-size validation. Startup
  diagnostics report documents total/usable/dropped, windows, byte coverage and dropped tail.
  See `docs/DATA.md`. (#95)
- **Corpus BPB evaluation.** `eval_bpb` and the `fblt-eval` wrapper score a checkpoint over a
  raw byte corpus and write one JSON record per run: causal BPB, optional masked BPB and
  accuracy, and SHA-256 provenance for the checkpoint, corpus and entropy LM plus the git
  commit. See `docs/EVALUATION.md`. (#93)
- **Device peak table and MBU.** A per-device FLOPS/bandwidth table supplies MFU and MBU
  without manual flags; `--peak-tflops` / `--peak-bandwidth` still override it, and the lookup
  resolves Ti/PCIe/L40S variants. Both values are printed in the startup and `RESULT` lines,
  including inference MBU in `infer_bench`. (#79, #88)
- **`entropy_lm_eval`.** Reports next-byte cross-entropy in nats and bits, patch-count
  statistics and how often the patcher hits its patch cap for a trained entropy LM. (#63)
- **Opt-in fused attention forward** (`--flash-attn`, CUDA): flash-attention-style computation
  for the attention core. (#78)
- **Batched KV-cache storage and a batched greedy-generation helper.** The cache now carries a
  batch dimension and `generate_greedy` accepts multiple rows; the CLI still runs batch size 1.
  (#69, #70)
- **Native PyTorch BLT/BLT-D implementation** under `pysrc/fblt_torch`: encoder/decoder parity
  tests against the C model, checkpoint round-trip, and a tiny training entry point. (#65)
- **`fblt.data_pipeline` offline helpers** (`load_manifest`, `doc_spans`, `window_offsets`,
  `window_summary`) that mirror the C window rule, so a corpus and manifest can be inspected
  without starting a run. (#95)
- **Python-side checkpoint helpers** (`fblt/_checkpoint.py`): atomic writes, save-directory
  layout, and latest/best/resume lookup, plus a `model_info` CLI that prints checkpoint
  metadata. The C trainer does not accept `--save-dir`/`--resume` yet, so these are building
  blocks rather than a wired feature. (#80)
- **Dataset download/cache scaffolding** (`fblt/dataset.py`): registry, SHA-256 verification,
  file locking, retries and a written manifest. The registry entry currently ships without a
  URL, so a download writes an empty manifest instead of fetching data. (#82, #83)
- **Speedrun and leaderboard CLI scaffolding** (`fblt/speedrun.py`): result aggregation over
  JSON files; the runner does not invoke training yet. (#81)
- **Hugging Face export/import CLI scaffolding** (`fblt/hf_export.py`): writes a minimal
  `config.json` and copies the checkpoint. It is not a tensor-level conversion to a Hugging
  Face format. (#87)
- CI: CUDA syntax check on Ubuntu 24.04, plus the Python CLI, eval and regression tests in the
  `linux-test` job. (#62, #91, #95)

### Changed

- The default entropy-patching threshold is 3.0 (was 2.5) for inference and `entropy_lm_eval`.
  (#64)
- `--dtype` is the single mixed-precision switch; `--fp8-format` is rejected unless an FP8
  dtype is selected. (#92)
- Backward-pass work on CUDA: fused add+RMSNorm backward when the backend provides it, async
  H2D copies in the mask builder, and fewer scratch allocations. Profiling notes are in
  `docs/experiments/E009_backward_profile.md`. (#72, #76, #77)
- `fcntl` is optional in `fblt/dataset.py`, so importing it on Windows no longer fails. (#83)

### Removed

- `fblt.data_pipeline.pack_documents`, `stream_corpus`, `windows_for_docs` and the sampler-state
  helpers. `pack_documents` emitted windows that crossed document boundaries and
  `stream_corpus` closed its mmap as soon as it yielded. (#95)

### Fixed

- Gradient clipping read uninitialized device memory when a model had more than 64 parameter
  tensors; the norm buffer is now sized up front and the clip factor is folded into the optimizer
  update. The old CUDA path dropped 64 of the 67 contributions, so the norm and clip factor
  were wrong; CUDA now matches the CPU norm. (#89)
- BF16 matmuls on CUDA read 2-byte BF16 buffers as FP32 in `cublasSgemm`. Mixed GEMMs now
  stage the FP32 operand into the shadow dtype and keep FP32 accumulation. (#90)
- Gradient accumulation divided by the configured count instead of the number actually
  accumulated, wrote several loss-log lines per optimizer step, and expressed epochs and
  throughput in micro-batches. It now averages by the real count, logs one line per step, and
  reports windows processed; `--grad-accum` is validated before the corpus loads and the second
  gradient buffer is preflighted with a flag-specific error. The Python trainer now logs the
  micro-batch mean. (#91)
- `--doc-boundaries` was parsed and then ignored: every trainer call passed `NULL` boundaries
  and windows were cut from the flat stream, so `off` and `mask` produced identical losses.
  `mask` now uses real manifest offsets, and `sentinel` is rejected instead of silently doing
  nothing. (#95)
- The evaluation harness emitted zeroed metrics. `fblt-eval` now drives `eval_bpb` and records
  real BPB values with reproducible provenance. (#93)
- Entropy-LM training progress is flushed and output parent directories are created. (#61)

## [1.0.1] - 2026-10-25

### Added
- Lightweight research workflow docs, experiment log, and experiment report template
- Public roadmap (`todo.md`) and nanochat-inspired follow-up ideas
- Stack-smol code experiment and training config (`configs/train/stack_smol_c.yaml`)
- Results page and comparison notes with `facebookresearch/blt`

### Changed
- README and starter guide restructure; documentation map and links
- Starter guide entropy-LM example now uses a learnable SGD learning rate (`--lr 0.1`)

### Fixed
- Standalone entropy-LM training used the main model AdamW learning rate (`1e-4`) for plain SGD, so it never learned; now uses `0.1`
- Checkpoint metadata null termination in `read_meta`
- Relative Markdown links and diff checks for docs changes

## [1.0.0] - 2026-10-22

### Added
- Best-checkpoint saving on eval metric (`--save-best`, `--best-metric`, `--save-best-optim`)
- Metrics JSONL event stream (`--metrics-log`) and optional W&B integration
- LR warmup + cosine/min-lr schedule (`--warmup-steps`, `--lr-schedule`, `--min-lr`)
- Training telemetry: param count, ms/step, B/s, TFLOP/s, MFU (`--peak-tflops`)
- Inference `--top-k` and `--num-samples`
- Profiling wrapper script and docs
- Checkpoint metadata v2 (JSON header) with v1 backward compatibility
- Selective weight decay (norms excluded by default; `--weight-decay-norms`)
- macOS CPU CI job
- Packaging metadata and extras; robust binary resolution
- Finetune recipe config
- Ablation configs under `configs/ablations/`

### Changed
- README layout (`run/` -> `csrc/`), known limitations and support matrix
- AGENTS.md profiling link (`docs/Plan.md` -> `docs/PROFILING.md`)

### Fixed
- CLI reference stale tool paths and omitted built tools documented

