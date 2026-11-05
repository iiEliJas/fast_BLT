# Changelog

All notable changes to fBLT are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- Gradient clipping read uninitialized device memory when a model had more than 64 parameter
  tensors; the norm buffer is now sized up front and the clip factor is folded into the optimizer
  update. The old CUDA path dropped 64 of the 67 contributions, so the norm and clip factor
  were wrong; CUDA now matches the CPU norm.

### Added

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

