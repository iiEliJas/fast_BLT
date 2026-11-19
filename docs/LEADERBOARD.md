# Leaderboard and speedrun

`scripts/speedrun.sh` takes a fresh checkout through build → corpus → train → BPB eval →
generation, and writes one JSON record describing the run. `fblt-leaderboard` collects those
records into a table, so results can be compared without hand-editing a document.

The point is reproducibility, not peak numbers. A record carries the git commit, the exact
command, the resolved config, the hardware, and the SHA-256 of the checkpoint and corpora, so a
suspicious row can be traced back to the bytes that produced it. A record without that
provenance is marked, not silently ranked.

## Quickstart

```bash
bash scripts/speedrun.sh --out runs/speedrun
```

The first run builds `train_blt_d`, `eval_bpb` and `infer` (CPU). Pass `--backend cuda` for a
CUDA build, `--no-build` when the binaries already exist, and `--dry-run` to print the planned
commands without executing anything.

```bash
# a few seeds and configs; extra flags are forwarded to the speedrun
bash scripts/miniseries.sh --seeds 1,2,3 --configs configs/train/debug.yaml --out runs/mini
# aggregate every summary under a directory (recursive)
python -m fblt.leaderboard --results runs --out-md runs/LEADERBOARD.md
```

A single run writes:

```
runs/speedrun/
  summary.json          the leaderboard record (schema below)
  eval.json             the full BPB record from fblt-eval
  generated.bin         bytes generated from the trained checkpoint
  metrics.jsonl         per-step training metrics
  ckpt/checkpoints/     step_*.fblt, step_*.fbop, latest.txt
  runs/speedrun-*/      resolved_config.yaml and training logs
  LEADERBOARD.md        written by fblt-leaderboard, not by the speedrun
```

Both output paths are relative to the working directory when you start the script.

## What the run does

1. Builds the CPU or CUDA targets (`cmake -S . -B build[-cuda]`, then the three binaries).
2. Builds `data/train.bin` and `data/heldout.bin` with `fblt/scripts/build_sample_corpus.py`
   when they are missing.
3. Trains the built-in tiny config (`embed=32`, `hidden=64`, 1 layer, `window=128`,
   `block_size=8`, AdamW, 100 steps at `lr=0.05`, seed 7) or the `--config` you pass, with
   `--save-dir <out>/ckpt`.
4. Scores the resulting checkpoint on the held-out corpus with `fblt.eval`, fixed-stride
   patching, `layout=paper`, `masked_acc` included.
5. Generates `--new-bytes` greedy bytes from `--prompt`.
6. Writes `summary.json`.

`--steps`, `--window`, `--block-size`, `--lr`, `--seed`, `--windows`, `--new-bytes` and
`--prompt` override the config. A run with `--steps 2` is a CI smoke test, not a result: it is
recorded and it is valid, but the BPB will be near the uniform-byte baseline of 8.

## Record schema (`fblt-speedrun/v1`)

Every summary is a single JSON object. `format` and `schema_version` identify the schema;
readers reject anything else rather than misinterpreting it.

| Field | Type | Meaning |
| --- | --- | --- |
| `format` | string | Always `fblt-speedrun/v1` |
| `schema_version` | int | `1` |
| `run_id` | string | Unique id; `fblt-leaderboard` dedupes on it (later `created` wins) |
| `created` | string | UTC ISO-8601 timestamp |
| `synthetic` | bool | Set by the author when the numbers are not a real training result |
| `dry_run` | bool | `true` only for `--dry-run` output; those records are never written |
| `command` | string | Exact command line that produced the record |
| `git_commit` | string | `HEAD` at run time, `unknown` outside a git checkout |
| `git_dirty` | bool or null | Untracked/modified files at run time |
| `backend` | string | `cpu` or `cuda`; what ran, not what was available |
| `config_path` | string or null | YAML passed with `--config` |
| `config` | object | Effective architecture and training knobs (`embed`, `hidden`, `layers`, `window`, `block_size`, `grad_accum`, `optimizer`, `diffusion`, `lr`, `steps`, `seed`, `corpus`) |
| `hardware` | object | `cpu`, `cpu_count`, `os`, `python`, `gpu` (list), `cuda_visible` |
| `data` | object | `train` and `eval`, each `{path, bytes, sha256}` |
| `model` | object | `params` from the checkpoint's tensor headers |
| `training` | object | `steps`, `wall_seconds` |
| `throughput` | object | `steps_per_second`, `bytes_per_second` |
| `metrics` | object | `causal_bpb`, `masked_bpb`, `masked_acc` — copied from the eval record, `null` when unavailable |
| `generation` | object | `prompt`, `new_bytes`, `output_sha256`, `output_bytes`, `preview` |
| `artifacts` | object | `out_dir`, `checkpoint`, `checkpoint_sha256`, `eval`, `generated`, `summary` |

`throughput.bytes_per_second` is derived: `steps_per_second * window * grad_accum`. It is the
training stream rate implied by the step rate, not a measured I/O number, and it is labelled as
derived here so nobody quotes it as bandwidth.

`metrics` is whatever the evaluator produced. A null `masked_bpb`/`masked_acc` means diffusion
scoring was off for that checkpoint; it is not zero.

## Table columns

`fblt-leaderboard` prints one row per record:

| Column | Source |
| --- | --- |
| run | `run_id` |
| model | `config` as `e{embed}/h{hidden}/L{layers}` (per-stack counts appended when they differ) |
| status | `ok`, `unverified`, `synthetic` or `invalid` (see below) |
| model params | `model.params` |
| data (MB) | `data.train.bytes / 1e6` |
| steps | `training.steps` |
| hardware | `hardware.cpu` (truncated) plus the first GPU name when present |
| backend | `backend` |
| causal BPB | `metrics.causal_bpb`, 4 decimals |
| masked acc | `metrics.masked_acc`, 4 decimals |
| steps/s | `throughput.steps_per_second`, 1 decimal |
| commit | `git_commit[:8]`, with `-dirty` when `git_dirty` |
| flags | why the row is not `ok` |

Rows sort by status (`ok`, `unverified`, `synthetic`, `invalid`) and then ascending causal BPB.
Lower BPB is better. `--sort steps_per_second` and `--sort params` change the secondary order.

## Status rules

- `ok` — commit is known, `command` is present, `config` is non-empty, `checkpoint_sha256` is
  present, `git_dirty` is false or null.
- `unverified` — as above but the tree was dirty. The numbers may be fine; they are not
  reproducible from the commit alone.
- `synthetic` — the author set `synthetic: true`, or `git_commit` is missing/`unknown`, or
  `command` is empty, or `config` is empty. These rows stay in the JSONL and are hidden from
  the markdown table unless `--include-synthetic` is passed.
- `invalid` — a required field is missing, `format` is wrong, `schema_version` is not 1, the
  BPB is non-finite or negative, or `artifacts.checkpoint_sha256` is absent. Invalid records
  are never ranked; `--include-synthetic` lists them so they can be fixed.

Synthetic rows are useful for smoke tests and ablations that were not trained from a clean
tree. They are not comparable to real runs and the table says so.

## Submitting a result

1. Run the speedrun on a clean tree:

   ```bash
   git status --porcelain   # must be empty
   bash scripts/speedrun.sh --backend cuda --out runs/myrun
   ```

2. Check your own row first:

   ```bash
   python -m fblt.leaderboard --results runs/myrun
   ```

   The status must be `ok`. `unverified` means you had uncommitted changes; commit them and
   re-run, or mark the entry synthetic on purpose.

3. Add `<run_id>.json` (the `summary.json`) under `leaderboard/` and open a PR with the
   filled-in form from [`leaderboard-entry-template.md`](leaderboard-entry-template.md).

The PR must include the command and the config. A row without both is not reviewable: the
reviewer cannot tell what was trained on what.

## Limits

- The built-in config is a smoke test. It is not a scaling result, and its BPB is not
  comparable to the TinyStories checkpoints in `docs/BENCHMARKS.md` (different data, different
  parameter count, different steps).
- `data.train.bytes` counts the whole corpus file, not the bytes consumed by the run. Use
  `steps * window * grad_accum` for consumed bytes if you need them.
- CPU and CUDA runs on the same config produce numerically close but not identical losses; the
  backend column is part of the record so the two are never silently averaged.
- A record is only as good as its provenance fields. Editing `summary.json` by hand to raise a
  row breaks the checkpoint hash and the reviewer can tell.
