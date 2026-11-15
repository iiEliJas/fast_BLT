# Evaluation

`fblt-eval` wraps the `eval_bpb` C tool. It scores a checkpoint over a raw byte corpus
and writes one JSON record per run. Every number in that record is produced by the C
binary; the Python wrapper adds file hashes and the git commit so a record can be traced
back to exact inputs.

## What a BPB record is

Bits per byte (BPB) is the model's average cross-entropy over the corpus divided by
`ln(2)`. Byte-level models predict bytes, so a byte is the natural unit: it does not
depend on a tokenizer, a vocabulary, or a merge table. Two checkpoints are compared on
BPB, not on token-level perplexity, because token-level numbers change with the
tokenizer.

A record holds three groups of numbers:

- `metrics.causal_bpb` — the causal decoder's BPB over every byte in each window except
  the first, since the first byte has no context.
- `metrics.masked_bpb` / `metrics.masked_acc` — BPB and exact-byte accuracy over the
  block-diffusion masked cells, or `null` when diffusion scoring is off.
- `model.params` and `metrics.clean_bytes` / `metrics.masked_cells` — the weight count and
  the number of scored units behind the two BPBs.

## Quickstart

```bash
cmake --build build --target eval_bpb

fblt-train --backend cpu --corpus data/train.bin \
  --embed 32 --hidden 64 --layers 1 --window 128 --block-size 8 \
  --steps 150 --lr 0.05 --save-weights runs/toy/model.fblt

python -m fblt.eval \
  --checkpoint runs/toy/model.fblt --corpus data/heldout.bin \
  --window 128 --windows 8 --block-size 8 --patcher fixed --layout paper \
  --out runs/toy/bpb.json
```

`python -m fblt.eval` and the installed `fblt-eval` entry point are the same function.
The command prints a one-row markdown table and the full record to stdout, and writes
the same record to `--out`.

## Flags

Checkpoint and corpus are required and checked before the binary is resolved: a bad path
raises instead of producing a record of zeros.

| Flag | Default | Description |
|------|---------|-------------|
| `--task {bpb}` | `bpb` | Evaluation task. Phase 1 supports only `bpb`. |
| `--checkpoint FILE` | *required* | Model checkpoint (`.fblt`). |
| `--corpus FILE` | *required* | Raw byte corpus. |
| `--backend {cpu,cuda}` | `cpu` | Execution backend. |
| `--out FILE` | `eval.json` | JSON record destination. |
| `--csv FILE` | none | Optional flattened CSV of the record. |
| `--window W` | checkpoint metadata, then 512 | Bytes per window. |
| `--windows N` | 20 | Windows requested. |
| `--skip S` | 0 | Windows skipped before the first one. |
| `--patcher {fixed,entropy}` | `fixed` | Segmentation strategy. |
| `--patch-len L` | 4 | Fixed-stride patch length. |
| `--entropy-lm FILE` | none | Entropy LM weights; required by `--patcher entropy`. |
| `--entropy-threshold F` | 3.0 (C default) | Global entropy threshold for entropy patching. |
| `--block-size B` | checkpoint metadata, then 4 | Diffusion block size. |
| `--layout {paper,novel}` | `paper` | Masked-cell layout. |
| `--t F` | 1.0 | Diffusion timestep used for batch masking. |
| `--seed N` | 7 | Base RNG seed for masking. |
| `--embed N` | checkpoint metadata | Embedding width. |
| `--hidden N` | checkpoint metadata | Hidden width. |
| `--enc-layers N` | checkpoint metadata | Encoder layers. |
| `--glob-layers N` | checkpoint metadata | Global layers. |
| `--dec-layers N` | checkpoint metadata | Decoder layers. |
| `--cross-attn {all,last}` | checkpoint metadata, then `all` | Cross-attention placement. |
| `--d0 {learned,zeros}` | checkpoint metadata, then `learned` | Block-row D_0 policy. |
| `--diffusion {0,1}` | checkpoint metadata, then 1 | Score masked block cells. |

v2 checkpoints store the model shape and window/block size, so the shape flags are
usually unnecessary. v1 checkpoints carry no metadata; pass `--embed --hidden
--enc-layers --glob-layers --dec-layers` (and `--window`, `--block-size`) explicitly.

## Record schema

Top-level keys: `format` (`fblt-eval/bpb-v1`), `task`, `checkpoint`, `corpus`,
`checkpoint_sha256`, `corpus_sha256`, `backend`, `protocol`, `model`, `metrics`,
`git_commit`, `timestamp`. `entropy_lm_sha256` is added after `protocol` when
`--entropy-lm` is used. The wrapper computes the three hashes and `git_commit`; the C
tool writes everything else, including `timestamp` (UTC).

This is a real record from a toy run (32/64/1, 4 windows of 128 bytes on
`data/heldout.bin`, fixed patching, paper layout):

```json
{
  "format": "fblt-eval/bpb-v1",
  "task": "bpb",
  "checkpoint": "/tmp/eval_bpb_check/model.fblt",
  "corpus": "data/heldout.bin",
  "checkpoint_sha256": "f1b00382a6109da9ee43c35243d3477ff60ed0e11bf3557710f4ad8a1bb78468",
  "corpus_sha256": "f92f9df1da35185bba7f6720b88a67cbf46d44397a28d81a6ff6515db3341080",
  "backend": "cpu",
  "protocol": {
    "window": 128,
    "windows_requested": 4,
    "windows_evaluated": 4,
    "skip": 0,
    "patcher": "fixed",
    "patch_len": 4,
    "entropy_lm": null,
    "entropy_threshold": null,
    "block_size": 8,
    "layout": "paper",
    "t": 1.0,
    "seed": 7
  },
  "model": {
    "embed": 32,
    "hidden": 64,
    "enc_layers": 1,
    "glob_layers": 1,
    "dec_layers": 1,
    "cross_attn": "all",
    "d0_mode": "learned",
    "diffusion": 1,
    "params": 66976
  },
  "metrics": {
    "causal_bpb": 7.399506,
    "masked_bpb": 7.339484,
    "masked_acc": 0.038934,
    "clean_bytes": 508,
    "masked_cells": 976
  },
  "git_commit": "5b3b493fea4a2e0b7243f50f6889ceec2b0acc57",
  "timestamp": "2026-11-15T21:07:52Z"
}
```

## Comparability

Two records are only comparable when their `protocol` blocks match field for field:
`window`, `windows_requested`, `skip`, `patcher`, `patch_len`, `entropy_lm`,
`entropy_threshold`, `block_size`, `layout`, `t` and `seed`. They must also share the same
`corpus_sha256`; a different corpus makes the numbers unrelated regardless of protocol.

`layout: paper` scores the training layout, where the masked cells are the block rows
the trainer optimizes. `layout: novel` scores one aligned-novel block per window, so it
reports far fewer `masked_cells` (32 versus 976 in the example above, 4 windows of 128
bytes at block size 8) and a noisier `masked_acc`. The two layouts answer different
questions and their `masked_*` numbers should not be tabulated together.

Entropy patching is stochastic: the segmentation depends on the entropy LM's softmax
over the window, and masking additionally draws timesteps from `--seed`. Reproduce a
record only with the same entropy LM file, the same `entropy_lm`/`entropy_lm_sha256`
pair, the same threshold and the same seed. The record stores all four.

Byte-level BPB is not directly comparable to token-level benchmark numbers. A token
carries several bytes, so token perplexities, token-level accuracies and byte BPB live on
different denominators.

## CPU vs CUDA

The same checkpoint and protocol produce the same record except for `backend`. On the
toy checkpoint above (8 windows of 128 bytes, block size 8, paper layout), CPU and CUDA
were bit-identical: `causal_bpb 7.401415`, `masked_bpb 7.328648`, `masked_acc 0.032275`.
The metrics are printed to six decimals, so a marginal difference can still be hidden by
rounding; larger models and longer windows may drift in the last digits because of
different accumulation orders.

## Scope

Phase 1 is corpus BPB only. `--task` accepts `bpb` and nothing else; an unknown task is
an argparse error rather than a fabricated score. The CORE task harness (ARC, MMLU,
GSM8K, HumanEval, few-shot prompt rendering, centered accuracy, `bench/eval_suite.py`)
is phase 2 and is not implemented.
