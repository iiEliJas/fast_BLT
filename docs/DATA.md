# Data: corpus layout and document boundaries

`train_blt_d` trains on a raw byte stream. By default it slices that stream into fixed
non-overlapping windows, ignoring whatever text structure the bytes happen to have. If the
corpus ships with a manifest that records where each source document starts, the trainer can
instead carve every training window out of a single document, so no window (and no
next-byte target) ever spans two documents.

Two flags control this: `--doc-boundaries off|mask` and `--corpus-manifest FILE`, with
`--manifest-split NAME` selecting the manifest split (default `train`).

## Corpus layout

- One flat file of raw bytes: no header, no framing, no per-document length prefix.
- Documents are joined by exactly **one** `\n`. Both prep scripts in `fblt/scripts/` follow
  this convention (`prep_tinystories.py`, `prep_stack_smol.py` append a `\n` after each
  document; `build_sample_corpus.py` inserts `SEP = b"\n"` between files).
- The corpus is read into memory once at startup, and the manifest is parsed into a
  temporary arena that is released right after the window table is built.
- `--eval-corpus` takes the same layout. Evaluation always slices it as a flat stream; it
  does not consult a manifest (see [Current limits](#current-limits)).

## Manifest layouts

Two layouts are accepted. The trainer picks `index.<split>` first and falls back to
`<split>_files`.

### `index.<split>` — offsets and lengths

Used by `data/tinystories/manifest.json` and `data/the-stack-smol/manifest.json`.

```json
{
  "bytes": {
    "train": {"size_bytes": 343}
  },
  "index": {
    "train": [
      {"offset": 0, "length": 101},
      {"offset": 102, "length": 40},
      {"offset": 143, "length": 200}
    ]
  }
}
```

Each entry needs `offset` and `length`. Extra keys (the prep scripts write `path`) are
ignored. `bytes.<split>.size_bytes` is optional; when present it must equal the corpus size.

### `<split>_files` — sizes only

Used by `data/sample_corpus_manifest.json`.

```json
{
  "train_size_bytes": 37,
  "train_files": [{"size": 10}, {"size": 20}, {"size": 5}]
}
```

This layout records sizes but no offsets, so offsets are reconstructed by accumulating
`offset += size + 1` — the `+1` is the separator. `<split>_size_bytes` is optional; when
present it must equal the corpus size. Because the reconstruction assumes the corpus really
is those documents joined by single newlines, the trainer also checks the separator byte at
`corpus[offset + size]` for every entry except the last, and refuses a corpus that does not
match.

## Validation

Every check below is fatal (message on stderr, exit 1). Nothing is silently skipped.

| Check | Applies to |
|-------|-----------|
| `index.<split>` or `<split>_files` exists | both |
| `offset`/`length`/`size` are integers | both |
| `length > 0`, `size > 0`, `offset >= 0` | both |
| entries are ordered and do not overlap (`offset >= previous end`) | `index.<split>` |
| `offset + length <= corpus size` | both |
| declared `size_bytes` equals the corpus size | both, when the key is present |
| `corpus[offset + length] == '\n'` for every entry but the last | `<split>_files` |
| the split yields at least one window | both |

`--manifest-split` defaults to `train`; `heldout` is a valid split name for corpora that
carry one, but it is only usable as a `--corpus` if you build that split's `.bin`.

`sentinel` is **not** implemented. It is still accepted by the argument parser only so the
trainer can reject it with an explicit message:

```
train_blt_d: --doc-boundaries sentinel is not implemented; use mask or off
```

## Boundary modes

| Mode | Behaviour |
|------|-----------|
| `off` (default) | Flat stream. Window `i` starts at `i * window`; `count = (corpus_size - window) / window`. No manifest allowed. |
| `mask` | Windows are carved from the manifest. Requires `--corpus-manifest`. |
| `sentinel` | Rejected; see above. |

The `mask` rule, per manifest entry of `length` bytes at `offset`:

```
n = (length - 1) // window   if length > window   else 0
window j starts at offset + j * window,  j = 0 .. n-1
```

The `length - 1` is the next-byte target rule: a window of `window` bytes also needs the
byte at `start + window` as its prediction target, and that target must stay inside the same
document. A document therefore needs to be strictly longer than one window to contribute
anything at all; documents with `length <= window` are dropped and counted in
`dropped_docs`. The tail of each document that does not fill a whole window is dropped too.

That is the price of strict alignment: whatever does not fill a whole window is never
trained on. On the sample corpus (window 32) the tails are 2783 of 1417887 indexed bytes
(0.2%); on TinyStories at window 64 they are 6175822 of 171447886 (3.6%). `bytes_indexed`
(sum of entry lengths, separators excluded) versus `bytes_covered` (`windows * window`) is
reported as `coverage`, and the uncovered remainder as `tail`.

`off` and `mask` are mutually exclusive with each other's flags: `--corpus-manifest` without
`mask` is an error, and `mask` without `--corpus-manifest` is an error.

## Diagnostics

The trainer prints one line at startup describing the window table, before the first step:

```text
[TRAIN] doc-boundaries=off windows=44313 (flat stream)
[TRAIN] doc-boundaries=mask manifest=... split=train docs=172 usable_docs=171 dropped_docs=1 windows=44222 coverage=99.80% tail=2783B
```

`docs` counts manifest entries considered, `usable_docs` those that produced at least one
window, `dropped_docs` the rest, `windows` the total window count, and `coverage` /
`tail` the tradeoff above.

Measured examples (CPU build, one step is enough — the line is printed before training).
`build_sample_corpus.py` regenerates the sample corpus from the working tree, so its document
and window counts grow as source files are added; the numbers below are a 2026-11-16
snapshot. TinyStories is a fixed dataset, so those numbers are stable.

```bash
# Sample corpus (172 source files) with the <split>_files layout.
build/train_blt_d --corpus data/train.bin --backend cpu --window 32 \
  --doc-boundaries mask --corpus-manifest data/sample_corpus_manifest.json --steps 1
```

```text
[TRAIN] doc-boundaries=mask manifest=data/sample_corpus_manifest.json split=train docs=172 usable_docs=171 dropped_docs=1 windows=44222 coverage=99.80% tail=2783B
```

```bash
# TinyStories (189,957 stories) with the index.<split> layout.
build/train_blt_d --corpus data/tinystories/train.bin --backend cpu --window 64 \
  --doc-boundaries mask --corpus-manifest data/tinystories/manifest.json --steps 1
```

```text
[TRAIN] doc-boundaries=mask manifest=data/tinystories/manifest.json split=train docs=189957 usable_docs=189957 dropped_docs=0 windows=2582376 coverage=96.40% tail=6175822B
```

For the same corpus at `--window 32` the flat mode reports more windows, but some of them
cross a document boundary — that is exactly what `mask` removes.

## Memory

The corpus is fully resident (`corpus_size` bytes) and so is the parsed manifest; the
manifest arena is sized from the file length and freed once the window table exists. The
window table itself is one `size_t` per window when `mask` is used — roughly 350 KB for the
sample corpus at window 32 and 20 MB for TinyStories at window 64 — and no table at all for
`off`, where offsets are computed as `i * window`.

## Current limits

- The periodic monitor and the `--eval-corpus` path still slice **flat** windows. A run
  trained with `--doc-boundaries mask` is therefore evaluated on boundary-crossing windows.
- No sharding (`--corpus-shards`/`--shard-index`/`--shard-count`), no streaming corpus
  reader, and no stateful sampler: windows are visited by the same deterministic affine
  permutation over `[0, windows)` in both modes.
- No sentinel token is inserted at document boundaries, and none is reserved in the byte
  alphabet.
- Resuming a run does not persist the window table; it is rebuilt from `--corpus`,
  `--corpus-manifest`, `--manifest-split` and `--window`, so those must be passed again.

## Offline inspection

`fblt/data_pipeline.py` reproduces the `mask` rule for tooling that must not start a
training run:

```python
from fblt.data_pipeline import load_manifest, doc_spans, window_offsets, window_summary

manifest = load_manifest("data/sample_corpus_manifest.json")
spans = doc_spans(manifest, "train", corpus_size=1418058)  # [(offset, length), ...]
offsets = window_offsets(manifest, "train", 32)  # [start, ...]
window_summary(manifest, "train", 32)  # trainer diagnostics as a dict
```

`window_summary` returns `docs_total`, `docs_used`, `docs_dropped`, `windows`,
`bytes_indexed`, `bytes_covered`, `tail_bytes` and `coverage` — the same numbers the
diagnostics line prints. `doc_spans` raises `ValueError` for a missing split, non-positive
lengths/sizes, overlapping entries, entries past the corpus, and a declared size that
disagrees with `corpus_size`.

The Python module is a mirror, not the implementation: `csrc/train_data.c` is authoritative
for what a run actually trains on, and an e2e test keeps the two honest.
