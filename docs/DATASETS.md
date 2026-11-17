# Datasets: download and cache

`fblt.dataset` fetches a dataset by name into a shared cache and records what it fetched.
It handles the download, the checksum and the manifest; it does not convert the bytes into a
training corpus. That is still the job of the prep scripts in `fblt/scripts/` (see
[Corpus preparation](#corpus-preparation)).

## Cache layout

The cache root is `$FBLT_CACHE`, defaulting to `~/.cache/fblt`:

```
$FBLT_CACHE/datasets/<name>/
  .lock                  advisory lock held during download
  shard-00000.bin        raw bytes exactly as published by the source
  manifest.json          written last, after the checksum matches
```

`shard-00000.bin` keeps no extension of the upstream file. Datasets are stored as one flat
file because that is what the trainer and the prep scripts expect.

## CLI

```bash
python -m fblt.dataset --list
python -m fblt.dataset --download tinyshakespeare
python -m fblt.dataset --verify tinyshakespeare
```

`--list` prints every registered dataset with its size, license and URL. `--download` prints
the manifest path and is a no-op when the manifest already exists. `--verify` re-hashes the
shard and compares it against the manifest, exiting non-zero on a mismatch or a missing
file. Download and verify failures exit 1 with the reason on stderr.

To put the cache somewhere disposable:

```bash
FBLT_CACHE=/tmp/fblt-cache python -m fblt.dataset --download tinyshakespeare
FBLT_CACHE=/tmp/fblt-cache python -m fblt.dataset --verify tinyshakespeare
```

## Registered datasets

| Name | Bytes | License | Source | Upstream file |
| --- | ---: | --- | --- | --- |
| `tinyshakespeare` | 1,115,394 | public domain | `karpathy/char-rnn` | `data/tinyshakespeare/input.txt` |
| `tinystories` | 22,502,601 | CDLA-Sharing-1.0 | `roneneldan/TinyStories` | `TinyStoriesV2-GPT4-valid.txt` |

`tinystories` points at the GPT-4-valid split, not the full training set: it is small enough
to fetch on a laptop and is the split you want for held-out evaluation anyway. The full
corpus is assembled by `fblt/scripts/prep_tinystories.py` through the `datasets` library.

### Checksums

The recorded SHA-256 values were computed on 2026-11-17 by downloading each URL twice and
comparing the two digests; both entries matched on both downloads. To refresh one after an
upstream change, download it twice, confirm the digests agree, then update `sha256` and
`size` in `fblt/dataset.py` and this table in the same commit.

```bash
curl -sL -o /tmp/a "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
curl -sL -o /tmp/b "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
sha256sum /tmp/a /tmp/b
```

Do not edit a checksum from a single download: a truncated or redirected fetch produces a
digest that looks fine and fails for everyone else.

## Manifest

```json
{
  "name": "tinyshakespeare",
  "url": "https://raw.githubusercontent.com/.../input.txt",
  "version": "char-rnn@master:input.txt",
  "license": "public-domain",
  "source": "karpathy/char-rnn (Shakespeare)",
  "size": 1115394,
  "sha256": "86c4e6aa...dc565ed",
  "shards": 1
}
```

`sha256` and `size` come from the registry, i.e. they are the expected values, and the
downloaded bytes are only published once they match. `version` identifies the upstream
revision as precisely as the source allows (branch plus filename); it is a label, not a
pin — the checksum is what actually pins the content.

## Adding an entry

Add a `DatasetSpec` to `REGISTRY` in `fblt/dataset.py` with `name`, `url`, `sha256`, `size`,
`license`, `source` and `version`, then add a row to the table above. A spec is rejected
before any file is written when the URL is empty or the checksum is not 64 lowercase hex
characters, and `tests/py/test_dataset.py::test_registry_entries_are_verified` fails the
same way, so a placeholder entry cannot merge.

Stay with sources that serve the file directly over HTTPS. Hugging Face `resolve` URLs work
(the `tinystories` entry goes through the CDN redirect); dataset viewer pages do not.

## Corpus preparation

Once the bytes are cached, the prep scripts convert them:

```bash
python fblt/scripts/prep_tinystories.py --out-dir data/tinystories
python fblt/scripts/build_sample_corpus.py
```

The output is a flat byte stream plus a manifest with per-document offsets, which is the
layout `--corpus-manifest` reads. See [docs/DATA.md](DATA.md).

## Current limits

- Single-file datasets only. `shards` is always written as 1; per-shard checksums and
  multi-shard downloads are not implemented.
- Retries start from byte zero. There is no HTTP `Range` resume, so a retry re-downloads the
  whole file; only complete, verified bodies are published.
- The lock is POSIX `flock`. On Windows (`fcntl` unavailable) a download is not protected
  against a concurrent process, and there is no non-blocking or timeout mode, so a stale
  holder blocks until its process exits.
- No free-space check before downloading.
- The cache entry is the upstream file; no decompression or splitting is applied.

## Tests

`tests/py/test_dataset.py` covers the registry invariants, the local-file download and
verify roundtrip, the no-op rerun, a tampered shard, a checksum mismatch and the two
validation failures that must not leave a manifest behind. The tests use local files only;
they do not hit the network.
