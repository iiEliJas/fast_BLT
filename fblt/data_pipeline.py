"""Offline manifest helpers mirroring the trainer's document-aware windows.

The authoritative implementation of the window rule is `blt_data_windows_manifest`
in `csrc/train_data.c`; `build/train_blt_d --doc-boundaries mask` is what decides
which windows a run actually trains on. This module only reproduces that rule for
offline inspection (window counts, offsets, coverage) so a corpus and its manifest
can be checked without starting a run. If the two ever disagree, the trainer wins.
"""

import json


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def _declared_size_index(manifest, split):
    return manifest.get("bytes", {}).get(split, {}).get("size_bytes")


def _declared_size_files(manifest, split):
    return manifest.get(f"{split}_size_bytes")


def _spans_from_index(manifest, split, corpus_size):
    entries = manifest["index"][split]
    spans = []
    prev_end = 0
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"index.{split}[{i}] is not an object")
        try:
            offset = int(entry["offset"])
            length = int(entry["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"index.{split}[{i}] needs integer offset and length") from exc
        if offset < 0 or length <= 0:
            raise ValueError(f"index.{split}[{i}] has offset {offset} and length {length}")
        if corpus_size is not None and offset + length > corpus_size:
            raise ValueError(
                f"index.{split}[{i}] (offset {offset} length {length}) runs past the corpus "
                f"({corpus_size} bytes)"
            )
        if offset < prev_end:
            raise ValueError(
                f"index.{split}[{i}] starts at {offset}, "
                f"before the previous entry ends at {prev_end}"
            )
        prev_end = offset + length
        spans.append((offset, length))
    return spans


def _spans_from_files(manifest, split, corpus_size):
    entries = manifest[f"{split}_files"]
    spans = []
    offset = 0
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{split}_files[{i}] is not an object")
        try:
            length = int(entry["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{split}_files[{i}] needs a positive integer size") from exc
        if length <= 0:
            raise ValueError(f"{split}_files[{i}] has non-positive size {length}")
        if corpus_size is not None and offset + length > corpus_size:
            raise ValueError(
                f"{split}_files[{i}] (offset {offset} length {length}) runs past the corpus "
                f"({corpus_size} bytes)"
            )
        spans.append((offset, length))
        # The corpus joins the documents with exactly one '\n'.
        offset += length + 1
    return spans


def doc_spans(manifest, split="train", corpus_size=None):
    """Return the `(offset, length)` document spans of `split` in manifest order.

    `index.<split>` entries carry both fields; `<split>_files` entries carry sizes
    only and are joined by a single `\\n` in the corpus, so offsets accumulate as
    `offset += size + 1`. When `corpus_size` is given, entries are bounds-checked
    against it and a declared size that disagrees is an error.
    """
    index = manifest.get("index", {})
    if isinstance(index.get(split), list):
        declared = _declared_size_index(manifest, split)
        if declared is not None and corpus_size is not None and int(declared) != int(corpus_size):
            raise ValueError(
                f"manifest bytes.{split}.size_bytes={declared} does not match the corpus "
                f"({corpus_size} bytes)"
            )
        return _spans_from_index(manifest, split, corpus_size)

    if isinstance(manifest.get(f"{split}_files"), list):
        declared = _declared_size_files(manifest, split)
        if declared is not None and corpus_size is not None and int(declared) != int(corpus_size):
            raise ValueError(
                f"manifest {split}_size_bytes={declared} does not match the corpus "
                f"({corpus_size} bytes)"
            )
        return _spans_from_files(manifest, split, corpus_size)

    raise ValueError(f"manifest has neither index.{split} nor {split}_files")


def _windows_in_doc(length, window):
    if window <= 0:
        raise ValueError(f"window must be >= 1, got {window}")
    # One byte per document is reserved for the next-byte target of the last
    # window, which sits at start + window and must stay inside the document.
    return (length - 1) // window if length > window else 0


def window_offsets(manifest, split, window, corpus_size=None):
    """Start offsets of every training window, in manifest order."""
    offsets = []
    for offset, length in doc_spans(manifest, split, corpus_size):
        offsets.extend(offset + j * window for j in range(_windows_in_doc(length, window)))
    return offsets


def window_summary(manifest, split, window, corpus_size=None):
    """Diagnostics matching the trainer's `[TRAIN] doc-boundaries=mask` line."""
    spans = doc_spans(manifest, split, corpus_size)
    counts = [_windows_in_doc(length, window) for _, length in spans]
    windows = sum(counts)
    docs_used = sum(1 for n in counts if n > 0)
    bytes_indexed = sum(length for _, length in spans)
    bytes_covered = windows * window
    return {
        "docs_total": len(spans),
        "docs_used": docs_used,
        "docs_dropped": len(spans) - docs_used,
        "windows": windows,
        "bytes_indexed": bytes_indexed,
        "bytes_covered": bytes_covered,
        "tail_bytes": bytes_indexed - bytes_covered,
        "coverage": 100.0 * bytes_covered / bytes_indexed if bytes_indexed else 0.0,
    }
