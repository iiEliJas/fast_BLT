"""Offline document-window helpers, checked against the C rule in csrc/train_data.c."""

import json

import pytest

from fblt.data_pipeline import doc_spans, load_manifest, window_offsets, window_summary

WINDOW = 32
CORPUS_SIZE = 343

INDEX_MANIFEST = {
    "bytes": {"train": {"size_bytes": CORPUS_SIZE}},
    "index": {
        "train": [
            {"offset": 0, "length": 101},
            {"offset": 102, "length": 40},
            {"offset": 143, "length": 200},
        ]
    },
}

INDEX_SPANS = [(0, 101), (102, 40), (143, 200)]
INDEX_OFFSETS = [0, 32, 64, 102, 143, 175, 207, 239, 271, 303]


def inside_spans(start, spans, window=WINDOW):
    # The next-byte target at start + window must also stay inside the document.
    return any(start >= off and start + window < off + length for off, length in spans)


def test_load_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(INDEX_MANIFEST))
    assert load_manifest(path) == INDEX_MANIFEST


def test_index_layout_spans_and_offsets():
    assert doc_spans(INDEX_MANIFEST, "train") == INDEX_SPANS
    assert window_offsets(INDEX_MANIFEST, "train", WINDOW) == INDEX_OFFSETS
    assert len(window_offsets(INDEX_MANIFEST, "train", WINDOW)) == 10


def test_index_layout_windows_never_leave_their_document():
    spans = doc_spans(INDEX_MANIFEST, "train")
    for start in window_offsets(INDEX_MANIFEST, "train", WINDOW):
        assert inside_spans(start, spans), f"window at {start} crosses a document boundary"


def test_files_layout_offsets_match_the_c_unit_test():
    manifest = {
        "train_size_bytes": 37,
        "train_files": [{"size": 10}, {"size": 20}, {"size": 5}],
    }
    assert doc_spans(manifest, "train") == [(0, 10), (11, 20), (32, 5)]
    # One '\n' separator between the joined documents shifts each entry by one.
    assert window_offsets(manifest, "train", 4) == [0, 4, 11, 15, 19, 23, 32]


def test_short_document_is_dropped_and_counted():
    manifest = {"index": {"train": [{"offset": 0, "length": 32}, {"offset": 33, "length": 100}]}}
    summary = window_summary(manifest, "train", WINDOW)
    assert summary["docs_total"] == 2
    assert summary["docs_used"] == 1
    assert summary["docs_dropped"] == 1
    assert summary["windows"] == 3
    assert window_offsets(manifest, "train", WINDOW) == [33, 65, 97]


def test_window_summary_numbers():
    summary = window_summary(INDEX_MANIFEST, "train", WINDOW, CORPUS_SIZE)
    assert summary == {
        "docs_total": 3,
        "docs_used": 3,
        "docs_dropped": 0,
        "windows": 10,
        "bytes_indexed": 341,
        "bytes_covered": 320,
        "tail_bytes": 21,
        "coverage": pytest.approx(100.0 * 320 / 341),
    }
    assert round(summary["coverage"], 2) == 93.84


def test_window_summary_of_empty_manifest_is_zero():
    summary = window_summary({"index": {"train": []}}, "train", WINDOW)
    assert summary == {
        "docs_total": 0,
        "docs_used": 0,
        "docs_dropped": 0,
        "windows": 0,
        "bytes_indexed": 0,
        "bytes_covered": 0,
        "tail_bytes": 0,
        "coverage": 0.0,
    }


def test_missing_split_raises():
    with pytest.raises(ValueError, match="neither index.heldout nor heldout_files"):
        doc_spans(INDEX_MANIFEST, "heldout")


def test_overlapping_index_entries_raise():
    manifest = {"index": {"train": [{"offset": 0, "length": 101}, {"offset": 100, "length": 40}]}}
    with pytest.raises(ValueError, match="before the previous entry ends"):
        doc_spans(manifest, "train")


def test_entry_past_corpus_raises():
    manifest = {"index": INDEX_MANIFEST["index"]}
    with pytest.raises(ValueError, match="runs past the corpus"):
        doc_spans(manifest, "train", CORPUS_SIZE - 1)


def test_declared_size_mismatch_raises():
    manifest = {
        "bytes": {"train": {"size_bytes": 999}},
        "index": {"train": [{"offset": 0, "length": 101}]},
    }
    with pytest.raises(ValueError, match="does not match the corpus"):
        doc_spans(manifest, "train", 101)


def test_files_declared_size_mismatch_raises():
    manifest = {"train_size_bytes": 999, "train_files": [{"size": 10}]}
    with pytest.raises(ValueError, match="does not match the corpus"):
        doc_spans(manifest, "train", 10)


@pytest.mark.parametrize("length", [0, -5])
def test_non_positive_length_raises(length):
    manifest = {"index": {"train": [{"offset": 0, "length": length}]}}
    with pytest.raises(ValueError):
        doc_spans(manifest, "train")


def test_non_positive_file_size_raises():
    manifest = {"train_files": [{"size": 10}, {"size": 0}]}
    with pytest.raises(ValueError, match="non-positive size"):
        doc_spans(manifest, "train")
