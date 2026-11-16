"""End-to-end checks for train_blt_d --doc-boundaries.

These drive the real binary rather than reimplementing its window rule, and are
skipped when the trainer has not been built so the plain Python job still
collects the module cleanly.
"""

import math
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("FBLT_BUILD_DIR", REPO_ROOT / "build"))
_CANDIDATES = [BUILD_DIR / "train_blt_d", BUILD_DIR / "train_blt_d.exe"]
TRAIN = next((p for p in _CANDIDATES if p.is_file()), _CANDIDATES[0])

pytestmark = pytest.mark.skipif(not TRAIN.is_file(), reason=f"train_blt_d is not built at {TRAIN}")

WINDOW = 32
DOCS = [b"A" * 101, b"B" * 40, b"C" * 200]
DOC_SPANS = [(0, 101), (102, 40), (143, 200)]


def write_fixtures(tmp_path):
    corpus = tmp_path / "corpus.bin"
    corpus.write_bytes(b"\n".join(DOCS))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        "{\n"
        '  "bytes": {"train": {"size_bytes": 343}},\n'
        '  "index": {"train": [\n'
        '    {"offset": 0, "length": 101},\n'
        '    {"offset": 102, "length": 40},\n'
        '    {"offset": 143, "length": 200}\n'
        "  ]}\n"
        "}\n"
    )
    return corpus, manifest


def base_args(corpus):
    return [
        "--corpus",
        str(corpus),
        "--backend",
        "cpu",
        "--embed",
        "16",
        "--hidden",
        "32",
        "--layers",
        "1",
        "--window",
        str(WINDOW),
        "--seed",
        "7",
        "--monitor-every",
        "100000",
        "--diffusion",
        "1",
    ]


def run_train(args, timeout=300):
    proc = subprocess.run(
        [str(TRAIN), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert proc.returncode == 0, (
        f"train_blt_d failed ({proc.returncode})\n"
        f"command: {TRAIN} {' '.join(args)}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc


def run_train_fails(args, timeout=120):
    proc = subprocess.run(
        [str(TRAIN), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert proc.returncode != 0, f"expected a failure, got exit 0\nstdout:\n{proc.stdout}"
    return proc


def read_loss_log(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        step, value = line.split()
        rows.append((int(step), float(value)))
    return rows


def read_batch_offsets(path):
    # step window_offset whash num_patches avg_patch_len ...
    return [int(line.split()[1]) for line in path.read_text().splitlines() if line.strip()]


def test_mask_and_off_differ(tmp_path):
    corpus, manifest = write_fixtures(tmp_path)
    off_log = tmp_path / "off.log"
    mask_log = tmp_path / "mask.log"

    run_train([*base_args(corpus), "--steps", "1", "--loss-log", str(off_log)])
    run_train(
        [
            *base_args(corpus),
            "--doc-boundaries",
            "mask",
            "--corpus-manifest",
            str(manifest),
            "--steps",
            "1",
            "--loss-log",
            str(mask_log),
        ]
    )

    off = read_loss_log(off_log)
    mask = read_loss_log(mask_log)
    assert len(off) == 1 and len(mask) == 1
    assert math.isfinite(off[0][1]) and math.isfinite(mask[0][1])
    assert off[0][1] != mask[0][1]


def test_mask_windows_stay_inside_documents(tmp_path):
    corpus, manifest = write_fixtures(tmp_path)
    batch_log = tmp_path / "batch.log"

    run_train(
        [
            *base_args(corpus),
            "--doc-boundaries",
            "mask",
            "--corpus-manifest",
            str(manifest),
            "--steps",
            "6",
            "--batch-log",
            str(batch_log),
        ]
    )

    offsets = read_batch_offsets(batch_log)
    assert len(offsets) == 6
    for offset in offsets:
        # The target byte at offset + window must also be inside the document.
        assert any(
            doc_offset <= offset and offset + WINDOW < doc_offset + doc_length
            for doc_offset, doc_length in DOC_SPANS
        ), f"window at {offset} crosses a document boundary"


def test_diagnostics_counts(tmp_path):
    corpus, manifest = write_fixtures(tmp_path)

    mask = run_train(
        [
            *base_args(corpus),
            "--doc-boundaries",
            "mask",
            "--corpus-manifest",
            str(manifest),
            "--steps",
            "1",
        ]
    )
    line = next(row for row in mask.stdout.splitlines() if "doc-boundaries=mask" in row)
    assert "docs=3" in line
    assert "usable_docs=3" in line
    assert "dropped_docs=0" in line
    assert "windows=10" in line

    off = run_train([*base_args(corpus), "--steps", "1"])
    # (343 - 32) // 32 = 9
    assert "[TRAIN] doc-boundaries=off windows=9 (flat stream)" in off.stdout


def test_sentinel_rejected(tmp_path):
    corpus, _ = write_fixtures(tmp_path)
    proc = run_train_fails([*base_args(corpus), "--doc-boundaries", "sentinel", "--steps", "1"])
    assert "sentinel is not implemented" in proc.stderr


def test_mask_requires_manifest(tmp_path):
    corpus, _ = write_fixtures(tmp_path)
    proc = run_train_fails([*base_args(corpus), "--doc-boundaries", "mask", "--steps", "1"])
    assert "requires --corpus-manifest" in proc.stderr


def test_manifest_requires_mask(tmp_path):
    corpus, manifest = write_fixtures(tmp_path)
    proc = run_train_fails([*base_args(corpus), "--corpus-manifest", str(manifest), "--steps", "1"])
    assert "requires --doc-boundaries mask" in proc.stderr


@pytest.mark.skipif(
    not (REPO_ROOT / "data/train.bin").is_file()
    or not (REPO_ROOT / "data/sample_corpus_manifest.json").is_file(),
    reason="sample corpus or its manifest is missing",
)
def test_sample_manifest_layout_end_to_end(tmp_path):
    proc = run_train(
        [
            "--corpus",
            str(REPO_ROOT / "data/train.bin"),
            "--backend",
            "cpu",
            "--embed",
            "16",
            "--hidden",
            "32",
            "--layers",
            "1",
            "--window",
            "64",
            "--seed",
            "7",
            "--monitor-every",
            "100000",
            "--diffusion",
            "1",
            "--doc-boundaries",
            "mask",
            "--corpus-manifest",
            str(REPO_ROOT / "data/sample_corpus_manifest.json"),
            "--steps",
            "1",
        ]
    )
    line = next(row for row in proc.stdout.splitlines() if "doc-boundaries=mask" in row)
    assert "docs=172" in line
    windows = re.search(r"windows=(\d+)", line)
    assert windows and int(windows.group(1)) > 0
