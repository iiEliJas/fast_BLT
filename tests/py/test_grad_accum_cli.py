"""End-to-end checks for train_blt_d --grad-accum.

These tests exercise the real binary instead of reimplementing its loop. They
are skipped when the trainer has not been built, so the plain Python CI job
still collects the module cleanly.
"""

import math
import os
import struct
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("FBLT_BUILD_DIR", REPO_ROOT / "build"))
_CANDIDATES = [BUILD_DIR / "train_blt_d", BUILD_DIR / "train_blt_d.exe"]
TRAIN = next((p for p in _CANDIDATES if p.is_file()), _CANDIDATES[0])

pytestmark = pytest.mark.skipif(not TRAIN.is_file(), reason=f"train_blt_d is not built at {TRAIN}")


def write_corpus(path, size=960):
    # Deterministic bytes spanning the full range; no dependency on random().
    path.write_bytes(bytes((i * 37 + (i // 7) * 11) & 0xFF for i in range(size)))


def base_args(corpus, *, window=32, embed=16, hidden=32, layers=1):
    return [
        "--corpus",
        str(corpus),
        "--backend",
        "cpu",
        "--diffusion",
        "1",
        "--embed",
        str(embed),
        "--hidden",
        str(hidden),
        "--layers",
        str(layers),
        "--block-size",
        "4",
        "--window",
        str(window),
        "--seed",
        "7",
        "--monitor-every",
        "100000",
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


def read_checkpoint(path):
    """Parse the v2 .fblt format written by blt_model_save_ex.

    Layout: magic, u32 version, u32 meta_len, meta bytes, u32 tensor count,
    then per tensor: u16 name_len, name, u32 ndim, ndim * size_t dims, f32 data.
    """
    raw = path.read_bytes()
    off = 0
    assert raw[off : off + 4] == b"FBLT"
    off += 4
    (version,) = struct.unpack_from("<I", raw, off)
    off += 4
    assert version == 2
    (meta_len,) = struct.unpack_from("<I", raw, off)
    off += 4
    meta = raw[off : off + meta_len]
    off += meta_len
    (count,) = struct.unpack_from("<I", raw, off)
    off += 4

    tensors = {}
    for _ in range(count):
        (name_len,) = struct.unpack_from("<H", raw, off)
        off += 2
        name = raw[off : off + name_len].decode("utf-8")
        off += name_len
        (ndim,) = struct.unpack_from("<I", raw, off)
        off += 4
        # size_t is 8 bytes on every platform this suite runs on.
        dims = struct.unpack_from(f"<{ndim}Q", raw, off)
        off += 8 * ndim
        numel = math.prod(dims) if ndim else 1
        tensors[name] = struct.unpack_from(f"<{numel}f", raw, off)
        off += 4 * numel
    assert off == len(raw), f"trailing bytes in {path}"
    return meta, tensors


def read_loss_log(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        step, value = line.split()
        rows.append((int(step), float(value)))
    return rows


def test_grad_accum_zero_is_rejected(tmp_path):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)
    proc = subprocess.run(
        [str(TRAIN), *base_args(corpus), "--grad-accum", "0", "--steps", "1"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode != 0
    assert "grad-accum" in proc.stderr.lower()


def test_loss_log_is_one_averaged_line_per_optimizer_step(tmp_path):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)
    acc_log = tmp_path / "acc.log"
    single_log = tmp_path / "single.log"

    # lr=0 keeps the weights fixed, so the micro-batches in the accumulated run
    # see exactly the same losses as the corresponding one-window steps.
    run_train(
        [
            *base_args(corpus),
            "--optimizer",
            "sgd",
            "--lr",
            "0",
            "--grad-accum",
            "2",
            "--steps",
            "2",
            "--report-every",
            "100",
            "--loss-log",
            str(acc_log),
        ]
    )
    run_train(
        [
            *base_args(corpus),
            "--optimizer",
            "sgd",
            "--lr",
            "0",
            "--grad-accum",
            "1",
            "--steps",
            "4",
            "--report-every",
            "100",
            "--loss-log",
            str(single_log),
        ]
    )

    acc = read_loss_log(acc_log)
    single = read_loss_log(single_log)

    # One line per optimizer step, independent of --report-every.
    assert [step for step, _ in acc] == [1, 2]
    assert [step for step, _ in single] == [1, 2, 3, 4]

    # Each accumulated step is the mean of its micro-batch losses. Values are
    # printed with %.6f, so allow one rounding unit.
    for i, (_, value) in enumerate(acc):
        expected = 0.5 * (single[2 * i][1] + single[2 * i + 1][1])
        assert math.isclose(value, expected, rel_tol=0, abs_tol=2e-6), (
            f"step {i + 1}: loss log {value} != micro average {expected}"
        )


def test_epoch_and_throughput_account_for_accumulation(tmp_path):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)  # 960 bytes, window 128 -> (960 - 128) / 128 = 6 windows

    proc = run_train(
        [
            *base_args(corpus, window=128, embed=32, hidden=64, layers=2),
            "--optimizer",
            "sgd",
            "--lr",
            "0.01",
            "--grad-accum",
            "2",
            "--steps",
            "10",
            "--report-every",
            "10",
        ]
    )
    out = proc.stdout

    # 10 optimizer steps * 2 micro-batches = 20 windows over 6 windows/epoch.
    assert "6 windows (~4 epochs)" in out

    line = next(line for line in out.splitlines() if line.startswith("step "))
    assert "epoch 3/4" in line

    # B/s is window * accum / dt; ms/step is 1000 * dt / steps. Their product
    # is fixed by the accounting, so this catches a missing accum factor.
    import re

    match = re.search(r"ms/step\s+([0-9.]+)\s+B/s\s+([0-9.]+)", line)
    assert match, line
    ms_per_step = float(match.group(1))
    bytes_per_s = float(match.group(2))
    expected_bps = 128 * 2 * 1000.0 / ms_per_step
    assert math.isclose(bytes_per_s, expected_bps, rel_tol=0.05)


def test_accumulated_gradient_matches_manual_average(tmp_path):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)

    common = [
        *base_args(corpus),
        "--optimizer",
        "sgd",
        "--lr",
        "1",
        "--max-norm",
        "1e30",
        "--report-every",
        "100",
    ]
    # Three runs from the same seeded init: the first two see one micro-batch
    # each, the third accumulates and averages both. With SGD lr=1 the averaged
    # update makes the final weights the mean of the two single-window results.
    run_train(
        [*common, "--grad-accum", "1", "--steps", "1", "--save-weights", str(tmp_path / "a.fblt")]
    )
    run_train(
        [
            *common,
            "--grad-accum",
            "1",
            "--start-step",
            "1",
            "--steps",
            "1",
            "--save-weights",
            str(tmp_path / "b.fblt"),
        ]
    )
    run_train(
        [*common, "--grad-accum", "2", "--steps", "1", "--save-weights", str(tmp_path / "c.fblt")]
    )

    _, first = read_checkpoint(tmp_path / "a.fblt")
    _, second = read_checkpoint(tmp_path / "b.fblt")
    _, averaged = read_checkpoint(tmp_path / "c.fblt")
    assert first.keys() == second.keys() == averaged.keys()

    for name in first:
        for i, (x, y, z) in enumerate(zip(first[name], second[name], averaged[name])):
            expected = 0.5 * (x + y)
            assert math.isclose(z, expected, rel_tol=1e-4, abs_tol=1e-5), (
                f"{name}[{i}]: accumulated {z} != mean of single windows {expected}"
            )


def test_grad_accum_one_is_bit_identical_to_default(tmp_path):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)
    common = [*base_args(corpus), "--steps", "2", "--report-every", "100"]

    default_path = tmp_path / "default.fblt"
    explicit_path = tmp_path / "explicit.fblt"
    run_train([*common, "--save-weights", str(default_path)])
    run_train([*common, "--grad-accum", "1", "--save-weights", str(explicit_path)])

    assert default_path.read_bytes() == explicit_path.read_bytes()


def test_resume_matches_uninterrupted_run(tmp_path):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)
    common = [
        *base_args(corpus),
        "--optimizer",
        "adamw",
        "--lr",
        "0.01",
        "--grad-accum",
        "2",
        "--report-every",
        "100",
    ]

    full_weights = tmp_path / "full.fblt"
    full_optim = tmp_path / "full.opt"
    part_weights = tmp_path / "part.fblt"
    part_optim = tmp_path / "part.opt"
    resumed_weights = tmp_path / "resumed.fblt"
    resumed_optim = tmp_path / "resumed.opt"

    run_train(
        [
            *common,
            "--steps",
            "8",
            "--save-weights",
            str(full_weights),
            "--save-optim",
            str(full_optim),
        ]
    )
    run_train(
        [
            *common,
            "--steps",
            "4",
            "--save-weights",
            str(part_weights),
            "--save-optim",
            str(part_optim),
        ]
    )
    run_train(
        [
            *common,
            "--steps",
            "4",
            "--load-weights",
            str(part_weights),
            "--load-optim",
            str(part_optim),
            "--save-weights",
            str(resumed_weights),
            "--save-optim",
            str(resumed_optim),
        ]
    )

    # Resuming at step 4 must reproduce the uninterrupted 8-step trajectory,
    # including AdamW moments and the bias-correction counter.
    assert full_weights.read_bytes() == resumed_weights.read_bytes()
    assert full_optim.read_bytes() == resumed_optim.read_bytes()
