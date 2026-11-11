"""End-to-end checks for train_blt_d --dtype and --fp8-format.

These tests exercise the real binary instead of reimplementing its checks. The
CPU cases are skipped when the trainer has not been built; the CUDA cases are
additionally skipped when no CUDA build exists, and skipped at runtime when the
installed device is too old for the requested math dtype.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("FBLT_BUILD_DIR", REPO_ROOT / "build"))
_CANDIDATES = [BUILD_DIR / "train_blt_d", BUILD_DIR / "train_blt_d.exe"]
TRAIN = next((p for p in _CANDIDATES if p.is_file()), _CANDIDATES[0])

_CUDA_CANDIDATES = [
    REPO_ROOT / "build-cuda" / "train_blt_d",
    REPO_ROOT / "build-cuda" / "train_blt_d.exe",
]
CUDA_TRAIN = next((p for p in _CUDA_CANDIDATES if p.is_file()), _CUDA_CANDIDATES[0])

pytestmark = pytest.mark.skipif(not TRAIN.is_file(), reason=f"train_blt_d is not built at {TRAIN}")

requires_cuda_build = pytest.mark.skipif(
    not CUDA_TRAIN.is_file(), reason=f"CUDA train_blt_d is not built at {CUDA_TRAIN}"
)


def write_corpus(path, size=960):
    # Deterministic bytes spanning the full range; no dependency on random().
    path.write_bytes(bytes((i * 37 + (i // 7) * 11) & 0xFF for i in range(size)))


def base_args(corpus, *, backend="cpu", window=32, embed=16, hidden=32, layers=1):
    return [
        "--corpus",
        str(corpus),
        "--backend",
        backend,
        "--diffusion",
        "1",
        "--embed",
        str(embed),
        "--hidden",
        str(hidden),
        "--layers",
        str(layers),
        "--block-size",
        "8",
        "--window",
        str(window),
        "--seed",
        "7",
        "--monitor-every",
        "1000",
    ]


def run_train(binary, args, env=None, timeout=120):
    return subprocess.run(
        [str(binary), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def run_cuda(args, timeout=120):
    # An empty CUDA_VISIBLE_DEVICES hides the device from the CUDA runtime.
    env = dict(os.environ)
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = run_train(CUDA_TRAIN, args, env=env, timeout=timeout)
    if proc.returncode != 0 and "requires a device with compute capability" in proc.stderr:
        pytest.skip("device lacks the required compute capability")
    assert proc.returncode == 0, (
        f"train_blt_d failed ({proc.returncode})\n"
        f"command: {CUDA_TRAIN} {' '.join(args)}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc


def test_fp32_trains_on_cpu(tmp_path):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)

    proc = run_train(TRAIN, [*base_args(corpus), "--dtype", "fp32", "--steps", "2"])
    assert proc.returncode == 0, (
        f"train_blt_d failed ({proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "dtype=fp32" in proc.stdout


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp8"])
def test_low_precision_requires_cuda(tmp_path, dtype):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)

    proc = run_train(TRAIN, [*base_args(corpus), "--dtype", dtype, "--steps", "1"])
    assert proc.returncode == 1
    assert "requires --backend cuda" in proc.stderr


@pytest.mark.parametrize("flag,dtype_name", [("--bf16", "bf16"), ("--fp8", "fp8-e4m3")])
def test_dtype_aliases_require_cuda(tmp_path, flag, dtype_name):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)

    proc = run_train(TRAIN, [*base_args(corpus), flag, "--steps", "1"])
    assert proc.returncode == 1
    assert "requires --backend cuda" in proc.stderr
    # The alias resolves to the same dtype as the long form, so the message
    # names it exactly like --dtype would.
    assert dtype_name in proc.stderr


def test_unknown_dtype_is_rejected(tmp_path):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)

    proc = run_train(TRAIN, [*base_args(corpus), "--dtype", "int4", "--steps", "1"])
    assert proc.returncode != 0
    assert "unknown --dtype" in proc.stderr


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_fp8_format_requires_fp8_dtype(tmp_path, fmt):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)

    proc = run_train(TRAIN, [*base_args(corpus), "--fp8-format", fmt, "--steps", "1"])
    assert proc.returncode == 1
    assert "only applies to --dtype fp8" in proc.stderr


def test_unknown_fp8_format_is_rejected(tmp_path):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)

    proc = run_train(TRAIN, [*base_args(corpus), "--fp8-format", "e3m0", "--steps", "1"])
    assert proc.returncode == 1
    assert "unknown --fp8-format" in proc.stderr


@requires_cuda_build
def test_fp8_e4m3_trains_on_cuda(tmp_path):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)

    proc = run_cuda([*base_args(corpus, backend="cuda"), "--dtype", "fp8", "--steps", "1"])
    assert "dtype=fp8-e4m3" in proc.stdout


@requires_cuda_build
def test_fp8_e5m2_trains_on_cuda(tmp_path):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)

    proc = run_cuda(
        [
            *base_args(corpus, backend="cuda"),
            "--dtype",
            "fp8",
            "--fp8-format",
            "e5m2",
            "--steps",
            "1",
        ]
    )
    assert "dtype=fp8-e5m2" in proc.stdout


@requires_cuda_build
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_wide_float_trains_on_cuda(tmp_path, dtype):
    corpus = tmp_path / "corpus.bin"
    write_corpus(corpus)

    proc = run_cuda([*base_args(corpus, backend="cuda"), "--dtype", dtype, "--steps", "1"])
    assert f"dtype={dtype}" in proc.stdout
