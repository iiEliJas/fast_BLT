import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest

from fblt._binary import resolve_binary
from fblt.eval import main, run_bpb_eval, write_csv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_missing_checkpoint_raises(tmp_path):
    out = tmp_path / "eval.json"
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run_bpb_eval(str(tmp_path / "nope.fblt"), "data/train.bin", out=str(out))
    assert not out.exists()


def test_missing_corpus_raises(tmp_path):
    ckpt = tmp_path / "empty.fblt"
    ckpt.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="corpus"):
        run_bpb_eval(str(ckpt), str(tmp_path / "nope.bin"))


def test_cli_missing_checkpoint_returns_nonzero(tmp_path, capsys):
    out = tmp_path / "eval.json"
    rc = main(
        [
            "--checkpoint",
            str(tmp_path / "nope.fblt"),
            "--corpus",
            "data/train.bin",
            "--out",
            str(out),
        ]
    )
    captured = capsys.readouterr()
    assert rc != 0
    assert "error:" in captured.err
    assert "checkpoint" in captured.err
    assert not out.exists()


def test_cli_rejects_unknown_task(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--task",
                "core",
                "--checkpoint",
                str(tmp_path / "m.fblt"),
                "--corpus",
                "data/train.bin",
                "--out",
                str(tmp_path / "eval.json"),
            ]
        )


def test_write_csv_flattens(tmp_path):
    record = {
        "task": "bpb",
        "protocol": {"window": 64, "patcher": "fixed"},
        "model": {"params": 1234},
        "metrics": {"causal_bpb": 1.5, "masked_bpb": None},
    }
    path = tmp_path / "out.csv"
    write_csv([record], str(path))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    header = lines[0].split(",")
    assert "metrics_causal_bpb" in header
    assert "protocol_window" in header
    assert "model_params" in header
    assert "task" in header


def test_bpb_record_end_to_end(tmp_path):
    try:
        train = resolve_binary("train_blt_d", "cpu")
        resolve_binary("eval_bpb", "cpu")
    except FileNotFoundError:
        pytest.skip("C binaries not built; skipping end-to-end BPB eval")

    ckpt = tmp_path / "model.fblt"
    subprocess.run(
        [
            train,
            "--corpus",
            "data/train.bin",
            "--embed",
            "16",
            "--hidden",
            "32",
            "--layers",
            "1",
            "--window",
            "64",
            "--block-size",
            "4",
            "--steps",
            "3",
            "--lr",
            "0.01",
            "--save-weights",
            str(ckpt),
            "--report-every",
            "1000",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=True,
    )

    out = tmp_path / "eval.json"
    rec = run_bpb_eval(
        str(ckpt),
        "data/train.bin",
        window=64,
        windows=2,
        block_size=4,
        patch_len=4,
        out=str(out),
    )

    assert rec["format"] == "fblt-eval/bpb-v1"
    assert rec["task"] == "bpb"
    assert rec["protocol"]["windows_evaluated"] == 2
    assert rec["metrics"]["causal_bpb"] > 0
    assert math.isfinite(rec["metrics"]["causal_bpb"])
    assert rec["model"]["params"] > 0
    assert rec["checkpoint_sha256"] == _sha256(ckpt)
    assert rec["corpus_sha256"] == _sha256(REPO_ROOT / "data" / "train.bin")
    assert rec["git_commit"] != "unknown"
    assert json.loads(out.read_text()) == rec
