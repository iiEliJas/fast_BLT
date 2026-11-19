import json
import math
import struct
from pathlib import Path

import pytest

import fblt.speedrun as speedrun
from fblt._checkpoint import latest_checkpoint
from fblt.speedrun import DEFAULT_CONFIG, SUMMARY_FORMAT, aggregate_leaderboard, main, run_speedrun

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_checkpoint(path, params=66, step=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format_version": 2,
        "config": {"embed": 32, "hidden": 64, "layers": 1},
        "training": {"step": step, "optimizer": "adamw", "seed": 7},
    }
    raw = json.dumps(meta).encode()
    name = b"w"
    with open(path, "wb") as fh:
        fh.write(b"FBLT")
        fh.write(struct.pack("<I", 2))
        fh.write(struct.pack("<I", len(raw)))
        fh.write(raw)
        fh.write(struct.pack("<I", 1))
        fh.write(struct.pack("<H", len(name)))
        fh.write(name)
        fh.write(struct.pack("<I", 1))
        fh.write(struct.pack("<" + ("Q" if struct.calcsize("P") == 8 else "I"), params))
        fh.write(b"\x00" * (params * 4))


def _corpora(tmp_path):
    """Small raw byte files so unit tests never depend on data/ being built."""
    train = tmp_path / "corpus" / "train.bin"
    heldout = tmp_path / "corpus" / "heldout.bin"
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_bytes(b"unit test corpus " * 64)
    heldout.write_bytes(b"held out " * 32)
    return train, heldout


def _recorder(calls):
    def fake(argv, cwd=None, env=None, dry_run=False, log=print):
        argv = [str(part) for part in argv]
        calls.append({"argv": argv, "cwd": cwd, "env": env, "dry_run": dry_run})
        log("$ " + " ".join(argv))
        if dry_run:
            return 0
        if "fblt.train" in argv:
            _write_checkpoint(Path(cwd) / "ckpt" / "checkpoints" / "step_000002.fblt")
        if "fblt.infer" in argv:
            output = Path(argv[argv.index("--output") + 1])
            output.write_bytes(b"hello speedrun")
        return 0

    return fake


def test_dry_run_calls_no_command_and_creates_nothing(tmp_path, monkeypatch):
    calls = []
    lines = []
    monkeypatch.setattr(speedrun, "_run_command", _recorder(calls))
    monkeypatch.setattr(speedrun, "run_bpb_eval", lambda *a, **k: pytest.fail("eval in dry-run"))

    train, heldout = _corpora(tmp_path)
    out = tmp_path / "dry"
    record = run_speedrun(
        out, dry_run=True, steps=2, corpus=train, eval_corpus=heldout, log=lines.append
    )

    assert record["dry_run"] is True
    assert record["format"] == SUMMARY_FORMAT
    assert record["schema_version"] == 1
    assert record["metrics"] is None
    assert record["generation"] is None
    assert record["model"]["params"] is None
    assert record["training"]["wall_seconds"] is None
    assert record["throughput"] == {"steps_per_second": None, "bytes_per_second": None}
    assert record["hardware"]["cpu"]
    assert isinstance(record["hardware"]["gpu"], list)
    assert record["data"]["train"]["sha256"]
    assert all(call["dry_run"] is True for call in calls)
    assert not out.exists()

    joined = "\n".join(lines)
    assert "cmake" in joined
    assert "fblt.train" in joined
    assert "fblt.infer" in joined


def test_explicit_kwargs_override_config_file(tmp_path, monkeypatch):
    config = tmp_path / "cfg.yaml"
    config.write_text("steps: 50\nlr: 0.01\nseed: 3\nembed: 16\n")

    calls = []
    monkeypatch.setattr(speedrun, "_run_command", _recorder(calls))
    record = run_speedrun(config=config, out_dir=tmp_path / "out", dry_run=True, steps=2, lr=0.5)

    assert record["config"]["steps"] == 2
    assert record["config"]["lr"] == 0.5
    assert record["config"]["seed"] == 3
    assert record["config"]["embed"] == 16
    # Absent from the YAML, so the speedrun default survives instead of the
    # trainer default (hidden=128).
    assert record["config"]["hidden"] == DEFAULT_CONFIG["hidden"]
    assert record["config_path"] == str(config.resolve())

    train_argv = next(call["argv"] for call in calls if "fblt.train" in call["argv"])
    assert train_argv[train_argv.index("--steps") + 1] == "2"
    assert train_argv[train_argv.index("--lr") + 1] == "0.5"
    assert train_argv[train_argv.index("--seed") + 1] == "3"


def test_no_build_skips_cmake(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(speedrun, "_run_command", _recorder(calls))
    run_speedrun(tmp_path / "out", build=False, dry_run=True)

    assert all(call["argv"][0] != "cmake" for call in calls)


def test_main_dry_run_prints_plan(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(speedrun, "_run_command", _recorder([]))

    rc = main(["--out", str(tmp_path / "o"), "--dry-run", "--steps", "2"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "cmake" in captured.out
    assert "fblt.train" in captured.out
    assert "speedrun:" in captured.out
    assert not (tmp_path / "o").exists()


def test_summary_record_schema(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(speedrun, "_run_command", _recorder(calls))

    def fake_eval(checkpoint, corpus, **kwargs):
        record = {
            "format": "fblt-eval/bpb-v1",
            "metrics": {"causal_bpb": 5.5, "masked_bpb": None, "masked_acc": 0.25},
        }
        Path(kwargs["out"]).write_text(json.dumps(record))
        return record

    monkeypatch.setattr(speedrun, "run_bpb_eval", fake_eval)

    train, heldout = _corpora(tmp_path)
    out = tmp_path / "run"
    record = run_speedrun(
        out,
        build=False,
        steps=2,
        windows=2,
        new_bytes=8,
        corpus=train,
        eval_corpus=heldout,
        run_id="unit-run",
        log=lambda *_: None,
    )

    for key in (
        "format",
        "schema_version",
        "run_id",
        "created",
        "synthetic",
        "dry_run",
        "command",
        "git_commit",
        "git_dirty",
        "backend",
        "config_path",
        "config",
        "hardware",
        "data",
        "model",
        "training",
        "throughput",
        "metrics",
        "generation",
        "artifacts",
    ):
        assert key in record

    assert record["format"] == SUMMARY_FORMAT
    assert record["schema_version"] == 1
    assert record["run_id"] == "unit-run"
    assert record["dry_run"] is False
    assert isinstance(record["command"], str) and "fblt.speedrun" in record["command"]
    assert record["config"]["steps"] == 2
    assert record["config"]["corpus"] == str(train.resolve())
    assert record["metrics"] == {"causal_bpb": 5.5, "masked_bpb": None, "masked_acc": 0.25}
    assert record["model"]["params"] == 66
    assert record["training"]["steps"] == 2
    assert record["training"]["wall_seconds"] is not None
    assert record["throughput"]["steps_per_second"] is not None
    expected_bytes = record["throughput"]["steps_per_second"] * record["config"]["window"] * 1
    assert record["throughput"]["bytes_per_second"] == pytest.approx(expected_bytes, rel=1e-3)
    assert len(record["artifacts"]["checkpoint_sha256"]) == 64
    assert record["generation"]["prompt"] == "int main"
    assert record["generation"]["output_bytes"] == len(b"hello speedrun")
    assert len(record["generation"]["output_sha256"]) == 64
    assert record["generation"]["preview"].startswith("b'")

    on_disk = json.loads((out / "summary.json").read_text())
    assert on_disk == record


def test_aggregate_leaderboard_writes_json_array(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    record = {"format": SUMMARY_FORMAT, "run_id": "r1", "schema_version": 1}
    (nested / "summary.json").write_text(json.dumps(record))

    target = tmp_path / "lb.json"
    rows = aggregate_leaderboard(tmp_path, target)

    assert rows == [record]
    assert json.loads(target.read_text()) == [record]


@pytest.mark.skipif(
    not (REPO_ROOT / "build" / "train_blt_d").exists(), reason="requires a CPU build"
)
@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "train.bin").exists(), reason="requires data/train.bin"
)
def test_integration_real_run(tmp_path):
    out = tmp_path / "real"
    rc = main(
        [
            "--out",
            str(out),
            "--no-build",
            "--steps",
            "2",
            "--windows",
            "2",
            "--new-bytes",
            "8",
            "--run-id",
            "integration",
        ]
    )

    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["format"] == SUMMARY_FORMAT
    assert summary["git_commit"]
    assert summary["metrics"]["causal_bpb"] > 0
    assert math.isfinite(summary["metrics"]["causal_bpb"])
    assert summary["artifacts"]["checkpoint_sha256"]
    assert (out / "generated.bin").stat().st_size > 0
    assert latest_checkpoint(str(out / "ckpt")) is not None
