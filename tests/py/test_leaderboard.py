import json

from fblt.leaderboard import (
    classify,
    dedupe,
    flags,
    load_records,
    main,
    sort_records,
    to_markdown,
    validate,
)

FORMAT = "fblt-speedrun/v1"


def _record(run_id="r1", bpb=5.0, **overrides):
    record = {
        "format": FORMAT,
        "schema_version": 1,
        "run_id": run_id,
        "created": "2026-01-01T00:00:00Z",
        "synthetic": False,
        "dry_run": False,
        "command": "python -m fblt.speedrun --out runs/x",
        "git_commit": "abcdef1234567890",
        "git_dirty": False,
        "backend": "cpu",
        "config_path": None,
        "config": {
            "embed": 32,
            "hidden": 64,
            "layers": 1,
            "window": 128,
            "block_size": 8,
            "grad_accum": 1,
            "optimizer": "adamw",
            "diffusion": 1,
            "lr": 0.05,
            "steps": 100,
            "seed": 7,
            "corpus": "/corpus/train.bin",
        },
        "hardware": {
            "cpu": "x" * 60,
            "cpu_count": 8,
            "os": "Linux",
            "python": "3.11.0",
            "gpu": [],
            "cuda_visible": None,
        },
        "data": {
            "train": {"path": "/corpus/train.bin", "bytes": 1000000, "sha256": "a"},
            "eval": {"path": "/corpus/heldout.bin", "bytes": 20000, "sha256": "b"},
        },
        "model": {"params": 66976},
        "training": {"steps": 100, "wall_seconds": 4.2},
        "throughput": {"steps_per_second": 23.8, "bytes_per_second": 30461.0},
        "metrics": {"causal_bpb": bpb, "masked_bpb": None, "masked_acc": 0.5},
        "generation": {
            "prompt": "int main",
            "new_bytes": 32,
            "output_sha256": "c",
            "output_bytes": 32,
            "preview": "b'...'",
        },
        "artifacts": {
            "out_dir": "/out",
            "checkpoint": "/out/ckpt/checkpoints/step_000100.fblt",
            "checkpoint_sha256": "d",
            "eval": "/out/eval.json",
            "generated": "/out/generated.bin",
            "summary": "/out/summary.json",
        },
    }
    record.update(overrides)
    return record


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_recursive_discovery_ignores_other_formats(tmp_path):
    _write(tmp_path / "a" / "deep" / "summary.json", _record("r1", 5.0))
    _write(tmp_path / "b" / "summary.json", _record("r2", 4.0))
    # Eval records and resolved configs sit next to summaries; they are skipped
    # without a warning when discovered through a directory.
    _write(tmp_path / "a" / "eval.json", {"format": "fblt-eval/bpb-v1"})
    (tmp_path / "b" / "resolved_config.yaml").write_text("embed: 32\n")

    records, warnings = load_records([tmp_path])

    assert sorted(record["run_id"] for record in records) == ["r1", "r2"]
    assert warnings == []


def test_jsonl_yields_one_record_per_line(tmp_path):
    path = tmp_path / "records.jsonl"
    lines = [_record("r1", 5.0), _record("r2", 4.5), _record("r3", 4.0)]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    records, warnings = load_records([path])

    assert [record["run_id"] for record in records] == ["r1", "r2", "r3"]
    assert warnings == []


def test_bad_format_warns_only_when_explicit(tmp_path):
    bad = tmp_path / "eval.json"
    _write(bad, {"format": "fblt-eval/bpb-v1", "metrics": {"causal_bpb": 1.0}})

    through_dir, dir_warnings = load_records([tmp_path])
    explicit, explicit_warnings = load_records([bad])

    assert through_dir == []
    assert dir_warnings == []
    assert explicit == []
    assert any("eval.json" in warning for warning in explicit_warnings)


def test_malformed_json_warns_once(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{not json")

    records, warnings = load_records([tmp_path])

    assert records == []
    assert len(warnings) == 1


def test_dedupe_keeps_later_record(tmp_path):
    first = _record("same", 6.0, created="2026-01-01T00:00:00Z")
    second = _record("same", 4.0, created="2026-02-01T00:00:00Z")

    deduped = dedupe([first, second])

    assert len(deduped) == 1
    assert deduped[0]["created"] == "2026-02-01T00:00:00Z"
    assert deduped[0]["metrics"]["causal_bpb"] == 4.0


def test_validate_flags_missing_pieces():
    broken = _record()
    broken["metrics"] = {"causal_bpb": None}
    broken["artifacts"]["checkpoint_sha256"] = None

    problems = validate(broken)

    assert any("causal_bpb" in problem for problem in problems)
    assert any("checkpoint_sha256" in problem for problem in problems)
    assert validate(_record()) == []


def test_invalid_records_excluded_unless_include_synthetic():
    good = _record("good", 4.0)
    broken = _record("broken", 3.0)
    broken["artifacts"]["checkpoint_sha256"] = None

    default_md = to_markdown([good, broken])
    with_invalid = to_markdown([good, broken], include_synthetic=True)

    assert "good" in default_md
    assert "broken" not in default_md
    assert "broken" in with_invalid
    assert "invalid" in with_invalid


def test_classify_variants():
    assert classify(_record()) == "ok"
    assert classify(_record(git_dirty=True)) == "unverified"
    assert classify(_record(synthetic=True)) == "synthetic"
    assert classify(_record(git_commit=None)) == "synthetic"
    assert classify(_record(git_commit="unknown")) == "synthetic"
    assert classify(_record(command="")) == "synthetic"
    assert classify(_record(config={})) == "synthetic"


def test_flags_explain_status():
    assert flags(_record()) == []
    assert "dirty tree" in flags(_record(git_dirty=True))
    assert "no git commit" in flags(_record(git_commit=None))


def test_markdown_header_has_every_column():
    markdown = to_markdown([_record()])
    header = next(line for line in markdown.splitlines() if line.startswith("| run"))

    for column in (
        "run",
        "model",
        "status",
        "model params",
        "data (MB)",
        "steps",
        "hardware",
        "backend",
        "causal BPB",
        "masked acc",
        "steps/s",
        "commit",
        "flags",
    ):
        assert column in header


def test_markdown_model_column():
    simple = to_markdown([_record("r1", 4.0)])
    assert "e32/h64/L1" in simple

    split = _record("r2", 4.0)
    split["config"].update({"enc_layers": 2, "glob_layers": 1, "dec_layers": 2})
    assert "e32/h64/L1/2-1-2" in to_markdown([split])

    missing = _record("r3", 4.0)
    missing["config"] = {"steps": 100}
    assert "| - |" in to_markdown([missing])


def test_sort_puts_lower_bpb_first():
    ordered = sort_records([_record("high", 6.0), _record("low", 3.0)])

    assert [record["run_id"] for record in ordered] == ["low", "high"]


def test_sort_puts_synthetic_after_ok():
    synthetic = _record("syn", 1.0, git_commit=None)
    ok = _record("ok", 9.0)

    ordered = sort_records([synthetic, ok])

    assert [record["run_id"] for record in ordered] == ["ok", "syn"]


def test_cli_writes_markdown_and_jsonl(tmp_path):
    results = tmp_path / "runs"
    _write(results / "a" / "summary.json", _record("r1", 5.0))
    _write(results / "b" / "summary.json", _record("r2", 4.0))
    md = tmp_path / "LEADERBOARD.md"
    jsonl = tmp_path / "records.jsonl"

    rc = main(["--results", str(results), "--out-md", str(md), "--out-jsonl", str(jsonl)])

    assert rc == 0
    assert "causal BPB" in md.read_text()
    lines = jsonl.read_text().splitlines()
    assert len(lines) == 2
    assert jsonl.read_text().endswith("\n")
    assert [json.loads(line)["run_id"] for line in lines] == ["r2", "r1"]


def test_cli_returns_one_without_valid_records(tmp_path):
    results = tmp_path / "runs"
    broken = _record("broken", 4.0)
    broken["artifacts"]["checkpoint_sha256"] = None
    _write(results / "summary.json", broken)
    md = tmp_path / "LEADERBOARD.md"

    rc = main(["--results", str(results), "--out-md", str(md)])

    assert rc == 1
    assert md.exists()


def test_cli_returns_zero_with_no_input_files(tmp_path):
    rc = main(["--results", str(tmp_path / "absent")])

    assert rc == 0
