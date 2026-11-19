"""Speedrun leaderboard: discover, validate, classify and render summaries.

A speedrun summary is only useful if a reader can tell whether it was produced
on a clean checkout, on real data, with a recorded command. This module loads
every summary it can find, labels the ones that cannot prove that, and never
raises on a malformed record: a broken file is a warning, not a crash.
"""

import argparse
import json
import math
import sys
import typing
from pathlib import Path

REQUIRED_PATHS = (
    "format",
    "schema_version",
    "run_id",
    "backend",
    "config",
    "hardware",
    "metrics",
    "training",
    "artifacts",
)
SUMMARY_FORMAT = "fblt-speedrun/v1"

_STATUS_RANK = {"ok": 0, "unverified": 1, "synthetic": 2, "invalid": 3}


def _get(record: dict, dotted: str) -> typing.Any:
    cur: typing.Any = record
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _input_files(paths) -> typing.Tuple[typing.List[Path], typing.List[str]]:
    files: typing.List[Path] = []
    warnings: typing.List[str] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix in (".json", ".jsonl"):
                    files.append(candidate)
        elif path.is_file():
            files.append(path)
        else:
            warnings.append(f"{path}: no such file or directory")
    return files, warnings


def _absorb(obj, path: Path, explicit: bool, records, warnings) -> None:
    if not isinstance(obj, dict):
        warnings.append(f"{path}: expected a JSON object, got {type(obj).__name__}")
        return
    if obj.get("format") != SUMMARY_FORMAT:
        if explicit:
            warnings.append(f"{path}: not a speedrun summary (format={obj.get('format')!r})")
        return
    records.append(obj)


def _load_file(path: Path, explicit: bool, records, warnings) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.append(f"{path}: {exc}")
        return

    if path.suffix == ".jsonl":
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"{path}:{lineno}: invalid JSON ({exc.msg})")
                continue
            _absorb(obj, path, explicit, records, warnings)
        return

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        warnings.append(f"{path}: invalid JSON ({exc.msg})")
        return
    items = obj if isinstance(obj, list) else [obj]
    for item in items:
        _absorb(item, path, explicit, records, warnings)


def load_records(paths) -> typing.Tuple[typing.List[dict], typing.List[str]]:
    """Return (records, warnings) from files and directories.

    Directories are walked recursively in sorted order. Files whose `format` is
    not a speedrun summary are skipped silently inside a directory (eval
    records and resolved configs live next to summaries) and reported when the
    caller named them explicitly.
    """
    records: typing.List[dict] = []
    warnings: typing.List[str] = []
    pending: typing.List[typing.Tuple[Path, bool]] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix in (".json", ".jsonl"):
                    pending.append((candidate, False))
        elif path.is_file():
            pending.append((path, True))
        else:
            warnings.append(f"{path}: no such file or directory")
    for path, explicit in pending:
        _load_file(path, explicit, records, warnings)
    return records, warnings


def validate(record: dict) -> typing.List[str]:
    """Structural problems that make a record unusable; empty means valid."""
    problems: typing.List[str] = []
    for dotted in REQUIRED_PATHS:
        if _get(record, dotted) is None:
            problems.append(f"missing required field: {dotted}")
    fmt = record.get("format")
    if fmt is not None and fmt != SUMMARY_FORMAT:
        problems.append(f"wrong format: {fmt!r}")
    version = record.get("schema_version")
    if version is not None and version != 1:
        problems.append(f"schema_version must be 1, got {version!r}")
    bpb = _get(record, "metrics.causal_bpb")
    if bpb is None:
        problems.append("missing metrics.causal_bpb")
    elif not isinstance(bpb, (int, float)) or isinstance(bpb, bool):
        problems.append("metrics.causal_bpb is not a number")
    elif not math.isfinite(bpb) or bpb < 0:
        problems.append("metrics.causal_bpb must be finite and non-negative")
    if _get(record, "artifacts.checkpoint_sha256") is None:
        problems.append("missing artifacts.checkpoint_sha256")
    return problems


def classify(record: dict) -> str:
    """ok / unverified (dirty tree) / synthetic (cannot prove provenance)."""
    if record.get("synthetic") is True:
        return "synthetic"
    commit = record.get("git_commit")
    if not commit or commit == "unknown":
        return "synthetic"
    if not record.get("command"):
        return "synthetic"
    if not record.get("config"):
        return "synthetic"
    if record.get("git_dirty") is True:
        return "unverified"
    return "ok"


def flags(record: dict) -> typing.List[str]:
    """Human-readable reasons behind a record's status."""
    reasons: typing.List[str] = []
    if record.get("synthetic") is True:
        reasons.append("marked synthetic")
    commit = record.get("git_commit")
    if not commit or commit == "unknown":
        reasons.append("no git commit")
    if not record.get("command"):
        reasons.append("no command")
    if not record.get("config"):
        reasons.append("no config")
    if record.get("git_dirty") is True:
        reasons.append("dirty tree")
    for problem in validate(record):
        if problem not in reasons:
            reasons.append(problem)
    return reasons


def dedupe(records: typing.List[dict]) -> typing.List[dict]:
    """Last record wins per run_id, keeping first-seen ordering."""
    by_id: typing.Dict[typing.Any, dict] = {}
    for record in records:
        by_id[record.get("run_id")] = record
    return list(by_id.values())


def _sort_value(record: dict, key: str) -> typing.Any:
    if key == "steps_per_second":
        value = _get(record, "throughput.steps_per_second")
        return (0, -value) if isinstance(value, (int, float)) else (1, 0.0)
    if key == "params":
        value = _get(record, "model.params")
        return (0, -value) if isinstance(value, (int, float)) else (1, 0.0)
    value = _get(record, "metrics.causal_bpb")
    return (0, value) if isinstance(value, (int, float)) else (1, 0.0)


def sort_records(records: typing.List[dict], key: str = "causal_bpb") -> typing.List[dict]:
    """Status first (ok before unverified before synthetic before invalid)."""

    def sort_key(record: dict):
        status = "invalid" if validate(record) else classify(record)
        return (_STATUS_RANK.get(status, 3),) + _sort_value(record, key)

    return sorted(records, key=sort_key)


def _model_label(record: dict) -> str:
    cfg = record.get("config") or {}
    embed, hidden, layers = cfg.get("embed"), cfg.get("hidden"), cfg.get("layers")
    if embed is None or hidden is None or layers is None:
        return "-"
    label = f"e{embed}/h{hidden}/L{layers}"
    split = [cfg.get(key) for key in ("enc_layers", "glob_layers", "dec_layers")]
    if all(value is not None for value in split) and any(value != layers for value in split):
        label += "/" + "-".join(str(value) for value in split)
    return label


def _hardware_label(record: dict) -> str:
    hardware = record.get("hardware") or {}
    cpu = str(hardware.get("cpu") or "-")[:40]
    gpus = hardware.get("gpu") or []
    if isinstance(gpus, list) and gpus and isinstance(gpus[0], dict):
        name = gpus[0].get("name")
        if name:
            return f"{cpu} + {name}"
    return cpu


def _commit_label(record: dict) -> str:
    commit = record.get("git_commit")
    if not commit:
        return "-"
    label = str(commit)[:8]
    return label + "-dirty" if record.get("git_dirty") is True else label


def _row(record: dict) -> str:
    problems = validate(record)
    status = "invalid" if problems else classify(record)
    params = _get(record, "model.params")
    train_bytes = _get(record, "data.train.bytes")
    data_mb = f"{train_bytes / 1e6:.1f}" if isinstance(train_bytes, (int, float)) else "-"
    bpb = _get(record, "metrics.causal_bpb")
    bpb_text = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else "-"
    acc = _get(record, "metrics.masked_acc")
    acc_text = f"{acc:.4f}" if isinstance(acc, (int, float)) else "-"
    sps = _get(record, "throughput.steps_per_second")
    sps_text = f"{sps:.1f}" if isinstance(sps, (int, float)) else "-"
    steps = _get(record, "training.steps")
    steps_text = str(steps) if steps is not None else "-"
    reason_text = "; ".join(flags(record)) or "-"
    columns = [
        str(record.get("run_id") or "-"),
        _model_label(record),
        status,
        str(params) if params is not None else "-",
        data_mb,
        steps_text,
        _hardware_label(record),
        str(record.get("backend") or "-"),
        bpb_text,
        acc_text,
        sps_text,
        _commit_label(record),
        reason_text,
    ]
    return "| " + " | ".join(columns) + " |"


def to_markdown(records: typing.List[dict], include_synthetic: bool = False) -> str:
    visible = [record for record in records if include_synthetic or not validate(record)]
    ordered = sort_records(visible)
    lines = [
        "# fBLT speedrun leaderboard",
        "",
        "_Lower causal BPB is better; steps/s is training throughput in optimizer steps "
        "per second and data (MB) is the training corpus size._",
        "",
        "| run | model | status | model params | data (MB) | steps | hardware | backend "
        "| causal BPB | masked acc | steps/s | commit | flags |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(_row(record) for record in ordered)
    return "\n".join(lines) + "\n"


def write_markdown(records: typing.List[dict], path, include_synthetic: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_markdown(records, include_synthetic=include_synthetic), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="fblt-leaderboard",
        description="Render fBLT speedrun summaries as a leaderboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        action="append",
        default=None,
        help="Summary file or directory (repeatable)",
    )
    parser.add_argument("--out-md", default=None, help="Write markdown here")
    parser.add_argument("--out-jsonl", default=None, help="Write deduped records here")
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="List synthetic and invalid records too",
    )
    parser.add_argument(
        "--sort",
        choices=["causal_bpb", "steps_per_second", "params"],
        default="causal_bpb",
        help="Secondary sort key after status",
    )
    args = parser.parse_args(argv)

    paths = args.results or ["runs"]
    files, _ = _input_files(paths)
    records, warnings = load_records(paths)
    for warning in warnings:
        print(warning, file=sys.stderr)

    records = dedupe(records)
    ordered = sort_records(records, args.sort)

    if args.out_md:
        write_markdown(ordered, args.out_md, include_synthetic=args.include_synthetic)
    else:
        sys.stdout.write(to_markdown(ordered, include_synthetic=args.include_synthetic))

    if args.out_jsonl:
        target = Path(args.out_jsonl)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            for record in ordered:
                fh.write(json.dumps(record) + "\n")

    valid = [record for record in records if not validate(record)]
    if not valid and files:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
