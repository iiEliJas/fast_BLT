"""Reproducible speedrun: build, train, eval, generate, one summary record.

`bytes_per_second` is derived, not measured: it is
`steps_per_second * window * grad_accum`, i.e. how many corpus bytes were
consumed per wall-clock second at the configured batch geometry. Wall time is
measured around the training subprocess only.

Everything that touches the machine goes through `_run_command` so a test can
replace it with a recorder. The one exception is BPB evaluation, which reuses
`fblt.eval.run_bpb_eval` rather than shelling out to `eval_bpb` again.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import typing
from pathlib import Path

import yaml

from fblt._checkpoint import latest_checkpoint, model_info
from fblt._config import TrainConfig, load_config
from fblt._provenance import (
    atomic_write_json,
    git_commit,
    git_dirty,
    hardware_info,
    python_executable,
    sha256_file,
    utc_now,
)
from fblt.eval import run_bpb_eval

DEFAULT_CONFIG: typing.Dict[str, typing.Any] = {
    "embed": 32,
    "hidden": 64,
    "layers": 1,
    "window": 128,
    "block_size": 8,
    "steps": 100,
    "lr": 0.05,
    "seed": 7,
    "grad_accum": 1,
    "optimizer": "adamw",
    "diffusion": 1,
}
SUMMARY_FORMAT = "fblt-speedrun/v1"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIRS = {"cpu": "build", "cuda": "build-cuda"}


def _run_command(argv, cwd=None, env=None, dry_run=False, log=print) -> int:
    """Single funnel for external commands; dry-run logs without spawning."""
    log("$ " + shlex.join(str(part) for part in argv))
    if dry_run:
        return 0
    # Keep the logged command ahead of the child's own output when stdout is a pipe.
    sys.stdout.flush()
    sys.stderr.flush()
    proc = subprocess.run(
        [str(part) for part in argv], cwd=None if cwd is None else str(cwd), env=env
    )
    return proc.returncode


def _repo_env() -> typing.Dict[str, str]:
    """Environment with the repo root importable regardless of cwd."""
    env = dict(os.environ)
    root = str(_REPO_ROOT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing else root + os.pathsep + existing
    return env


def _merge_config(
    config_path, overrides: typing.Dict[str, typing.Any]
) -> typing.Dict[str, typing.Any]:
    """DEFAULT_CONFIG, then YAML values, then explicit non-None kwargs."""
    merged = dict(DEFAULT_CONFIG)
    if config_path is not None:
        path = os.fspath(config_path)
        # load_config validates keys and types; the raw mapping says which
        # values were actually present, so absent keys keep the speedrun default
        # instead of silently picking up the trainer default.
        validated = load_config(TrainConfig, path)
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
        for key in DEFAULT_CONFIG:
            for candidate in (key, key.replace("_", "-")):
                if candidate in raw and raw[candidate] is not None:
                    merged[key] = getattr(validated, key)
                    break
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _default_corpus(name: str) -> Path:
    return _REPO_ROOT / "data" / name


def _resolve_corpus(explicit, name: str) -> Path:
    return Path(explicit).resolve() if explicit is not None else _default_corpus(name).resolve()


def _file_entry(path: Path) -> typing.Dict[str, typing.Any]:
    entry: typing.Dict[str, typing.Any] = {"path": str(path), "bytes": None, "sha256": None}
    if path.is_file():
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    return entry


def _train_argv(backend, run_id, train_corpus, out_dir, cfg, metrics_path):
    return [
        python_executable(),
        "-m",
        "fblt.train",
        "--backend",
        backend,
        "--run-name",
        f"speedrun-{run_id}",
        "--corpus",
        str(train_corpus),
        "--save-dir",
        str(out_dir / "ckpt"),
        "--save-every",
        "0",
        "--metrics-log",
        str(metrics_path),
        "--embed",
        str(cfg["embed"]),
        "--hidden",
        str(cfg["hidden"]),
        "--layers",
        str(cfg["layers"]),
        "--window",
        str(cfg["window"]),
        "--block-size",
        str(cfg["block_size"]),
        "--steps",
        str(cfg["steps"]),
        "--lr",
        str(cfg["lr"]),
        "--seed",
        str(cfg["seed"]),
        "--grad-accum",
        str(cfg["grad_accum"]),
        "--optimizer",
        str(cfg["optimizer"]),
        "--diffusion",
        str(cfg["diffusion"]),
    ]


def _generate_argv(backend, checkpoint, prompt, new_bytes, block_size, output):
    return [
        python_executable(),
        "-m",
        "fblt.infer",
        "--backend",
        backend,
        "--checkpoint",
        str(checkpoint),
        "--prompt",
        prompt,
        "--new-bytes",
        str(new_bytes),
        "--block-size",
        str(block_size),
        # Training in the speedrun never enables entropy patching, so inference
        # must use the same fixed-stride segmentation instead of demanding an
        # entropy LM the run does not have.
        "--fixed-patches",
        "--output",
        str(output),
    ]


def _command_line(
    out_dir,
    backend,
    cfg,
    config_path,
    *,
    windows,
    new_bytes,
    prompt,
    corpus,
    eval_corpus,
    run_id,
    build,
) -> str:
    argv = [
        python_executable(),
        "-m",
        "fblt.speedrun",
        "--out",
        str(out_dir),
        "--backend",
        backend,
        "--steps",
        str(cfg["steps"]),
        "--window",
        str(cfg["window"]),
        "--block-size",
        str(cfg["block_size"]),
        "--lr",
        str(cfg["lr"]),
        "--seed",
        str(cfg["seed"]),
        "--windows",
        str(windows),
        "--new-bytes",
        str(new_bytes),
        "--prompt",
        prompt,
    ]
    if corpus is not None:
        argv.extend(["--corpus", str(corpus)])
    if eval_corpus is not None:
        argv.extend(["--eval-corpus", str(eval_corpus)])
    if run_id is not None:
        argv.extend(["--run-id", run_id])
    if config_path is not None:
        argv.extend(["--config", os.fspath(config_path)])
    argv.append("--build" if build else "--no-build")
    return shlex.join(argv)


def _build_commands(backend, build_dir):
    configure = ["cmake", "-S", str(_REPO_ROOT), "-B", str(build_dir)]
    if backend == "cuda":
        configure.append("-DUSE_CUDA=ON")
    compile_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "train_blt_d",
        "eval_bpb",
        "infer",
        "--parallel",
    ]
    return [configure, compile_cmd]


def _default_run_id(backend: str) -> str:
    commit = git_commit() or "nogit"
    return f"{backend}-{time.strftime('%Y%m%d-%H%M%S')}-{commit[:8]}"


def run_speedrun(
    out_dir,
    *,
    config=None,
    backend="cpu",
    build=True,
    steps=None,
    window=None,
    block_size=None,
    lr=None,
    seed=None,
    windows=8,
    new_bytes=32,
    prompt="int main",
    corpus=None,
    eval_corpus=None,
    run_id=None,
    dry_run=False,
    log=print,
) -> dict:
    out_dir = Path(out_dir).resolve()
    cfg = _merge_config(
        config,
        {
            "steps": steps,
            "window": window,
            "block_size": block_size,
            "lr": lr,
            "seed": seed,
        },
    )
    config_path = None if config is None else str(Path(config).resolve())
    run_id = run_id or _default_run_id(backend)
    commit = git_commit()
    dirty = git_dirty()
    hardware = hardware_info()

    train_corpus = _resolve_corpus(corpus, "train.bin")
    heldout_corpus = _resolve_corpus(eval_corpus, "heldout.bin")
    build_dir = _REPO_ROOT / _BUILD_DIRS["cuda" if backend == "cuda" else "cpu"]

    if build:
        for argv in _build_commands(backend, build_dir):
            rc = _run_command(argv, cwd=_REPO_ROOT, dry_run=dry_run, log=log)
            if rc != 0:
                raise RuntimeError(f"build command failed with exit code {rc}: {argv[0]} {argv[3]}")

    needs_corpus = not train_corpus.is_file() or not heldout_corpus.is_file()
    if needs_corpus:
        _run_command(
            [python_executable(), "fblt/scripts/build_sample_corpus.py"],
            cwd=_REPO_ROOT,
            dry_run=dry_run,
            log=log,
        )
        if not dry_run and (not train_corpus.is_file() or not heldout_corpus.is_file()):
            raise FileNotFoundError(
                f"sample corpus build did not produce {train_corpus} and {heldout_corpus}"
            )

    summary: typing.Dict[str, typing.Any] = {
        "format": SUMMARY_FORMAT,
        "schema_version": 1,
        "run_id": run_id,
        "created": utc_now(),
        "synthetic": False,
        "dry_run": bool(dry_run),
        "command": _command_line(
            out_dir,
            backend,
            cfg,
            config,
            windows=windows,
            new_bytes=new_bytes,
            prompt=prompt,
            corpus=corpus,
            eval_corpus=eval_corpus,
            run_id=run_id,
            build=build,
        ),
        "git_commit": commit,
        "git_dirty": dirty,
        "backend": backend,
        "config_path": config_path,
        "config": {
            "embed": cfg["embed"],
            "hidden": cfg["hidden"],
            "layers": cfg["layers"],
            "window": cfg["window"],
            "block_size": cfg["block_size"],
            "grad_accum": cfg["grad_accum"],
            "optimizer": cfg["optimizer"],
            "diffusion": cfg["diffusion"],
            "lr": cfg["lr"],
            "steps": cfg["steps"],
            "seed": cfg["seed"],
            "corpus": str(train_corpus),
        },
        "hardware": hardware,
        "data": {
            "train": _file_entry(train_corpus),
            "eval": _file_entry(heldout_corpus),
        },
        "model": {"params": None},
        "training": {"steps": cfg["steps"], "wall_seconds": None},
        "throughput": {"steps_per_second": None, "bytes_per_second": None},
        "metrics": None,
        "generation": None,
        "artifacts": {
            "out_dir": str(out_dir),
            "checkpoint": None,
            "checkpoint_sha256": None,
            "eval": str(out_dir / "eval.json"),
            "generated": str(out_dir / "generated.bin"),
            "summary": str(out_dir / "summary.json"),
        },
    }

    env = _repo_env()
    metrics_path = out_dir / "metrics.jsonl"
    train_cmd = _train_argv(backend, run_id, train_corpus, out_dir, cfg, metrics_path)
    generated_path = out_dir / "generated.bin"
    planned_checkpoint = out_dir / "ckpt" / "checkpoints" / f"step_{cfg['steps']:06d}.fblt"

    if dry_run:
        summary["artifacts"]["checkpoint"] = str(planned_checkpoint)
        _run_command(train_cmd, cwd=out_dir, env=env, dry_run=True, log=log)
        log(
            "eval: run_bpb_eval(checkpoint={}, corpus={}, windows={})".format(
                planned_checkpoint, heldout_corpus, windows
            )
        )
        _run_command(
            _generate_argv(
                backend, planned_checkpoint, prompt, new_bytes, cfg["block_size"], generated_path
            ),
            cwd=out_dir,
            env=env,
            dry_run=True,
            log=log,
        )
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    rc = _run_command(train_cmd, cwd=out_dir, env=env, dry_run=False, log=log)
    wall_seconds = time.monotonic() - started
    if rc != 0:
        raise RuntimeError(f"training failed with exit code {rc}")

    checkpoint = latest_checkpoint(str(out_dir / "ckpt"))
    if checkpoint is None:
        raise RuntimeError(
            f"training finished but no checkpoint was written under {out_dir / 'ckpt'}"
        )
    summary["model"]["params"] = model_info(checkpoint)["param_count"]
    summary["training"]["wall_seconds"] = round(wall_seconds, 3)
    steps_per_second = cfg["steps"] / wall_seconds if wall_seconds > 0 else None
    if steps_per_second is None:
        summary["throughput"] = {"steps_per_second": None, "bytes_per_second": None}
    else:
        summary["throughput"] = {
            "steps_per_second": round(steps_per_second, 3),
            "bytes_per_second": round(steps_per_second * cfg["window"] * cfg["grad_accum"], 3),
        }
    summary["artifacts"]["checkpoint"] = str(checkpoint)
    summary["artifacts"]["checkpoint_sha256"] = sha256_file(checkpoint)

    log(
        "eval: run_bpb_eval(checkpoint={}, corpus={}, windows={})".format(
            checkpoint, heldout_corpus, windows
        )
    )
    eval_record = run_bpb_eval(
        checkpoint,
        str(heldout_corpus),
        backend=backend,
        out=str(out_dir / "eval.json"),
        window=cfg["window"],
        windows=windows,
        patch_len=4,
        patcher="fixed",
        layout="paper",
        block_size=cfg["block_size"],
    )
    eval_metrics = eval_record.get("metrics") or {}
    summary["metrics"] = {
        "causal_bpb": eval_metrics.get("causal_bpb"),
        "masked_bpb": eval_metrics.get("masked_bpb"),
        "masked_acc": eval_metrics.get("masked_acc"),
    }

    gen_cmd = _generate_argv(
        backend, checkpoint, prompt, new_bytes, cfg["block_size"], generated_path
    )
    rc = _run_command(gen_cmd, cwd=out_dir, env=env, dry_run=False, log=log)
    if rc != 0:
        raise RuntimeError(f"inference failed with exit code {rc}")

    payload = generated_path.read_bytes()
    summary["generation"] = {
        "prompt": prompt,
        "new_bytes": new_bytes,
        "output_sha256": sha256_file(generated_path),
        "output_bytes": len(payload),
        "preview": repr(payload[:64]),
    }

    atomic_write_json(out_dir / "summary.json", summary)

    return summary


def aggregate_leaderboard(results_dir, out_json):
    """Backward-compatible shim: write the deduped records as a JSON array."""
    from fblt.leaderboard import dedupe, load_records

    records, warnings = load_records([results_dir])
    for warning in warnings:
        print(warning, file=sys.stderr)
    rows = dedupe(records)
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=2)
        fh.write("\n")
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fblt-speedrun",
        description="Reproducible build/train/eval/generate speedrun",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default="runs/speedrun", help="Output directory")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu", help="Compute backend")
    parser.add_argument("--config", default=None, help="YAML TrainConfig path")
    parser.add_argument("--steps", type=int, default=None, help="Training steps")
    parser.add_argument("--window", type=int, default=None, help="Bytes per training window")
    parser.add_argument("--block-size", type=int, default=None, help="Diffusion block size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--windows", type=int, default=8, help="Eval windows")
    parser.add_argument("--new-bytes", type=int, default=32, help="Bytes to generate")
    parser.add_argument("--prompt", default="int main", help="Generation prompt")
    parser.add_argument("--corpus", default=None, help="Training corpus (raw bytes)")
    parser.add_argument("--eval-corpus", default=None, help="Held-out eval corpus")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Configure and compile before running",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands, create nothing")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        record = run_speedrun(
            args.out,
            config=args.config,
            backend=args.backend,
            build=args.build,
            steps=args.steps,
            window=args.window,
            block_size=args.block_size,
            lr=args.lr,
            seed=args.seed,
            windows=args.windows,
            new_bytes=args.new_bytes,
            prompt=args.prompt,
            corpus=args.corpus,
            eval_corpus=args.eval_corpus,
            run_id=args.run_id,
            dry_run=args.dry_run,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"speedrun: error: {exc}", file=sys.stderr)
        return 1

    bpb = (record.get("metrics") or {}).get("causal_bpb")
    bpb_text = "-" if bpb is None else f"{bpb:.4f}"
    print(f"speedrun: {record['run_id']} bpb={bpb_text} -> {Path(args.out) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
