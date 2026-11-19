import argparse
import dataclasses
import os
import subprocess
import sys
import threading
import time
import typing

import yaml

from fblt._binary import ensure_wsl_cuda_env, resolve_binary
from fblt._config import TrainConfig, add_config_args, load_config
from fblt._runner import run_binary


def _parse_args() -> typing.Tuple[argparse.Namespace, typing.Dict[str, typing.Any]]:
    """Returns the namespace plus every explicitly-passed config field."""
    overrides: typing.Dict[str, typing.Any] = {}
    parser = argparse.ArgumentParser(
        prog="fblt-train",
        description="Train BLT-D model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda", "torch"],
        required=True,
        help="Compute backend",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="key=value override (repeatable)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--profile", action="store_true", help="Run under nsys profile (CUDA only)")
    parser.add_argument("--wandb", action="store_true", help="Stream metrics to W&B")
    parser.add_argument("--wandb-project", type=str, default="fblt", help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")

    # Every remaining TrainConfig field becomes a real flag, so the C binary's
    # full option surface is reachable without --override.
    add_config_args(parser, TrainConfig, skip=("backend",), overrides=overrides)

    args = parser.parse_args()
    overrides.update({k: v for k, v in vars(args).items() if v is not None})
    return args, overrides


def _check_yaml_no_backend(yaml_path: str) -> None:
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if isinstance(raw, dict) and "backend" in raw:
        raise SystemExit(
            f"Error: YAML config {yaml_path} contains 'backend' key. "
            "Use --backend on the command line instead."
        )


def _derive_run_name(config_path: typing.Optional[str]) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if config_path is not None:
        stem = os.path.splitext(os.path.basename(config_path))[0]
        return f"{stem}-{stamp}"
    return f"default-{stamp}"


_DEFAULT_OUTPUTS: typing.Dict[str, str] = {
    "save_weights": "checkpoint.fblt",
    "grad_norm_log": "grad_norm.log",
    "update_norm_log": "update_norm.log",
    "component_norm_log": "component_norm.log",
    "activation_dump_log": "activation_dump.log",
    "batch_log": "batch.log",
    "loss_log": "loss.log",
    "metrics_log": "metrics.jsonl",
}


def _fill_output_defaults(cfg, run_dir: str) -> None:
    """Fill unset output paths; --save-dir already names its own checkpoint."""
    for field_name, filename in _DEFAULT_OUTPUTS.items():
        if field_name == "save_weights" and cfg.save_dir is not None:
            continue
        if getattr(cfg, field_name) is None:
            setattr(cfg, field_name, os.path.join(run_dir, filename))


def main() -> None:
    args, explicit = _parse_args()

    # Reject backend in YAML
    if args.config is not None:
        _check_yaml_no_backend(args.config)

    # Build config
    cfg = load_config(TrainConfig, args.config, args.override)

    # Explicit flags win over --override and over YAML.
    for key, val in explicit.items():
        setattr(cfg, key, val)
    cfg.backend = args.backend
    if cfg.backend == "torch":
        _venv = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "pysrc",
            ".venv",
            "bin",
            "python",
        )
        _same = os.path.abspath(sys.executable) == os.path.abspath(_venv)
        _reexec = os.environ.get("FBLT_TORCH_REEXEC")
        if os.path.exists(_venv) and (not _same) and (not _reexec):
            os.environ["FBLT_TORCH_REEXEC"] = "1"
            _mod = "train" if "train.py" in __file__ else "infer"
            os.execv(_venv, [_venv, "-m", "fblt." + _mod] + sys.argv[1:])

    # Derive run name and directory
    run_name = args.run_name if args.run_name is not None else _derive_run_name(args.config)
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Fill default output paths for any field still None. --save-dir already
    # names its own checkpoint files, so filling save_weights here would write
    # a second copy of the same weights.
    _fill_output_defaults(cfg, run_dir)

    # Write resolved config
    resolved_path = os.path.join(run_dir, "resolved_config.yaml")
    with open(resolved_path, "w") as fh:
        yaml.dump(dataclasses.asdict(cfg), fh)

    # Print run directory
    print(run_dir)

    if cfg.backend == "torch":
        _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pysrc")
        sys.path.insert(0, _p)
        from fblt_torch.cli import run_train

        rc = run_train(cfg, args)
        sys.exit(rc)

    if cfg.backend == "cuda":
        ensure_wsl_cuda_env()

    # Resolve binary and run
    binary = resolve_binary("train_blt_d", cfg.backend)
    argv = cfg.to_argv()

    if args.profile:
        if cfg.backend != "cuda":
            raise SystemExit("--profile requires --backend cuda")
        import shutil

        nsys = shutil.which("nsys")
        if nsys is None:
            raise SystemExit("nsys not found; install Nsight Systems or run the binary directly")
        cmd = [
            nsys,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "-o",
            os.path.join(run_dir, "profile"),
            binary,
        ] + argv
        rc = subprocess.call(cmd)
        sys.exit(rc)

    metrics_path = getattr(cfg, "metrics_log", None)
    if getattr(args, "wandb", False):
        try:
            import wandb
        except ImportError:
            raise SystemExit("wandb not installed; pip install 'fblt[tracking]'")
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or run_name,
            entity=args.wandb_entity,
            config=dataclasses.asdict(cfg),
        )
        stop = threading.Event()

        def on_event(ev):
            evt = ev.get("event")
            if evt == "train":
                run.log(
                    {
                        "train/loss": ev.get("loss"),
                        "train/lr": ev.get("lr"),
                        "train/grad_norm": ev.get("grad_norm"),
                    },
                    step=ev.get("step"),
                )
            elif evt == "eval":
                run.log(
                    {
                        "eval/causal_bpb": ev.get("causal_bpb"),
                        "eval/masked_acc": ev.get("masked_acc"),
                    },
                    step=ev.get("step"),
                )
            elif evt == "monitor":
                run.log(
                    {"monitor/l_clean": ev.get("l_clean"), "monitor/gap": ev.get("gap")},
                    step=ev.get("step"),
                )
            elif evt == "result":
                run.log(
                    {"result/causal_bpb": ev.get("causal_bpb"), "result/params": ev.get("params")}
                )

        from fblt._runner import tail_jsonl

        t = threading.Thread(target=tail_jsonl, args=(metrics_path, on_event, stop), daemon=True)
        t.start()
        try:
            rc = run_binary(binary, argv)
        finally:
            stop.set()
            t.join(timeout=5)
            run.finish()
        sys.exit(rc)

    rc = run_binary(binary, argv)
    sys.exit(rc)


if __name__ == "__main__":
    main()
