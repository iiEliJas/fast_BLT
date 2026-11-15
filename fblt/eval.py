"""Corpus BPB evaluation wrapper for the `eval_bpb` binary.

This is item 4 phase 1: bits-per-byte scoring of a checkpoint over a raw byte
corpus, with provenance captured in one JSON record. The CORE task harness
(ARC/MMLU/GSM8K/HumanEval, prompt rendering, centered accuracy) is phase 2 and
is not implemented here; `--task` accepts only `bpb` so an unsupported task
fails instead of fabricating a score.

The binary reads the model shape from v2 checkpoint metadata when the shape
flags are omitted. v1 checkpoints carry no metadata and need explicit flags.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import typing
from pathlib import Path

from fblt._binary import ensure_wsl_cuda_env, resolve_binary
from fblt._runner import run_binary_captured

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> typing.Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def _insert_after(record: dict, anchor: str, extras: dict) -> dict:
    """Rebuilds `record` with `extras` placed right after `anchor`."""
    out: dict = {}
    inserted = False
    for key, value in record.items():
        if key in extras:
            continue
        out[key] = value
        if key == anchor:
            out.update(extras)
            inserted = True
    if not inserted:
        out.update(extras)
    return out


def _atomic_write_json(path, record: dict) -> None:
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".fblt-eval-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def run_bpb_eval(
    checkpoint,
    corpus,
    *,
    backend="cpu",
    out=None,
    window=None,
    windows=20,
    skip=0,
    patcher="fixed",
    patch_len=4,
    entropy_lm=None,
    entropy_threshold=None,
    block_size=None,
    layout="paper",
    t=1.0,
    seed=7,
    embed=None,
    hidden=None,
    enc_layers=None,
    glob_layers=None,
    dec_layers=None,
    cross_attn=None,
    d0=None,
    diffusion=None,
) -> dict:
    """Scores `checkpoint` over `corpus` and returns the BPB record.

    Paths are checked up front so a typo raises instead of yielding a record of
    zeros. `out`, when given, receives the same merged record as returned.
    """
    checkpoint = os.fspath(checkpoint)
    corpus = os.fspath(corpus)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not os.path.isfile(corpus):
        raise FileNotFoundError(f"corpus not found: {corpus}")
    if entropy_lm is not None:
        entropy_lm = os.fspath(entropy_lm)
        if not os.path.isfile(entropy_lm):
            raise FileNotFoundError(f"entropy LM not found: {entropy_lm}")

    if backend == "cuda":
        ensure_wsl_cuda_env()

    binary = resolve_binary("eval_bpb", backend)

    fd, tmp_json = tempfile.mkstemp(prefix="fblt-eval-", suffix=".json")
    os.close(fd)
    try:
        argv: typing.List[str] = [
            "--checkpoint",
            checkpoint,
            "--corpus",
            corpus,
            "--backend",
            backend,
            "--output-json",
            tmp_json,
        ]
        optional = [
            ("--window", window),
            ("--windows", windows),
            ("--skip", skip),
            ("--patcher", patcher),
            ("--patch-len", patch_len),
            ("--entropy-lm", entropy_lm),
            ("--entropy-threshold", entropy_threshold),
            ("--block-size", block_size),
            ("--layout", layout),
            ("--t", t),
            ("--seed", seed),
            ("--embed", embed),
            ("--hidden", hidden),
            ("--enc-layers", enc_layers),
            ("--glob-layers", glob_layers),
            ("--dec-layers", dec_layers),
            ("--cross-attn", cross_attn),
            ("--d0", d0),
            ("--diffusion", diffusion),
        ]
        for flag, value in optional:
            if value is not None:
                argv.extend([flag, str(value)])

        returncode, _stdout, stderr = run_binary_captured(binary, argv)
        if returncode != 0:
            raise RuntimeError(stderr.strip() or f"eval_bpb exited with {returncode}")
        with open(tmp_json, "r") as fh:
            record = json.load(fh)
    finally:
        try:
            os.unlink(tmp_json)
        except OSError:
            pass

    provenance = {
        "checkpoint_sha256": _sha256_file(checkpoint),
        "corpus_sha256": _sha256_file(corpus),
    }
    record = _insert_after(record, "corpus", provenance)
    if entropy_lm is not None:
        record = _insert_after(record, "protocol", {"entropy_lm_sha256": _sha256_file(entropy_lm)})

    record["git_commit"] = _git_commit() or record.get("git_commit") or "unknown"

    if out is not None:
        _atomic_write_json(out, record)
    return record


def _flatten(record: dict) -> dict:
    flat: dict = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def write_csv(records, path) -> None:
    if not records:
        return
    rows = [_flatten(r) for r in records]
    fieldnames: typing.List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cell(value) -> str:
    return "-" if value is None else str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="fblt-eval",
        description="Corpus BPB evaluation (item 4 phase 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=["bpb"], default="bpb", help="Evaluation task")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.fblt)")
    parser.add_argument("--corpus", required=True, help="Raw byte corpus")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu", help="Compute backend")
    parser.add_argument("--out", default="eval.json", help="JSON record path")
    parser.add_argument("--csv", default=None, help="Optional flattened CSV output")
    parser.add_argument("--window", type=int, default=None, help="Bytes per window")
    parser.add_argument("--windows", type=int, default=20, help="Windows requested")
    parser.add_argument("--skip", type=int, default=0, help="Windows to skip")
    parser.add_argument(
        "--patcher", choices=["fixed", "entropy"], default="fixed", help="Segmentation"
    )
    parser.add_argument("--patch-len", type=int, default=4, help="Fixed-stride patch length")
    parser.add_argument("--entropy-lm", default=None, help="Entropy LM weights")
    parser.add_argument("--entropy-threshold", type=float, default=None, help="Entropy threshold")
    parser.add_argument("--block-size", type=int, default=None, help="Diffusion block size")
    parser.add_argument(
        "--layout", choices=["paper", "novel"], default="paper", help="Masked layout"
    )
    parser.add_argument("--t", type=float, default=1.0, help="Diffusion timestep")
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed")
    parser.add_argument("--embed", type=int, default=None, help="Embedding width")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden width")
    parser.add_argument("--enc-layers", type=int, default=None, help="Encoder layers")
    parser.add_argument("--glob-layers", type=int, default=None, help="Global layers")
    parser.add_argument("--dec-layers", type=int, default=None, help="Decoder layers")
    parser.add_argument(
        "--cross-attn", choices=["all", "last"], default=None, help="Cross-attention"
    )
    parser.add_argument("--d0", choices=["learned", "zeros"], default=None, help="D_0 policy")
    parser.add_argument(
        "--diffusion", type=int, choices=[0, 1], default=None, help="Score masked cells"
    )
    args = parser.parse_args(argv)

    try:
        record = run_bpb_eval(
            args.checkpoint,
            args.corpus,
            backend=args.backend,
            out=args.out,
            window=args.window,
            windows=args.windows,
            skip=args.skip,
            patcher=args.patcher,
            patch_len=args.patch_len,
            entropy_lm=args.entropy_lm,
            entropy_threshold=args.entropy_threshold,
            block_size=args.block_size,
            layout=args.layout,
            t=args.t,
            seed=args.seed,
            embed=args.embed,
            hidden=args.hidden,
            enc_layers=args.enc_layers,
            glob_layers=args.glob_layers,
            dec_layers=args.dec_layers,
            cross_attn=args.cross_attn,
            d0=args.d0,
            diffusion=args.diffusion,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.csv:
        write_csv([record], args.csv)

    metrics = record.get("metrics", {})
    protocol = record.get("protocol", {})
    print(
        "| checkpoint | corpus | backend | patcher | layout | windows | "
        "causal_bpb | masked_bpb | masked_acc |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    print(
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            record.get("checkpoint"),
            record.get("corpus"),
            record.get("backend"),
            protocol.get("patcher"),
            protocol.get("layout"),
            protocol.get("windows_evaluated"),
            _cell(metrics.get("causal_bpb")),
            _cell(metrics.get("masked_bpb")),
            _cell(metrics.get("masked_acc")),
        )
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
