"""Provenance helpers shared by record-writing wrappers.

Every record that claims reproducibility has to say which code, which machine
and which input bytes produced it. The helpers here are read-only: they never
raise when a tool is missing, they return None or an empty list instead, so a
record can still be written on a machine without git or an NVIDIA GPU.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import typing
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks so size does not matter."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> typing.Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def git_commit() -> typing.Optional[str]:
    """Full HEAD sha, or None when git is unavailable or this is not a repo."""
    proc = _git("rev-parse", "HEAD")
    if proc is None or proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def git_dirty() -> typing.Optional[bool]:
    """True when `git status --porcelain` lists anything, None without git."""
    proc = _git("status", "--porcelain")
    if proc is None or proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def _gpu_info() -> typing.List[typing.Dict[str, str]]:
    query = "--query-gpu=name,memory.total,driver_version"
    try:
        proc = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    gpus: typing.List[typing.Dict[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[0]:
            continue
        gpus.append({"name": parts[0], "memory_total": parts[1], "driver": parts[2]})
    return gpus


def hardware_info() -> typing.Dict[str, typing.Any]:
    """Best-effort machine description; every field degrades instead of raising."""
    cpu = platform.processor() or platform.machine() or "unknown"
    return {
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpu": _gpu_info(),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def utc_now() -> str:
    """ISO 8601 UTC timestamp with a Z suffix and second precision."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path, record: dict) -> None:
    """Write JSON through a temp file in the same directory, then rename."""
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".fblt-record-", suffix=".json")
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


def python_executable() -> str:
    return sys.executable or "python3"
