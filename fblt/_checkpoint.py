"""Read-only helpers for fBLT checkpoint containers and save directories.

The container is the one written by ``blt_model_save_ex`` in
``csrc/models/checkpoint.c`` (little-endian):

    magic       4 bytes, "FBLT"
    u32         version (1 = no metadata block, 2 = JSON metadata block)
    u32         meta_len, then meta_len bytes of UTF-8 JSON  (version 2 only)
    u32         num_tensors
    per tensor:
        u16     name_len
        bytes   name
        u32     ndim
        ndim x  size_t dims
        f32     payload, prod(dims) values
"""

import glob
import json
import os
import re
import struct
import typing

_MAGIC = b"FBLT"
_MAX_META_LEN = 1024 * 1024
# dims are written with sizeof(size_t), which is the platform pointer width
_DIM_CHAR = "Q" if struct.calcsize("P") == 8 else "I"
_DIM_FMT = "<" + _DIM_CHAR
_DIM_BYTES = struct.calcsize(_DIM_FMT)
_STEP_RE = re.compile(r"step_(\d+)")
_METRIC_PREFERENCE = ("causal_bpb", "masked_acc")
_MAXIMISED_METRICS = frozenset({"masked_acc"})


def _read_exact(fh, size, what):
    data = fh.read(size)
    if len(data) != size:
        raise ValueError(f"checkpoint truncated: short read of {what}")
    return data


def _parse_header_raw(fh):
    """Read magic/version/metadata; returns (version, raw_meta_bytes, parsed_meta)."""
    if fh.read(4) != _MAGIC:
        raise ValueError("not an fblt checkpoint: bad magic")
    (version,) = struct.unpack("<I", _read_exact(fh, 4, "version"))
    if version == 1:
        return version, None, None
    if version != 2:
        if version > 2:
            raise ValueError(
                f"unsupported checkpoint version {version}: file is newer than this reader"
            )
        raise ValueError(f"unsupported checkpoint version {version}")
    (meta_len,) = struct.unpack("<I", _read_exact(fh, 4, "meta_len"))
    if meta_len > _MAX_META_LEN:
        raise ValueError("meta_len too large")
    if meta_len == 0:
        return version, b"", {}
    raw = _read_exact(fh, meta_len, "metadata")
    return version, raw, json.loads(raw.decode("utf-8"))


def _parse_header(fh):
    """Read magic/version/metadata from an open handle; returns (version, meta)."""
    version, _, meta = _parse_header_raw(fh)
    return version, meta


def _read_header(path):
    with open(path, "rb") as fh:
        return _parse_header(fh)


def read_meta(path: str) -> typing.Optional[dict]:
    """Return the metadata dict of a v2 checkpoint, or None for a v1 file.

    Raises ValueError for a non-checkpoint (bad magic), a version this reader
    does not know, or a truncated header/metadata block.
    """
    return _read_header(path)[1]


def _tensor_count(fh):
    (count,) = struct.unpack("<I", _read_exact(fh, 4, "tensor count"))
    return count


class TensorLayout(typing.NamedTuple):
    name: str
    shape: typing.Tuple[int, ...]
    offset: int
    nbytes: int


def _iter_tensor_layouts(fh, count, file_size):
    """Yield TensorLayout per tensor, seeking past payloads instead of reading them."""
    for index in range(count):
        raw_len = fh.read(2)
        if len(raw_len) != 2:
            raise ValueError("checkpoint truncated at tensor header")
        (name_len,) = struct.unpack("<H", raw_len)
        raw_name = fh.read(name_len)
        if len(raw_name) != name_len:
            raise ValueError(f"checkpoint truncated at tensor header (tensor {index})")
        name = raw_name.decode("utf-8", "replace")
        raw_ndim = fh.read(4)
        if len(raw_ndim) != 4:
            raise ValueError(f"checkpoint truncated at tensor '{name}'")
        (ndim,) = struct.unpack("<I", raw_ndim)
        raw_dims = fh.read(ndim * _DIM_BYTES)
        if len(raw_dims) != ndim * _DIM_BYTES:
            raise ValueError(f"checkpoint truncated at tensor '{name}'")
        dims = struct.unpack(f"<{ndim}{_DIM_CHAR}", raw_dims) if ndim else ()
        numel = 1
        for dim in dims:
            numel *= dim
        start = fh.tell()
        nbytes = numel * 4
        if start + nbytes > file_size:
            raise ValueError(f"checkpoint truncated at tensor '{name}'")
        fh.seek(start + nbytes)
        yield TensorLayout(name, tuple(dims), start, nbytes)


def _iter_tensors(fh, count, file_size):
    """Yield (name, numel) per tensor, seeking past payloads instead of reading them."""
    for layout in _iter_tensor_layouts(fh, count, file_size):
        yield layout.name, layout.nbytes // 4


def tensor_layout(path):
    """Return (version, raw_metadata_bytes_or_None, TensorLayouts) in file order.

    Only header fields are read; payloads are seeked over. Raises ValueError for
    a bad magic, an unsupported version or a truncated header/entry.
    """
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        version, raw, _ = _parse_header_raw(fh)
        count = _tensor_count(fh)
        layouts = list(_iter_tensor_layouts(fh, count, size))
    return version, raw, layouts


def read_meta_raw(path):
    """Return the exact metadata bytes of a v2 checkpoint, or None for v1."""
    with open(path, "rb") as fh:
        _, raw, _ = _parse_header_raw(fh)
    return raw


def _walk(path):
    """Stream a container; returns (version, meta, tensor_count, param_count)."""
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        version, meta = _parse_header(fh)
        count = _tensor_count(fh)
        params = 0
        seen = 0
        for _, numel in _iter_tensors(fh, count, size):
            params += numel
            seen += 1
    return version, meta, seen, params


def _is_valid_checkpoint(path):
    try:
        _walk(path)
    except (OSError, ValueError):
        return False
    return True


def checkpoint_metadata(path):
    """Return the normalised metadata of a checkpoint.

    Keys: step, optimizer, seed, best_metric, config, format_version. Missing
    values are None. Raises ValueError for files that are not readable
    checkpoints.
    """
    version, meta = _read_header(path)
    meta = meta if isinstance(meta, dict) else {}
    training = meta.get("training")
    training = training if isinstance(training, dict) else {}
    fmt = meta.get("format_version")
    return {
        "step": training.get("step"),
        "optimizer": training.get("optimizer"),
        "seed": training.get("seed"),
        "best_metric": training.get("best_metric"),
        "config": meta.get("config"),
        "format_version": version if fmt is None else fmt,
    }


def checkpoint_step(path):
    """Return the training step recorded in metadata or in the file name.

    Falls back to a ``step_(\\d+)`` substring of the basename when the file
    carries no metadata. Returns None when neither source has a step.
    """
    step = checkpoint_metadata(path)["step"]
    if isinstance(step, int) and not isinstance(step, bool):
        return step
    match = _STEP_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else None


def atomic_write(path, data_bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data_bytes)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def save_dir_layout(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)


def checkpoint_path(save_dir, name):
    return os.path.join(save_dir, "checkpoints", name)


def _read_pointer(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _resolve_recorded(recorded, ckdir, save_dir):
    """Resolve a path recorded in latest.txt/best.txt.

    Tried in order: as recorded (absolute or cwd-relative), relative to
    <dir>/checkpoints/, then relative to <dir>. Returns None if none exist.
    """
    if not recorded:
        return None
    for cand in (recorded, os.path.join(ckdir, recorded), os.path.join(save_dir, recorded)):
        if os.path.isfile(cand):
            return cand
    return None


def _scan(save_dir):
    return sorted(glob.glob(os.path.join(save_dir, "checkpoints", "*.fblt")))


def latest_checkpoint(save_dir):
    """Return the newest usable checkpoint under save_dir, or None.

    Honours <dir>/checkpoints/latest.txt when the path it names still exists
    and parses; otherwise takes the valid *.fblt with the highest
    checkpoint_step (ties broken by the lexicographically greater name).
    Truncated or otherwise unreadable files are never returned.
    """
    ckdir = os.path.join(save_dir, "checkpoints")
    hit = _resolve_recorded(_read_pointer(os.path.join(ckdir, "latest.txt")), ckdir, save_dir)
    if hit is not None and _is_valid_checkpoint(hit):
        return hit

    best_key = None
    best_path = None
    for path in _scan(save_dir):
        if not _is_valid_checkpoint(path):
            continue
        step = checkpoint_step(path)
        key = (-1 if step is None else step, os.path.basename(path))
        if best_key is None or key > best_key:
            best_key = key
            best_path = path
    return best_path


def _metric_better(name, value, path, incumbent):
    cur_value, cur_path = incumbent
    if name in _MAXIMISED_METRICS:
        better = value > cur_value
    else:
        better = value < cur_value
    return better or (value == cur_value and path > cur_path)


def best_checkpoint(save_dir):
    """Return the checkpoint with the best recorded metric under save_dir.

    Honours <dir>/checkpoints/best.txt when the path it names still exists and
    parses. Otherwise scans *.fblt for metadata training.best_metric, minimising
    causal_bpb and maximising masked_acc; causal_bpb wins when both metrics are
    present because it is the trainer's primary metric. Metrics are never
    compared across names.

    Only when no checkpoint carries best-metric metadata does this degenerate
    to latest_checkpoint().
    """
    ckdir = os.path.join(save_dir, "checkpoints")
    hit = _resolve_recorded(_read_pointer(os.path.join(ckdir, "best.txt")), ckdir, save_dir)
    if hit is not None and _is_valid_checkpoint(hit):
        return hit

    incumbents = {}
    for path in _scan(save_dir):
        try:
            metric = checkpoint_metadata(path)["best_metric"]
        except (OSError, ValueError):
            continue
        if not isinstance(metric, dict):
            continue
        name = metric.get("name")
        value = metric.get("value")
        if name not in _METRIC_PREFERENCE or not isinstance(value, (int, float)):
            continue
        value = float(value)
        current = incumbents.get(name)
        if current is None or _metric_better(name, value, path, current):
            incumbents[name] = (value, path)

    for name in _METRIC_PREFERENCE:
        if name in incumbents:
            return incumbents[name][1]
    return latest_checkpoint(save_dir)


def resume_checkpoint(save_dir, mode="auto"):
    """Return the checkpoint to resume from: auto prefers best, then latest."""
    if mode not in ("auto", "latest", "best"):
        raise ValueError(f"unknown resume mode '{mode}' (expected auto, latest or best)")
    if mode == "latest":
        return latest_checkpoint(save_dir)
    if mode == "best":
        return best_checkpoint(save_dir)
    return best_checkpoint(save_dir) or latest_checkpoint(save_dir)


def model_info(path):
    """Return path, size, metadata, tensor/parameter counts and step of a checkpoint.

    The tensor section is walked by seeking over payloads, so memory use does
    not grow with the checkpoint size. A truncated payload raises ValueError.
    """
    size = os.path.getsize(path)
    _, meta, tensor_count, param_count = _walk(path)
    metadata = checkpoint_metadata(path)
    return {
        "path": path,
        "size_bytes": size,
        "meta": meta,
        "tensor_count": tensor_count,
        "param_count": param_count,
        "step": metadata["step"],
        "best_metric": metadata["best_metric"],
    }
