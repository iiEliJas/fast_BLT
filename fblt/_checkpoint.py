import glob
import json
import os
import struct
import typing


def read_meta(path: str) -> typing.Optional[dict]:
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != b"FBLT":
            return None
        (version,) = struct.unpack("<I", fh.read(4))
        if version == 1:
            return None
        if version != 2:
            raise ValueError(f"unsupported checkpoint version {version}")
        (meta_len,) = struct.unpack("<I", fh.read(4))
        if meta_len > 1024 * 1024:
            raise ValueError("meta_len too large")
        meta_bytes = fh.read(meta_len)
        if len(meta_bytes) != meta_len:
            raise ValueError("truncated metadata")
        text = meta_bytes.decode("utf-8")
        return json.loads(text) if text else {}


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


def latest_checkpoint(save_dir):
    ckdir = os.path.join(save_dir, "checkpoints")
    if not os.path.isdir(ckdir):
        return None
    cands = sorted(glob.glob(os.path.join(ckdir, "*.fblt")))
    return cands[-1] if cands else None


def best_checkpoint(save_dir):
    # best by meta "best_loss" if present else latest
    ckdir = os.path.join(save_dir, "checkpoints")
    if not os.path.isdir(ckdir):
        return None
    best = None
    best_loss = None
    for p in glob.glob(os.path.join(ckdir, "*.fblt")):
        try:
            m = read_meta(p)
        except Exception:
            m = None
        loss = None
        if isinstance(m, dict):
            loss = m.get("best_loss", m.get("loss"))
        if loss is not None and (best_loss is None or loss < best_loss):
            best_loss = loss
            best = p
    if best is None:
        cands = sorted(glob.glob(os.path.join(ckdir, "*.fblt")))
        return cands[-1] if cands else None
    return best


def resume_checkpoint(save_dir, mode="auto"):
    if mode == "latest":
        return latest_checkpoint(save_dir)
    if mode == "best":
        return best_checkpoint(save_dir)
    # auto: prefer best if exists else latest
    b = best_checkpoint(save_dir)
    return b if b else latest_checkpoint(save_dir)


def model_info(path):
    meta = read_meta(path)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    return {"path": path, "size_bytes": size, "meta": meta}
