import json
import os
import struct

import pytest

from fblt._checkpoint import (
    atomic_write,
    best_checkpoint,
    checkpoint_metadata,
    checkpoint_step,
    latest_checkpoint,
    model_info,
    read_meta,
    resume_checkpoint,
    save_dir_layout,
)

TENSORS = (("a", (2, 3)), ("b", (4,)))


def make_ckpt(path, meta=None, tensors=TENSORS, version=2, truncate=None):
    data = b"FBLT" + struct.pack("<I", version)
    if version == 2:
        blob = json.dumps({} if meta is None else meta).encode()
        data += struct.pack("<I", len(blob)) + blob
    data += struct.pack("<I", len(tensors))
    for name, dims in tensors:
        raw = name.encode()
        numel = 1
        for dim in dims:
            numel *= dim
        data += struct.pack("<H", len(raw)) + raw + struct.pack("<I", len(dims))
        data += b"".join(struct.pack("<Q", dim) for dim in dims)
        data += b"\0" * (4 * numel)
    if truncate is not None:
        data = data[:truncate]
    path = str(path)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def meta(step=None, metric=None, value=None):
    doc = {"format": "fblt-checkpoint", "format_version": 2, "config": {"embed": 8}}
    training = {"step": step, "optimizer": "adamw", "seed": 7}
    if metric is not None:
        training["best_metric"] = {"name": metric, "value": value, "step": step}
    doc["training"] = training
    return doc


def ckdir(tmp_path):
    save_dir_layout(str(tmp_path))
    return tmp_path / "checkpoints"


def test_atomic_write_roundtrip(tmp_path):
    path = tmp_path / "x.bin"
    atomic_write(str(path), b"hello")
    assert path.read_bytes() == b"hello"
    assert not os.path.exists(str(path) + ".tmp")


def test_latest_honours_pointer(tmp_path):
    d = ckdir(tmp_path)
    old = make_ckpt(d / "step_000001.fblt", meta(1))
    make_ckpt(d / "step_000002.fblt", meta(2))
    (d / "latest.txt").write_text(old + "\n")
    assert latest_checkpoint(str(tmp_path)) == old


def test_latest_stale_pointer_falls_back(tmp_path):
    d = ckdir(tmp_path)
    make_ckpt(d / "step_000007.fblt", meta(7))
    newer = make_ckpt(d / "step_000009.fblt", meta(9))
    (d / "latest.txt").write_text(str(d / "step_000008.fblt"))
    assert latest_checkpoint(str(tmp_path)) == newer


def test_latest_skips_truncated(tmp_path):
    d = ckdir(tmp_path)
    good = make_ckpt(d / "step_000001.fblt", meta(1))
    make_ckpt(d / "step_999999.fblt", meta(999999), truncate=20)
    assert latest_checkpoint(str(tmp_path)) == good


def test_latest_ignores_garbage_pointer(tmp_path):
    d = ckdir(tmp_path)
    good = make_ckpt(d / "step_000003.fblt", meta(3))
    broken = make_ckpt(d / "broken.fblt", meta(4), truncate=4)
    (d / "latest.txt").write_text(broken)
    assert latest_checkpoint(str(tmp_path)) == good


def test_latest_nothing_valid(tmp_path):
    d = ckdir(tmp_path)
    make_ckpt(d / "step_000004.fblt", meta(4), truncate=6)
    assert latest_checkpoint(str(tmp_path)) is None
    assert latest_checkpoint(str(tmp_path / "missing")) is None


def test_best_honours_pointer(tmp_path):
    d = ckdir(tmp_path)
    picked = make_ckpt(d / "step_000001.fblt", meta(1, "causal_bpb", 2.0))
    make_ckpt(d / "step_000002.fblt", meta(2, "causal_bpb", 0.5))
    (d / "best.txt").write_text(str(picked))
    assert best_checkpoint(str(tmp_path)) == picked


def test_best_picks_lowest_causal_bpb(tmp_path):
    d = ckdir(tmp_path)
    make_ckpt(d / "step_000001.fblt", meta(1, "causal_bpb", 1.2))
    best = make_ckpt(d / "step_000002.fblt", meta(2, "causal_bpb", 0.9))
    make_ckpt(d / "step_000003.fblt", meta(3))
    assert best_checkpoint(str(tmp_path)) == best


def test_best_picks_highest_masked_acc(tmp_path):
    d = ckdir(tmp_path)
    make_ckpt(d / "step_000001.fblt", meta(1, "masked_acc", 0.5))
    best = make_ckpt(d / "step_000002.fblt", meta(2, "masked_acc", 0.7))
    assert best_checkpoint(str(tmp_path)) == best


def test_best_prefers_causal_bpb_over_masked_acc(tmp_path):
    d = ckdir(tmp_path)
    ckpt = make_ckpt(d / "step_000001.fblt", meta(1, "causal_bpb", 1.1))
    make_ckpt(d / "step_000002.fblt", meta(2, "masked_acc", 0.99))
    assert best_checkpoint(str(tmp_path)) == ckpt


def test_best_without_metric_degenerates_to_latest(tmp_path):
    d = ckdir(tmp_path)
    make_ckpt(d / "step_000001.fblt", meta(1))
    latest = make_ckpt(d / "step_000002.fblt", meta(2))
    assert best_checkpoint(str(tmp_path)) == latest


def test_checkpoint_step_sources(tmp_path):
    named = make_ckpt(tmp_path / "step_000042.fblt", meta())
    assert checkpoint_step(named) == 42
    stamped = make_ckpt(tmp_path / "weights.fblt", meta(120))
    assert checkpoint_step(stamped) == 120
    plain = make_ckpt(tmp_path / "weights2.fblt", meta())
    assert checkpoint_step(plain) is None


def test_read_meta_version_1_returns_none(tmp_path):
    v1 = make_ckpt(tmp_path / "v1.fblt", version=1)
    assert read_meta(v1) is None
    assert checkpoint_metadata(v1)["format_version"] == 1


def test_read_meta_bad_magic(tmp_path):
    path = make_ckpt(tmp_path / "bad.fblt", meta(1))
    with open(path, "wb") as fh:
        fh.write(b"XXXX" + b"\0" * 12)
    with pytest.raises(ValueError, match="bad magic"):
        read_meta(path)


def test_read_meta_newer_version(tmp_path):
    path = make_ckpt(tmp_path / "new.fblt", meta(1), version=3)
    with pytest.raises(ValueError, match="newer than this reader"):
        read_meta(path)


def test_read_meta_truncated_metadata(tmp_path):
    path = make_ckpt(tmp_path / "cut.fblt", meta(1), truncate=12)
    with pytest.raises(ValueError):
        read_meta(path)


def test_model_info_counts(tmp_path):
    path = make_ckpt(tmp_path / "step_000120.fblt", meta(120, "causal_bpb", 0.98))
    info = model_info(path)
    assert info["tensor_count"] == 2
    assert info["param_count"] == 10
    assert info["step"] == 120
    assert info["best_metric"]["name"] == "causal_bpb"
    assert info["meta"]["format"] == "fblt-checkpoint"
    assert info["size_bytes"] == os.path.getsize(path)


def test_model_info_truncated_payload(tmp_path):
    make_ckpt(tmp_path / "full.fblt", meta(120))
    cut = tmp_path / "cut.fblt"
    cut.write_bytes((tmp_path / "full.fblt").read_bytes()[:-2])
    with pytest.raises(ValueError, match="truncated at tensor 'b'"):
        model_info(str(cut))


def test_resume_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="unknown resume mode"):
        resume_checkpoint(str(tmp_path), "maybe")
